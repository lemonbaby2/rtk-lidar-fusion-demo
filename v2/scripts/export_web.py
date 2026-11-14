#!/usr/bin/env python3
"""Export the whole result set to one self-contained HTML viewer.

Why one file with embedded data, and a hand-written WebGL renderer rather than
three.js from a CDN: the deliverable has to open by double-clicking it.  A CDN
import fails behind a firewall, and a separate ``.bin`` fetched with ``fetch()``
fails on ``file://`` because of CORS.  Both of those turn "here is my result" into
"here is a blank page", so the data goes in as base64 and the renderer is ~200
lines of WebGL2 with an orbit camera.

The viewer shows, in one scene:
    * the fused map at 0.4 m LOD, coloured from the two KITTI cameras,
    * the estimated trajectory, the RTK reference and the LiDAR-only baseline,
    * the occupancy layer used by the planner, as a floor texture,
    * the hierarchical plan and the flat baseline plan, if stage 6 ran,
plus a side panel with every number the report quotes, read from ``results/*.json``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import bundle as B  # noqa: E402
from navstack import levels, store as ST, tiles as T  # noqa: E402

MAX_POINTS = 130_000


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def load_json(p: str):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--map", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--web", default="/home/nvidia/rtk_fusion_demo/navstack_v2/web")
    ap.add_argument("--max-points", type=int, default=MAX_POINTS)
    a = ap.parse_args()
    os.makedirs(a.web, exist_ok=True)

    b = B.load_bundle(a.data)
    st = ST.MapStore(a.map).open()

    # ---- map cloud at mid LOD -------------------------------------------- #
    print("== cloud ==", flush=True)
    pts, rgb, info = st.load_all(lod="mid")
    if len(pts) > a.max_points:
        rng = np.random.default_rng(4)
        sel = rng.choice(len(pts), a.max_points, replace=False)
        pts, rgb = pts[sel], rgb[sel]
    print(f"  {len(pts)} points (leaf {info['leaf_m']} m)", flush=True)

    # frame0 origin so the JS camera maths stays in small numbers
    org = pts[:, :3].mean(0).astype(np.float64)
    P = (pts[:, :3] - org).astype(np.float32)
    C = rgb.astype(np.uint8)

    # ---- trajectories ----------------------------------------------------- #
    def traj(name: str):
        p = os.path.join(a.out, f"{name}_poses.npy")
        if not os.path.exists(p):
            return None
        T_ = np.load(p)
        return (T_[:, :3, 3] - org).astype(np.float32)

    trajs = {}
    for name, label in (("best", "fused (RTK+LIO)"), ("graph_lio", "LiDAR only"),
                        ("dr", "LIO dead reckoning")):
        t = traj(name)
        if t is not None:
            trajs[name] = {"label": label, "pts": t}
    if "rtk_pos" in b:
        trajs["rtk"] = {"label": "RTK reference",
                        "pts": (b["rtk_pos"] - org).astype(np.float32)}

    # ---- occupancy layer -------------------------------------------------- #
    grid, ginfo = levels.minimap(st, res_m=1.0)
    print(f"  occupancy {grid.shape} @ {ginfo['res_m']} m", flush=True)
    gridb64 = b64((grid * 80).astype(np.uint8))   # 0/80/160 = unknown/observed/occupied
    gorigin = np.array(ginfo["origin"], float) - org[:2]

    # ---- navigation paths ------------------------------------------------- #
    nav6 = load_json(os.path.join(a.out, "stage6_nav.json"))
    paths = {"coarse": None, "fine": None, "flat": None}
    if nav6:
        sample = nav6.get("overlay")
        if sample is None and nav6.get("pairs"):
            ok = [q for q in nav6["pairs"] if q.get("hier_ok")]
            if ok:
                sample = max(ok, key=lambda q: q.get("coarse_len_m") or 0)
        if sample:
            for k in ("coarse", "fine", "flat"):
                if sample.get(k):
                    paths[k] = (np.asarray(sample[k], float) - org[:2]).round(2).tolist()

    # ---- numbers for the panel ------------------------------------------- #
    metrics = {}
    for key, fn in (("stage1", "stage1_ablation.json"), ("stage2", "stage2_map.json"),
                    ("stage3", "stage3_vision.json"), ("stage4", "stage4_levels.json"),
                    ("stage5", "stage5_relocal.json"), ("stage6", "stage6_nav.json")):
        d = load_json(os.path.join(a.out, fn))
        if d is not None:
            metrics[key] = d

    ablation = metrics.get("stage1", {}).get("ablation", [])
    payload = {
        "meta": b["meta"],
        "n_points": int(len(P)),
        "origin": org.round(3).tolist(),
        "grid": {"shape": list(grid.shape), "res": ginfo["res_m"],
                 "origin": gorigin.round(2).tolist(), "b64": gridb64},
        "cloud": {"xyz_b64": b64(P), "rgb_b64": b64(C), "n": int(len(P))},
        "trajs": {k: {"label": v["label"], "b64": b64(v["pts"]), "n": int(len(v["pts"]))}
                  for k, v in trajs.items()},
        "paths": paths,
        "ablation": ablation,
        "metrics": metrics,
    }

    # binary copies too, for anyone who wants the data rather than the picture
    P.tofile(os.path.join(a.web, "cloud_mid_xyz.f32"))
    C.tofile(os.path.join(a.web, "cloud_mid_rgb.u8"))
    with open(os.path.join(a.web, "viewer_data.json"), "w") as f:
        json.dump({k: v for k, v in payload.items() if k != "metrics"}, f)

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    path = os.path.join(a.web, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(path)
    print(f"  wrote {path} ({size / 1e6:.2f} MB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>navstack v2 — 大地图与多楼层导航栈</title>
<style>
  :root{--bg:#0d1117;--panel:#151b23;--line:#243040;--fg:#e6edf3;--dim:#8b98a8;
        --acc:#58a6ff;--red:#f85149;--grn:#3fb950;--yel:#d29922;--pur:#bc8cff}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
            font:13px/1.55 -apple-system,"Segoe UI","Noto Sans SC",sans-serif;overflow:hidden}
  #app{display:flex;height:100%}
  #view{flex:1;position:relative;min-width:0}
  canvas{display:block;width:100%;height:100%;cursor:grab}
  canvas:active{cursor:grabbing}
  #side{width:380px;flex:0 0 380px;background:var(--panel);border-left:1px solid var(--line);
        overflow-y:auto;padding:14px 16px}
  h1{font-size:15px;margin:0 0 2px;letter-spacing:.2px}
  h2{font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:var(--dim);
     margin:16px 0 6px;font-weight:600}
  .sub{color:var(--dim);font-size:11px;margin-bottom:10px}
  table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
  th,td{text-align:right;padding:2px 4px;font-size:11.5px;border-bottom:1px solid #1d2530}
  th:first-child,td:first-child{text-align:left}
  th{color:var(--dim);font-weight:500}
  .best{color:var(--grn);font-weight:600}
  .k{display:flex;justify-content:space-between;padding:2px 0;font-size:12px}
  .k span:first-child{color:var(--dim)}
  .k b{font-variant-numeric:tabular-nums;font-weight:600}
  .hint{position:absolute;left:14px;bottom:12px;color:var(--dim);font-size:11px;
        background:rgba(13,17,23,.7);padding:6px 9px;border-radius:6px;
        border:1px solid var(--line)}
  .legend{position:absolute;left:14px;top:12px;background:rgba(13,17,23,.78);
          border:1px solid var(--line);border-radius:8px;padding:9px 11px;font-size:11.5px}
  .legend div{display:flex;align-items:center;gap:7px;margin:2px 0}
  .sw{width:16px;height:3px;border-radius:2px}
  #stats{position:absolute;right:12px;top:12px;font-size:11px;color:var(--dim);
         background:rgba(13,17,23,.7);padding:5px 8px;border-radius:6px;
         border:1px solid var(--line);font-variant-numeric:tabular-nums}
  .note{color:var(--dim);font-size:11px;margin-top:5px;line-height:1.45}
  .tag{display:inline-block;padding:0 5px;border-radius:4px;font-size:10px;
       border:1px solid var(--line);color:var(--dim);margin-left:4px}
</style>
</head>
<body>
<div id="app">
  <div id="view">
    <canvas id="c"></canvas>
    <div class="legend" id="legend"></div>
    <div class="hint">拖拽旋转 · 滚轮缩放 · 右键/Shift 拖拽平移 · 空格切换视图</div>
    <div id="stats"></div>
  </div>
  <div id="side">
    <h1>navstack v2</h1>
    <div class="sub" id="sub"></div>
    <div id="panel"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('c');
const gl = cv.getContext('webgl2', {antialias:true});
if(!gl){document.body.innerHTML='<p style="padding:20px">需要 WebGL2</p>';}

function b64f32(s){const b=atob(s),n=b.length,a=new Uint8Array(n);
  for(let i=0;i<n;i++)a[i]=b.charCodeAt(i);return new Float32Array(a.buffer);}
function b64u8(s){const b=atob(s),n=b.length,a=new Uint8Array(n);
  for(let i=0;i<n;i++)a[i]=b.charCodeAt(i);return a;}

const P = b64f32(DATA.cloud.xyz_b64), COL = b64u8(DATA.cloud.rgb_b64);
const N = DATA.cloud.n;
const TRAJ = {};
for(const k in DATA.trajs) TRAJ[k] = b64f32(DATA.trajs[k].b64);

const VS = `#version 300 es
in vec3 aPos; in vec4 aCol; uniform mat4 uMVP; uniform float uSize;
out vec4 vCol;
void main(){ gl_Position = uMVP*vec4(aPos,1.0); gl_PointSize = uSize; vCol = aCol; }`;
const FS = `#version 300 es
precision mediump float; in vec4 vCol; out vec4 o;
void main(){ o = vCol; }`;

function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
  if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o;}
const prog = gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS));
gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(prog);
if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw gl.getProgramInfoLog(prog);
gl.useProgram(prog);
const aPos = gl.getAttribLocation(prog,'aPos'), aCol = gl.getAttribLocation(prog,'aCol');
const uMVP = gl.getUniformLocation(prog,'uMVP'), uSize = gl.getUniformLocation(prog,'uSize');

function buf(arr){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
  gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);return b;}
const bPos = buf(P), bCol = buf(COL);
const tPos = {}, tCol = {};
function trajBuff(k,hex){
  const a=TRAJ[k]; tPos[k]=buf(a);
  const c=new Uint8Array(a.length/3*4);
  for(let i=0;i<a.length/3;i++){c[i*4]=hex[0];c[i*4+1]=hex[1];c[i*4+2]=hex[2];c[i*4+3]=255;}
  tCol[k]=buf(c);
}

// ---- occupancy floor + planned paths ---------------------------------- #
// The ground height is taken from the RTK track rather than from each path's own
// z, which does not exist: stage 6 stores planar paths, and subtracting a 2-vector
// origin from a 3-vector point leaves absolute z (about -1.8 m here) instead of a
// frame-relative one.  Taking z from the track puts the overlay on the road.
function groundZ(){
  const a=TRAJ.rtk||TRAJ.best; if(!a||a.length<3) return 0;
  const zs=[]; for(let i=2;i<a.length;i+=3) zs.push(a[i]);
  zs.sort((p,q)=>p-q); return zs[Math.floor(zs.length/2)];
}
const GZ = groundZ();

let gBufPos=null, gBufCol=null, gCount=0;
if(DATA.grid){
  const gs=DATA.grid.shape, res=DATA.grid.res, o=DATA.grid.origin;
  const raw=b64u8(DATA.grid.b64);
  const pos=[], col=[];
  for(let i=0;i<gs[0];i++)for(let j=0;j<gs[1];j++){
    const v=raw[i*gs[1]+j];
    if(v===0) continue;                       // never observed: leave it empty
    pos.push(o[0]+(i+0.5)*res, o[1]+(j+0.5)*res, GZ-0.03);
    if(v>=160){col.push(214,86,72,255);}else{col.push(34,58,76,255);}
  }
  gCount=pos.length/3;
  if(gCount){
    gBufPos=buf(new Float32Array(pos));
    gBufCol=buf(new Uint8Array(col));
  }
}

const pPos={}, pCol={}, pN={};
const PATH_STYLE={coarse:[150,158,168], fine:[80,200,255], flat:[255,110,220]};
// numpy's tolist() gives [[x,y],...], not a flat [x,y,x,y,...].  Reading it as flat
// yields undefined -> NaN vertices, and NaN vertices are not an error: the draw
// call succeeds and rasterises nothing, so the viewer looked like it had no paths
// while the buffers were "populated" and the legend counted three of them.
function pathVerts(arr){
  if(!arr||!arr.length) return null;
  const m=arr.length, f=new Float32Array(m*3);
  if(Array.isArray(arr[0])){
    for(let i=0;i<m;i++){f[i*3]=arr[i][0];f[i*3+1]=arr[i][1];f[i*3+2]=GZ+0.10;}
  }else{
    for(let i=0;i<m/2;i++){f[i*3]=arr[i*2];f[i*3+1]=arr[i*2+1];f[i*3+2]=GZ+0.10;}
  }
  return f;
}
for(const k in (DATA.paths||{})){
  const f=pathVerts(DATA.paths[k]);
  if(!f || f.length<12) continue;
  const m=f.length/3;
  const c=new Uint8Array(m*4), st=PATH_STYLE[k]||[255,255,255];
  for(let i=0;i<m;i++){c[i*4]=st[0];c[i*4+1]=st[1];c[i*4+2]=st[2];c[i*4+3]=255;}
  pPos[k]=buf(f); pCol[k]=buf(c); pN[k]=m;
}

// ---- camera ----
// Default framing has to show the whole 205 x 174 m map at once: at dist 120 a
// 0.85 rad fov covers ~110 m, so the planned paths (the point of the stage-6
// overlay) fell outside the first view.
const A={yaw:-1.15,pitch:0.92,dist:215,tx:0,ty:0,tz:0};
let persp=true;
function mul(a,b){const o=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
    for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;}
function perspMat(fov,asp,n,f){const t=1/Math.tan(fov/2),o=new Float32Array(16);
  o[0]=t/asp;o[5]=t;o[10]=(f+n)/(n-f);o[11]=-1;o[14]=2*f*n/(n-f);return o;}
function orthoMat(r,n,f){const o=new Float32Array(16);
  o[0]=1/r;o[5]=1/r;o[10]=-2/(f-n);o[14]=-(f+n)/(f-n);o[15]=1;return o;}
function lookAt(eye,ctr,up){
  const z=norm(sub(eye,ctr)), x=norm(cross(up,z)), y=cross(z,x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -dot(x,eye),-dot(y,eye),-dot(z,eye),1]);}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function norm(a){const l=Math.hypot(...a)||1;return [a[0]/l,a[1]/l,a[2]/l];}

function draw(){
  const w=cv.clientWidth,h=cv.clientHeight,dpr=Math.min(devicePixelRatio||1,2);
  if(cv.width!==w*dpr||cv.height!==h*dpr){cv.width=w*dpr;cv.height=h*dpr;}
  gl.viewport(0,0,cv.width,cv.height);
  gl.clearColor(0.051,0.067,0.09,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  const cp=Math.cos(A.pitch), eye=[A.tx+A.dist*cp*Math.cos(A.yaw),
      A.ty+A.dist*cp*Math.sin(A.yaw), A.tz+A.dist*Math.sin(A.pitch)];
  // y-up -> z-up: the map is z-up, so swap axes in the view matrix
  const view = lookAt([eye[0],eye[2],eye[1]],[A.tx,A.tz,A.ty],[0,0,1]);
  const proj = persp ? perspMat(0.85,w/h,0.5,4000) : orthoMat(80,-4000,4000);
  // model: (x,y,z) -> (x,z,y)
  const model = new Float32Array([1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1]);
  const mvp = mul(proj, mul(view, model));
  gl.uniformMatrix4fv(uMVP,false,mvp);

  gl.enableVertexAttribArray(aPos); gl.enableVertexAttribArray(aCol);
  gl.bindBuffer(gl.ARRAY_BUFFER,bPos); gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);
  gl.bindBuffer(gl.ARRAY_BUFFER,bCol); gl.vertexAttribPointer(aCol,4,gl.UNSIGNED_BYTE,true,0,0);
  gl.uniform1f(uSize, persp?2.4:2.0);
  gl.drawArrays(gl.POINTS,0,N);

  if(gBufPos){
    gl.bindBuffer(gl.ARRAY_BUFFER,gBufPos); gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,gBufCol); gl.vertexAttribPointer(aCol,4,gl.UNSIGNED_BYTE,true,0,0);
    gl.uniform1f(uSize, persp?3.0:2.4);
    gl.drawArrays(gl.POINTS,0,gCount);
  }

  // The plan is an annotation, not scene geometry: inside a 130 k-point cloud any
  // depth-tested line is buried by whatever points sit in front of it.  Drawing it
  // with the depth test off is the difference between "the viewer contains the
  // paths" and "you can see the paths", and WebGL ignores lineWidth above 1, so
  // the same vertices are drawn as points to give the line thickness.
  gl.disable(gl.DEPTH_TEST);
  for(const k in pPos){
    gl.bindBuffer(gl.ARRAY_BUFFER,pPos[k]); gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,pCol[k]); gl.vertexAttribPointer(aCol,4,gl.UNSIGNED_BYTE,true,0,0);
    gl.uniform1f(uSize, persp?4.0:3.2);
    gl.drawArrays(gl.POINTS,0,pN[k]);
    gl.uniform1f(uSize,1.0);
    gl.drawArrays(gl.LINE_STRIP,0,pN[k]);
  }
  gl.enable(gl.DEPTH_TEST);
  gl.uniform1f(uSize,1.0);
  for(const k in tPos){
    gl.bindBuffer(gl.ARRAY_BUFFER,tPos[k]); gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,tCol[k]); gl.vertexAttribPointer(aCol,4,gl.UNSIGNED_BYTE,true,0,0);
    gl.drawArrays(gl.LINE_STRIP,0,TRAJ[k].length/3);
  }
  const pn=Object.keys(pN).length;
  document.getElementById('stats').textContent =
    `${N.toLocaleString()} pts · ${gCount?gCount.toLocaleString()+' 格占位 · ':''}` +
    `${pn?pn+' 条规划路径 · ':''}${persp?'透视':'俯视'} · ${A.dist.toFixed(0)} m`;
  requestAnimationFrame(draw);
}
let drag=null;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};});
addEventListener('mouseup',()=>drag=null);
addEventListener('mousemove',e=>{
  if(!drag)return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  drag.x=e.clientX; drag.y=e.clientY;
  if(drag.pan){const s=A.dist*0.0016;
    A.tx-=Math.cos(A.yaw)*dx*s - Math.sin(A.yaw)*dy*s;
    A.ty-=Math.sin(A.yaw)*dx*s + Math.cos(A.yaw)*dy*s;}
  else{A.yaw-=dx*0.006; A.pitch=Math.max(-1.45,Math.min(1.45,A.pitch+dy*0.006));}
});
cv.addEventListener('wheel',e=>{e.preventDefault();
  A.dist=Math.max(8,Math.min(1200,A.dist*Math.exp(e.deltaY*0.0011)));},{passive:false});
cv.addEventListener('contextmenu',e=>e.preventDefault());
addEventListener('keydown',e=>{if(e.code==='Space'){persp=!persp;e.preventDefault();}});

// ---- panel ----
const M=DATA.metrics||{};
const LEG=[['best','#58a6ff'],['rtk','#3fb950'],['graph_lio','#f85149'],['dr','#d29922']];
const PLEG=[['coarse','#969ea8','粗层路径 1.0 m'],['fine','#50c8ff','分层精化 0.25 m'],
            ['flat','#ff6edc','单层基线 0.25 m']];
document.getElementById('legend').innerHTML =
  LEG.filter(([k])=>TRAJ[k]).map(([k,hex])=>{
    const l=(DATA.trajs[k]||{}).label||k;
    return `<div><span class="sw" style="background:${hex}"></span>${l}</div>`;}).join('')
  + (Object.keys(pN).length
      ? PLEG.filter(([k])=>pN[k]).map(([k,hex,label])=>
          `<div><span class="sw" style="background:${hex}"></span>${label}</div>`).join('')
      : '')
  + (gCount ? `<div><span class="sw" style="background:#d65648"></span>占位格（1 m，来自 stage 4 小地图）</div>` : '');

function kv(k,v){return `<div class="k"><span>${k}</span><b>${v}</b></div>`;}
function fmt(x,d=3){return (x===null||x===undefined)?'—':Number(x).toFixed(d);}

let h='';
const meta=DATA.meta||{};
h+=`<h2>数据集</h2>`;
h+=kv('序列','KITTI 2011_09_26_drive_0005');
h+=kv('帧数 / 时长',`${meta.frames} / ${fmt(meta.duration_s,1)} s`);
h+=kv('路径长度',`${fmt(meta.path_length_m,1)} m`);
h+=kv('IMU',`${fmt(meta.imu_hz,0)} Hz`);
h+=kv('立体基线',`${fmt(meta.stereo_baseline_m,3)} m`);
h+=kv('地图点数',`${DATA.n_points.toLocaleString()}`);

const ab=M.stage1&&M.stage1.ablation;
if(ab){
  h+=`<h2>① 因子图消融 <span class="tag">ATE, m</span></h2><table>
    <tr><th>variant</th><th>mean</th><th>rmse</th><th>in gap</th></tr>`;
  const best=Math.min(...ab.filter(r=>!r.name.includes('longgap')&&!r.name.includes('degen'))
                        .map(r=>r.ate_rmse_m));
  for(const r of ab){
    const cls=(r.ate_rmse_m===best)?' class="best"':'';
    h+=`<tr${cls}><td>${r.name}</td><td>${fmt(r.ate_mean_m)}</td>
        <td>${fmt(r.ate_rmse_m)}</td><td>${fmt(r.ate_in_outage_m)}</td></tr>`;
  }
  h+=`</table><div class="note">失锁 40–90 帧；<b>degen_*</b> 为 LiDAR 与 RTK 同时退化 60–90 帧，
      IMU 在此把窗口内误差从 0.952 m 降到 0.069 m。</div>`;
}
const s2=M.stage2;
if(s2){
  h+=`<h2>② 地图存储与按需加载</h2>`;
  h+=kv('关键帧写入',`${fmt(s2.write.keyframe_ms_mean,1)} ms 均值 / ${fmt(s2.write.keyframe_ms_max,1)} ms 峰值`);
  h+=kv('整包写入',`${fmt(s2.write.total_s,1)} s / ${s2.keyframes} 帧`);
  const c=(s2.cache_cold||{}).fine;
  if(c) h+=kv('区域加载(冷,0.25 m)',`${fmt(c.ms_median,2)} ms`);
  h+=kv('区域 vs 全图点数',`${fmt(s2.read.fine.point_ratio_pct,1)}%`);
  h+=kv('区域 vs 全图耗时',`${fmt(s2.read.fine.time_ratio_pct,1)}%`);
  h+=kv('崩溃恢复',`${s2.recovery.recovered}/${s2.recovery.expected_missing} 精确`);
  h+=kv('schema 守卫',s2.schema_guard.refused_newer_schema?'拒绝更新版本':'—');
}
const s3=M.stage3;
if(s3){
  const d=s3.dynamic_mask, r=d.registration;
  h+=`<h2>③ 视觉层</h2>`;
  h+=kv('动态实例检出率',`${fmt(d.instance_detection_rate*100,1)}%`);
  h+=kv('LiDAR 被掩膜点',`${fmt(d.lidar_point_frac_mean*100,2)}%`);
  h+=kv('配准残差(掩膜前→后)',`${fmt(r.rmse_all_median)} → ${fmt(r.rmse_masked_median)} m (${fmt(r.rmse_improve_pct,1)}%)`);
  h+=kv('视觉描述子 R@1',fmt(s3.descriptor.recall_at_1));
  const st=(s3.stereo_vs_lidar||[]).filter(x=>x.median_abs_m);
  if(st.length){const mm=st.reduce((a,x)=>a+x.median_abs_m,0)/st.length;
    h+=kv('立体深度 vs LiDAR',`${fmt(mm,2)} m 中位误差`);}
  if(s3.vision_vote) h+=kv('视觉投票 R@1',`${fmt(s3.vision_vote.lidar_only_recall_at_1)} → ${fmt(s3.vision_vote.fused_recall_at_1)}`);
}
const s4=M.stage4;
if(s4){
  h+=`<h2>④ 多楼层</h2>`;
  h+=kv('真实数据楼层数',s4.real.levels_detected);
  h+=kv('合成双层楼层数',s4.synthetic_duplex.levels_detected);
  h+=kv('楼层分配准确率',fmt(s4.synthetic_duplex.assignment_accuracy));
  const sv=s4.synthetic_duplex;
  h+=kv('按层加载节省点数',`${fmt(sv.level_load_saving_vs_all_pct,1)}%`);
}
const s5=M.stage5;
if(s5){
  const m=s5.metrics;
  h+=`<h2>⑤ 自动重定位</h2>`;
  h+=kv('L1 召回 @1 / @5',`${fmt(m.l1_recall_at_1)} / ${fmt(m.l1_recall_at_5)}`);
  h+=kv('L2 成功率',fmt(m.l2_success_rate));
  h+=kv('位置误差中位',`${fmt(m.trans_err_median_m)} m`);
  h+=kv('L1 / L2 耗时',`${fmt(m.l1_ms_median,2)} / ${fmt(m.l2_ms_median,1)} ms`);
  h+=kv('乱扫拒绝率',fmt(s5.absence_test.rejection_rate));
}
const s6=M.stage6;
if(s6){
  const g=s6.aggregate;
  h+=`<h2>⑥ 分层导航</h2>`;
  h+=kv('成功率（分层/单层）',`${fmt(g.hier_success*100,0)}% / ${fmt(g.flat_success*100,0)}%`);
  h+=kv('分层冷启动 / 热启动',`${fmt(g.hier_cold_ms_median,1)} / ${fmt(g.hier_warm_ms_median,1)} ms`);
  h+=kv('单层 0.25 m 基线',`${fmt(g.flat_total_ms_median,1)} ms（A* 仅 ${fmt(g.flat_astar_ms_median,2)} ms）`);
  h+=kv('加速比（冷 / 热）',`×${fmt(g.speedup_cold_median,2)} / ×${fmt(g.speedup_warm_median,2)}`);
  h+=kv('A* 展开格数（粗层 vs 细层）',`${fmt(g.coarse_expanded_median,0)} vs ${fmt(g.flat_expanded_median,0)}（×${fmt(g.expansion_ratio_coarse_vs_flat,1)}）`);
  h+=kv('路径长度偏差',`${fmt(g.len_delta_mean_pct,2)}%`);
  h+=kv('最小净空（分层 / 单层）',`${fmt(g.hier_clearance_min_m,2)} / ${fmt(g.flat_clearance_min_m,2)} m`);
  h+=kv('落在障碍格内顶点',`${fmt(g.hier_in_obstacle_pct_max,2)}% / ${fmt(g.flat_in_obstacle_pct_max,2)}%`);
  h+=kv('窗口回退率',fmt(g.fallback_window_rate,3));
}
document.getElementById('panel').innerHTML=h;
document.getElementById('sub').textContent =
  `FAST-LIO2 风格前端 + GTSAM 因子图 + x_nav 六主题 · ${(DATA.meta||{}).source||''}`;

requestAnimationFrame(draw);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
