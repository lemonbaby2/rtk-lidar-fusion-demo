#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the fused point cloud to an MP4 with a GPU z-buffer rasteriser.

Layout per frame, 1280x720:

  left    chase view of the accumulated coloured map.  Points are rasterised with
          a torch z-buffer (scatter_reduce amin) so nearer surfaces actually win,
          supersampled 2x and splatted 2x so the cloud does not look like noise.
  right   camera 02 frame with this frame's LiDAR returns projected onto it and
          coloured by depth, i.e. the visual/lidar fusion itself, above
  right   a top-down map with the odometry, RTK and fused trajectories, the RTK
          outage window, and the live error numbers.

Only about 17 % of a 360 deg sweep can be coloured at all: the colour cameras look
forward over roughly 81 deg, and 02/03 are a stereo pair facing the same way, so
the second one only adds parallax margin.  The rest of the sweep keeps grey, which
is why the chase view is largely grey and the camera panel exists.

  python render_video.py            # -> out/frames/*.png, out/fusion_demo.mp4
  python render_video.py --stress 25 # sustained load, reports GPU utilisation
"""
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import torch

OUT = os.environ.get("FUSE_OUT", "/home/nvidia/rtk_fusion_demo/out")
RAW = os.environ.get("FUSE_RAW", "/home/nvidia/rtk_fusion_demo/raw")
DATA = os.environ.get("FUSE_DATA", "/home/nvidia/rtk_fusion_demo/data")
NPZ = f"{OUT}/fusion_result.npz"
FRAMES = f"{OUT}/frames"
VIDEO = f"{OUT}/fusion_demo.mp4"
DRIVE = os.path.join(RAW, "2011_09_26/2011_09_26_drive_0005_sync")

VIEW_W, VIEW_H = 880, 720          # 3D panel
PANEL_W = 400                       # right column
CAM_H = 121                         # camera panel height (1242x375 aspect)
MAP_Y = CAM_H + 8                   # top-down panel starts here
MAP_H = VIEW_H - MAP_Y
W, H = VIEW_W + PANEL_W, VIEW_H
FPS = 20
FOV_DEG = 62.0
SS = 2                              # supersample factor for the 3D view
NEAR_M = 55.0                       # points kept for the chase view
CHASE_BACK, CHASE_UP, CHASE_AHEAD = 13.0, 6.5, 18.0
DEPTH_MAX = 60.0                    # depth colour range on the camera panel
GAIN = 1.25                         # display-only brightening; KITTI is underexposed
BG = (14, 16, 22)                   # BGR, dark so the coloured points read well
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# GPU rasteriser
# --------------------------------------------------------------------------- #
def look_at(eye, target, up):
    """World -> camera basis. +z forward, +x right, +y down (image convention)."""
    zc = target - eye
    zc = zc / zc.norm()
    xc = torch.cross(zc, up, dim=0)
    xc = xc / xc.norm()
    yc = torch.cross(zc, xc, dim=0)
    return torch.stack([xc, yc, zc])            # rows


def rasterise(xyz, rgb, eye, basis, f, w, h, splat=1):
    """Project points and resolve occlusion with a z-buffer on the GPU.

    `splat` paints each surviving point into a splat*splat block.  One pixel per
    point leaves the frame looking like noise at this density, and enlarging
    afterwards would merge colours across depth edges; splatting before the
    z-buffer keeps the nearest surface winning.
    """
    rel = xyz - eye
    cam = rel @ basis.T                          # (N,3) in camera axes
    z = cam[:, 2]
    front = z > 0.4
    cam, rgb, z = cam[front], rgb[front], z[front]

    u = f * cam[:, 0] / z + w * 0.5
    v = f * cam[:, 1] / z + h * 0.5
    inside = (u >= 0) & (u < w - splat) & (v >= 0) & (v < h - splat)
    u, v, z, rgb = u[inside].long(), v[inside].long(), z[inside], rgb[inside]
    if z.numel() == 0:
        return None, None

    base = v * w + u
    if splat == 1:
        idx, zz = base, z
    else:
        offs = torch.tensor([dy * w + dx for dx in range(splat) for dy in range(splat)],
                            device=z.device, dtype=torch.long).reshape(-1, 1)
        idx = (base.unsqueeze(0) + offs).reshape(-1)
        zz = z.repeat(offs.shape[0])
        rgb = rgb.repeat(offs.shape[0], 1)

    zbuf = torch.full((w * h,), 1e9, device=xyz.device, dtype=z.dtype)
    zbuf.scatter_reduce_(0, idx, zz, reduce="amin", include_self=True)

    nearest = zbuf[idx] <= zz + 1e-4
    idx, rgb = idx[nearest], rgb[nearest]

    img = torch.empty((w * h, 3), device=xyz.device, dtype=torch.float32)
    img[:] = torch.tensor(BG[::-1], device=xyz.device, dtype=torch.float32)
    img[idx] = rgb
    return img.reshape(h, w, 3), int(nearest.sum())


# --------------------------------------------------------------------------- #
# camera panel: LiDAR returns projected into the colour image
# --------------------------------------------------------------------------- #
def load_projection():
    """P_rect_02 @ R_rect_00 @ Tr_velo_to_cam, straight from the calibration files."""
    base = os.path.join(RAW, "2011_09_26")

    def mat(path, key, shape):
        for line in open(os.path.join(base, path)):
            if line.startswith(key + ":"):
                v = np.array([float(x) for x in line.split(":")[1].split()])
                return v.reshape(shape)
        raise KeyError(key)

    R_v2c = mat("calib_velo_to_cam.txt", "R", (3, 3))
    T_v2c = mat("calib_velo_to_cam.txt", "T", (3,))
    R_rect = mat("calib_cam_to_cam.txt", "R_rect_00", (3, 3))
    P2 = mat("calib_cam_to_cam.txt", "P_rect_02", (3, 4))
    Tr = np.eye(4)
    Tr[:3, :3] = R_v2c
    Tr[:3, 3] = T_v2c
    Rr = np.eye(4)
    Rr[:3, :3] = R_rect
    return P2 @ Rr @ Tr


_TURBO = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                           cv2.COLORMAP_TURBO).reshape(-1, 3)


def camera_panel(img_path, xyz, proj, w_out=PANEL_W):
    """Photo with the current frame's returns painted on top, coloured by depth."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return np.full((CAM_H, w_out, 3), BG, dtype=np.uint8)
    ih, iw = img.shape[:2]
    h_out = int(round(w_out * ih / iw))
    panel = cv2.resize(img, (w_out, h_out), interpolation=cv2.INTER_AREA)

    xyz = np.asarray(xyz, dtype=np.float64)
    d = np.linalg.norm(xyz, axis=1)
    keep = d < DEPTH_MAX
    xyz, d = xyz[keep], d[keep]
    h4 = np.hstack([xyz, np.ones((len(xyz), 1))])
    uv = (proj @ h4.T).T
    z = uv[:, 2]
    ok = z > 1e-3
    u = (uv[ok, 0] / z[ok]) * (w_out / iw)
    v = (uv[ok, 1] / z[ok]) * (h_out / ih)
    inside = (u >= 0) & (u < w_out - 1) & (v >= 0) & (v < h_out - 1)
    ui = u[inside].astype(np.int32)
    vi = v[inside].astype(np.int32)
    dd = d[ok][inside]
    col = _TURBO[np.clip(dd / DEPTH_MAX * 255, 0, 255).astype(np.int32)]
    for dy in (0, 1):
        for dx in (0, 1):
            panel[np.clip(vi + dy, 0, h_out - 1), np.clip(ui + dx, 0, w_out - 1)] = col

    cv2.putText(panel, "camera 02 + lidar, colour = depth (0-60 m)",
                (8, h_out - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1,
                cv2.LINE_AA)
    return panel


# --------------------------------------------------------------------------- #
# top-down panel
# --------------------------------------------------------------------------- #
def draw_topdown(p_rtk, p_odo, p_fused, i, drop, w, h, scale, origin, err_o, err_f):
    """Bird's-eye view of the three trajectories with the outage shaded."""
    img = np.full((h, w, 3), BG, dtype=np.uint8)

    def to_px(p):
        """World metres -> panel pixels.  Accepts a single point or an (N,3) trail."""
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        x = (p[:, 0] - origin[0]) * scale + w * 0.5
        y = (p[:, 1] - origin[1]) * scale + h * 0.5
        return np.stack([x, h - y - 1], axis=-1)

    def poly(p, col, width=2, upto=None, dotted=False):
        q = to_px(p if upto is None else p[:upto]).astype(np.int32)
        if len(q) < 2:
            return
        if dotted:
            for k in range(0, len(q) - 1, 3):
                cv2.line(img, tuple(q[k]), tuple(q[min(k + 1, len(q) - 1)]), col, width,
                         cv2.LINE_AA)
        else:
            cv2.polylines(img, [q], False, col, width, cv2.LINE_AA)

    step = 20.0 * scale
    if step > 8:
        ox, oy = to_px(origin)[0]
        k = 1
        while ox - k * step > 0:
            cv2.line(img, (int(ox - k * step), 0), (int(ox - k * step), h), (34, 36, 44), 1)
            k += 1
        k = 1
        while ox + k * step < w:
            cv2.line(img, (int(ox + k * step), 0), (int(ox + k * step), h), (34, 36, 44), 1)
            k += 1
        k = 1
        while oy - k * step > 0:
            cv2.line(img, (0, int(oy - k * step)), (w, int(oy - k * step)), (34, 36, 44), 1)
            k += 1
        k = 1
        while oy + k * step < h:
            cv2.line(img, (0, int(oy + k * step)), (w, int(oy + k * step)), (34, 36, 44), 1)
            k += 1

    d0, d1 = int(drop[0]), int(drop[1])
    poly(p_rtk, (120, 200, 120), 2, upto=d0 + 1, dotted=True)      # RTK truth
    poly(p_rtk, (90, 110, 90), 2, upto=d1 + 1, dotted=True)        # outage arc
    poly(p_odo, (150, 150, 150), 2)                                 # odometry drift
    poly(p_fused, (60, 60, 235), 3)                                 # fused

    cur = to_px(p_fused[i])[0].astype(int)
    cv2.circle(img, tuple(cur), 6, (60, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(cur), 9, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "TOP-DOWN  (20 m grid)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (215, 215, 215), 1, cv2.LINE_AA)
    for k, (lab, col) in enumerate([("fused (RTK+ICP)", (60, 60, 235)),
                                    ("odometry only", (150, 150, 150)),
                                    ("RTK/INS truth", (120, 200, 120))]):
        y = 46 + k * 19
        cv2.line(img, (10, y - 4), (30, y - 4), col, 3, cv2.LINE_AA)
        cv2.putText(img, lab, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (205, 205, 205), 1, cv2.LINE_AA)

    outage = d0 <= i < d1
    cv2.putText(img, "RTK OUTAGE" if outage else "RTK OK", (10, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                (60, 180, 255) if outage else (120, 220, 120), 2, cv2.LINE_AA)
    cv2.putText(img, f"ATE odo   {err_o[i]:5.2f} m", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (175, 175, 175), 1, cv2.LINE_AA)
    cv2.putText(img, f"ATE fused {err_f[i]:5.2f} m", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 255), 1, cv2.LINE_AA)
    return img


class GpuSampler(threading.Thread):
    """Best-effort utilisation sampler; GB10 may not report every field."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4).stdout.strip()
                if out.split("\n")[0].strip().isdigit():
                    self.samples.append(int(out.split("\n")[0]))
            except Exception:
                pass
            time.sleep(0.25)


def load_inputs():
    d = np.load(NPZ)
    return {
        "world": torch.from_numpy(d["world"]).to(DEV),
        "wrgb": (torch.from_numpy(d["world_rgb"]).to(DEV).float() * GAIN).clamp(0, 255),
        "offsets": d["frame_offsets"],
        "rot": torch.from_numpy(d["rot_rtk"]).to(DEV),
        "p_rtk": d["p_rtk"], "p_odo": d["p_odo"], "p_fused": d["p_fused"],
        "err_odo": d["err_odo"], "err_fused": d["err_fused"],
        "drop": d["drop_window"],
        "n": len(d["frame_offsets"]) - 1,
    }


def stress(seconds=25.0):
    """Sustained rasterisation load, to report what the GPU does at peak.

    A 300 k point chase view finishes in 20 ms, far too short for nvidia-smi to
    show more than a few percent, and reporting that as the hardware ceiling would
    be misleading.  This pushes the whole accumulated cloud through the same
    rasteriser back to back and samples utilisation throughout.
    """
    g = load_inputs()
    world, wrgb = g["world"], g["wrgb"]
    f = 0.5 * VIEW_W * SS / np.tan(np.deg2rad(FOV_DEG) * 0.5)
    up_world = torch.tensor([0.0, 0.0, 1.0], device=DEV)

    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    frames = 0
    while time.time() - t0 < seconds:
        i = frames % g["n"]
        R = g["rot"][i]
        fwd = R @ torch.tensor([1.0, 0.0, 0.0], device=DEV)
        up = R @ up_world
        ego = torch.from_numpy(g["p_fused"][i]).to(DEV).float()
        eye = ego - CHASE_BACK * fwd + CHASE_UP * up
        basis = look_at(eye, ego + CHASE_AHEAD * fwd, up)
        rasterise(world, wrgb, eye, basis, f, VIEW_W * SS, VIEW_H * SS, splat=2)
        frames += 1
        torch.cuda.synchronize()
    sampler.stop = True
    sampler.join(timeout=3)
    el = time.time() - t0
    out = {
        "seconds": round(el, 1), "frames": frames,
        "points_per_frame": int(world.shape[0]),
        "raster_points_per_second": int(frames * world.shape[0] / el),
        "gpu_util_mean_pct": round(float(np.mean(sampler.samples)), 1) if sampler.samples else None,
        "gpu_util_max_pct": int(np.max(sampler.samples)) if sampler.samples else None,
        "samples": len(sampler.samples),
    }
    json.dump(out, open(f"{OUT}/gpu_stress.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--stress":
        stress(float(sys.argv[2]) if len(sys.argv) > 2 else 25.0)
        return

    os.makedirs(FRAMES, exist_ok=True)
    g = load_inputs()
    world, wrgb, rot = g["world"], g["wrgb"], g["rot"]
    p_rtk, p_odo, p_fused = g["p_rtk"], g["p_odo"], g["p_fused"]
    err_o, err_f, drop, n = g["err_odo"], g["err_fused"], g["drop"], g["n"]
    print(f"device={DEV} frames={n} global_points={world.shape[0]}", flush=True)

    proj = load_projection()
    PTS = np.load(os.path.join(DATA, "pts.npy"))     # per-frame lidar, 37 MB
    allp = np.concatenate([p_rtk, p_odo, p_fused])
    lo, hi = allp.min(0), allp.max(0)
    origin = (lo + hi) * 0.5
    span = max(hi[0] - lo[0], hi[1] - lo[1], 20.0) * 1.15
    scale = min(PANEL_W, MAP_H) / span

    f = 0.5 * VIEW_W * SS / np.tan(np.deg2rad(FOV_DEG) * 0.5)
    up_world = torch.tensor([0.0, 0.0, 1.0], device=DEV)

    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    shown = 0
    canvas = None
    for i in range(n):
        hi_i = int(g["offsets"][i + 1])
        xyz, rgb = world[:hi_i], wrgb[:hi_i]

        R = rot[i]
        fwd = R @ torch.tensor([1.0, 0.0, 0.0], device=DEV)
        up = R @ up_world
        ego = torch.from_numpy(p_fused[i]).to(DEV).float()
        eye = ego - CHASE_BACK * fwd + CHASE_UP * up
        basis = look_at(eye, ego + CHASE_AHEAD * fwd, up)

        near = torch.linalg.norm(xyz - ego, dim=1) < NEAR_M
        view, res = rasterise(xyz[near], rgb[near], eye, basis, f,
                              VIEW_W * SS, VIEW_H * SS, splat=2)
        if view is None:
            view_np = np.full((VIEW_H, VIEW_W, 3), BG, dtype=np.uint8)
        else:
            hi_res = view.clamp(0, 255).to(torch.uint8).cpu().numpy()[:, :, ::-1]
            view_np = cv2.resize(hi_res, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
            shown = res

        cam = camera_panel(os.path.join(DRIVE, "image_02", "data", f"{i:010d}.png"),
                           PTS[i], proj)

        top = draw_topdown(p_rtk, p_odo, p_fused, i, drop, PANEL_W, MAP_H, scale,
                           origin, err_o, err_f)
        right = np.full((VIEW_H, PANEL_W, 3), BG, dtype=np.uint8)
        right[:CAM_H] = cam[:CAM_H]
        right[MAP_Y:] = top

        canvas = np.hstack([view_np, right])
        cv2.putText(canvas, "RTK + LiDAR + camera fusion  |  KITTI raw drive_0005",
                    (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, f"frame {i+1:03d}/{n}   points in view {shown}   "
                           f"map {hi_i}",
                    (16, VIEW_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (190, 190, 190), 1, cv2.LINE_AA)
        cv2.imwrite(f"{FRAMES}/f{i:04d}.png", canvas)
        if i % 40 == 0:
            print(f"  {i}/{n}  {shown} pts  {(time.time()-t0):.1f}s", flush=True)
    sampler.stop = True
    sampler.join(timeout=3)
    dt = time.time() - t0

    for k in range(FPS * 2):                    # hold the last frame
        cv2.imwrite(f"{FRAMES}/f{n + k:04d}.png", canvas)

    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{FRAMES}/f%04d.png",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", VIDEO],
                   check=True, capture_output=True)
    cv2.imwrite(f"{OUT}/poster.png", canvas)

    stats = {
        "frames": n, "fps": FPS, "resolution": [W, H],
        "render_seconds": round(dt, 1),
        "seconds_per_frame": round(dt / n, 3),
        "gpu_util_samples": len(sampler.samples),
        "gpu_util_mean_pct": round(float(np.mean(sampler.samples)), 1) if sampler.samples else None,
        "gpu_util_max_pct": int(np.max(sampler.samples)) if sampler.samples else None,
        "video": VIDEO, "video_mb": round(os.path.getsize(VIDEO) / 1e6, 2),
    }
    json.dump(stats, open(f"{OUT}/render_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
