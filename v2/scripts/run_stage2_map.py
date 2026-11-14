#!/usr/bin/env python3
"""Stage 2: write the map keyframe by keyframe, then read it back by region.

Measures the two numbers theme 1 and theme 2 are supposed to buy:

* **write latency per keyframe** (must stay under one frame period, 0.11 s, or
  the mapping run cannot keep up in real time), and
* **the region-load ratio** -- points touched / milliseconds spent, when loading
  a 30 m disc versus loading the whole map at the same LOD.

It also proves the crash story rather than asserting it: the map is copied, the
manifest is rolled back to a mid-run state to imitate a process that died before
its last commit, and ``recover()`` is asked to rebuild the tail from the journal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import bundle as B  # noqa: E402
from navstack import schema as S  # noqa: E402
from navstack import store as ST  # noqa: E402
from navstack import tiles as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--map", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--poses", default="best")
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--queries", type=int, default=24)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    b = B.load_bundle(a.data)
    poses = B.poses_from(a.out, prefer=(a.poses, "graph_full", "best", "dr"))
    if poses is None:
        raise SystemExit("no poses; run stage1 first")
    print(f"== writing map from {len(poses)} poses ==", flush=True)

    if os.path.isdir(a.map):
        shutil.rmtree(a.map)
    st = ST.MapStore(a.map)
    st.create(map_id="kitti_2011_09_26_drive_0005", window=12,
              source="KITTI raw drive 0005, KITTI Odometry convention")

    t0 = time.perf_counter()
    n_flushed = 0
    for kf, pts, rgb in B.make_keyframes(poses, b["pts"], b.get("rgb"), b["frame_t"]):
        st.add_keyframe(kf, pts, rgb)
        st.set_cursor({"frame": kf.seq, "t": kf.t, "keyframes": len(st.keyframes),
                       "done": False})
        if (kf.id + 1) % st.flush_every == 0:
            fl = st.flush_chunks()
            n_flushed += 1
        if kf.id % 30 == 0:
            print(f"  kf {kf.id:3d} seq {kf.seq:3d} {len(pts):5d} pts  "
                  f"write {st.write_log[-1]['t_keyframe_ms']:.1f} ms", flush=True)
    st.set_cursor({"frame": int(poses.shape[0] - 1), "t": float(b["frame_t"][-1]),
                   "keyframes": len(st.keyframes), "done": True})
    fl = st.commit()
    t_write = time.perf_counter() - t0
    print(f"  wrote {len(st.keyframes)} keyframes in {t_write:.1f} s "
          f"({n_flushed + 1} chunk flushes)", flush=True)

    ws = st.write_stats()
    print(f"  keyframe write {ws['keyframe_ms_mean']:.1f} ms mean / "
          f"{ws['keyframe_ms_p95']:.1f} ms p95 / {ws['keyframe_ms_max']:.1f} ms max",
          flush=True)
    print(f"  chunk flush {ws['flush_ms_total']:.0f} ms total over "
          f"{ws['flush_calls']} calls ({ws['flush_files']} LOD files)", flush=True)

    # ---------------------------------------------------------- read back ----
    print("== reopening ==", flush=True)
    st2 = ST.MapStore(a.map).open()
    print(f"  {st2.meta.n_keyframes} keyframes, schema {st2.meta.schema_version}, "
          f"chunks {sum(len(v) for v in st2.level_chunks.values())}", flush=True)

    # --------------------------------------------------- region vs whole ----
    print(f"== region load: {a.radius} m disc, {a.queries} queries ==", flush=True)
    rng = np.random.default_rng(3)
    n = len(st2.keyframes)
    idx = rng.choice(n, size=min(a.queries, n), replace=False)
    rows = []
    for lod in ("coarse", "mid", "fine"):
        for i in idx:
            c = st2.keyframes[i].pose[:3, 3]
            pts, rgb, info = st2.load_region(c, radius=a.radius, lod=lod)
            rows.append({"kind": "region", "lod": lod, **{k: info[k] for k in
                        ("chunks_requested", "chunks_loaded", "points", "ms")}})
        full_pts, _, finfo = st2.load_all(lod=lod)
        rows.append({"kind": "full", "lod": lod, "chunks_requested": -1,
                     "chunks_loaded": -1, "points": finfo["points"],
                     "ms": finfo["ms"]})
        reg = [r for r in rows if r["kind"] == "region" and r["lod"] == lod]
        print(f"  lod {lod:6s}: region {np.median([r['points'] for r in reg]):8.0f} pts / "
              f"{np.median([r['ms'] for r in reg]):7.1f} ms   "
              f"whole map {finfo['points']:8d} pts / {finfo['ms']:7.1f} ms", flush=True)

    summary = {}
    for lod in ("coarse", "mid", "fine"):
        reg = [r for r in rows if r["kind"] == "region" and r["lod"] == lod]
        full = next(r for r in rows if r["kind"] == "full" and r["lod"] == lod)
        summary[lod] = {
            "region_points_median": float(np.median([r["points"] for r in reg])),
            "region_ms_median": round(float(np.median([r["ms"] for r in reg])), 2),
            "region_ms_p95": round(float(np.percentile([r["ms"] for r in reg], 95)), 2),
            "region_chunks_median": float(np.median([r["chunks_loaded"] for r in reg])),
            "full_points": full["points"], "full_ms": full["ms"],
            "point_ratio_pct": round(100.0 * np.median([r["points"] for r in reg])
                                     / max(full["points"], 1), 3),
            "time_ratio_pct": round(100.0 * np.median([r["ms"] for r in reg])
                                    / max(full["ms"], 1e-6), 3),
        }

    # cache behaviour: repeat the same query, then walk a trajectory
    st2.cache.hits = st2.cache.misses = st2.cache.evictions = 0
    for i in idx:
        st2.load_region(st2.keyframes[i].pose[:3, 3], radius=a.radius, lod="fine")
    warm = st2.cache.stats()

    # the warm number above flatters the loader, so cold is measured too: a fresh
    # cache with nothing resident, one region query, I/O time separated out
    print("== cold cache region load ==", flush=True)
    cold = {}
    for lod in ("coarse", "mid", "fine"):
        st2.cache = T.ChunkCache()
        vals, ios = [], []
        for i in idx[:5]:
            _, _, info = st2.load_region(st2.keyframes[int(i)].pose[:3, 3],
                                         radius=a.radius, lod=lod)
            vals.append(info["ms"])
            ios.append(info["io_ms"])
            st2.cache = T.ChunkCache()          # every query starts empty
        cold[lod] = {"ms_median": round(float(np.median(vals)), 2),
                     "io_ms_median": round(float(np.median(ios)), 2),
                     "points_median": float(info["points"])}
        print(f"  lod {lod:6s}: cold {cold[lod]['ms_median']:7.2f} ms "
              f"(of which I/O {cold[lod]['io_ms_median']:7.2f} ms)", flush=True)

    # ------------------------------------------------ chunks are derived ----
    print("== rebuild chunks from keyframes ==", flush=True)
    i0 = int(idx[0])
    p_before, _, _ = st2.load_region(st2.keyframes[i0].pose[:3, 3],
                                    radius=a.radius, lod="fine")
    rb = st2.rebuild_chunks()
    p_after, _, _ = st2.load_region(st2.keyframes[i0].pose[:3, 3],
                                   radius=a.radius, lod="fine")
    identical = (len(p_before) == len(p_after) and
                 np.allclose(p_before, p_after, atol=1e-4)) if len(p_before) else False
    print(f"  rebuilt {rb['chunks']} chunks in {rb['total_ms']:.0f} ms, "
          f"{len(p_before)} -> {len(p_after)} pts, identical={identical}", flush=True)

    # ------------------------------------------------ crash + recover test --
    print("== crash / recover ==", flush=True)
    crash_dir = a.map + "_crash"
    if os.path.isdir(crash_dir):
        shutil.rmtree(crash_dir)
    shutil.copytree(a.map, crash_dir)
    with open(os.path.join(crash_dir, "manifest.json")) as f:
        doc = json.load(f)
    full_kf = len(doc["keyframes"])
    cut = max(1, int(full_kf * 0.6))
    doc["keyframes"] = doc["keyframes"][:cut]
    doc["meta"]["n_keyframes"] = cut
    doc["journal_lines"] = cut
    with open(os.path.join(crash_dir, "manifest.json"), "w") as f:
        json.dump(doc, f, indent=1)
    stc = ST.MapStore(crash_dir).open()
    before = len(stc.keyframes)
    rec = stc.recover()
    after = len(stc.keyframes)
    print(f"  manifest said {before}, journal held {after} "
          f"(recovered {rec['recovered']}, expected {full_kf - cut})", flush=True)
    shutil.rmtree(crash_dir)

    # ---------------------------------------------------------- versioning --
    bad = a.map + "_badver"
    if os.path.isdir(bad):
        shutil.rmtree(bad)
    shutil.copytree(a.map, bad)
    with open(os.path.join(bad, "manifest.json")) as f:
        doc = json.load(f)
    doc["schema_version"] = 99
    with open(os.path.join(bad, "manifest.json"), "w") as f:
        json.dump(doc, f)
    refused = False
    try:
        ST.MapStore(bad).open()
    except RuntimeError as e:
        refused = True
        msg = str(e)
    shutil.rmtree(bad)

    payload = {
        "map_dir": a.map,
        "keyframes": len(st.keyframes),
        "write": {**ws, "total_s": round(t_write, 2),
                  "frame_period_s": round(float(np.median(np.diff(b["frame_t"]))), 4)},
        "read": summary,
        "cache_warm": warm,
        "cache_cold": cold,
        "chunks_are_derived": {"rebuilt": rb, "points_before": int(len(p_before)),
                               "points_after": int(len(p_after)),
                               "identical": bool(identical)},
        "recovery": {"manifest_before": before, "keyframes_after": after,
                     "recovered": rec["recovered"],
                     "expected_missing": full_kf - cut,
                     "exact": bool(after == full_kf)},
        "schema_guard": {"refused_newer_schema": refused,
                         "message": msg if refused else None},
        "config": {"chunk_size_m": S.CHUNK_SIZE_M,
                   "buffer_m": S.CHUNK_BUFFER_M,
                   "lod_leaf_m": T.LOD_LEAF, "radius_m": a.radius},
    }
    with open(os.path.join(a.out, "stage2_map.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  disk {ws['disk_total'] / 1e6:.2f} MB for {len(st.keyframes)} keyframes")
    print(f"saved {a.out}/stage2_map.json")


if __name__ == "__main__":
    main()
