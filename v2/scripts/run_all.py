#!/usr/bin/env python3
"""Run the whole pipeline in order and write results/summary.json.

Stages are separate processes on purpose: stage 3 holds a Mask R-CNN on the GPU,
stage 6 allocates two occupancy grids, and keeping them isolated means a failure
in one reports its own traceback instead of poisoning a long-lived interpreter.
Each stage writes its own ``stageN_*.json``; this script only collects them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGES = [
    ("stage1_graph", "run_stage1.py",
     "LIO registration + GTSAM factor graph, with ablation", ()),
    ("stage2_map", "run_stage2_map.py",
     "incremental map store: atomic writes, journal, region loading, LOD", ()),
    ("stage3_vision", "run_stage3_vision.py",
     "dynamic masks, semantic cost, stereo depth, visual descriptors",
     ("raw", "media")),
    ("stage4_levels", "run_stage4_levels.py",
     "floor segmentation, per-level bake, global minimap", ()),
    ("stage5_relocal", "run_stage5_relocal.py",
     "Scan Context L1 retrieval + multi-hypothesis verification", ()),
    ("stage6_nav", "run_stage6_nav.py",
     "coarse/fine occupancy layers and hierarchical planning", ()),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--raw", default="/home/nvidia/rtk_fusion_demo/raw")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--media", default="/home/nvidia/rtk_fusion_demo/navstack_v2/media")
    ap.add_argument("--web", default="/home/nvidia/rtk_fusion_demo/navstack_v2/web")
    ap.add_argument("--only", default="", help="comma list of stage ids to run")
    ap.add_argument("--skip", default="", help="comma list of stage ids to skip")
    ap.add_argument("--with-viz", action="store_true", default=True)
    ap.add_argument("--no-viz", dest="with_viz", action="store_false")
    a = ap.parse_args()

    only = {s.strip() for s in a.only.split(",") if s.strip()}
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    os.makedirs(a.out, exist_ok=True)

    common = ["--data", a.data, "--out", a.out]
    times = {}
    failed = {}
    for sid, script, desc, extra in STAGES:
        if only and sid not in only:
            continue
        if sid in skip:
            print(f"== skip {sid} ==", flush=True)
            continue
        print(f"\n{'=' * 70}\n== {sid}: {desc}\n{'=' * 70}", flush=True)
        cmd = [PY, os.path.join(HERE, script)] + common
        # 只有真正声明了 --raw 的脚本才转发，别一刀切（stage2/4/5/6 都不收这个参数）
        if "raw" in extra:
            cmd += ["--raw", a.raw]
        if "media" in extra:
            cmd += ["--media", a.media]
        t0 = time.perf_counter()
        r = subprocess.run(cmd)
        times[sid] = round(time.perf_counter() - t0, 1)
        if r.returncode != 0:
            failed[sid] = r.returncode
            print(f"!! {sid} exited {r.returncode}", flush=True)

    if a.with_viz and not only:
        for sid, script, desc in (("viz_web", "export_web.py", "self-contained viewer"),
                                  ("viz_video", "render_video.py", "BEV + camera video")):
            print(f"\n== {sid}: {desc} ==", flush=True)
            cmd = [PY, os.path.join(HERE, script)] + common
            if script == "export_web.py":
                cmd += ["--web", a.web]
            else:
                cmd += ["--raw", a.raw, "--media", a.media]
            t0 = time.perf_counter()
            subprocess.run(cmd)
            times[sid] = round(time.perf_counter() - t0, 1)

    files = {"stage1": "stage1_ablation.json", "stage2": "stage2_map.json",
             "stage3": "stage3_vision.json", "stage4": "stage4_levels.json",
             "stage5": "stage5_relocal.json", "stage6": "stage6_nav.json",
             "imu_agreement": "imu_lio_agreement.json", "bias_sweep": "tune_bias.json"}
    summary = {"generated_unix": time.time(), "stage_seconds": times,
               "failed": failed, "results": {}}
    for k, fn in files.items():
        p = os.path.join(a.out, fn)
        if os.path.exists(p):
            with open(p) as f:
                summary["results"][k] = json.load(f)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'=' * 70}")
    for k, v in times.items():
        print(f"  {k:22s} {v:7.1f} s")
    if failed:
        print(f"  !! failed: {failed}")
    print(f"  summary -> {a.out}/summary.json")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
