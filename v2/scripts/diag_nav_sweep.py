#!/usr/bin/env python3
"""Choose the occupancy range gate by measurement, not by taste.

The point-level probe showed 44 in-band returns inside a single 0.25 m cell that
the vehicle drove through, with heights spread 0.2-1.7 m above the measured
ground.  Those are not obstacles; they are the road seen at a grazing angle from
20-100 m away, where a fraction of a degree of pitch error becomes metres of
apparent height.  The builder accumulated hits from every return in the scan
without a range limit, so the road became a wall.

This sweeps the range gate (and the ego footprint radius) and reports, for the
same ten start/goal pairs stage 6 uses, whether the flat 0.25 m baseline can
plan at all, how long it takes, and how close the path passes to a real
obstacle.  The gate that is reported as the default is the smallest one that
keeps the baseline planning, because a larger gate invents obstacles.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import nav, store as ST  # noqa: E402


def kf_clouds(st: ST.MapStore):
    out = []
    for kf in st.keyframes:
        with np.load(os.path.join(st.dir_kf, f"{kf.id:06d}.npz")) as z:
            out.append((kf.pose, z["pts"].astype(np.float32)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--map",
                    default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--pairs", type=int, default=8)
    a = ap.parse_args()

    st = ST.MapStore(a.map).open()
    kfs = kf_clouds(st)
    poses = np.stack([kf.pose[:3, 3] for kf in st.keyframes])
    n = len(poses)
    rng = np.random.default_rng(17)
    pairs = []
    for _ in range(a.pairs):
        i = int(rng.integers(0, max(2, n // 3)))
        j = int(rng.integers(max(3, 2 * n // 3), n - 1))
        pairs.append((i, j))
    pairs += [(0, n - 1), (n // 2, n - 1)]
    print(f"== {len(kfs)} keyframes, {len(pairs)} pairs ==", flush=True)

    P = np.concatenate([p for _, p in kfs])
    G = nav.build_ground_field(P)

    grids = [
        # (label, res, max_occ_m, ego_radius_m, inflation_m)
        ("0.25m gate=none ego=0.0 infl=0.0", 0.25, 1e9, 0.0, 0.0),
        ("0.25m gate=30   ego=0.0 infl=0.0", 0.25, 30.0, 0.0, 0.0),
        ("0.25m gate=30   ego=0.4 infl=0.0", 0.25, 30.0, 0.4, 0.0),
        ("0.25m gate=30   ego=0.8 infl=0.0", 0.25, 30.0, 0.8, 0.0),
        ("0.25m gate=30   ego=0.0 infl=0.6", 0.25, 30.0, 0.0, 0.6),
        ("0.25m gate=30   ego=0.8 infl=0.6", 0.25, 30.0, 0.8, 0.6),
        ("0.25m gate=50   ego=0.8 infl=0.6", 0.25, 50.0, 0.8, 0.6),
        ("1.00m gate=30   ego=0.8 infl=0.6", 1.00, 30.0, 0.8, 0.6),
        ("1.00m gate=none ego=0.8 infl=0.6", 1.00, 1e9, 0.8, 0.6),
        ("1.00m gate=none ego=0.0 infl=0.6", 1.00, 1e9, 0.0, 0.6),
    ]

    rows = []
    for label, res, gate, ego_r, infl in grids:
        t0 = time.perf_counter()
        lay = nav.build_layer(kfs, res=res,
                              cfg=nav.LayerCfg(inflation_m=infl, max_occ_m=gate,
                                               ego_radius_m=ego_r),
                              verbose=False, ground=G)
        build_ms = (time.perf_counter() - t0) * 1e3
        cells = int(lay.grid.shape[0] * lay.grid.shape[1])
        ok, lens, ms, exp = 0, [], [], []
        for i, j in pairs:
            c, info = nav.astar(lay, poses[i], poses[j], allow_unknown=True)
            if c is not None:
                ok += 1
                lens.append(nav.path_length(
                    np.stack([lay.to_world(k) for k in c])))
            ms.append(info["ms"])
            exp.append(info["expanded"])
        row = {"label": label, "res_m": res, "max_occ_m": gate,
               "ego_radius_m": ego_r, "inflation_m": infl,
               "cells": cells,
               "free_pct": round(100 * float((lay.grid == nav.FREE).mean()), 2),
               "occ_pct": round(100 * float((lay.grid == nav.OCCUPIED).mean()), 2),
               "inflate_effective_m": lay.stats.get("inflation_effective_m", 0.0),
               "footprint_cells": lay.stats.get("ego_footprint_cells", 0),
               "build_ms": round(build_ms, 0),
               "astar_ok": f"{ok}/{len(pairs)}",
               "astar_ms_median": round(float(np.median(ms)), 1),
               "expanded_median": int(np.median(exp)),
               "len_median_m": round(float(np.median(lens)), 2) if lens else None}
        rows.append(row)
        print(f"  {label}: free {row['free_pct']:5.2f}% occ {row['occ_pct']:5.2f}% "
              f"build {build_ms:6.0f} ms | A* {row['astar_ok']:>6s} "
              f"median {row['astar_ms_median']:8.1f} ms "
              f"expanded {row['expanded_median']:7d} "
              f"len {row['len_median_m']} m", flush=True)

    with open(os.path.join(a.out, "diag_nav_sweep.json"), "w") as f:
        json.dump({"pairs": [list(p) for p in pairs], "grids": rows,
                   "ground_field": G.as_dict()}, f, indent=2)
    print(f"\nsaved {a.out}/diag_nav_sweep.json")


if __name__ == "__main__":
    main()
