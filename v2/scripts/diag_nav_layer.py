#!/usr/bin/env python3
"""Why does the flat 0.25 m baseline refuse to plan?

Stage 6 measured `flat 0.25 m: expanded 1, 0.1 ms` on every pair, i.e. A* popped
the start cell and found not one passable neighbour.  The stage-6 report blames
the fine layer, but "the fine layer is bad" is not a diagnosis.  This script
measures the actual cell values around the ego pose at each resolution and
answers three concrete questions:

1. What is the z band that the layer builder uses, and is it the band I think it
   is?  (``g = median(pts[:,2])`` on a cloud in a world frame whose ground is not
   at zero is a suspicious choice.)
2. What fraction of cells is OCCUPIED, and what fraction is OCCUPIED *because of
   inflation*?
3. Are the ego cells isolated -- each one set FREE and re-asserted after
   inflation, but with no FREE neighbour, which produces expanded==1 exactly?

Run with no arguments; it prints a table per resolution and writes
``out/diag_nav_layer.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import nav, store as ST  # noqa: E402

SYM = {nav.UNKNOWN: ".", nav.FREE: " ", nav.OCCUPIED: "#"}


def kf_clouds(st: ST.MapStore):
    out = []
    for kf in st.keyframes:
        with np.load(os.path.join(st.dir_kf, f"{kf.id:06d}.npz")) as z:
            out.append((kf.pose, z["pts"]))
    return out


def show(layer, cell, half=6, tag=""):
    i0, j0 = cell
    rows = []
    for di in range(-half, half + 1):
        line = ""
        for dj in range(-half, half + 1):
            i, j = i0 + di, j0 + dj
            if not layer.inside(i, j):
                line += "?"
            else:
                line += SYM[int(layer.grid[i, j])]
        rows.append(line)
    print(f"    {tag} grid around cell {cell} ('#'=occ, ' '=free, '.'=unknown):")
    for k, r in enumerate(rows):
        mark = "  <-- ego row" if k == half else ""
        print(f"      {r}{mark}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--map",
                    default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--inflation", type=float, default=0.60)
    a = ap.parse_args()

    st = ST.MapStore(a.map).open()
    kfs = kf_clouds(st)
    poses = np.stack([kf.pose[:3, 3] for kf in st.keyframes])
    print(f"== {len(kfs)} keyframes, extent {np.ptp(poses[:,0]):.1f} x "
          f"{np.ptp(poses[:,1]):.1f} m ==", flush=True)

    # ---- what does the layer builder think "ground" is? -------------------- #
    print("\n== z statistics of the stored clouds (world frame) ==", flush=True)
    zs, gs, band_frac = [], [], []
    for pose, pts in kfs[:20]:
        z = pts[:, 2]
        g = float(np.median(z))
        zs.append(float(np.percentile(z, 2)))
        gs.append(g)
        band_frac.append(float(np.mean((z - g >= 0.20) & (z - g <= 2.20))))
    print(f"  z p2 (true ground)   median over frames : {np.median(zs):+.3f} m")
    print(f"  median(z) per frame  median over frames : {np.median(gs):+.3f} m")
    print(f"  band [median+0.20, median+2.20] holds    : "
          f"{100*np.median(band_frac):.1f} % of points")
    print(f"  => the band in absolute z is roughly "
          f"[{np.median(gs)+0.20:+.2f}, {np.median(gs)+2.20:+.2f}] m", flush=True)

    payload = {"z_stats": {"ground_p2_m": round(float(np.median(zs)), 3),
                           "median_z_m": round(float(np.median(gs)), 3),
                           "band_point_pct": round(float(100 * np.median(band_frac)), 1)},
               "layers": []}

    s_xy, g_xy = poses[0][:2], poses[-1][:2]
    print(f"\n== start {s_xy.round(2)}  goal {g_xy.round(2)} ==", flush=True)

    for res in (1.0, 0.5, 0.25):
        print(f"\n===== resolution {res} m (inflation {a.inflation} m) =====", flush=True)
        lay = nav.build_layer(kfs, res=res,
                              cfg=nav.LayerCfg(inflation_m=a.inflation), verbose=False)
        raw = lay.grid_raw
        st_ = lay.stats
        n = st_["cells"]
        print(f"  cells {n}: free {100*st_['free_cells']/n:.1f}%  "
              f"occ {100*st_['occupied_cells']/n:.1f}%  "
              f"unknown {100*st_['unknown_cells']/n:.1f}%  ({st_['ms']:.0f} ms)")
        occ_raw = int((raw == nav.OCCUPIED).sum())
        print(f"  before inflation: occ {100*occ_raw/n:.1f}%  "
              f"=> inflation added {100*(st_['occupied_cells']-occ_raw)/n:.1f} pp")

        sc, gc = lay.to_cell(s_xy), lay.to_cell(g_xy)
        for tag, c in (("start", sc), ("goal", gc)):
            v = int(lay.grid[c]); vr = int(raw[c])
            # count passable 8-neighbours
            nb = sum(1 for di, dj in nav._NB
                     if lay.inside(c[0]+di, c[1]+dj) and int(lay.grid[c[0]+di, c[1]+dj]) != nav.OCCUPIED)
            print(f"  {tag} cell {c}: grid={v} raw={vr} passable_neighbours={nb}/8")

        show(lay, sc, half=6, tag="start")
        show(lay, gc, half=6, tag="goal")

        # how many ego cells are isolated?
        ego = np.floor((poses[:, :2] - lay.origin) / res).astype(np.int64)
        eok = ((ego[:, 0] >= 0) & (ego[:, 1] >= 0) &
               (ego[:, 0] < lay.grid.shape[0]) & (ego[:, 1] < lay.grid.shape[1]))
        ego = ego[eok]
        iso = 0
        for i, j in ego:
            if all(int(lay.grid[i+di, j+dj]) == nav.OCCUPIED
                   for di, dj in nav._NB
                   if lay.inside(i+di, j+dj)):
                iso += 1
        print(f"  ego cells: {len(ego)}, isolated (all 8 nb occupied): {iso} "
              f"({100*iso/max(len(ego),1):.0f}%)")

        # A* with and without inflation, with and without unknown
        astar = {}
        for infl_tag, layer_use in (("inflated", lay), ("raw", _reuse(lay, raw))):
            for allow in (True, False):
                cells, info = nav.astar(layer_use, s_xy, g_xy, allow_unknown=allow)
                astar[f"{infl_tag}_allow_unknown_{allow}"] = {
                    "found": cells is not None,
                    "expanded": info["expanded"], "ms": info["ms"],
                    "len_m": round(nav.path_length(
                        np.stack([layer_use.to_world(c) for c in cells])), 2)
                    if cells is not None else None}
                print(f"  A* ({infl_tag:8s}, allow_unknown={str(allow):5s}): "
                      f"found={cells is not None} expanded={info['expanded']:7d} "
                      f"{info['ms']:7.1f} ms")

        payload["layers"].append({"res_m": res, "stats": st_,
                                  "occ_before_inflation_pct": round(100*occ_raw/n, 1),
                                  "start_cell": list(sc), "goal_cell": list(gc),
                                  "ego_cells": int(len(ego)), "ego_isolated": int(iso),
                                  "astar": astar})

    with open(os.path.join(a.out, "diag_nav_layer.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved {a.out}/diag_nav_layer.json")


def _reuse(lay: nav.OccLayer, raw: np.ndarray) -> nav.OccLayer:
    return nav.OccLayer(raw, lay.res, lay.origin, lay.cfg, dict(lay.stats), raw)


if __name__ == "__main__":
    main()
