#!/usr/bin/env python3
"""Is the occupancy map marking the *road* as an obstacle?

Stage 6 found the flat 0.25 m baseline unable to leave its start cell.  A first
diagnostic (``diag_nav_layer.py``) showed the ego cell is FREE and every
neighbour is OCCUPIED, at both 1.0 m and 0.25 m -- so the surface the vehicle
actually drove on is being classified as an obstacle.  The original builder
anchored the obstacle band to a single global median per scan:

    band = [median(z) + 0.20, median(z) + 2.20]

With the ground at z = -1.78 m the lower edge sits 0.39 m above the road, and 154
accumulated scans turn three marginally high returns into a permanent wall.

This script measures that hypothesis, then measures the fix:

1. Build the 2 m ground height field (low percentile of z per cell).
2. Histogram every point's height above *that* ground.  If the old band was
   eating road noise, a spike sits below 0.39 m and the new band loses it.
3. Count hits per cell on the ego cells, old criterion vs new.
4. Run A* on a 0.25 m layer built with the local-ground band, with and without
   inflation, and compare with the old band.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import nav, store as ST  # noqa: E402


def kf_clouds(st: ST.MapStore):
    out = []
    for kf in st.keyframes:
        with np.load(os.path.join(st.dir_kf, f"{kf.id:06d}.npz")) as z:
            out.append((kf.pose, z["pts"].astype(np.float32)))
    return out


def hitmap(P, mask, org, res, shape):
    h = np.zeros(shape, np.int32)
    ij = np.floor((P[mask][:, :2] - org) / res).astype(np.int64)
    ok = ((ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
          (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1]))
    np.add.at(h, (ij[ok, 0], ij[ok, 1]), 1)
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--map",
                    default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--res", type=float, default=0.25)
    ap.add_argument("--z-min", type=float, default=0.20)
    ap.add_argument("--z-max", type=float, default=2.20)
    a = ap.parse_args()

    st = ST.MapStore(a.map).open()
    kfs = kf_clouds(st)
    poses = np.stack([kf.pose[:3, 3] for kf in st.keyframes])
    P = np.concatenate([p for _, p in kfs])
    print(f"== {len(kfs)} keyframes, {len(P)} points, extent "
          f"{np.ptp(poses[:,0]):.1f} x {np.ptp(poses[:,1]):.1f} m ==", flush=True)

    G = nav.build_ground_field(P)
    print(f"  ground field {G.shape[0]}x{G.shape[1]} @ {G.res} m, "
          f"z range [{np.nanmin(G.grid):+.2f}, {np.nanmax(G.grid):+.2f}] m", flush=True)

    dz = G.dz(P)
    good = np.isfinite(dz)
    print(f"  points with a ground reference: {100*good.mean():.2f} %", flush=True)

    # ---- hypothesis test --------------------------------------------------- #
    cfg = nav.LayerCfg(z_min=a.z_min, z_max=a.z_max)
    z_med_frame = float(np.median([np.median(p[:, 2]) for _, p in kfs]))
    lo_old, hi_old = z_med_frame + a.z_min, z_med_frame + a.z_max
    in_old = (P[:, 2] >= lo_old) & (P[:, 2] <= hi_old)
    in_new = np.isfinite(dz) & (dz >= a.z_min) & (dz <= a.z_max)
    print(f"\n== band membership ==", flush=True)
    print(f"  old band, absolute z [{lo_old:+.2f}, {hi_old:+.2f}] "
          f"(per-frame median anchor) : {100*in_old.mean():5.1f} % of points")
    print(f"  new band, height above local ground {a.z_min:.2f}-{a.z_max:.2f} m"
          f"          : {100*in_new.mean():5.1f} % of points")
    contam = in_old & good & (dz < a.z_min)
    print(f"  of the OLD band, {100*contam.sum()/max(in_old.sum(),1):.1f} % sits below "
          f"{a.z_min} m above local ground   <-- road / kerb / noise")
    edges = [-3.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.1, a.z_min]
    below = good & (dz < a.z_min)
    hi = np.histogram(dz[below], bins=edges)[0]
    print("  dz histogram of everything below the band (m above local ground):")
    for k in range(len(hi)):
        print(f"    [{edges[k]:+5.2f},{edges[k+1]:+5.2f}) : {hi[k]:9d}")

    # ---- hits on ego cells ------------------------------------------------- #
    print(f"\n== hits on ego cells @ {a.res} m ==", flush=True)
    org2 = P[:, :2].min(0) - 1.0
    ij = np.floor((P[:, :2] - org2) / a.res).astype(np.int64)
    shape2 = tuple((ij.max(0) + 2).tolist())
    h_old, h_new = hitmap(P, in_old, org2, a.res, shape2), \
        hitmap(P, in_new, org2, a.res, shape2)
    ego = np.floor((poses[:, :2] - org2) / a.res).astype(np.int64)
    ego = ego[(ego[:, 0] >= 0) & (ego[:, 1] >= 0) &
              (ego[:, 0] < shape2[0]) & (ego[:, 1] < shape2[1])]
    ego_hits = {}
    for tag, h in (("old", h_old), ("new", h_new)):
        v = h[ego[:, 0], ego[:, 1]]
        ego_hits[tag] = {"mean": round(float(v.mean()), 1),
                         "median": float(np.median(v)),
                         "ge3_pct": round(100 * float(np.mean(v >= 3)), 1),
                         "ge10_pct": round(100 * float(np.mean(v >= 10)), 1)}
        print(f"  {tag} band: mean {v.mean():6.1f}  median {np.median(v):5.1f}  "
              f">=3: {ego_hits[tag]['ge3_pct']:5.1f} %  "
              f">=10: {ego_hits[tag]['ge10_pct']:5.1f} %")

    # ---- A*: old band vs new band vs new band + inflation ------------------ #
    s, g = poses[0], poses[-1]
    runs = {}
    print(f"\n== A* @ {a.res} m ==", flush=True)
    variants = [
        ("old_band_inflate_0.60", None, 0.60),
        ("old_band_inflate_0.00", None, 0.00),
        ("ground_band_inflate_0.60", G, 0.60),
        ("ground_band_inflate_0.00", G, 0.00),
    ]
    for tag, gr, infl in variants:
        lay = nav.build_layer(kfs, res=a.res,
                              cfg=nav.LayerCfg(z_min=a.z_min, z_max=a.z_max,
                                               inflation_m=infl),
                              verbose=False, ground=gr)
        n = int(lay.grid.shape[0] * lay.grid.shape[1])
        cells, info = nav.astar(lay, s, g, allow_unknown=True)
        L = round(nav.path_length(np.stack([lay.to_world(c) for c in cells])), 2) \
            if cells is not None else None
        # what the planner actually saw at both ends
        ends = {}
        for nm, xy in (("start", s), ("goal", g)):
            c = lay.to_cell(xy)
            nb = sum(1 for di, dj in nav._NB
                     if lay.inside(c[0]+di, c[1]+dj)
                     and int(lay.grid[c[0]+di, c[1]+dj]) != nav.OCCUPIED)
            ends[nm] = {"cell": list(c), "grid": int(lay.grid[c]),
                        "raw": int(lay.grid_raw[c]) if lay.grid_raw is not None else None,
                        "passable_nb": nb}
        runs[tag] = {"cells": n,
                     "free_pct": round(100 * float((lay.grid == nav.FREE).mean()), 1),
                     "occ_pct": round(100 * float((lay.grid == nav.OCCUPIED).mean()), 1),
                     "inflation_effective_m": lay.stats.get("inflation_effective_m"),
                     "ends": ends,
                     "found": cells is not None, "expanded": info["expanded"],
                     "ms": info["ms"], "len_m": L,
                     "clearance_m": round(nav.clearance(lay, np.stack(
                         [lay.to_world(c) for c in cells])), 2) if cells is not None else None}
        print(f"  {tag:26s}: free {runs[tag]['free_pct']:5.1f}%  "
              f"occ {runs[tag]['occ_pct']:5.1f}%  inflate_eff "
              f"{(runs[tag]['inflation_effective_m'] or 0):.2f} m -> "
              f"found={cells is not None} expanded={info['expanded']:6d} "
              f"{info['ms']:7.1f} ms  len={L} m", flush=True)
        print(f"    start {ends['start']['cell']} grid={ends['start']['grid']} "
              f"raw={ends['start']['raw']} nb={ends['start']['passable_nb']}/8 | "
              f"goal {ends['goal']['cell']} grid={ends['goal']['grid']} "
              f"raw={ends['goal']['raw']} nb={ends['goal']['passable_nb']}/8",
              flush=True)

    payload = {
        "points": int(len(P)),
        "ground_field": G.as_dict(),
        "band_pct_all_points": {"old": round(100 * float(in_old.mean()), 1),
                                "new": round(100 * float(in_new.mean()), 1)},
        "old_band_pct_below_new_floor": round(
            100 * float(contam.sum() / max(in_old.sum(), 1)), 1),
        "dz_hist_edges": edges, "dz_hist_counts": hi.tolist(),
        "ego_cells": int(len(ego)), "ego_hits": ego_hits,
        "astar": runs,
    }
    with open(os.path.join(a.out, "diag_nav_ground.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved {a.out}/diag_nav_ground.json")


if __name__ == "__main__":
    main()
