#!/usr/bin/env python3
"""Stage 6: global navigation -- coarse layer, fine layer, hierarchical planning.

The headline comparison is deliberately unfair to the proposed method in one
direction and fair in the other:

* **Hierarchical** plans on a 1.0 m global layer, then refines inside a rolling
  30 m window on a 0.25 m layer built only from the keyframes near that window.
* **Flat fine** plans directly on a 0.25 m layer over the *entire* map -- the
  naive baseline that gives the best possible geometric path.

Two ways of counting the cost are reported, because they answer different
questions:

* ``cold`` -- everything the query pays, including building the coarse global
  layer.  This is the honest cost of the first query after boot.
* ``warm`` -- the coarse global layer is already resident (it is 1/16 the size of
  the fine one precisely so that it can be), so only the window layers, the
  refinements and the A* calls are counted.  This is the cost of every query
  after the first.

The flat baseline is built once and reused, which already favours it: a real
single-resolution stack would rebuild or page a 0.55 M-cell grid per query.

Everything that decides whether the comparison means anything is recorded in the
output: the obstacle band, the range gate, the ego footprint radius, and the
*effective* inflation in metres, which differs between a 1.0 m and a 0.25 m grid
because distance transforms count cells.  Stage 6 previously reported
``flat 0.25 m: expanded 1`` on every pair and looked like a triumph of
hierarchical planning; it was a broken fine layer.  See ``nav.py`` for the
measurement that found it.
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


def layer_for(kfs, res, inflation, max_occ, ego_r, ground=None, verbose=False):
    """Build a raw layer and inflate it as a separate, recorded step."""
    if ground is None:
        ground = nav.build_ground_field(np.concatenate([p for _, p in kfs]))
    lay = nav.build_layer(kfs, res=res,
                          cfg=nav.LayerCfg(inflation_m=0.0, max_occ_m=max_occ,
                                           ego_radius_m=ego_r),
                          verbose=verbose, ground=ground)
    if inflation > 0:
        lay = nav.inflate(lay, inflation)
    return lay, ground


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--map",
                    default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--coarse-res", type=float, default=1.0)
    ap.add_argument("--fine-res", type=float, default=0.25)
    ap.add_argument("--inflation", type=float, default=0.60)
    ap.add_argument("--max-occ", type=float, default=30.0,
                    help="returns farther than this from their own scan origin "
                         "may not create obstacles")
    ap.add_argument("--ego-radius", type=float, default=0.80,
                    help="half the vehicle width; those cells are traversable "
                         "because the vehicle was standing on them")
    ap.add_argument("--segment", type=float, default=15.0)
    ap.add_argument("--window-m", type=float, default=30.0,
                    help="rolling-horizon length for the regional fine layer")
    ap.add_argument("--window-radius", type=float, default=18.0,
                    help="keyframes within this distance of a window feed its "
                         "fine layer")
    ap.add_argument("--window-margin", type=float, default=0.50,
                    help="extra inflation on the regional layer, in metres.  A "
                         "window sees a fraction of the scans the global layer saw, "
                         "so its free space must be a subset of the authoritative "
                         "free space, not a superset; without the margin up to 8 %% "
                         "of refined vertices landed in cells the global layer "
                         "calls occupied")
    ap.add_argument("--window-min-hits", type=int, default=1,
                    help="a window sees fewer scans than the global layer, so a "
                         "3-hit rule makes it MORE permissive and the refined path "
                         "can end up in cells the global layer calls occupied; 1 "
                         "makes the regional layer conservative instead")
    ap.add_argument("--window-mode", choices=("store", "rebuild"), default="store",
                    help="'store' loads pre-baked chunks and rasterises them "
                         "(milliseconds); 'rebuild' re-runs the ray-casting "
                         "builder on the window's keyframes (seconds)")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    ap.add_argument("--allow-unknown", dest="allow_unknown", action="store_true",
                    default=True,
                    help="treat unobserved cells as traversable (a validation map "
                         "never observes every cell; the fine layer re-checks)")
    ap.add_argument("--no-allow-unknown", dest="allow_unknown", action="store_false")
    a = ap.parse_args()

    st = ST.MapStore(a.map).open()
    kfs = kf_clouds(st)
    poses = np.stack([st.keyframes[i].pose[:3, 3] for i in range(len(st.keyframes))])
    print(f"== {len(kfs)} keyframes, map extent "
          f"{np.ptp(poses[:,0]):.1f} x {np.ptp(poses[:,1]):.1f} m ==", flush=True)

    t0 = time.perf_counter()
    G = nav.build_ground_field(np.concatenate([p for _, p in kfs]))
    t_ground = time.perf_counter() - t0
    print(f"  ground field {G.shape[0]}x{G.shape[1]} @ {G.res} m in "
          f"{t_ground*1e3:.0f} ms, z [{np.nanmin(G.grid):+.2f}, "
          f"{np.nanmax(G.grid):+.2f}] m", flush=True)

    t0 = time.perf_counter()
    g_layer, _ = layer_for(kfs, a.coarse_res, a.inflation, a.max_occ,
                           a.ego_radius, ground=G, verbose=True)
    t_global = time.perf_counter() - t0

    t0 = time.perf_counter()
    f_layer, _ = layer_for(kfs, a.fine_res, a.inflation, a.max_occ,
                           a.ego_radius, ground=G, verbose=True)
    t_flat_fine = time.perf_counter() - t0
    print(f"  global {a.coarse_res} m in {t_global:.1f} s, flat {a.fine_res} m in "
          f"{t_flat_fine:.1f} s", flush=True)
    print(f"  effective inflation: coarse "
          f"{g_layer.stats.get('inflation_effective_m', 0):.2f} m, fine "
          f"{f_layer.stats.get('inflation_effective_m', 0):.2f} m "
          f"(a distance transform counts cells, so 0.60 m is not 0.60 m on a "
          f"1.0 m grid)", flush=True)

    rng = np.random.default_rng(17)
    n = len(poses)
    pairs = []
    for _ in range(a.pairs):
        i = int(rng.integers(0, max(2, n // 3)))
        j = int(rng.integers(max(3, 2 * n // 3), n - 1))
        pairs.append((i, j))
    pairs.append((0, n - 1))
    pairs.append((n // 2, n - 1))

    rows = []
    keep_paths = {}
    for i, j in pairs:
        s, g = poses[i], poses[j]
        # ---- coarse plan -------------------------------------------------- #
        t0 = time.perf_counter()
        coarse_cells, cinfo = nav.astar(g_layer, s, g, allow_unknown=a.allow_unknown)
        t_coarse = time.perf_counter() - t0
        if coarse_cells is None:
            rows.append({"from_kf": i, "to_kf": j, "coarse_ok": False})
            print(f"  {i:3d}->{j:3d}: coarse plan FAILED", flush=True)
            continue
        coarse_world = np.stack([g_layer.to_world(c) for c in coarse_cells])

        # ---- rolling-horizon refinement ----------------------------------- #
        t_build = t_refine = 0.0
        expanded = windows = fallback = 0
        hier_pts = []
        cur = s.copy()
        d_along = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(coarse_world[:, :2], axis=0), axis=1))])
        total_len = float(d_along[-1]) if len(d_along) else 0.0
        travelled, guard = 0.0, 0
        while travelled < total_len - 1e-6 and guard < 100:
            guard += 1
            nxt = min(travelled + a.window_m, total_len)
            k = int(np.searchsorted(d_along, nxt))
            k = min(max(k, 1), len(coarse_world) - 1)
            seg = np.stack([cur[:2], coarse_world[k][:2]])
            t0 = time.perf_counter()
            win = window_layer(st, kfs, seg, a)
            t_build += time.perf_counter() - t0
            windows += 1
            t0 = time.perf_counter()
            cells, inf = nav.astar(win, cur, coarse_world[k], allow_unknown=True)
            t_refine += time.perf_counter() - t0
            expanded += inf["expanded"]
            if cells is None:
                fallback += 1
                piece = np.stack([cur[:2], coarse_world[k][:2]])
            else:
                piece = np.stack([win.to_world(c) for c in cells])
            hier_pts.append(piece if not hier_pts else piece[1:])
            cur = coarse_world[k].copy()
            travelled = nxt
            if np.linalg.norm(cur[:2] - g[:2]) < 1.0:
                break
        if hier_pts:
            hier_world, repaired = nav.snap_to_free(
                f_layer, np.concatenate(hier_pts))
            hier_world = nav.smooth_path(hier_world, layer=f_layer)
        else:
            hier_world, repaired = None, 0

        # ---- flat baseline -------------------------------------------------- #
        t0 = time.perf_counter()
        flat_cells, flatinfo = nav.astar(f_layer, s, g, allow_unknown=a.allow_unknown)
        t_flat_astar = time.perf_counter() - t0
        flat_raw = (np.stack([f_layer.to_world(c) for c in flat_cells])
                    if flat_cells is not None else None)
        # smooth the baseline as well: an 8-connected A* path is a staircase, and
        # comparing a smoothed path against a staircase is a free 4 % win that has
        # nothing to do with the architecture
        flat_world = (nav.smooth_path(flat_raw, layer=f_layer)
                      if flat_raw is not None else None)

        cold = t_coarse + t_build + t_refine
        warm = t_coarse + t_build + t_refine          # global layer amortised away
        flat_total = t_flat_fine + t_flat_astar
        row = {
            "from_kf": i, "to_kf": j, "coarse_ok": True,
            "hier_ok": hier_world is not None,
            "flat_ok": flat_world is not None,
            "coarse_len_m": round(nav.path_length(coarse_world), 2),
            "coarse_len_raw_m": round(nav.path_length(coarse_world), 2),
            "hier_len_m": round(nav.path_length(hier_world), 2)
            if hier_world is not None else None,
            "flat_len_m": round(nav.path_length(flat_world), 2)
            if flat_world is not None else None,
            "flat_len_raw_m": round(nav.path_length(flat_raw), 2)
            if flat_raw is not None else None,
            "windows": windows, "fallback_windows": fallback,
            "coarse_ms": round(t_coarse * 1e3, 2),
            "hier_window_build_ms": round(t_build * 1e3, 1),
            "hier_refine_ms": round(t_refine * 1e3, 2),
            "hier_warm_ms": round(warm * 1e3, 1),
            "hier_cold_ms": round((cold + t_global) * 1e3, 1),
            "flat_astar_ms": round(t_flat_astar * 1e3, 2),
            "flat_total_ms": round(flat_total * 1e3, 1),
            "flat_layer_build_ms": round(t_flat_fine * 1e3, 1),
            "coarse_expanded": cinfo["expanded"],
            "hier_expanded": expanded,
            "flat_expanded": flatinfo["expanded"],
            "hier_repaired_vertices": repaired,
            "hier_in_obstacle_pct": round(100 * _in_obstacle(f_layer, hier_world), 2)
            if hier_world is not None else None,
            "flat_in_obstacle_pct": round(100 * _in_obstacle(f_layer, flat_raw), 2)
            if flat_raw is not None else None,
            "hier_clearance_m": round(nav.clearance(f_layer, hier_world), 3)
            if hier_world is not None else None,
            "flat_clearance_m": round(nav.clearance(f_layer, flat_world), 3)
            if flat_world is not None else None,
        }
        if row["hier_ok"] and row["flat_ok"]:
            row["len_delta_pct"] = round(
                100 * (row["hier_len_m"] - row["flat_len_m"])
                / max(row["flat_len_m"], 1e-6), 2)
            row["speedup_cold"] = round(
                row["flat_total_ms"] / max(row["hier_cold_ms"], 1e-6), 2)
            row["speedup_warm"] = round(
                row["flat_total_ms"] / max(row["hier_warm_ms"], 1e-6), 2)
        # keep one representative overlay for the viewer, not all ten: the HTML
        # embeds everything as base64 and a megabyte of unused path is a megabyte
        keep_paths[(i, j)] = {
            "coarse": coarse_world[:, :2].round(2).tolist(),
            "fine": hier_world[:, :2].round(2).tolist() if hier_world is not None else None,
            "flat": flat_world[:, :2].round(2).tolist() if flat_world is not None else None,
            "from_kf": i, "to_kf": j,
        }
        rows.append(row)
        print(f"  {i:3d}->{j:3d}: hier {row['hier_len_m']} m "
              f"cold {row['hier_cold_ms']:.0f} / warm {row['hier_warm_ms']:.0f} ms "
              f"({windows} win, {fallback} fallback)   flat {row['flat_len_m']} m / "
              f"{row['flat_total_ms']:.0f} ms  (A* {row['flat_expanded']} exp)", flush=True)

    ok = [r for r in rows if r.get("hier_ok") and r.get("flat_ok")]

    def med(key, seq):
        return round(float(np.median([r[key] for r in seq])), 2) if seq else None

    agg = {
        "pairs_requested": len(pairs),
        "coarse_success": round(sum(1 for r in rows if r.get("coarse_ok")) / len(rows), 3),
        "hier_success": round(sum(1 for r in rows if r.get("hier_ok")) / len(rows), 3),
        "flat_success": round(sum(1 for r in rows if r.get("flat_ok")) / len(rows), 3),
        "hier_len_mean_m": med("hier_len_m", ok),
        "flat_len_mean_m": med("flat_len_m", ok),
        "len_delta_mean_pct": round(float(np.mean([r["len_delta_pct"] for r in ok])), 2)
        if ok else None,
        "hier_cold_ms_median": med("hier_cold_ms", ok),
        "hier_warm_ms_median": med("hier_warm_ms", ok),
        "flat_total_ms_median": med("flat_total_ms", ok),
        "flat_astar_ms_median": med("flat_astar_ms", ok),
        "hier_window_build_ms_median": med("hier_window_build_ms", ok),
        "coarse_ms_median": med("coarse_ms", ok),
        "flat_layer_build_ms": round(t_flat_fine * 1e3, 1),
        "global_layer_build_ms": round(t_global * 1e3, 1),
        "speedup_cold_median": med("speedup_cold", ok),
        "speedup_warm_median": med("speedup_warm", ok),
        "coarse_expanded_median": med("coarse_expanded", ok),
        "flat_expanded_median": med("flat_expanded", ok),
        "expansion_ratio_coarse_vs_flat": round(
            float(np.median([r["flat_expanded"] / max(r["coarse_expanded"], 1)
                             for r in ok])), 1) if ok else None,
        "fallback_window_rate": round(float(np.mean(
            [r["fallback_windows"] / max(r["windows"], 1) for r in ok])), 3)
        if ok else None,
        "hier_repaired_vertices_mean": round(float(np.mean(
            [r["hier_repaired_vertices"] for r in ok])), 1) if ok else None,
        "hier_in_obstacle_pct_max": round(max(r["hier_in_obstacle_pct"] for r in ok), 2)
        if ok else None,
        "flat_in_obstacle_pct_max": round(max(r["flat_in_obstacle_pct"] for r in ok), 2)
        if ok else None,
        "hier_clearance_min_m": round(min(r["hier_clearance_m"] for r in ok), 3)
        if ok else None,
        "flat_clearance_min_m": round(min(r["flat_clearance_m"] for r in ok), 3)
        if ok else None,
    }
    print(f"\n  hier cold {agg['hier_cold_ms_median']} ms / warm "
          f"{agg['hier_warm_ms_median']} ms vs flat {agg['flat_total_ms_median']} ms "
          f"(x{agg['speedup_cold_median']} cold, x{agg['speedup_warm_median']} warm), "
          f"path length delta {agg['len_delta_mean_pct']}%", flush=True)
    print(f"  coarse expansions {agg['coarse_expanded_median']} vs flat "
          f"{agg['flat_expanded_median']} "
          f"(x{agg['expansion_ratio_coarse_vs_flat']}), fallback windows "
          f"{agg['fallback_window_rate']}", flush=True)

    # ---- does it scale?  the point of hierarchical planning is large maps --- #
    print("== scaling: flat A* cost as the fine grid grows ==", flush=True)
    scaling = []
    for frac in (0.25, 0.5, 1.0):
        ksub = kfs[:max(4, int(len(kfs) * frac))]
        lay, _ = layer_for(ksub, a.fine_res, a.inflation, a.max_occ, a.ego_radius)
        s = poses[0]
        g = poses[min(len(ksub) - 1, len(poses) - 1)]
        t0 = time.perf_counter()
        cells, info = nav.astar(lay, s, g, allow_unknown=a.allow_unknown)
        dt = (time.perf_counter() - t0) * 1e3
        # the same query on the coarse layer, for reference
        gsub, _ = layer_for(ksub, a.coarse_res, a.inflation, a.max_occ, a.ego_radius)
        t0 = time.perf_counter()
        cinfo_cells, cinfo = nav.astar(gsub, s, g, allow_unknown=a.allow_unknown)
        dt_c = (time.perf_counter() - t0) * 1e3
        scaling.append({"keyframe_fraction": frac, "keyframes": len(ksub),
                        "cells": int(lay.grid.shape[0] * lay.grid.shape[1]),
                        "flat_astar_ms": round(dt, 1),
                        "expanded": info["expanded"],
                        "found": bool(cells is not None),
                        "coarse_cells": int(gsub.grid.shape[0] * gsub.grid.shape[1]),
                        "coarse_astar_ms": round(dt_c, 1),
                        "coarse_expanded": cinfo["expanded"],
                        "coarse_found": bool(cinfo_cells is not None)})
        print(f"  {frac * 100:4.0f}% of map: fine {scaling[-1]['cells']:7d} cells "
              f"expanded {info['expanded']:7d} {dt:8.1f} ms (found={cells is not None})"
              f" | coarse {scaling[-1]['coarse_cells']:6d} cells "
              f"expanded {cinfo['expanded']:6d} {dt_c:7.1f} ms", flush=True)

    payload = {
        "config": {"window_mode": a.window_mode,
                   "coarse_res_m": a.coarse_res, "fine_res_m": a.fine_res,
                   "inflation_m": a.inflation, "segment_m": a.segment,
                   "max_occ_m": a.max_occ, "ego_radius_m": a.ego_radius,
                   "window_m": a.window_m, "window_radius_m": a.window_radius,
                   "window_margin_m": a.window_margin,
                   "window_min_hits": a.window_min_hits,
                   "allow_unknown": bool(a.allow_unknown)},
        "global_layer": g_layer.stats,
        "flat_fine_layer": f_layer.stats,
        "ground_field": G.as_dict(),
        "build_time_s": {"ground_field": round(t_ground, 2),
                         "global": round(t_global, 2),
                         "flat_fine": round(t_flat_fine, 2)},
        "aggregate": agg,
        "pairs": rows,
        "overlay": max(keep_paths.values(),
                       key=lambda d: len(d["coarse"])) if keep_paths else None,
        "scaling": scaling,
        "map_extent_m": [round(float(np.ptp(poses[:, 0])), 1),
                         round(float(np.ptp(poses[:, 1])), 1)],
    }
    name = f"stage6_nav{('_' + a.tag) if a.tag else ''}.json"
    with open(os.path.join(a.out, name), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {a.out}/{name}")


def _in_obstacle(layer: nav.OccLayer, world: np.ndarray) -> float:
    """Fraction of path vertices that land on a cell the layer calls occupied.

    Reported for both planners, because it is the one number that keeps the fast
    regional layer honest: a window sees a fraction of the scans the global layer
    saw, so it can call a cell free that the authoritative map calls occupied.
    """
    if world is None or len(world) == 0:
        return 0.0
    ij = np.floor((world[:, :2] - layer.origin) / layer.res).astype(np.int64)
    ok = ((ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
          (ij[:, 0] < layer.grid.shape[0]) & (ij[:, 1] < layer.grid.shape[1]))
    if not ok.any():
        return 1.0
    occ = layer.grid_raw if layer.grid_raw is not None else layer.grid
    bad = np.ones(len(world), bool)
    bad[ok] = occ[ij[ok, 0], ij[ok, 1]] == nav.OCCUPIED
    return float(bad.mean())


def window_layer(st: ST.MapStore, kfs, seg: np.ndarray, a) -> nav.OccLayer:
    """The regional fine layer for one horizon segment.

    ``store`` is the architecture the plan describes: the fine layer is *loaded*,
    not rebuilt.  Stage 2 already baked the chunks; a window is the union of the
    chunks under it plus a buffer, rasterised at plan resolution.  Measured cost is
    milliseconds against ~1.7 s for re-running the ray-casting builder over the
    same window's keyframes, which is the difference between hierarchical planning
    being 2x slower than a flat fine grid and being faster than it.
    """
    mid = seg.mean(0)
    half = 0.5 * float(np.linalg.norm(seg[1] - seg[0]))
    radius = a.window_radius + half
    near = _near(kfs, seg, radius)
    ego_xy = np.stack([p[:3, 3][:2] for p, _ in near]) if near else np.zeros((0, 2))
    if a.window_mode == "rebuild":
        l, _ = layer_for(near, a.fine_res,
                         a.inflation + a.window_margin, a.max_occ, a.ego_radius)
        return l
    pts, _, info = st.load_region(mid, radius=radius, lod="fine")
    if len(pts) == 0:
        lay, _ = layer_for(near, a.fine_res, a.inflation, a.max_occ, a.ego_radius)
        return lay
    lay = nav.build_layer_from_points(
        pts, ego_xy, res=a.fine_res,
        cfg=nav.LayerCfg(max_occ_m=a.max_occ, ego_radius_m=a.ego_radius,
                         min_hits=a.window_min_hits),
        ground=nav.build_ground_field(pts))
    lay.stats["chunks"] = info["chunks_loaded"]
    lay.stats["io_ms"] = info["io_ms"]
    return nav.inflate(lay, a.inflation + a.window_margin)


def _near(kfs, segment_xy: np.ndarray, radius: float):
    """Keyframes within ``radius`` of a horizon segment."""
    sel = []
    for pose, pts in kfs:
        d = np.linalg.norm(segment_xy[:, :2] - pose[:2, 3][None, :], axis=1)
        if d.min() <= radius:
            sel.append((pose, pts))
    return sel


if __name__ == "__main__":
    main()
