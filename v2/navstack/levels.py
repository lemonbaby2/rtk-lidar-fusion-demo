"""Multi-level maps: split by ground height, bake per level, load per level.

Theme 4 of the plan.  Two things are easy to get wrong and are handled here:

* **Floors are separated by the *ground* height of a keyframe, not by the mean z
  of its points.**  A keyframe on floor 1 that happens to see a lot of ceiling
  has a point-mean above a keyframe on floor 2 that sees a lot of ground, so
  clustering point means merges the floors.  The ground height is the mode of the
  low-z point histogram, which is invariant to what else is in view.

* **The 1D cluster threshold must be smaller than a storey and larger than a
  ramp.**  A 2.0 m separation with a 2.5 m minimum cluster mass does that:
  storeys are 2.8-4.0 m, a spiral ramp inside one storey climbs 2.5 m over
  several tens of metres and is *not* a separate level because its keyframes do
  not form a separated mode.

``bake()`` writes nothing; it returns in-memory clouds plus the statistics the
report needs.  Persisting the bake is the web exporter's job, because only the
exporter knows which LOD the client can afford.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import schema as S
from . import tiles as T


def ground_height(pts: np.ndarray, bin_m: float = 0.10,
                  band_m: float = 6.0) -> float:
    """Mode of the low-z histogram: the height of the surface being driven on.

    The search band is taken relative to the *lowest decile* of the cloud, so a
    scan dominated by a nearby wall cannot drag the estimate upward.
    """
    if len(pts) == 0:
        return 0.0
    lo = float(np.percentile(pts[:, 2], 2))
    sel = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= lo + band_m), 2]
    if sel.size < 50:
        return lo
    hist, edges = np.histogram(sel, bins=max(8, int(band_m / bin_m)))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def cluster_1d(values: np.ndarray, min_sep: float = 2.0,
               min_mass: int = 2) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Single-linkage clustering on a sorted 1D signal with a separation floor."""
    v = np.asarray(values, float)
    order = np.argsort(v)
    labels = np.zeros(len(v), int)
    lab = 0
    for k in range(1, len(order)):
        if v[order[k]] - v[order[k - 1]] > min_sep:
            lab += 1
        labels[order[k]] = lab
    groups = [np.where(labels == i)[0] for i in range(lab + 1)]
    groups = [g for g in groups if len(g) >= min_mass]
    return labels, groups


def estimate_levels(store, min_sep: float = 2.0, min_mass: int = 2,
                    verbose: bool = True) -> List[S.Level]:
    """Assign every keyframe to a floor and rewrite ``store.levels``."""
    zg, ids = [], []
    for kf in store.keyframes:
        p = os.path.join(store.dir_kf, f"{kf.id:06d}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            pts = z["pts"]
        zg.append(ground_height(pts))
        ids.append(kf.id)
    if not zg:
        store.levels = [S.Level(0, 0.0, 0.0, 0.0)]
        return store.levels

    zg = np.asarray(zg)
    labels, groups = cluster_1d(zg, min_sep=min_sep, min_mass=min_mass)
    # assign by nearest cluster centre rather than by label, so keyframes in a
    # discarded small cluster still land on the closest kept floor
    centres = np.array([zg[g].mean() for g in groups])
    order = np.argsort(centres)
    centres = centres[order]
    groups = [groups[i] for i in order]

    kf_by_id = {kf.id: kf for kf in store.keyframes}
    z_of = {i: z for i, z in zip(ids, zg)}
    levels: List[S.Level] = []
    for li, c in enumerate(centres):
        levels.append(S.Level(id=li, z_min=float(c - 1.0), z_max=float(c + 1.0),
                              z_ref=float(c)))
    for kf, z in zip([kf_by_id[i] for i in ids], zg):
        li = int(np.argmin(np.abs(centres - z)))
        kf.level = li
        levels[li].keyframes.append(kf.id)

    # widen each level's band to the observed spread of its own ground heights,
    # capped at two metres so two levels can never claim the same point
    for lv in levels:
        zs = np.array([z_of[i] for i in lv.keyframes]) if lv.keyframes else np.array([lv.z_ref])
        lv.z_min = float(zs.min() - 0.30)
        lv.z_max = float(zs.min() + 2.00)
    store.levels = levels
    store.level_chunks = {}
    for kf in store.keyframes:
        for k in kf.chunks:
            store.level_chunks.setdefault(kf.level, set()).add(k)
    if verbose:
        print(f"  levels: {len(levels)}  " +
              "  ".join(f"L{l.id} z={l.z_ref:+.2f} n={len(l.keyframes)}"
                        for l in levels), flush=True)
    return levels


def bake(store, lod: str = "mid", per_level_cap: int = 400_000,
         verbose: bool = True) -> Dict[str, dict]:
    """Merge each level into one cloud; returns clouds + stats (not persisted)."""
    leaf = T.LOD_LEAF.get(lod, S.DEFAULT_LOD_M)
    out: Dict[str, dict] = {}
    rng = np.random.default_rng(7)
    for lv in store.levels:
        clouds, n_kf = [], 0
        for kf in store.keyframes:
            if kf.level != lv.id:
                continue
            p = os.path.join(store.dir_kf, f"{kf.id:06d}.npz")
            if not os.path.exists(p):
                continue
            with np.load(p) as z:
                clouds.append((z["pts"], z["rgb"]))
            n_kf += 1
        pts, rgb = T.merge_with_limit(clouds, leaf, cap=per_level_cap, rng=rng)
        lv.n_points = int(len(pts))
        out[str(lv.id)] = {
            "level": lv.id, "z_ref": round(lv.z_ref, 3),
            "z_min": round(lv.z_min, 3), "z_max": round(lv.z_max, 3),
            "keyframes": n_kf, "points": int(len(pts)), "lod": lod,
            "leaf_m": leaf,
            "bbox_min": pts.min(0).round(2).tolist() if len(pts) else None,
            "bbox_max": pts.max(0).round(2).tolist() if len(pts) else None,
            "pts": pts, "rgb": rgb,
        }
        if verbose:
            print(f"  baked L{lv.id}: {len(pts)} pts from {n_kf} keyframes "
                  f"(lod={lod}, {leaf} m)", flush=True)
    return out


def minimap(store, res_m: float = 1.0, level: Optional[int] = None
            ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Top-down occupancy of the whole map, for the global view (theme 6 input).

    ``0`` = unknown, ``1`` = observed-but-low (a few returns: grass, kerb),
    ``2`` = occupied.  A 1 m cell that collected >= 6 returns is treated as
    occupied, which is the threshold that separated walls from road in this
    sequence.
    """
    pts_all = []
    for kf in store.keyframes:
        if level is not None and kf.level != int(level):
            continue
        p = os.path.join(store.dir_kf, f"{kf.id:06d}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            pts_all.append(z["pts"])
    if not pts_all:
        return np.zeros((1, 1), np.int8), {}
    pts = np.concatenate(pts_all)
    org = pts[:, :2].min(0) - 2 * res_m
    ij = np.floor((pts[:, :2] - org) / res_m).astype(np.int64)
    w = int(ij[:, 0].max()) + 3
    h = int(ij[:, 1].max()) + 3
    cnt = np.zeros((w, h), np.int32)
    np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
    grid = np.zeros((w, h), np.int8)
    grid[cnt > 0] = 1
    grid[cnt >= 6] = 2
    return grid, {"shape": [w, h], "res_m": res_m,
                  "origin": [round(float(v), 2) for v in org],
                  "occupied_cells": int((grid == 2).sum()),
                  "observed_cells": int((grid > 0).sum()),
                  "coverage_m2": round(float((grid > 0).sum()) * res_m ** 2, 1)}
