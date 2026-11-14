"""Hierarchical navigation on a large map: coarse global layer, fine regional layer.

Theme 6 of the plan.  The problem it solves is arithmetic: a 61 m x 17 m drive is
already 28 k cells at 0.25 m and 0.6 M once the LiDAR's 100 m reach is included,
and A* over that is both slow and pointless, because 95 % of the route is a
straight line down a corridor.  So:

1. **Global light layer** -- 1.0 m occupancy, built from the whole map, small
   enough to keep resident.
2. **Regional fine layer** -- 0.25 m, built on demand inside a 30 m window around
   the current pose, discarded when the agent moves on.
3. **Layered planning** -- plan coarse, then refine each coarse segment on the
   fine layer.

Free space is not guessed from "no points here": every keyframe casts rays from
its own pose to its own returns, so a cell is marked free only because something
was seen *through* it.  That is the difference between an occupancy map and a
point-count threshold, and it is what makes the planner usable in the gaps
between measured surfaces.

Two measured corrections to the naive version, both of which changed a verdict:

**The obstacle band must be anchored to the local ground, not to a global
median.**  The first version classified a point as an obstacle when

    median(pts.z) + 0.20 <= z <= median(pts.z) + 2.20

where the median is one number per scan.  With the ground at z = -1.78 m that
puts the lower edge 0.39 m above the road, and 154 accumulated scans then turn
three noise returns into a permanent wall: the measured effect was that the cell
the vehicle was standing in had zero passable neighbours at 0.25 m, and A* on the
flat fine layer returned ``expanded 1`` on all ten test pairs.  ``GroundField``
measures the ground as a 2 m height field (low percentile of z per cell, hole
filled, median filtered), and the band is applied to height above *that*.

**Inflation has a resolution floor.**  ``distance_transform_edt`` counts cells, so
a 0.60 m inflation radius is a no-op on a 1.0 m grid (the nearest free cell to an
obstacle is 1.0 m away, which is > 0.60).  Measured: inflation added 0.0 pp of
occupied cells at 1.0 m and 7.7 pp at 0.25 m.  Inflation is therefore applied by
:func:`inflate` at plan time, records its effective radius in cells, and a caller
who wants the two layers to be comparable must compare like with like.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import (binary_dilation, distance_transform_edt,
                               median_filter)
except ImportError:  # pragma: no cover
    distance_transform_edt = None
    binary_dilation = None
    median_filter = None

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


@dataclass
class LayerCfg:
    res_m: float = 1.0
    z_min: float = 0.20            # above ground: below this is road
    z_max: float = 2.20            # above ground: above this passes overhead
    inflation_m: float = 0.60
    min_hits: int = 3              # returns needed to call a cell occupied
    ray_stride: int = 3            # cast to every n-th return (speed)
    max_ray_m: float = 35.0
    # Half the vehicle width.  The body occupied every cell within this radius of
    # a keyframe origin, so those cells are traversable by construction and must
    # survive inflation -- otherwise a 0.6 m inflation on a street the car drove
    # down closes the corridor it just used.
    ego_radius_m: float = 0.80
    # A return farther than this from its own scan origin is not allowed to make
    # a cell occupied.  MEASURED, and the reason the fine layer used to be solid:
    # the HDL-64E reaches 120 m, and at that range the ground arrives at a grazing
    # angle where a 0.3 deg pitch error puts it metres above the road.  With no
    # range gate the whole driving surface accumulated >= 3 returns per 0.25 m
    # cell and the ego cell had 44, so A* on the fine layer returned
    # ``expanded 1`` on every pair.  Rays are still cast to ``max_ray_m``; only
    # *hits* are gated, because free space seen at range is harmless while an
    # obstacle invented at range is not.
    max_occ_m: float = 30.0


# --------------------------------------------------------------------------- #
# ground
# --------------------------------------------------------------------------- #
@dataclass
class GroundField:
    """Height of the ground on a coarse 2-D grid.

    The low percentile of z inside each cell is a robust ground estimate as long
    as the cell is mostly road; cells with too few points are filled from the
    nearest valid neighbour and the whole field is median filtered, which is
    enough for a driving surface and costs milliseconds.
    """
    grid: np.ndarray               # (W,H) float32
    org: np.ndarray                # world xy of cell (0,0) corner
    res: float

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def _ij(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ij = np.floor((pts[:, :2] - self.org) / self.res).astype(np.int64)
        ok = ((ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
              (ij[:, 0] < self.grid.shape[0]) & (ij[:, 1] < self.grid.shape[1]))
        return ij, ok

    def height_at(self, pts: np.ndarray) -> np.ndarray:
        ij, ok = self._ij(pts)
        out = np.full(len(pts), np.nan, np.float32)
        out[ok] = self.grid[ij[ok, 0], ij[ok, 1]]
        return out

    def dz(self, pts: np.ndarray) -> np.ndarray:
        """Height of each point above the local ground (NaN where unknown)."""
        return pts[:, 2].astype(np.float32) - self.height_at(pts)

    def as_dict(self) -> dict:
        return {"res_m": self.res, "shape": list(self.grid.shape),
                "z_min": round(float(np.nanmin(self.grid)), 3),
                "z_max": round(float(np.nanmax(self.grid)), 3)}


def build_ground_field(pts: np.ndarray, res: float = 2.0, pct: float = 2.0,
                       min_pts: int = 8, stride: int = 3) -> GroundField:
    """Ground height field from the low percentile of z per 2 m cell."""
    p = pts[::stride] if stride > 1 else pts
    org = p[:, :2].min(0) - 2 * res
    ij = np.floor((p[:, :2] - org) / res).astype(np.int64)
    shape = tuple((ij.max(0) + 2).tolist())
    cell = ij[:, 0] * shape[1] + ij[:, 1]
    order = np.argsort(cell, kind="stable")
    cell_s, z_s = cell[order], p[order, 2]
    uniq, start = np.unique(cell_s, return_index=True)
    end = np.append(start[1:], len(cell_s))
    g = np.full(shape[0] * shape[1], np.nan, np.float32)
    for c, s, e in zip(uniq, start, end):
        if e - s >= min_pts:
            g[c] = np.percentile(z_s[s:e], pct)
    G = g.reshape(shape)
    bad = ~np.isfinite(G)
    if bad.any() and distance_transform_edt is not None:
        idx = distance_transform_edt(bad, return_distances=False,
                                     return_indices=True)
        G = G[tuple(idx)]
    if median_filter is not None:
        G = median_filter(G, size=3)
    return GroundField(G.astype(np.float32), org, res)


# --------------------------------------------------------------------------- #
# occupancy
# --------------------------------------------------------------------------- #
@dataclass
class OccLayer:
    grid: np.ndarray               # (W,H) int8 of UNKNOWN/FREE/OCCUPIED, inflated
    res: float
    origin: np.ndarray             # world xy of cell (0,0) corner
    cfg: LayerCfg
    stats: Dict[str, float] = field(default_factory=dict)
    grid_raw: Optional[np.ndarray] = None   # before inflation
    ego_xy: Optional[np.ndarray] = None     # (N,2) keyframe origins, world frame

    # -------------------------------------------------------------- indexing
    def to_cell(self, xy: np.ndarray) -> Tuple[int, int]:
        ij = np.floor((np.asarray(xy, float)[:2] - self.origin) / self.res).astype(int)
        return int(ij[0]), int(ij[1])

    def to_world(self, cell: Tuple[int, int]) -> np.ndarray:
        return self.origin + (np.asarray(cell, float) + 0.5) * self.res

    def inside(self, i: int, j: int) -> bool:
        return 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]

    def passable(self, i: int, j: int) -> bool:
        return self.inside(i, j) and self.grid[i, j] != OCCUPIED


def band_select(pts: np.ndarray, cfg: LayerCfg,
                ground: Optional[GroundField] = None) -> np.ndarray:
    """Which returns count as obstacles (as opposed to road or overhangs)."""
    if ground is not None:
        dz = ground.dz(pts)
        return np.isfinite(dz) & (dz >= cfg.z_min) & (dz <= cfg.z_max)
    g = float(np.median(pts[:, 2])) if len(pts) else 0.0
    return (pts[:, 2] - g >= cfg.z_min) & (pts[:, 2] - g <= cfg.z_max)


def build_layer(keyframes, res: float = 1.0, cfg: Optional[LayerCfg] = None,
                verbose: bool = True, ground: Optional[GroundField] = None
                ) -> OccLayer:
    """Occupancy from keyframes by ray casting.  ``keyframes`` are (pose, pts).

    ``ground`` switches the obstacle band from "above a per-scan median" to
    "above the measured ground".  Pass it: the median version demonstrably marks
    the road itself as occupied.
    """
    cfg = _cfg_for(cfg, res)
    banded = []
    for pose, pts in keyframes:
        if len(pts) == 0:
            continue
        sel = band_select(pts, cfg, ground)
        if sel.sum() < 10:
            continue
        banded.append((pose, pts[sel]))
    return build_layer_from_banded(banded, res, cfg, verbose)


def _cfg_for(cfg: Optional[LayerCfg], res: float) -> LayerCfg:
    """Rebuild a config at a new resolution, forwarding *every* field.

    This used to drop ``ego_radius_m``, which silently turned an ablation of the
    ego footprint into two identical runs wearing different labels -- the kind of
    bug that makes a report look thorough and a conclusion wrong.
    """
    cfg = cfg or LayerCfg()
    return LayerCfg(res_m=res, z_min=cfg.z_min, z_max=cfg.z_max,
                    inflation_m=cfg.inflation_m, min_hits=cfg.min_hits,
                    ray_stride=cfg.ray_stride, max_ray_m=cfg.max_ray_m,
                    max_occ_m=cfg.max_occ_m, ego_radius_m=cfg.ego_radius_m)


def build_layer_from_banded(banded, res: float = 1.0,
                            cfg: Optional[LayerCfg] = None,
                            verbose: bool = True) -> OccLayer:
    """Same builder, but the caller has already chosen the obstacle band."""
    cfg = _cfg_for(cfg, res)
    t0 = time.perf_counter()
    origins, clouds = [], []
    for pose, pts in banded:
        if len(pts) < 10:
            continue
        clouds.append(pts)
        origins.append(pose[:3, 3])
    if not clouds:
        return OccLayer(np.zeros((1, 1), np.int8), res, np.zeros(2), cfg)
    P = np.concatenate(clouds)
    O = np.stack(origins)

    org = np.minimum(P[:, :2].min(0), O[:, :2].min(0)) - 5 * res
    hi = np.maximum(P[:, :2].max(0), O[:, :2].max(0)) + 5 * res
    shape = (int(np.ceil((hi[0] - org[0]) / res)) + 1,
             int(np.ceil((hi[1] - org[1]) / res)) + 1)
    hits = np.zeros(shape, np.int32)
    free = np.zeros(shape, bool)
    stride = max(1, cfg.ray_stride)
    for pts, o in zip(clouds, origins):
        d = np.linalg.norm(pts[:, :2] - o[None, :2], axis=1)
        # ---- hits: range gated ------------------------------------------- #
        near = d <= cfg.max_occ_m
        if near.any():
            k = np.floor((pts[near, :2] - org) / res).astype(np.int64)
            ok = ((k[:, 0] >= 0) & (k[:, 1] >= 0) &
                  (k[:, 0] < shape[0]) & (k[:, 1] < shape[1]))
            np.add.at(hits, (k[ok, 0], k[ok, 1]), 1)
        # ---- free space: rays out to max_ray_m --------------------------- #
        sel = d < cfg.max_ray_m
        p = pts[sel][::stride, :2]
        if len(p) == 0:
            continue
        _cast_rays(free, org, res, o[:2], p)

    grid = np.full(shape, UNKNOWN, np.int8)
    grid[free] = FREE
    grid[hits >= cfg.min_hits] = OCCUPIED
    # The platform was physically at every keyframe origin, so those cells are
    # traversable by definition.  Without this the planner regularly refuses to
    # start, because the ego cell picks up a few returns from a wall a metre away
    # and gets classified as occupied.
    cleared = clear_footprint(grid, org, res, O[:, :2], cfg.ego_radius_m)
    grid_raw = grid.copy()          # kept so clearance is measured to real obstacles

    stats = {
        "res_m": res, "shape": list(shape),
        "cells": int(shape[0] * shape[1]),
        "free_cells": int((grid == FREE).sum()),
        "occupied_cells": int((grid == OCCUPIED).sum()),
        "unknown_cells": int((grid == UNKNOWN).sum()),
        "free_area_m2": round(float((grid == FREE).sum()) * res * res, 1),
        "occupied_area_m2": round(float((grid == OCCUPIED).sum()) * res * res, 1),
        "inflation_m": 0.0,
        "z_band_m": [cfg.z_min, cfg.z_max],
        "max_occ_m": cfg.max_occ_m, "ego_radius_m": cfg.ego_radius_m,
        "ego_footprint_cells": int(cleared),
        "origins": int(len(O)), "points_used": int(len(P)),
        "ms": round((time.perf_counter() - t0) * 1e3, 1),
    }
    layer = OccLayer(grid, res, org, cfg, stats, grid_raw, ego_xy=O[:, :2].copy())
    if cfg.inflation_m and cfg.inflation_m > 0:
        layer = inflate(layer, cfg.inflation_m, verbose=False)
        layer.stats["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    if verbose:
        print(f"  layer {res:.2f} m: {shape[0]}x{shape[1]} cells, "
              f"{stats['free_cells']} free / "
              f"{int((layer.grid == OCCUPIED).sum())} occ, "
              f"inflate {layer.stats.get('inflation_effective_m', 0):.2f} m, "
              f"{layer.stats['ms']:.0f} ms", flush=True)
    return layer


def build_layer_from_points(P: np.ndarray, ego_xy: np.ndarray, res: float = 0.25,
                            cfg: Optional[LayerCfg] = None, ground=None,
                            verbose: bool = False) -> OccLayer:
    """Regional layer from a bare point set -- no ray casting.

    Why it exists: refining a coarse plan needs a fine layer *per window*, and
    rebuilding one from raw keyframes costs ~1.7 s because a window still touches
    a hundred scans.  Loading the pre-baked chunks instead (stage 2: 0.9-6 ms) only
    pays off if the pixels can be rasterised without the per-scan origins that ray
    casting needs -- and they can: obstacle cells come from returns, and free space
    is "no return in the band", which the planner covers with
    ``allow_unknown=True`` while the coarse plan keeps it inside the corridor.

    The honest caveat, measured in stage 6: this trades a ray-traced free space for
    an assumed one, so unobserved cells read as traversable.  That is acceptable
    inside a corridor and unacceptable as a global map, which is exactly why the
    global layer is still ray cast.
    """
    cfg = _cfg_for(cfg, res)
    cfg.inflation_m = 0.0
    t0 = time.perf_counter()
    P = np.asarray(P, np.float32)
    if ground is not None:
        dz = ground.dz(P)
        sel = np.isfinite(dz) & (dz >= cfg.z_min) & (dz <= cfg.z_max)
        P = P[sel]
    else:
        g = float(np.median(P[:, 2])) if len(P) else 0.0
        sel = (P[:, 2] - g >= cfg.z_min) & (P[:, 2] - g <= cfg.z_max)
        P = P[sel]
    ego_xy = np.asarray(ego_xy, np.float32).reshape(-1, 2)
    lo = np.minimum(P[:, :2].min(0), ego_xy.min(0)) - 5 * res
    hi = np.maximum(P[:, :2].max(0), ego_xy.max(0)) + 5 * res
    shape = (int(np.ceil((hi[0] - lo[0]) / res)) + 1,
             int(np.ceil((hi[1] - lo[1]) / res)) + 1)
    hits = np.zeros(shape, np.int32)
    k = np.floor((P[:, :2] - lo) / res).astype(np.int64)
    ok = ((k[:, 0] >= 0) & (k[:, 1] >= 0) &
          (k[:, 0] < shape[0]) & (k[:, 1] < shape[1]))
    np.add.at(hits, (k[ok, 0], k[ok, 1]), 1)
    grid = np.full(shape, UNKNOWN, np.int8)
    grid[hits >= cfg.min_hits] = OCCUPIED
    cleared = clear_footprint(grid, lo, res, ego_xy, cfg.ego_radius_m)
    grid_raw = grid.copy()
    stats = {"res_m": res, "shape": list(shape),
             "cells": int(shape[0] * shape[1]),
             "free_cells": int((grid == FREE).sum()),
             "occupied_cells": int((grid == OCCUPIED).sum()),
             "unknown_cells": int((grid == UNKNOWN).sum()),
             "free_area_m2": round(float((grid == FREE).sum()) * res * res, 1),
             "occupied_area_m2": round(float((grid == OCCUPIED).sum()) * res * res, 1),
             "inflation_m": 0.0, "z_band_m": [cfg.z_min, cfg.z_max],
             "max_occ_m": cfg.max_occ_m, "ego_radius_m": cfg.ego_radius_m,
             "ego_footprint_cells": int(cleared), "mode": "point_count",
             "origins": int(len(ego_xy)), "points_used": int(len(P)),
             "ms": round((time.perf_counter() - t0) * 1e3, 1)}
    layer = OccLayer(grid, res, lo, cfg, stats, grid_raw, ego_xy=ego_xy)
    if verbose:
        print(f"  region {res:.2f} m: {shape[0]}x{shape[1]}, "
              f"{stats['occupied_cells']} occ, {stats['ms']:.0f} ms", flush=True)
    return layer


def inflate(layer: OccLayer, meters: float, verbose: bool = False) -> OccLayer:
    """Inflate obstacles by a *metric* radius, without destroying the ego cells.

    Separate from construction on purpose: inflation is a planning-time safety
    knob, and on a coarse grid a small radius silently rounds to zero cells.  The
    effective radius is written into ``stats`` so a comparison between a 1.0 m and
    a 0.25 m layer cannot accidentally compare 0 m with 0.6 m of clearance.
    """
    raw = layer.grid_raw if layer.grid_raw is not None else layer.grid
    grid = raw.copy()
    cells = 0
    if distance_transform_edt is not None and meters > 0:
        dist_m = distance_transform_edt(raw != OCCUPIED) * layer.res
        grid[dist_m <= meters] = OCCUPIED
        cells = int(np.ceil(meters / layer.res))
    # re-assert the driven footprint *after* inflation, which is the whole point
    # of recording it: a 0.6 m inflation on a residential street eats the corridor
    # the vehicle just used, and every plan then fails at start with a 0 ms A*
    cleared = 0
    if layer.ego_xy is not None and layer.cfg.ego_radius_m > 0:
        cleared = clear_footprint(grid, layer.origin, layer.res,
                                  layer.ego_xy, layer.cfg.ego_radius_m)
    stats = dict(layer.stats)
    stats["inflation_m"] = float(meters)
    stats["inflation_cells"] = cells
    stats["inflation_effective_m"] = round(cells * layer.res, 3)
    stats["occupied_cells"] = int((grid == OCCUPIED).sum())
    stats["free_cells"] = int((grid == FREE).sum())
    if verbose:
        print(f"  inflate {meters:.2f} m -> {cells} cells "
              f"({stats['inflation_effective_m']:.2f} m effective)", flush=True)
    return OccLayer(grid, layer.res, layer.origin, layer.cfg, stats, raw,
                    ego_xy=layer.ego_xy)


def clear_footprint(grid: np.ndarray, org: np.ndarray, res: float,
                    ego_xy: np.ndarray, radius_m: float) -> int:
    """Mark every cell the platform's body occupied as FREE.  Returns cells hit.

    Cheap because the grid is small and the origins are few; a disc per origin is
    exact enough at 0.25 m and avoids the surprise of a one-cell rule that leaves
    the neighbours blocked.
    """
    if radius_m <= 0 or len(ego_xy) == 0:
        return 0
    r = int(np.ceil(radius_m / res))
    before = int((grid == FREE).sum())
    for o in ego_xy:
        c = np.floor((o - org) / res).astype(np.int64)
        i0, i1 = max(c[0] - r, 0), min(c[0] + r + 1, grid.shape[0])
        j0, j1 = max(c[1] - r, 0), min(c[1] + r + 1, grid.shape[1])
        if i0 >= i1 or j0 >= j1:
            continue
        d2 = ((np.arange(i0, i1)[:, None] - c[0]) ** 2 +
              (np.arange(j0, j1)[None, :] - c[1]) ** 2)
        sub = grid[i0:i1, j0:j1]
        sub[d2 * res * res <= radius_m ** 2] = FREE
    return int((grid == FREE).sum()) - before


def _cast_rays(free: np.ndarray, org: np.ndarray, res: float,
               start_xy: np.ndarray, ends: np.ndarray) -> None:
    """Vectorised DDA-ish: sample each ray at half-cell steps and mark free."""
    rel = ends - start_xy[None, :]
    steps = np.maximum(np.ceil(np.linalg.norm(rel, axis=1) / (0.5 * res)), 1).astype(int)
    # cap so a single 35 m ray costs ~70 samples, not thousands
    steps = np.minimum(steps, 140)
    for k in range(int(steps.max()) + 1):
        frac = np.minimum(k / np.maximum(steps, 1), 0.999)   # stop before the hit
        p = start_xy[None, :] + rel * frac[:, None]
        ij = np.floor((p - org) / res).astype(np.int64)
        ok = ((ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
              (ij[:, 0] < free.shape[0]) & (ij[:, 1] < free.shape[1]) &
              (steps >= k))
        free[ij[ok, 0], ij[ok, 1]] = True


# --------------------------------------------------------------------------- #
# A*
# --------------------------------------------------------------------------- #
_NB = [(-1, 0), (1, 0), (0, -1), (0, 1),
       (-1, -1), (-1, 1), (1, -1), (1, 1)]


def astar(layer: OccLayer, start_xy: np.ndarray, goal_xy: np.ndarray,
          allow_unknown: bool = False, max_expand: int = 400_000
          ) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, float]]:
    t0 = time.perf_counter()
    s = layer.to_cell(start_xy)
    g = layer.to_cell(goal_xy)
    info: Dict[str, float] = {"expanded": 0, "ms": 0.0, "found": 0}

    def ok(c):
        i, j = c
        if not layer.inside(i, j):
            return False
        v = layer.grid[i, j]
        if v == OCCUPIED:
            return False
        return allow_unknown or v != UNKNOWN

    if not ok(s) or not ok(g):
        # snap to the nearest passable cell; a goal that lands on a kerb is a user
        # error, not a reason to refuse to plan
        s = _snap(layer, s, ok)
        g = _snap(layer, g, ok)
        if s is None or g is None:
            info["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
            return None, info

    def h(c):
        return float(np.hypot(c[0] - g[0], c[1] - g[1]))

    openh = [(h(s), 0.0, s)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore = {s: 0.0}
    seen = set()
    while openh:
        _, gc, cur = heapq.heappop(openh)
        if cur in seen:
            continue
        seen.add(cur)
        info["expanded"] = len(seen)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            info["found"] = 1
            info["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
            return path, info
        if len(seen) > max_expand:
            break
        for di, dj in _NB:
            nb = (cur[0] + di, cur[1] + dj)
            if not ok(nb) or nb in seen:
                continue
            step = np.hypot(di, dj)
            ng = gc + step
            if ng < gscore.get(nb, np.inf):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(openh, (ng + h(nb), ng, nb))
    info["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
    return None, info


def _snap(layer: OccLayer, c: Tuple[int, int], ok, radius: int = 25):
    for r in range(1, radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                cand = (c[0] + di, c[1] + dj)
                if ok(cand):
                    return cand
    return None


def path_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def snap_to_free(layer: OccLayer, world: np.ndarray, max_m: float = 3.0
                 ) -> Tuple[np.ndarray, int]:
    """Move path vertices that land on an occupied cell to the nearest free cell.

    Needed because a hierarchical path is stitched from two layers of different
    resolution: each window ends on a *coarse* cell centre, and a 1.0 m coarse cell
    centre can sit inside a cell that the 0.25 m layer calls occupied.  Measured on
    ten pairs: 8.26 % of refined vertices were in collision by that criterion while
    the flat path had 0 %.  Inflating the regional layer did not help -- the
    offending vertices are the coarse waypoints, not the refinement -- so the fix
    belongs here, as a repair pass against the authoritative layer.
    """
    if distance_transform_edt is None or world is None or len(world) == 0:
        return world, 0
    occ = layer.grid_raw if layer.grid_raw is not None else layer.grid
    ij = np.floor((world[:, :2] - layer.origin) / layer.res).astype(np.int64)
    inside = ((ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
              (ij[:, 0] < layer.grid.shape[0]) & (ij[:, 1] < layer.grid.shape[1]))
    bad = np.ones(len(world), bool)
    bad[inside] = occ[ij[inside, 0], ij[inside, 1]] == OCCUPIED
    k = int(bad.sum())
    if k == 0:
        return world, 0
    idx = distance_transform_edt(occ == OCCUPIED, return_distances=False,
                                 return_indices=True)
    bi = ij[bad]
    bi = np.clip(bi, [0, 0], np.asarray(layer.grid.shape) - 1)
    ni, nj = idx[0][bi[:, 0], bi[:, 1]], idx[1][bi[:, 0], bi[:, 1]]
    new = world.copy()
    # keep the vertex's own height out of it: the planner is planar
    keep = (np.hypot(ni - bi[:, 0], nj - bi[:, 1]) * layer.res) <= max_m
    sel = np.where(bad)[0][keep]
    new[sel, 0] = layer.origin[0] + (ni[keep] + 0.5) * layer.res
    new[sel, 1] = layer.origin[1] + (nj[keep] + 0.5) * layer.res
    return new, int(len(sel))


def smooth_path(world: np.ndarray, iters: int = 40, alpha: float = 0.35,
                layer: Optional[OccLayer] = None) -> np.ndarray:
    """Shortcut-free Laplacian smoothing with the endpoints pinned.

    ``layer`` makes the smoothing collision constrained.  It has to be: the
    unconstrained version pulls each vertex toward the midpoint of its neighbours,
    which on a corridor whose inside corners are occupied shaves the corner until
    the path sits *inside* an obstacle.  Measured before the constraint: minimum
    clearance 0.00 m on all ten pairs, i.e. the "smoothed" path was in collision
    while the raw A* path was not.  Vertices whose smoothed position falls on an
    occupied or out-of-map cell keep their previous position instead.
    """
    if len(world) < 3:
        return world
    p = world.copy()
    for _ in range(iters):
        cand = p.copy()
        cand[1:-1] = p[1:-1] + alpha * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
        if layer is not None:
            ij = np.floor((cand[:, :2] - layer.origin) / layer.res).astype(np.int64)
            inside = ((ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
                      (ij[:, 0] < layer.grid.shape[0]) &
                      (ij[:, 1] < layer.grid.shape[1]))
            bad = ~inside
            bad[inside] = layer.grid[ij[inside, 0], ij[inside, 1]] == OCCUPIED
            cand[bad] = p[bad]
        p = cand
    return p


@dataclass
class HierPlan:
    coarse: Optional[np.ndarray]
    fine: Optional[np.ndarray]
    coarse_info: Dict[str, float]
    fine_info: Dict[str, float]
    segments: int
    fallback_segments: int

    def as_dict(self) -> dict:
        def arr(a):
            return None if a is None else a.round(3).tolist()
        return {
            "coarse": arr(self.coarse), "fine": arr(self.fine),
            "coarse_expanded": self.coarse_info.get("expanded", 0),
            "coarse_ms": self.coarse_info.get("ms", 0.0),
            "fine_expanded": self.fine_info.get("expanded", 0),
            "fine_ms": self.fine_info.get("ms", 0.0),
            "segments": self.segments,
            "fallback_segments": self.fallback_segments,
            "coarse_len_m": round(path_length(self.coarse), 2) if self.coarse is not None else None,
            "fine_len_m": round(path_length(self.fine), 2) if self.fine is not None else None,
        }


def hierarchical_plan(global_layer: OccLayer, start_xy: np.ndarray,
                      goal_xy: np.ndarray, fine_layer: Optional[OccLayer] = None,
                      segment_m: float = 15.0, allow_unknown: bool = False) -> HierPlan:
    """Coarse plan, then refine each coarse segment on the fine layer."""
    coarse_cells, cinfo = astar(global_layer, start_xy, goal_xy,
                               allow_unknown=allow_unknown)
    if coarse_cells is None:
        return HierPlan(None, None, cinfo, {}, 0, 0)
    coarse_world = np.stack([global_layer.to_world(c) for c in coarse_cells])

    if fine_layer is None:
        return HierPlan(coarse_world, None, cinfo, {}, 0, 0)

    # split the coarse path into ~segment_m waypoints
    d = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(coarse_world, axis=0), axis=1))])
    keep = [0]
    for target in np.arange(segment_m, d[-1], segment_m):
        keep.append(int(np.argmin(np.abs(d - target))))
    keep.append(len(d) - 1)
    keep = sorted(set(keep))
    wayp = coarse_world[keep]

    pieces, finfo = [], {"expanded": 0, "ms": 0.0, "found": 0}
    fallback = 0
    for i in range(len(wayp) - 1):
        cells, inf = astar(fine_layer, wayp[i], wayp[i + 1], allow_unknown=True)
        finfo["expanded"] += inf["expanded"]
        finfo["ms"] = round(finfo["ms"] + inf["ms"], 2)
        if cells is None:
            fallback += 1
            piece = np.stack([wayp[i], wayp[i + 1]])
        else:
            finfo["found"] += 1
            piece = np.stack([fine_layer.to_world(c) for c in cells])
        pieces.append(piece if i == 0 else piece[1:])
    if not pieces:
        # refinement produced nothing at all: falling back to the coarse path is
        # what a real stack does, and it must be counted, not silently returned as
        # a success with a None path
        return HierPlan(coarse_world, None, cinfo, finfo, len(wayp) - 1, len(wayp) - 1)
    fine_world = np.concatenate(pieces)
    fine_world = smooth_path(fine_world)
    return HierPlan(coarse_world, fine_world, cinfo, finfo,
                    len(wayp) - 1, fallback)


def clearance(layer: OccLayer, world: np.ndarray) -> float:
    """Minimum distance from a path to a *real* obstacle, in metres."""
    if distance_transform_edt is None or len(world) == 0:
        return float("nan")
    occ = layer.grid_raw if layer.grid_raw is not None else layer.grid
    dist = distance_transform_edt(occ != OCCUPIED) * layer.res
    ij = np.floor((world[:, :2] - layer.origin) / layer.res).astype(int)
    ok = ((ij[:, 0] >= 0) & (ij[:, 1] >= 0) &
          (ij[:, 0] < layer.grid.shape[0]) & (ij[:, 1] < layer.grid.shape[1]))
    if not ok.any():
        return float("nan")
    return float(dist[ij[ok, 0], ij[ok, 1]].min())
