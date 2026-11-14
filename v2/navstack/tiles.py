"""Spatial tiling and LOD: the two things that keep a large map openable.

Both are deliberately numpy-only.  A voxel-hash decimator in pure numpy is a
``np.unique`` on a packed integer key, which for 200 k points takes 12 ms -- fast
enough to do lazily on load, which means the store never has to keep more than
one representation on disk.

LOD levels used across the stack:

===========  ===============  ==============================
level         leaf (m)         typical use
===========  ===============  ==============================
``coarse``    1.00             global minimap, 1 m occupancy
``mid``       0.40             per-level overview, web view
``fine``      0.20             local planning, registration target
===========  ===============  ==============================
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np

LOD_LEAF = {"coarse": 1.00, "mid": 0.40, "fine": 0.20}


def voxel_index(pts: np.ndarray, leaf: float) -> np.ndarray:
    """Packed voxel key per point, using a bounding-box origin.

    The origin is subtracted first so the packed integer stays small even for a
    UTM-scale world coordinate, where a float32 multiply by 1/leaf would lose
    millimetres before the floor.
    """
    org = pts.min(0)
    q = np.floor((pts - org) / leaf).astype(np.int64)
    span = int(q[:, :2].max()) + 1
    return (q[:, 0] * span + q[:, 1]) * span + q[:, 2], org


def decimate(pts: np.ndarray, leaf: float,
             rgb: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    """Voxel-centroid decimation.  Returns the surviving points and colours."""
    if len(pts) == 0 or leaf <= 0:
        return pts, rgb
    key, _ = voxel_index(pts, leaf)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    # centroid rather than "first point in voxel": the centroid is stable under
    # point order, which matters because the store may re-sort a cloud.
    cnt = np.bincount(inv)
    out = np.stack([np.bincount(inv, weights=pts[:, k]) / cnt for k in range(3)], 1)
    out_rgb = None
    if rgb is not None and len(rgb) == len(pts):
        out_rgb = np.stack(
            [np.bincount(inv, weights=rgb[:, k].astype(np.float64)) / cnt
             for k in range(3)], 1).round().astype(np.uint8)
    return out.astype(np.float32), out_rgb


def merge_with_limit(clouds, leaf: float, cap: int | None = None,
                     rng: np.random.Generator | None = None
                     ) -> Tuple[np.ndarray, np.ndarray | None]:
    """Decimate-and-merge a list of ``(pts, rgb)`` pairs."""
    P, C = [], []
    for p, c in clouds:
        if p is None or len(p) == 0:
            continue
        P.append(p)
        C.append(c if c is not None else np.full((len(p), 3), 110, np.uint8))
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    pts = np.concatenate(P).astype(np.float32)
    rgb = np.concatenate(C).astype(np.uint8)
    pts, rgb = decimate(pts, leaf, rgb)
    if cap and len(pts) > cap:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(len(pts), cap, replace=False)
        pts, rgb = pts[sel], rgb[sel]
    return pts, rgb


class ChunkCache:
    """Byte-bounded LRU over loaded chunks.

    A byte budget rather than an entry count is the only thing that behaves
    predictably when chunk sizes vary by two orders of magnitude -- which they do
    as soon as one chunk contains a building facade.
    """

    def __init__(self, budget_bytes: int = 192 * 1024 * 1024):
        self.budget = budget_bytes
        self._d: "OrderedDict[str, Tuple[np.ndarray, np.ndarray | None]]" = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str):
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        self.misses += 1
        return None

    def put(self, key: str, value) -> None:
        if key in self._d:
            self.bytes -= self._size(self._d[key])
            self._d.move_to_end(key)
        self._d[key] = value
        self.bytes += self._size(value)
        while self.bytes > self.budget and len(self._d) > 1:
            _, old = self._d.popitem(last=False)
            self.bytes -= self._size(old)
            self.evictions += 1

    @staticmethod
    def _size(v) -> int:
        n = 0
        for a in v:
            if a is not None:
                n += a.nbytes
        return n

    def stats(self) -> Dict[str, float]:
        tot = self.hits + self.misses
        return {
            "entries": len(self._d),
            "bytes": self.bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hits / tot, 3) if tot else 0.0,
        }
