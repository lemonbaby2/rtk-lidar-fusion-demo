"""Map schema: the on-disk contract between the mapping run and every consumer.

Design notes (these are the parts that bite later, so they are stated up front):

* **Schema version is checked, not migrated.**  ``SCHEMA_VERSION`` is compared on
  open; a newer store refuses to load in an older reader instead of silently
  mis-indexing.  Map formats change shape far more often than they change
  meaning, and a half-parsed map is worse than no map.

* **One keyframe = one file.**  A 40 m drive is 154 files of ~200 KB.  Rewriting
  the whole map on every new keyframe would be O(n^2); appending one file and
  fsyncing the directory is O(1).  The manifest is the only file that is
  rewritten, and it is rewritten atomically.

* **Poses live in ``frame0``**, the IMU body frame at the first keyframe with
  x along travel and z along measured gravity (see ``imu.frame0_rotation``).
  Every consumer that renders or plans in metres therefore needs no per-source
  transform, which is the single most common source of "the map is rotated 90
  degrees" bugs.

* **Levels are a first-class field, not a filter applied later.**  A keyframe
  belongs to exactly one level; the loader can then skip whole floors without
  touching their chunks (theme 4).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 2

# Chunk geometry.  30 m chunks with a 5 m load buffer is the x_nav default: a
# 30 m chunk covers ~1.5 s of driving at 70 km/h, so a local planner never needs
# more than the current chunk plus its neighbours, and the buffer stops the
# loader from thrashing at a chunk boundary the way a zero-margin layout does.
CHUNK_SIZE_M = 30.0
CHUNK_BUFFER_M = 5.0
DEFAULT_LOD_M = 0.20


@dataclass
class Level:
    """One floor of the map."""

    id: int
    z_min: float
    z_max: float
    z_ref: float                      # median ground height, the planner's datum
    keyframes: List[int] = field(default_factory=list)
    n_points: int = 0

    def contains(self, z: float, margin: float = 0.3) -> bool:
        return (self.z_min - margin) <= z <= (self.z_max + margin)


@dataclass
class Keyframe:
    """A registered scan plus everything needed to use it without the raw data.

    ``pose`` is ``frame0 <- body`` and the stored cloud is expected to be **in
    frame0 (world) coordinates**, already rotated and translated by ``pose``.
    Storing body-frame clouds instead looks tidier but breaks every consumer:
    chunk keys, occupancy grids and the multi-level bake are all world-space
    operations, and any one of them silently produces a map that is a pile of
    scans on top of each other.  ``bundle.to_world()`` is the one place that
    conversion happens.
    """

    id: int
    t: float                          # unix seconds, camera clock
    pose: np.ndarray                  # (4,4) frame0 <- body
    n_points: int
    level: int
    chunks: List[str]                 # chunk keys this scan contributes to
    seq: int = 0                      # index in the original sequence
    lod_points: int = 0
    # vision annotations (theme 3) -- kept per keyframe so a re-bake never needs
    # to re-run the network
    dynamic_frac: float = 0.0
    descriptor: Optional[np.ndarray] = None

    def xy(self) -> np.ndarray:
        return self.pose[:3, 3]

    def yaw(self) -> float:
        return float(np.arctan2(self.pose[1, 0], self.pose[0, 0]))


@dataclass
class MapMeta:
    schema_version: int = SCHEMA_VERSION
    map_id: str = "unnamed"
    created_unix: float = field(default_factory=time.time)
    frames: int = 0
    duration_s: float = 0.0
    path_length_m: float = 0.0
    chunk_size_m: float = CHUNK_SIZE_M
    chunk_buffer_m: float = CHUNK_BUFFER_M
    level_count: int = 1
    bbox_min: Optional[List[float]] = None
    bbox_max: Optional[List[float]] = None
    total_points: int = 0
    n_keyframes: int = 0
    source: str = ""
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def chunk_key(xy: np.ndarray, size: float = CHUNK_SIZE_M) -> str:
    """Chunk key for a world xy.  Signed floor, so -0.4 and +0.4 differ."""
    return f"{int(np.floor(xy[0] / size)):+03d}_{int(np.floor(xy[1] / size)):+03d}"


def chunks_for_points(pts: np.ndarray, size: float = CHUNK_SIZE_M,
                      min_pts: int = 12) -> List[str]:
    """Every chunk a scan touches, dropping chunks with a negligible footprint.

    The ``min_pts`` floor matters: a single stray return 40 m away would
    otherwise create a chunk that holds three points and costs a file open
    forever.
    """
    if len(pts) == 0:
        return []
    ci = np.floor(pts[:, 0] / size).astype(np.int64)
    cj = np.floor(pts[:, 1] / size).astype(np.int64)
    flat = ci * 100000 + cj
    uniq, cnt = np.unique(flat, return_counts=True)
    out = []
    for f, c in zip(uniq, cnt):
        if c >= min_pts:
            out.append(f"{int(f // 100000):+03d}_{int(f % 100000):+03d}")
    return sorted(out)


def neighbours_of(key: str, radius: int = 1) -> List[str]:
    """Chunk keys within a square radius, for the load buffer."""
    i, j = (int(x) for x in key.replace("+", "").split("_"))
    out = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            out.append(f"{i + di:+03d}_{j + dj:+03d}")
    return out


def keys_in_disc(center_xy: np.ndarray, radius: float,
                 size: float = CHUNK_SIZE_M) -> List[str]:
    """Chunk keys whose cell intersects a disc.  Used by the on-demand loader."""
    lo = np.floor((center_xy - radius) / size).astype(int)
    hi = np.floor((center_xy + radius) / size).astype(int)
    out = []
    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            # cheap cell-vs-disc reject on the closest point of the cell
            cx = (i + 0.5) * size
            cy = (j + 0.5) * size
            dx = max(abs(cx - center_xy[0]) - size / 2, 0.0)
            dy = max(abs(cy - center_xy[1]) - size / 2, 0.0)
            if dx * dx + dy * dy <= radius * radius:
                out.append(f"{i:+03d}_{j:+03d}")
    return out


def to_json(obj: Any) -> str:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(f"not serialisable: {type(o)}")
    return json.dumps(obj, indent=2, default=default)


def bbox_of(points: np.ndarray) -> Tuple[List[float], List[float]]:
    return points.min(0).round(3).tolist(), points.max(0).round(3).tolist()
