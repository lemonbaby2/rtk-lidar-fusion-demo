"""Shared plumbing: load the numpy bundle, turn poses into keyframes."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import schema as S

KEYS = ("pts", "rgb", "pose_rtk", "rtk_pos", "rtk_R_w2f0", "frame_t",
        "imu_t", "imu_acc", "imu_gyro")


def load_bundle(d: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k in KEYS:
        p = os.path.join(d, k + ".npy")
        if os.path.exists(p):
            out[k] = np.load(p)
    with open(os.path.join(d, "meta.json")) as f:
        out["meta"] = json.load(f)
    return out


def poses_from(out_dir: str, prefer: Sequence[str] = ("graph_full", "graph_lio_rtk",
                                                      "best", "dr")) -> Optional[np.ndarray]:
    """First available pose file from a previous stage, in preference order."""
    for name in prefer:
        p = os.path.join(out_dir, f"{name}_poses.npy")
        if os.path.exists(p):
            return np.load(p)
    return None


def to_world(pts: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Body-frame cloud -> frame0 cloud.  The single place this happens."""
    R = pose[:3, :3]
    t = pose[:3, 3]
    return (pts @ R.T + t[None, :]).astype(np.float32)


def to_body(pts_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    R = pose[:3, :3]
    t = pose[:3, 3]
    return ((pts_world - t[None, :]) @ R).astype(np.float32)


def make_keyframes(poses: np.ndarray, pts: np.ndarray, rgb: Optional[np.ndarray],
                   frame_t: np.ndarray, stride: int = 1,
                   level: int = 0, world: bool = True
                   ) -> List[Tuple[S.Keyframe, np.ndarray, Optional[np.ndarray]]]:
    """(Keyframe, cloud, colour) triples, decimating the source by ``stride``.

    With ``world=True`` the cloud is already in frame0, which is what the store
    and every downstream stage expect.
    """
    out = []
    n = min(len(poses), len(pts), len(frame_t))
    kid = 0
    for i in range(0, n, max(1, stride)):
        c = pts[i]
        keep = np.linalg.norm(c, axis=1) < 79.0
        c = c[keep]
        col = rgb[i][keep] if rgb is not None else None
        if world:
            c = to_world(c, poses[i])
        kf = S.Keyframe(id=kid, t=float(frame_t[i]),
                        pose=np.asarray(poses[i], np.float64),
                        n_points=int(len(c)), level=level,
                        chunks=[], seq=int(i))
        out.append((kf, c, col))
        kid += 1
    return out
