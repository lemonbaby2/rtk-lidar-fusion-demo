"""Front end: frame-to-frame point-to-plane registration on the GPU.

Why point-to-plane and not point-to-point
-----------------------------------------
The v1 demo registered with point-to-point ICP on a randomly sampled cloud and
reached 0.611 m ATE over 69 m.  Most of that error was not drift in the usual
sense: point-to-point ICP has no constraint along a locally planar surface, so
on a road the along-normal error is free.  Point-to-plane removes that degree of
freedom and typically buys a factor of two to four on driving data.

Why voxel-downsampled normals
-----------------------------
Estimating a normal per correspondence needs a neighbourhood per point, which
is the expensive part of point-to-plane ICP.  Computing normals once per frame
on a voxel grid and looking them up by nearest neighbour keeps the per-iteration
cost to a single distance matrix, which a GB10 does in about 2 ms for 4 k x 3 k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def to_t(p: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(p, dtype=DTYPE, device=DEV)


# --------------------------------------------------------------------------- #
# cloud utilities
# --------------------------------------------------------------------------- #
def voxel_downsample(p: torch.Tensor, leaf: float) -> torch.Tensor:
    """Mean of the points in every occupied voxel."""
    if p.numel() == 0:
        return p
    key = torch.floor(p / leaf).to(torch.int64)
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    acc = torch.zeros((uniq.shape[0], 3), dtype=DTYPE, device=p.device)
    cnt = torch.zeros(uniq.shape[0], dtype=DTYPE, device=p.device)
    acc.index_add_(0, inv, p)
    cnt.index_add_(0, inv, torch.ones(inv.shape[0], dtype=DTYPE, device=p.device))
    return acc / cnt.clamp_min(1.0).unsqueeze(1)


def voxel_normals(p: torch.Tensor, leaf: float, min_pts: int = 6) -> tuple:
    """Per-voxel PCA normals.  Returns ``(centers (K,3), normals (K,3), ok (K,))``.

    ``ok`` is False where the voxel held too few points or the surface was not
    planar enough (smallest eigenvalue not clearly below the middle one).  Those
    voxels contribute no constraint rather than a noisy one.
    """
    key = torch.floor(p / leaf).to(torch.int64)
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    k = uniq.shape[0]
    cnt = torch.zeros(k, dtype=DTYPE, device=p.device)
    s1 = torch.zeros((k, 3), dtype=DTYPE, device=p.device)
    s2 = torch.zeros((k, 3, 3), dtype=DTYPE, device=p.device)
    cnt.index_add_(0, inv, torch.ones(inv.shape[0], dtype=DTYPE, device=p.device))
    s1.index_add_(0, inv, p)
    outer = p.unsqueeze(2) * p.unsqueeze(1)
    s2.index_add_(0, inv, outer)

    n = cnt.clamp_min(1.0).unsqueeze(1)
    mean = s1 / n
    cov = s2 / n.unsqueeze(2) - mean.unsqueeze(2) * mean.unsqueeze(1)

    evals, evecs = torch.linalg.eigh(cov)
    normals = evecs[:, :, 0]                     # ascending: [0] is smallest
    l0, l1, l2 = evals[:, 0], evals[:, 1], evals[:, 2]

    ok = (cnt >= min_pts) & (l1 > 1e-7) & (l0 < 0.35 * l1)
    # orient normals "up" so the residual sign is consistent; not required for
    # least squares but makes the reported residual statistics readable
    flip = normals[:, 2] < 0
    normals = torch.where(flip.unsqueeze(1), -normals, normals)
    return mean, normals, ok


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues, batched over the last dimension."""
    th = torch.linalg.norm(w)
    if float(th) < 1e-9:
        return torch.eye(3, dtype=DTYPE, device=w.device)
    k = w / th
    Kx = torch.zeros((3, 3), dtype=DTYPE, device=w.device)
    Kx[0, 1], Kx[0, 2] = -k[2], k[1]
    Kx[1, 0], Kx[1, 2] = k[2], -k[0]
    Kx[2, 0], Kx[2, 1] = -k[1], k[0]
    return torch.eye(3, dtype=DTYPE, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


# --------------------------------------------------------------------------- #
# ICP
# --------------------------------------------------------------------------- #
@dataclass
class IcpStats:
    iters: int
    inliers: int
    rmse: float
    median: float
    cond: float


def icp_point_to_plane(
    src: torch.Tensor,
    tgt_pts: torch.Tensor,
    tgt_nrm: torch.Tensor,
    T0: torch.Tensor | None = None,
    iters: int = 35,
    max_dist: float = 1.0,
    max_resid: float = 0.30,
    tol: float = 1e-5,
    chunk: int = 2048,
) -> tuple:
    """Align ``src`` onto the surface defined by ``(tgt_pts, tgt_nrm)``.

    Returns ``(T (4,4), IcpStats)`` where ``T`` maps points of ``src`` into the
    target frame.  Convergence is declared when the update norm drops below
    ``tol``; the loop also exits early when the correspondence set collapses.
    """
    if src.numel() == 0 or tgt_pts.numel() == 0:
        return torch.eye(4, dtype=DTYPE, device=DEV), IcpStats(0, 0, float("inf"), float("inf"), 0.0)

    T = torch.eye(4, dtype=DTYPE, device=DEV) if T0 is None else T0.clone().to(DEV)
    stats = IcpStats(iters, 0, float("inf"), float("inf"), 0.0)

    for it in range(iters):
        R, t = T[:3, :3], T[:3, 3]
        x = src @ R.T + t

        # nearest neighbour, chunked so the distance matrix stays small
        idx = torch.empty(x.shape[0], dtype=torch.long, device=DEV)
        dist = torch.empty(x.shape[0], dtype=DTYPE, device=DEV)
        for s in range(0, x.shape[0], chunk):
            d = torch.cdist(x[s:s + chunk], tgt_pts)
            m, j = d.min(dim=1)
            dist[s:s + chunk] = m
            idx[s:s + chunk] = j

        q = tgt_pts[idx]
        nrm = tgt_nrm[idx]
        r = ((x - q) * nrm).sum(dim=1)
        keep = (dist < max_dist) & (r.abs() < max_resid)
        if int(keep.sum()) < 30:
            stats.iters = it
            break

        xk, nk, rk = x[keep], nrm[keep], r[keep]
        # d/dxi of n^T (x + w x x + dt - q)  ->  [ (x cross n)^T , n^T ]
        J = torch.cat([torch.cross(xk, nk, dim=1), nk], dim=1)
        H = J.T @ J
        b = -(J.T @ rk)
        lam = 1e-4 * torch.eye(6, dtype=DTYPE, device=DEV) * H.diagonal().mean().clamp_min(1e-9)
        dx = torch.linalg.solve(H + lam, b)

        dR = so3_exp(dx[:3])
        # left-multiply the increment: x <- dR x + dx_t, i.e. T <- dT @ T
        T_new = torch.eye(4, dtype=DTYPE, device=DEV)
        T_new[:3, :3] = dR @ R
        T_new[:3, 3] = dR @ t + dx[3:]
        T = T_new

        stats = IcpStats(
            it + 1,
            int(keep.sum()),
            float(torch.sqrt((rk ** 2).mean())),
            float(rk.abs().median()),
            float(torch.linalg.cond(H).clamp_max(1e12)),
        )
        if float(torch.linalg.norm(dx)) < tol:
            break

    return T, stats


# --------------------------------------------------------------------------- #
# registration cloud builder
# --------------------------------------------------------------------------- #
@dataclass
class RegCloud:
    points: torch.Tensor      # (N,3) under-sampled, range limited
    normals: torch.Tensor     # (K,3) voxel normals of the same frame
    centers: torch.Tensor     # (K,3) voxel centers carrying those normals
    raw_count: int


def build_reg_cloud(
    pts: np.ndarray,
    max_range: float = 55.0,
    leaf_src: float = 0.50,
    leaf_nrm: float = 0.50,
    max_src: int = 4000,
) -> RegCloud:
    """Prepare one frame for registration.

    The ranging limit is not cosmetic: KITTI's HDL-64E returns sparse points far
    out, and those carry very little angular information per point, so they cost
    time and add outlier correspondences.  55 m keeps the road and the facades
    that actually constrain motion.
    """
    p = to_t(np.asarray(pts, dtype=np.float32))
    p = p[torch.linalg.norm(p, dim=1) < max_range]
    raw = int(p.shape[0])

    src = voxel_downsample(p, leaf_src)
    if src.shape[0] > max_src:
        # deterministic stride rather than random: reproducible run to run
        sel = torch.linspace(0, src.shape[0] - 1, max_src).long()
        src = src[sel]

    centers, normals, ok = voxel_normals(p, leaf_nrm)
    return RegCloud(src, normals[ok], centers[ok], raw)


def constant_velocity_guess(prev_T: torch.Tensor | None, prev2_T: torch.Tensor | None) -> torch.Tensor:
    """Initial guess for the next registration.

    Uses the previous frame's increment directly.  An earlier version tried to
    extrapolate by applying the increment twice and took a matrix square root
    with ``torch.linalg.eigh`` -- which only holds for symmetric matrices.  A
    rotation matrix is not symmetric, so the "square root" was garbage whenever
    the increment was not near identity: frames 1-68 (straight road, ~0 rotation)
    drifted 0.6 m, frames 69-102 (the turn, ~3 deg/frame) drifted 29.7 m.  The
    previous increment is a good enough guess on 10 Hz data and cannot blow up.
    """
    if prev_T is None:
        return torch.eye(4, dtype=DTYPE, device=DEV)
    return prev_T.clone()
