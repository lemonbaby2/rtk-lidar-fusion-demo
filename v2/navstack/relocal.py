"""Place recognition and relocalisation: cheap recall first, expensive certainty after.

Theme 5 of the plan.  The pipeline is deliberately two-stage, because the two
stages fail in different ways and can therefore be reported separately:

**L1 -- Scan Context retrieval.**  Project the scan into a polar grid (20 rings x
60 sectors, max-z per bin) and compare with a cosine distance over column
vectors, minimising over all 60 sector rotations.  That gives rotation
invariance for free, and its ring key (mean z per ring) is a 20-D vector that can
be searched exhaustively in 0.2 ms for 154 places.  L1 maximises *recall*: it
must not lose the right place.

**L2 -- multi-hypothesis disambiguation.**  L1 returns places whose global shape
matches, which in a residential street means every straight section matches every
other.  L2 takes the top-K, aligns each by the L1 yaw shift, and verifies with
point-to-plane ICP on the actual clouds.  When several hypotheses survive, the
one whose verified transform agrees with the largest cluster of candidates wins
-- a single-frame ICP can be fooled by a symmetric scene, a cluster of
independent votes cannot be fooled consistently.

The evaluation is leave-one-out with a *temporal exclusion window*: for query
frame i, keyframes within +/-10 frames are removed from the database.  Without
that exclusion every query returns itself and Recall@1 is 100 % by construction,
which measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


# --------------------------------------------------------------------------- #
# L1: Scan Context
# --------------------------------------------------------------------------- #
@dataclass
class ScanContextCfg:
    max_radius: float = 50.0
    n_rings: int = 20
    n_sectors: int = 60
    z_clip: float = 4.0          # metres above the *local ground* that matter
    z_floor: float = -0.5        # a little below ground, so kerbs are kept


def scan_context(pts: np.ndarray, cfg: ScanContextCfg = ScanContextCfg()) -> np.ndarray:
    """(n_rings, n_sectors) descriptor of max height in each polar bin.

    Heights are measured **relative to the local ground**, estimated as the 2nd
    percentile of z.  Clipping absolute z instead is a silent disaster on body
    frame clouds: the sensor sits about 1.7 m up, so ``clip(z, 0, 3)`` throws away
    the road, the kerb and the lower half of every building, leaving a descriptor
    built from treetops.  Measured recall@1 was 0.068 with absolute clipping.
    """
    sc = np.zeros((cfg.n_rings, cfg.n_sectors), np.float32)
    if len(pts) == 0:
        return sc
    xy = pts[:, :2]
    r = np.linalg.norm(xy, axis=1)
    ok = (r > 1e-3) & (r < cfg.max_radius)
    if not ok.any():
        return sc
    r, xy, z = r[ok], xy[ok], pts[ok, 2]
    ground = float(np.percentile(z, 2))
    ring = np.minimum((r / cfg.max_radius * cfg.n_rings).astype(np.int64),
                      cfg.n_rings - 1)
    ang = (np.arctan2(xy[:, 1], xy[:, 0]) + np.pi) / (2 * np.pi)
    sec = np.minimum((ang * cfg.n_sectors).astype(np.int64), cfg.n_sectors - 1)
    zc = np.clip(z - ground, cfg.z_floor, cfg.z_clip)
    flat = ring * cfg.n_sectors + sec
    # max-z accumulation: sort by flat then take the last (max) of each run
    order = np.lexsort((zc, flat))
    f, zs = flat[order], zc[order]
    last = np.concatenate([[True], np.diff(f) > 0])
    idx = np.where(last)[0]
    sc.reshape(-1)[f[idx]] = zs[idx]
    return sc


def ring_key(sc: np.ndarray) -> np.ndarray:
    return sc.mean(1)


def sc_distance(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """Best Scan Context distance over sector rotations, and the shift found.

    This is the paper's distance, **column-wise cosine averaged over sectors**,
    not one cosine over the whole matrix.  A global cosine is dominated by the
    handful of tall bins (a building corner), so two unrelated streets with one
    similar facade score as a match; averaging per sector lets the many empty
    sectors outvote it, which is the whole point of the descriptor.

    A sector counts only if both column norms are non-zero, so an unobserved
    sector is ignored rather than rewarded.
    """
    if a.shape != b.shape:
        raise ValueError(f"scan context shape mismatch {a.shape} vs {b.shape}")
    ns = a.shape[1]
    na_col = np.linalg.norm(a, axis=0)
    best, best_shift = np.inf, 0
    for s in range(ns):
        br = np.roll(b, -s, axis=1)
        nb_col = np.linalg.norm(br, axis=0)
        valid = (na_col > 1e-9) & (nb_col > 1e-9)
        if not valid.any():
            continue
        cos = (a[:, valid] * br[:, valid]).sum(0) / (na_col[valid] * nb_col[valid])
        d = 1.0 - float(cos.mean())
        if d < best:
            best, best_shift = d, s
    return best, best_shift


# --------------------------------------------------------------------------- #
# L2: verification
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    kid: int
    seq: int
    l1_distance: float
    shift: int
    yaw_rad: float
    t_guess: np.ndarray
    icp_rmse: float = float("inf")
    icp_inliers: int = 0
    verified: bool = False
    trans_err_m: float = float("nan")
    yaw_err_deg: float = float("nan")


@dataclass
class RelocalResult:
    query_seq: int
    best: Optional[Hypothesis]
    hypotheses: List[Hypothesis]
    l1_ms: float
    l2_ms: float
    n_candidates: int
    cluster_votes: int


class Relocalizer:
    """Database of scan contexts + a lazily-built cloud cache."""

    def __init__(self, cfg: Optional[ScanContextCfg] = None,
                 topk: int = 10, verify_k: int = 5,
                 icp_accept_rmse: float = 0.18, icp_iters: int = 25,
                 max_l1_distance: float = 0.35, prefilter: Optional[int] = None,
                 icp_schedule: Sequence[Tuple[float, int]] = ((3.5, 12), (1.5, 12),
                                                              (0.8, 15))):
        self.cfg = cfg or ScanContextCfg()
        self.topk = topk
        self.verify_k = verify_k
        self.icp_accept_rmse = icp_accept_rmse
        self.icp_iters = icp_iters
        self.icp_schedule = tuple(icp_schedule)
        self.max_l1_distance = max_l1_distance
        # ``prefilter`` uses the ring key as a cheap shortlist, as in the original
        # Scan Context paper.  Left at None (full evaluation over every candidate)
        # because measured recall@5 was 0.25 with a 10-candidate ring-key
        # shortlist: the 20-D ring mean is too lossy to be a safe pre-filter on a
        # self-similar street, and full evaluation of 144 candidates costs ~15 ms.
        self.prefilter = prefilter
        self.seq_of: List[int] = []
        self.kid_of: List[int] = []
        self.sc: List[np.ndarray] = []
        self.rk: Optional[np.ndarray] = None
        self.poses: List[np.ndarray] = []
        self._clouds: Dict[int, np.ndarray] = {}
        self._kid_index: Dict[int, int] = {}

    # ------------------------------------------------------------------ build
    def add(self, kid: int, seq: int, pts: np.ndarray, pose: np.ndarray) -> None:
        self._kid_index[kid] = len(self.seq_of)
        self.seq_of.append(seq)
        self.kid_of.append(kid)
        self.sc.append(scan_context(pts, self.cfg))
        self.poses.append(np.asarray(pose, float))
        self._clouds[kid] = pts
        self.rk = None

    def finalize(self) -> None:
        self.rk = (np.stack([ring_key(s) for s in self.sc])
                   if self.sc else np.zeros((0, self.cfg.n_rings)))

    # --------------------------------------------------------------- retrieval
    def query(self, pts: np.ndarray, query_seq: int,
              exclude_window: int = 10, truth_pose: Optional[np.ndarray] = None,
              tol_frames: int = 5) -> RelocalResult:
        import time as _t
        if self.rk is None:
            self.finalize()
        q = scan_context(pts, self.cfg)
        qk = ring_key(q)

        t0 = _t.perf_counter()
        # shortlist: every candidate outside the temporal exclusion window, unless
        # a ring-key prefilter was explicitly requested
        cand_all = [i for i in range(len(self.seq_of))
                    if abs(self.seq_of[i] - query_seq) > exclude_window]
        n_cand = len(cand_all)
        if self.prefilter:
            dk = np.linalg.norm(self.rk[cand_all] - qk[None, :], axis=1) / np.sqrt(len(qk))
            cand = [cand_all[int(k)] for k in np.argsort(dk)[:self.prefilter]]
        else:
            cand = cand_all
        scored = []
        for i in cand:
            d, s = sc_distance(q, self.sc[i])
            scored.append(Hypothesis(
                kid=self.kid_of[i], seq=self.seq_of[i], l1_distance=d, shift=s,
                yaw_rad=2 * np.pi * s / self.cfg.n_sectors,
                t_guess=np.zeros(3)))
        scored.sort(key=lambda h: h.l1_distance)
        t_l1 = (_t.perf_counter() - t0) * 1e3

        # ---- L2 verify ------------------------------------------------ #
        t1 = _t.perf_counter()
        verified: List[Hypothesis] = []
        for h in scored[:self.verify_k]:
            cand_pose = self._pose_of(h.kid)
            cand_pts = self._clouds[h.kid]
            # yaw shift: b rolled left by s matches a, so the query is rotated by
            # -yaw relative to the candidate
            Rz = _rotz(-h.yaw_rad)
            T0 = np.eye(4)
            T0[:3, :3] = Rz
            T_ref, rmse, inl = self._verify(pts, cand_pts, T0)
            h.icp_rmse, h.icp_inliers = rmse, inl
            # acceptance needs both a good geometric fit and a plausible global
            # descriptor: rmse alone accepts any place whose surfaces happen to be
            # roughly planar, which on a residential street is every place
            h.verified = (rmse <= self.icp_accept_rmse
                          and h.l1_distance <= self.max_l1_distance)
            # The verified pose must come from the *refined* transform, not from
            # the candidate keyframe's own pose.  Returning the keyframe pose caps
            # the achievable accuracy at the keyframe spacing: with a 10-frame
            # temporal exclusion on a 0.45 m/frame drive the nearest usable
            # keyframe is ~4.5 m away, so a 3 m success criterion could never be
            # met no matter how good the matcher was (measured: 0.000 success with
            # 0.795 recall@1).
            T_est = cand_pose @ T_ref
            h.t_guess = T_est[:3, 3].copy()
            if truth_pose is not None:
                h.trans_err_m = float(np.linalg.norm(T_est[:3, 3] - truth_pose[:3, 3]))
                R_rel = T_est[:3, :3].T @ truth_pose[:3, :3]
                c = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
                h.yaw_err_deg = float(np.degrees(np.arccos(c)))
            verified.append(h)
        t_l2 = (_t.perf_counter() - t1) * 1e3

        # ---- cluster voting ------------------------------------------- #
        votes = 0
        best = None
        if verified:
            acc = [h for h in verified if h.verified]
            pool = acc if acc else verified
            # cluster by spatial proximity in the *map* (3.0 m) and pick the
            # largest cluster; break ties by ICP rmse
            labels = [-1] * len(pool)
            for i, h in enumerate(pool):
                if labels[i] >= 0:
                    continue
                grp = [j for j, g in enumerate(pool)
                       if np.linalg.norm(g.t_guess - h.t_guess) <= 3.0]
                for j in grp:
                    labels[j] = i
            groups: Dict[int, List[Hypothesis]] = {}
            for i, h in enumerate(pool):
                groups.setdefault(labels[i], []).append(h)
            biggest = max(groups.values(), key=lambda g: (len(g), -min(x.icp_rmse for x in g)))
            votes = len(biggest)
            best = min(biggest, key=lambda h: h.icp_rmse)
        return RelocalResult(query_seq=query_seq, best=best, hypotheses=verified,
                             l1_ms=round(t_l1, 3), l2_ms=round(t_l2, 3),
                             n_candidates=n_cand, cluster_votes=votes)

    # ------------------------------------------------------------------ utils
    def _pose_of(self, kid: int) -> np.ndarray:
        return self.poses[self._kid_index[kid]]

    def _verify(self, q: np.ndarray, c: np.ndarray, T0: np.ndarray
                ) -> Tuple[np.ndarray, float, int]:
        """Refined ``query_body -> candidate_body`` transform, its rmse and inliers.

        Coarse to fine, because a single point-to-plane pass from a yaw-only
        initial guess cannot cross the gap to the candidate: with a temporal
        exclusion window the nearest usable keyframe is several metres away, far
        outside the 1.2 m correspondence radius, so a one-shot ICP returns
        essentially its own initial guess.  Measured: rmse 0.10 (the surfaces do
        match) but 4-5 m of position error, i.e. the keyframe spacing leaking
        straight through.  Widening the radius first, then tightening it, lets the
        solution slide along the road before it is asked for precision.
        """
        if torch is None:
            return T0, float("inf"), 0
        from .lio import build_reg_cloud, icp_point_to_plane
        tgt = build_reg_cloud(c, max_range=45.0)
        src = build_reg_cloud(q, max_range=45.0)
        T = torch.as_tensor(T0, dtype=torch.float32)
        st = None
        for max_dist, iters in self.icp_schedule:
            T, st = icp_point_to_plane(src.points, tgt.centers, tgt.normals,
                                       T0=T, iters=iters, max_dist=max_dist,
                                       max_resid=0.40)
        return (T.detach().cpu().numpy().astype(np.float64),
                float(st.rmse), int(st.inliers))


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
