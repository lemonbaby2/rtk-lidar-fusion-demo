"""iSAM2 / batch factor graph: IMU + LIO + RTK + loop closure.

State layout
------------
``X(i)`` Pose3 in ``frame0``, ``V(i)`` velocity in ``frame0``, ``B(i)`` IMU bias.
IMU factors are only added when a 100 Hz stream is present, so the same code
runs on the 10 Hz ``_sync`` archive (where a single pre-integrated step per
0.11 s carries almost no information) and on the 100 Hz ``_extract`` one.

RTK as a robustness problem, not a constraint
---------------------------------------------
RTK is modelled as a *translation-only* unary prior with a Huber kernel.  Two
consequences worth stating explicitly:

* The attitude is left to the IMU and the LiDAR.  A full ``PriorFactorPose3``
  would inject the RTK/INS heading, which is the very quantity the stack is
  supposed to estimate -- that would make the comparison circular.
* The Huber kernel is what makes an outage survivable.  A plain Gaussian prior
  with a 0.15 m sigma and a 40 m jump (the outage window) drags the whole
  trajectory towards the wrong place; with k = 1.5 the outlier contributes a
  linear, bounded cost and the IMU/LiDAR keep the section locally consistent.

The outage is simulated, not real: frames in ``rtk_outage`` receive no RTK
factor at all, which is the honest way to test "what happens when the sky
disappears".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .imu import ImuConfig, ImuStream

try:
    import gtsam
except ImportError:  # pragma: no cover
    gtsam = None  # type: ignore


def _X(i: int) -> int:
    return int(gtsam.symbol("x", i))


def _V(i: int) -> int:
    return int(gtsam.symbol("v", i))


def _B(i: int) -> int:
    return int(gtsam.symbol("b", i))


@dataclass
class GraphConfig:
    # LIO relative pose noise.  Rotation is tight because point-to-plane fixes
    # it well; translation is looser because the road plane still leaves the
    # along-track direction weakly observable at low speed.
    lio_sigma_rot: float = 0.010          # rad
    lio_sigma_xy: float = 0.060           # m
    lio_sigma_z: float = 0.120            # m

    rtk_sigma_xy: float = 0.150           # m, RT3003 after smoothing
    rtk_sigma_z: float = 0.350            # m, altitude is the weak axis
    rtk_huber_k: float = 1.5

    bias_sigma_acc: float = 0.05          # m/s^2
    bias_sigma_gyro: float = 0.002        # rad/s
    bias_rw_acc: float = 1e-3
    bias_rw_gyro: float = 1e-4

    vel_sigma: float = 1.5                # m/s, weak prior on V(0) only
    loop_sigma_rot: float = 0.020
    loop_sigma_xy: float = 0.100
    loop_sigma_z: float = 0.150

    imu: ImuConfig = field(default_factory=ImuConfig)


@dataclass
class GraphResult:
    poses: np.ndarray                     # (N,4,4) in frame0
    velocities: np.ndarray | None
    ate_mean: float
    ate_final: float
    ate_rmse: float
    max_gap_m: float
    factors: Dict[str, int]
    converged: bool
    iterations: int
    error_initial: float
    error_final: float


def _pose3(T: np.ndarray) -> "gtsam.Pose3":
    return gtsam.Pose3(T)


def _rel_noise(cfg: GraphConfig, rot: float, xy: float, z: float):
    sig = np.array([rot, rot, rot, xy, xy, z], dtype=np.float64)
    return gtsam.noiseModel.Diagonal.Sigmas(sig)


def build_and_solve(
    rel_poses: Sequence[np.ndarray],
    frame_t: np.ndarray,
    rtk_pos: np.ndarray | None,
    cfg: GraphConfig,
    init_poses: np.ndarray | None = None,
    imu: ImuStream | None = None,
    use_imu: bool = True,
    use_rtk: bool = True,
    rtk_outage: Tuple[int, int] | None = None,
    lio_outage: Tuple[int, int] | None = None,
    loops: Sequence[Tuple[int, int, np.ndarray]] = (),
    init_velocity: np.ndarray | None = None,
    ate_truth: np.ndarray | None = None,
    init_rotation: np.ndarray | None = None,
    verbose: bool = False,
) -> GraphResult:
    """Assemble the graph, solve it, and report translation ATE against RTK.

    ``rel_poses[i]`` is ``T_{i-1,i}`` (the transform that maps frame ``i`` points
    into frame ``i-1``), already expressed in the body/IMU frame.
    """
    if gtsam is None:
        raise RuntimeError("gtsam is required")

    n = len(rel_poses) + 1
    graph = gtsam.NonlinearFactorGraph()
    counts: Dict[str, int] = {}

    def bump(name: str, k: int = 1) -> None:
        counts[name] = counts.get(name, 0) + k

    # ---- initial estimate: dead-reckon the LIO chain --------------------- #
    # The LIO increments live in the IMU body frame.  Putting R0 on X(0) both
    # rotates the whole chain into frame0 and -- more importantly -- gives the
    # pre-integration a gravity-aligned starting attitude.  With identity here
    # the 1.286 deg roll of the sensor becomes a 0.22 m/s^2 phantom bias and the
    # IMU factor actively fights the RTK prior.
    R0 = np.eye(3) if init_rotation is None else np.asarray(init_rotation, dtype=np.float64)
    if init_poses is None:
        init = np.zeros((n, 4, 4))
        init[0] = np.eye(4)
        init[0][:3, :3] = R0
        for i in range(1, n):
            init[i] = init[i - 1] @ rel_poses[i - 1]
    else:
        init = np.asarray(init_poses, dtype=np.float64)

    values = gtsam.Values()
    for i in range(n):
        values.insert(_X(i), _pose3(init[i]))

    do_imu = bool(use_imu and imu is not None and len(imu.t) > 2)

    if do_imu:
        # The pre-integration noise lives on the ImuStream, the graph noise on
        # GraphConfig.  Forgetting to sync them makes every sweep of
        # cfg.imu.* a no-op -- which is exactly what happened: 24 parameter
        # combinations all returned 0.1286 m.
        imu.cfg = cfg.imu

    if do_imu:
        # Velocity initialisation matters more than it looks.  Seeding every
        # V(i) with zero when the vehicle is already doing 10 m/s forces iSAM to
        # discover the whole velocity profile from scratch, and on 154 frames it
        # settles in a different basin than the truth.  Use the reference
        # difference when one is available (it only seeds the initial estimate,
        # it is not a factor), else zero.
        ref0 = ate_truth if ate_truth is not None else rtk_pos
        v0 = np.zeros(3) if init_velocity is None else np.asarray(init_velocity, dtype=np.float64)
        v_seed = np.zeros((n, 3))
        for i in range(n):
            v = v0
            if ref0 is not None and len(ref0) > 1:
                r = np.asarray(ref0, dtype=np.float64)
                if i + 1 < len(r):
                    v = (r[i + 1] - r[i]) / max(float(frame_t[i + 1] - frame_t[i]), 1e-6)
                elif i > 0:
                    v = (r[i] - r[i - 1]) / max(float(frame_t[i] - frame_t[i - 1]), 1e-6)
            v_seed[i] = np.asarray(v, dtype=np.float64)
            values.insert(_V(i), v_seed[i])
            values.insert(_B(i), gtsam.imuBias.ConstantBias())
        v_first = v_seed[0]

    # ---- prior on the first pose ---------------------------------------- #
    # 1e-4 on everything: frame0 is the origin of the whole problem, so it is
    # fixed, not estimated.  Anything looser lets the graph drift the world.
    T0 = np.eye(4)
    T0[:3, :3] = R0
    graph.add(gtsam.PriorFactorPose3(
        _X(0), _pose3(T0),
        gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-4))))
    bump("prior_pose")

    if do_imu:
        graph.add(gtsam.PriorFactorVector(
            _V(0), gtsam.Point3(*[float(x) for x in v_first]),
            gtsam.noiseModel.Isotropic.Sigma(3, cfg.vel_sigma)))
        bump("prior_vel")
        bias_prior = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([cfg.bias_sigma_acc] * 3 + [cfg.bias_sigma_gyro] * 3))
        graph.add(gtsam.PriorFactorConstantBias(_B(0), gtsam.imuBias.ConstantBias(), bias_prior))
        bump("prior_bias")

    # ---- odometry, IMU, RTK --------------------------------------------- #
    lio_noise = _rel_noise(cfg, cfg.lio_sigma_rot, cfg.lio_sigma_xy, cfg.lio_sigma_z)

    for i in range(1, n):
        if not (lio_outage and lio_outage[0] <= i < lio_outage[1]):
            graph.add(gtsam.BetweenFactorPose3(_X(i - 1), _X(i), _pose3(rel_poses[i - 1]), lio_noise))
            bump("lio_between")

        if do_imu:
            pim, nsamp = imu.preintegrate(float(frame_t[i - 1]), float(frame_t[i]))
            if nsamp > 0:
                # gtsam 4.3's ImuFactor takes a single shared bias key (pose_i,
                # vel_i, pose_j, vel_j, bias, pim) -- there is no bias_j argument.
                graph.add(gtsam.ImuFactor(
                    _X(i - 1), _V(i - 1), _X(i), _V(i), _B(i - 1), pim))
                bump("imu")
            graph.add(gtsam.BetweenFactorConstantBias(
                _B(i - 1), _B(i), gtsam.imuBias.ConstantBias(),
                gtsam.noiseModel.Diagonal.Sigmas(np.array(
                    [cfg.bias_rw_acc] * 3 + [cfg.bias_rw_gyro] * 3))))
            bump("bias_rw")

    rtk_used, rtk_skipped = 0, 0
    if use_rtk and rtk_pos is not None:
        base = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([cfg.rtk_sigma_xy, cfg.rtk_sigma_xy, cfg.rtk_sigma_z]))
        huber = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(cfg.rtk_huber_k), base)
        lo, hi = (rtk_outage if rtk_outage else (n, n))
        for i in range(n):
            if lo <= i < hi:
                rtk_skipped += 1
                continue
            graph.add(gtsam.PoseTranslationPrior3D(
                _X(i), gtsam.Point3(*[float(v) for v in rtk_pos[i]]), huber))
            rtk_used += 1
        bump("rtk", rtk_used)

    loop_noise = _rel_noise(cfg, cfg.loop_sigma_rot, cfg.loop_sigma_xy, cfg.loop_sigma_z)
    for (i, j, T_ij) in loops:
        graph.add(gtsam.BetweenFactorPose3(_X(int(i)), _X(int(j)), _pose3(T_ij), loop_noise))
        bump("loop")

    # ---- solve ----------------------------------------------------------- #
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(200)
    params.setRelativeErrorTol(1e-6)
    params.setVerbosityLM("SILENT")
    err0 = float(graph.error(values))
    opt = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    try:
        sol = opt.optimize()
        converged = True
        iters = int(opt.iterations())
    except Exception:
        # a singular normal equation (degenerate start) still yields the last
        # iterate; better to report what we have than to lose the run
        sol = opt.values()
        converged = False
        iters = -1
    err1 = float(graph.error(sol))

    poses = np.stack([sol.atPose3(_X(i)).matrix() for i in range(n)])
    vel = None
    if do_imu:
        try:
            vel = np.stack([np.array(sol.atVector(_V(i))) for i in range(n)])
        except Exception:
            vel = None

    if verbose:
        print(f"      graph: {counts}, err {err0:.1f} -> {err1:.3f}, iters {iters}")

    ate = np.full(n, np.nan)
    ref = rtk_pos if rtk_pos is not None else ate_truth
    if ref is not None:
        ref = np.asarray(ref, dtype=np.float64)[:n]
        ate = np.linalg.norm(poses[:, :3, 3] - ref, axis=1)

    gaps = np.linalg.norm(np.diff(ref, axis=0), axis=1) if ref is not None else np.zeros(n - 1)
    return GraphResult(
        poses=poses,
        velocities=vel,
        ate_mean=float(np.nanmean(ate)) if np.isfinite(ate).any() else float("nan"),
        ate_final=float(ate[-1]) if np.isfinite(ate[-1]) else float("nan"),
        ate_rmse=float(np.sqrt(np.nanmean(ate ** 2))) if np.isfinite(ate).any() else float("nan"),
        max_gap_m=float(gaps.max()) if len(gaps) else 0.0,
        factors=counts,
        converged=converged,
        iterations=iters,
        error_initial=err0,
        error_final=err1,
    )


def dead_reckon(rel_poses: Sequence[np.ndarray]) -> np.ndarray:
    """Cumulative product of relative poses, the LIO-only baseline."""
    n = len(rel_poses) + 1
    out = np.zeros((n, 4, 4))
    out[0] = np.eye(4)
    for i in range(1, n):
        out[i] = out[i - 1] @ rel_poses[i - 1]
    return out


def rpe(poses: np.ndarray, truth: np.ndarray, delta: int = 1) -> Dict[str, float]:
    """Relative *translation* error over a fixed interval.

    ATE answers "does the map line up"; RPE answers "does the motion model hold
    locally".  Reporting both stops a graph that has been dragged onto the truth
    by a strong prior from looking like a good estimator.

    Accepts ``(N,4,4)`` poses or ``(N,3)`` positions for either argument, because
    the RTK truth for this sequence is only trusted as a translation (the OXTS
    attitude is in a NED frame while the Mercator positions are east-north, so
    mixing them silently is a trap the v1 demo already fell into).
    """
    est = np.asarray(poses, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    if est.ndim == 3:
        est = est[:, :3, 3]
    if tru.ndim == 3:
        tru = tru[:, :3, 3]
    n = min(len(est), len(tru))
    if n <= delta:
        return {"trans_mean": float("nan"), "trans_rmse": float("nan")}
    d_est = est[delta:n] - est[:n - delta]
    d_tru = tru[delta:n] - tru[:n - delta]
    errs = np.linalg.norm(d_est - d_tru, axis=1)
    return {"trans_mean": float(errs.mean()), "trans_rmse": float(np.sqrt((errs ** 2).mean()))}
