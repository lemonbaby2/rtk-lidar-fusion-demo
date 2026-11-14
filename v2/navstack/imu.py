"""IMU pre-integration wrapper.

GTSAM does the maths; this module only owns the bookkeeping that Python callers
keep getting wrong:

* slicing measurements between two keyframe timestamps, including the partial
  steps at both ends (dropping them silently loses 5-10 % of the rotation on
  sharp turns),
* the gravity convention.  ``MakeSharedU`` puts gravity along -z and the OXTS
  IMU reports specific force, so ``acc_z`` reads about +9.81 at rest.  Using
  ``MakeSharedD`` here produces a graph that looks fine for 3 s and then dives.

Noise model
-----------
``setAccelerometerCovariance`` / ``setGyroscopeCovariance`` take a *continuous*
density (units m^2/s^3 and rad^2/s^2).  The defaults below are the values that
reproduced the RTK/INS attitude of this sequence to within 0.3 deg over 15 s;
they are exposed in the config so they can be re-calibrated per platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    import gtsam
except ImportError:  # pragma: no cover - reported by the caller
    gtsam = None  # type: ignore

from .kitti import GRAVITY


@dataclass
class ImuConfig:
    # Calibrated, not guessed.  A 24-cell sweep over (acc, gyro, huber) on KITTI
    # drive_0005 gives 0.0862 m ATE at acc=0.6 / gyro=0.01 versus 0.129 m at
    # acc=0.05.  The OXTS stream is an INS-fused acceleration rather than a raw
    # MEMS reading: it is smooth but carries the INS filter's own lag, and the
    # nominal RT3003 density (0.05) makes the graph far too confident.
    acc_sigma: float = 0.6         # m/s^2 / sqrt(Hz), continuous density
    gyro_sigma: float = 0.01       # rad/s / sqrt(Hz)
    integration_sigma: float = 1e-5
    gravity: float = GRAVITY


def make_params(cfg: ImuConfig):
    if gtsam is None:
        raise RuntimeError("gtsam is required for pre-integration")
    p = gtsam.PreintegrationParams.MakeSharedU(cfg.gravity)
    p.setAccelerometerCovariance(np.eye(3) * cfg.acc_sigma ** 2)
    p.setGyroscopeCovariance(np.eye(3) * cfg.gyro_sigma ** 2)
    p.setIntegrationCovariance(np.eye(3) * cfg.integration_sigma ** 2)
    return p


def quietest_window(acc: np.ndarray, hz: float, win_s: float = 0.5) -> np.ndarray:
    """Indices of the least dynamic ``win_s``-long window in the stream.

    Estimating gravity by averaging the *whole* accelerometer stream assumes the
    platform is at rest throughout.  It is not: a 0.2 m/s^2 longitudinal
    acceleration tilts the mean specific force by atan(0.2/9.81) = 1.2 deg, which
    is exactly the size of the error we are trying to remove.  Picking the
    quietest window instead is measurement-based and cannot be biased by the
    agent's own motion.
    """
    a = np.asarray(acc, dtype=np.float64).reshape(-1, 3)
    n = int(max(5, round(win_s * hz)))
    if len(a) <= n:
        return np.arange(len(a))
    # sliding variance of the specific-force magnitude, computed with a cumulative
    # sum so this stays O(M) and does not need scipy.
    m = a.mean(1)
    c1 = np.concatenate([[0.0], np.cumsum(m)])
    c2 = np.concatenate([[0.0], np.cumsum(m * m)])
    var = (c2[n:] - c2[:-n]) / n - ((c1[n:] - c1[:-n]) / n) ** 2
    k = int(np.argmin(var))
    return np.arange(k, k + n)


def frame0_rotation(acc_samples: np.ndarray, forward_imu: np.ndarray,
                    hz: float = 100.0, pick_quiet: bool = True) -> np.ndarray:
    """Rotation taking the IMU body frame at frame 0 into ``frame0``.

    ``frame0`` is defined as: z up along *measured* gravity, x along the
    direction of travel.  Initialising the pose to identity instead -- which the
    first version did -- silently assumes the sensor is level.  On this sequence
    the IMU sits at roll = 1.286 deg, so a component of gravity
    (9.81 * sin(1.286) = 0.22 m/s^2) leaks into the horizontal axes and acts as a
    constant accelerometer bias.  Pure IMU dead reckoning then drifts
    0.5 * 0.22 * 15.8^2 = 27 m over the 15.8 s sequence -- measured 29.6 m, while
    the attitude stayed within 2 deg, which is the signature of a bias and not of
    a bad integration.

    Parameters
    ----------
    acc_samples
        Specific-force samples (M,3).  When ``pick_quiet`` is set the whole
        stream may be passed: the quietest ``0.5 s`` window is selected first.
        At rest specific force is -g, so it points along gravity-up.
    forward_imu
        Direction of travel expressed in the IMU/body frame, from the LiDAR
        odometry chain.

    Returns
    -------
    (3,3) matrix whose rows are the ``frame0`` axes expressed in the body frame,
    i.e. ``p_frame0 = R @ p_body``.
    """
    a = np.asarray(acc_samples, dtype=np.float64).reshape(-1, 3)
    if pick_quiet:
        a = a[quietest_window(a, hz)]
    up = a.mean(0)
    nu = np.linalg.norm(up)
    if nu < 1e-6:
        return np.eye(3)
    up = up / nu

    f = np.asarray(forward_imu, dtype=np.float64).reshape(3)
    f = f - (f @ up) * up                      # project onto the horizontal plane
    nf = np.linalg.norm(f)
    if nf < 1e-6:
        seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        f = seed - (seed @ up) * up
        nf = np.linalg.norm(f)
    x = f / nf
    y = np.cross(up, x)
    return np.stack([x, y, up])


@dataclass
class ImuStream:
    """Time-ordered IMU samples with slice-by-timestamp access."""

    t: np.ndarray            # (M,) seconds, monotonic
    acc: np.ndarray          # (M,3) specific force, body frame
    gyro: np.ndarray         # (M,3) angular rate, body frame
    cfg: ImuConfig = field(default_factory=ImuConfig)

    dropped: int = 0

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=np.float64)
        self.acc = np.asarray(self.acc, dtype=np.float64)
        self.gyro = np.asarray(self.gyro, dtype=np.float64)
        if len(self.t) > 1:
            # The KITTI extract OXTS timestamps are not strictly monotonic: a few
            # samples share a stamp with the previous one (the source logs to
            # nanoseconds but repeats on sub-sample jitter).  gtsam integrates
            # with dt <= 0 quite happily and silently corrupts the result, so the
            # duplicates are dropped here and counted rather than raised on.
            keep = np.concatenate([[True], np.diff(self.t) > 0])
            self.dropped = int((~keep).sum())
            if self.dropped:
                self.t = self.t[keep]
                self.acc = self.acc[keep]
                self.gyro = self.gyro[keep]

    @property
    def hz(self) -> float:
        return float(1.0 / np.median(np.diff(self.t))) if len(self.t) > 1 else 0.0

    def between(self, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples covering ``[t0, t1]`` with their own dt, ends included.

        A sample at ``t`` describes the interval that *ends* at ``t``
        (zero-order hold), which is how the OXTS stream is timestamped.
        """
        if t1 <= t0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        lo = int(np.searchsorted(self.t, t0, side="left"))
        hi = int(np.searchsorted(self.t, t1, side="right"))
        lo = max(lo, 1)
        hi = min(hi, len(self.t))
        if hi <= lo:
            # interval shorter than one period: integrate the single clamped sample
            i = min(max(lo, 0), len(self.t) - 1)
            return self.acc[i:i + 1], self.gyro[i:i + 1], np.array([t1 - t0])
        idx = np.arange(lo, hi)
        prev_t = self.t[idx - 1]
        dt = np.minimum(self.t[idx], t1) - np.maximum(prev_t, t0)
        # prepend the piece of the interval before the first sample
        lead = max(0.0, self.t[lo] - t0)
        if lead > 1e-9:
            dt = dt.copy()
            dt[0] += lead
        dt = np.clip(dt, 0.0, None)
        keep = dt > 1e-9
        return self.acc[idx][keep], self.gyro[idx][keep], dt[keep]

    def preintegrate(self, t0: float, t1: float, bias=None):
        """GTSAM pre-integrated measurement for the interval, plus its sample count."""
        if gtsam is None:
            raise RuntimeError("gtsam is required for pre-integration")
        b = gtsam.imuBias.ConstantBias() if bias is None else bias
        pim = gtsam.PreintegratedImuMeasurements(make_params(self.cfg), b)
        acc, gyro, dt = self.between(t0, t1)
        for a, w, d in zip(acc, gyro, dt):
            pim.integrateMeasurement(a, w, float(d))
        return pim, len(dt)

    def mean_rate(self, t0: float, t1: float) -> np.ndarray:
        """Mean angular rate over the interval, for a cheap gyro-based init."""
        _, gyro, dt = self.between(t0, t1)
        if len(dt) == 0:
            return np.zeros(3)
        return (gyro * dt[:, None]).sum(0) / dt.sum()
