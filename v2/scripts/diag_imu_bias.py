#!/usr/bin/env python3
"""Is the OXTS acceleration usable, and if so what bias does it carry?

Everything else in this stack is already trustworthy: the LiDAR registration sits
at 0.084 m median residual, the RTK truth is centimetre class, and the clocks are
aligned to 7 ms.  The IMU factor is the only component that *loses* to doing
nothing, so the question is whether the accelerometer stream is consistent with
the motion at all.

Method, in the order the data allows:

1. Pre-integrate the IMU alone from t0 with zero bias and compare the resulting
   position with RTK at every keyframe.  Over 15.8 s this is dominated by
   ``0.5 * b * t^2``, so it is a clean read on the horizontal bias.
2. Recover that bias by least squares.  The sensitivity of position to a constant
   body-frame bias is ``S_i = -(integral integral R dt dt)``, which is available
   from the same integration, so the fit is a 3x3 linear solve -- no optimiser.
3. Re-integrate with the recovered bias.  If the drift collapses, the stream is
   self-consistent and any remaining problem is in how the graph handles bias.
   If it does not, the accelerometer is not consistent with the trajectory and no
   amount of factor tuning will rescue it.

Both answers are publishable; guessing between them is not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtsam  # noqa: E402

from navstack import imu as I  # noqa: E402
from navstack.bundle import load_bundle  # noqa: E402


def integrate(imu: I.ImuStream, t0: float, t1: float, bias: np.ndarray):
    """(p, v, R) at t1 relative to t0, by the same equations gtsam pre-integrates.

    Returns the sensitivities too: ``S`` with ``dp/db = -S`` and ``Sr`` for the
    rotation, since they come for free from the same loop.
    """
    acc, gyro, dt = imu.between(t0, t1)
    p = np.zeros(3)
    v = np.zeros(3)
    R = np.eye(3)
    S = np.zeros((3, 3))          # dp/db = -S
    Sv = np.zeros((3, 3))         # dv/db = -Sv
    g = np.array([0.0, 0.0, -imu.cfg.gravity])
    for a, w, d in zip(acc, gyro, dt):
        a = a - bias
        # midpoint-ish update: gtsam uses a first-order scheme on the same
        # quantities, and this reproduces it to ~1e-4 m over 0.1 s
        aw = R @ a + g
        p = p + v * d + 0.5 * aw * d * d
        v = v + aw * d
        # sensitivities: d(R a)/db = -R, so dp/db accumulates -0.5 R d^2
        S = S + 0.5 * R * (d * d)
        Sv = Sv + R * d
        th = w * d
        ang = np.linalg.norm(th)
        if ang > 1e-12:
            K = np.array([[0, -th[2], th[1]], [th[2], 0, -th[0]], [-th[1], th[0], 0]])
            dR = np.eye(3) + np.sin(ang) / ang * K + (1 - np.cos(ang)) / ang ** 2 * (K @ K)
        else:
            dR = np.eye(3)
        R = R @ dR
    return p, v, R, S, Sv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    b = load_bundle(a.data)
    imu = I.ImuStream(b["imu_t"], b["imu_acc"], b["imu_gyro"])
    truth = b["rtk_pos"]
    ft = b["frame_t"]
    n = min(len(truth), len(ft))
    idx = list(range(0, n, max(1, a.stride)))
    t0 = float(ft[0])

    print(f"== pure IMU from t0, {len(idx)} checkpoints ==", flush=True)
    est, sens = [], []
    for i in idx:
        p, v, R, S, Sv = integrate(imu, t0, float(ft[i]), np.zeros(3))
        est.append(p)
        sens.append(S)
    est = np.asarray(est)
    tru = truth[idx]
    tru = tru - tru[0]
    err = np.linalg.norm(est - tru, axis=1)
    print(f"  drift with zero bias: mean {err.mean():.2f} m, final {err[-1]:.2f} m",
          flush=True)

    # least squares: err_i ~= -S_i b  (per axis, stacked)
    A = -np.concatenate(sens, axis=0).reshape(-1, 3)
    y = (est - tru).reshape(-1)
    bias, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = est + (A @ bias).reshape(est.shape)
    err2 = np.linalg.norm(pred - tru, axis=1)
    print(f"  recovered acc bias: {np.round(bias, 4)} m/s^2 "
          f"(|b| = {np.linalg.norm(bias):.4f})", flush=True)
    print(f"  drift after bias removal: mean {err2.mean():.2f} m, "
          f"final {err2[-1]:.2f} m", flush=True)

    # re-integrate properly with the recovered bias (not just the linear model)
    est3 = []
    for i in idx:
        p, v, R, S, Sv = integrate(imu, t0, float(ft[i]), bias)
        est3.append(p)
    est3 = np.asarray(est3)
    err3 = np.linalg.norm(est3 - tru, axis=1)
    print(f"  drift with bias re-integrated: mean {err3.mean():.2f} m, "
          f"final {err3[-1]:.2f} m", flush=True)

    # gravity direction implied by the bias, if the bias were pure gravity leak
    gdir = np.degrees(np.arccos(np.clip(bias[2] / max(np.linalg.norm(bias), 1e-9), -1, 1)))
    print(f"  if the bias were pure gravity leak: tilt = {gdir:.2f} deg",
          flush=True)

    # ---- what does gtsam's own pre-integration say? -------------------- #
    pim, ns = imu.preintegrate(t0, float(ft[-1]))
    nav = pim.predict(gtsam.NavState(gtsam.Pose3(), np.zeros(3)),
                      gtsam.imuBias.ConstantBias())
    dp = np.array(nav.position()) - np.array(gtsam.Pose3().translation())
    p_true = truth[min(n, len(truth)) - 1] - truth[0]
    print(f"  gtsam PIM end position {np.round(dp, 2)} vs truth {np.round(p_true, 2)} "
          f"(err {np.linalg.norm(dp - p_true):.2f} m, {ns} samples)", flush=True)

    payload = {
        "checkpoints": len(idx),
        "zero_bias": {"drift_mean_m": round(float(err.mean()), 3),
                      "drift_final_m": round(float(err[-1]), 3)},
        "recovered_acc_bias": [round(float(x), 5) for x in bias],
        "recovered_acc_bias_norm": round(float(np.linalg.norm(bias)), 5),
        "after_bias_linear": {"drift_mean_m": round(float(err2.mean()), 3),
                              "drift_final_m": round(float(err2[-1]), 3)},
        "after_bias_reintegrated": {"drift_mean_m": round(float(err3.mean()), 3),
                                    "drift_final_m": round(float(err3[-1]), 3)},
        "implied_tilt_deg": round(float(gdir), 3),
        "gtsam_pim": {"end_pos": [round(float(x), 3) for x in dp],
                      "truth_end_pos": [round(float(x), 3) for x in p_true],
                      "err_m": round(float(np.linalg.norm(dp - p_true)), 3),
                      "samples": int(ns)},
        "imu_hz": round(imu.hz, 2),
    }
    with open(os.path.join(a.out, "imu_bias_fit.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {a.out}/imu_bias_fit.json")


if __name__ == "__main__":
    main()
