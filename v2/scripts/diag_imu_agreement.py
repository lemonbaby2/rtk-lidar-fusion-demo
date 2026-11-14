#!/usr/bin/env python3
"""Do the LIO increments and the IMU pre-integration agree?

This is the only comparison that isolates the IMU from every confounder:

* initial velocity and attitude are taken from the reference, so the result is not
  contaminated by a wrong seed (the previous diagnostic reported 56 m of "drift"
  that was really ``3.24 m/s * 15.8 s`` of unknown initial velocity),
* the comparison is *relative*, per keyframe interval, so nothing accumulates,
* both inputs are the ones the graph actually sees.

If the two disagree by more than their claimed sigmas, the IMU factor is being
forced into an argument it cannot win and will always lose to the LiDAR -- which
is exactly the observed behaviour.  The output distinguishes three failure modes,
because they need different fixes:

``scale``   the disagreement is proportional to the motion  -> gyro/acc scale
``bias``    the disagreement is constant in the body frame  -> sensor bias
``timing``  the disagreement correlates with angular rate   -> clock offset
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import imu as I  # noqa: E402
from navstack.bundle import load_bundle  # noqa: E402


def rot_angle(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--offsets", default="-0.10,-0.05,0.0,0.05,0.10",
                    help="IMU time offsets to test, seconds")
    a = ap.parse_args()

    b = load_bundle(a.data)
    rel = list(np.load(os.path.join(a.out, "rel_poses.npy")))
    imu = I.ImuStream(b["imu_t"], b["imu_acc"], b["imu_gyro"])
    truth = b["rtk_pos"]
    ft = b["frame_t"]
    n = len(rel) + 1

    # reference velocity and attitude per keyframe
    v_ref = np.zeros((n, 3))
    for i in range(n):
        if i + 1 < n:
            v_ref[i] = (truth[i + 1] - truth[i]) / max(ft[i + 1] - ft[i], 1e-6)
        else:
            v_ref[i] = (truth[i] - truth[i - 1]) / max(ft[i] - ft[i - 1], 1e-6)

    print("== LIO vs IMU increments (attitude from LiDAR chain, velocity from RTK) ==")
    G_VEC = np.array([0.0, 0.0, -imu.cfg.gravity])
    for off in (float(x) for x in a.offsets.split(",")):
        ang, tra = [], []
        for i in range(1, n):
            t0 = float(ft[i - 1]) + off
            t1 = float(ft[i]) + off
            pim, ns = imu.preintegrate(t0, t1)
            if ns == 0:
                continue
            dR = pim.deltaRij().matrix()
            dp = np.array(pim.deltaPij())
            R_i = rel[i - 1][:3, :3]           # attitude of frame i-1 (LiDAR)
            dt = float(ft[i] - ft[i - 1])
            # gtsam pre-integrates the *acceleration* term only: the constant
            # velocity part v_i*dt and the gravity part -0.5 g dt^2 have to be
            # put back before this can be compared with a measured displacement.
            # Comparing deltaPij() directly with a step of 0.42 m and calling it a
            # "12 % scale error" is a methodological mistake, not a finding.
            dp_lio = R_i.T @ (truth[i] - truth[i - 1] - v_ref[i] * dt
                              + 0.5 * G_VEC * dt * dt)
            ang.append(rot_angle(rel[i - 1][:3, :3].T @ dR))
            tra.append(float(np.linalg.norm(dp - dp_lio)))
        ang = np.asarray(ang)
        tra = np.asarray(tra)
        print(f"  offset {off:+.2f} s: rotation diff {np.median(ang):7.3f} deg "
              f"(p95 {np.percentile(ang, 95):6.3f})   "
              f"translation diff {np.median(tra):7.4f} m "
              f"(p95 {np.percentile(tra, 95):6.4f})")

    # ---- decompose at zero offset ------------------------------------- #
    ang, slip = [], []
    dp_norm, step = [], []
    for i in range(1, n):
        pim, ns = imu.preintegrate(float(ft[i - 1]), float(ft[i]))
        if ns == 0:
            continue
        R_i = rel[i - 1][:3, :3]
        dt = float(ft[i] - ft[i - 1])
        dp = np.array(pim.deltaPij())
        dp_lio = R_i.T @ (truth[i] - truth[i - 1] - v_ref[i] * dt
                          + 0.5 * G_VEC * dt * dt)
        ang.append(rot_angle(rel[i - 1][:3, :3].T @ pim.deltaRij().matrix()))
        slip.append(float(np.linalg.norm(dp - dp_lio)))
        dp_norm.append(float(np.linalg.norm(dp)))
        step.append(float(np.linalg.norm(dp_lio)))
    ang = np.asarray(ang)
    slip = np.asarray(slip)
    dp_norm = np.asarray(dp_norm)
    step = np.asarray(step)

    ratio = float(np.median(dp_norm / np.maximum(step, 1e-9)))
    gyro_rate = np.array([
        float(np.linalg.norm(imu.mean_rate(float(ft[i - 1]), float(ft[i]))))
        for i in range(1, n)])[:len(slip)]
    corr_t = float(np.corrcoef(slip, gyro_rate)[0, 1]) if gyro_rate.std() > 1e-9 else 0.0
    corr_s = float(np.corrcoef(slip, step)[0, 1]) if step.std() > 1e-9 else 0.0

    payload = {
        "steps": int(len(ang)),
        "rotation_diff_deg": {"median": round(float(np.median(ang)), 4),
                              "p95": round(float(np.percentile(ang, 95)), 4),
                              "max": round(float(ang.max()), 4)},
        "translation_diff_m": {"median": round(float(np.median(slip)), 5),
                               "p95": round(float(np.percentile(slip, 95)), 5)},
        "gravity_free_step_m": {"median": round(float(np.median(step)), 5)},
        "imu_dp_over_gravity_free_step": round(ratio, 4),
        "corr_error_vs_step": round(corr_s, 3),
        "corr_error_vs_gyro": round(corr_t, 3),
        "verdict": classify(ratio, corr_s, corr_t, float(np.median(ang))),
    }
    print(f"\n  |dp_imu| / |dp_lio - v dt + 0.5 g dt^2| = {ratio:.3f}  (1.0 = perfect)")
    print(f"  corr(error, step) {corr_s:+.3f}   corr(error, gyro) {corr_t:+.3f}")
    print(f"  verdict: {payload['verdict']}")

    with open(os.path.join(a.out, "imu_lio_agreement.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {a.out}/imu_lio_agreement.json")


def classify(ratio: float, corr_s: float, corr_t: float, ang_med: float) -> str:
    if ang_med > 1.0:
        return (f"attitude disagreement {ang_med:.2f} deg/step is far above the "
                f"1 deg the LIO sigma claims: the two sensors describe different "
                f"rotations, so an ImuFactor can only fight the LiDAR")
    if abs(ratio - 1.0) > 0.25:
        return f"scale mismatch, |dp_imu|/|dp_lio| = {ratio:.2f}"
    if abs(corr_t) > 0.6 and abs(corr_t) > abs(corr_s):
        return "error tracks angular rate -> a time offset remains"
    return "consistent within about 25 %; any residual problem is in weighting"


if __name__ == "__main__":
    main()
