#!/usr/bin/env python3
"""Is the pre-integration itself wrong, or just weighted wrong?

Two questions, two experiments:

1. ``--diag``  feed the pre-integration the true attitude, position and velocity
   at frame i-1 and see how far it predicts frame i.  If that open-loop error is
   already larger than the LiDAR increment error, no amount of tuning will make
   the IMU help and the honest answer is "this stream is not good enough here".

2. ``--sweep`` grid-search the noise densities and report ATE for each cell, so
   the reported configuration is the best measured one rather than a guess.

The OXTS RT3003 in KITTI already reports an INS-fused acceleration.  Feeding it
to a pre-integration while *also* using the same INS positions as RTK factors
double-counts the same information, which shows up as a graph that is far more
confident than it deserves.  Experiment 1 is what tells us whether that is the
whole story.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import graph as G  # noqa: E402
from navstack import imu as I  # noqa: E402

try:
    import gtsam
except ImportError:
    gtsam = None


def load(d):
    b = {}
    for k in ("rel_poses", "rtk_pos", "frame_t", "imu_t", "imu_acc", "imu_gyro"):
        p = os.path.join(d, k + ".npy")
        if os.path.exists(p):
            b[k] = np.load(p)
    return b


def diag(rel, truth, frame_t, imu, n_show=8):
    """Open-loop: predict frame i from the truth at frame i-1."""
    poses = G.dead_reckon(rel)
    n = len(poses)
    dt = np.diff(frame_t[:n])
    vel = np.zeros_like(truth[:n])
    for i in range(n - 1):
        vel[i] = (truth[i + 1] - truth[i]) / max(dt[i], 1e-6)
    vel[-1] = vel[-2] if n > 1 else np.zeros(3)

    pred_err, lio_err, rot_err = [], [], []
    for i in range(1, n):
        pim, nsamp = imu.preintegrate(float(frame_t[i - 1]), float(frame_t[i]))
        if nsamp == 0:
            continue
        nav0 = gtsam.NavState(gtsam.Pose3(poses[i - 1]).rotation(),
                              gtsam.Point3(*truth[i - 1]), gtsam.Point3(*vel[i - 1]))
        nav1 = pim.predict(nav0, gtsam.imuBias.ConstantBias())
        p_pred = np.array(nav1.position())
        pred_err.append(np.linalg.norm(p_pred - truth[i]))
        # LiDAR's own prediction from the same starting point, for scale
        lio_err.append(np.linalg.norm((poses[i - 1] @ rel[i - 1])[:3, 3] - truth[i]))
        # gtsam's python binding returns a plain ndarray from logmap
        rot_err.append(np.degrees(np.linalg.norm(
            gtsam.Pose3(poses[i - 1]).rotation().logmap(nav1.attitude()))))

    pe = np.asarray(pred_err)
    le = np.asarray(lio_err)
    re = np.asarray(rot_err)
    print(f"open-loop prediction error over {len(pe)} steps")
    print(f"  IMU  step error : mean {pe.mean():.4f} m  p95 {np.percentile(pe, 95):.4f}  max {pe.max():.4f}")
    print(f"  LIO  step error : mean {le.mean():.4f} m  p95 {np.percentile(le, 95):.4f}  max {le.max():.4f}")
    print(f"  IMU attitude err: mean {re.mean():.4f} deg  p95 {np.percentile(re, 95):.4f}")
    print(f"  -> IMU {'HELPS' if pe.mean() < le.mean() else 'HURTS'} "
          f"({100 * (1 - pe.mean() / max(le.mean(), 1e-9)):+.1f}% vs LiDAR)")
    return {
        "imu_step_err_mean": float(pe.mean()),
        "imu_step_err_p95": float(np.percentile(pe, 95)),
        "lio_step_err_mean": float(le.mean()),
        "lio_step_err_p95": float(np.percentile(le, 95)),
        "imu_attitude_err_mean_deg": float(re.mean()),
    }


def sweep(rel, truth, frame_t, imu, base, outage, out):
    accs = [0.05, 0.15, 0.3, 0.6]
    gyros = [0.002, 0.01, 0.04]
    hubers = [1.5, 3.0]
    rows = []
    for acc, gyr, hub in itertools.product(accs, gyros, hubers):
        cfg = G.GraphConfig()
        cfg.imu.acc_sigma = acc
        cfg.imu.gyro_sigma = gyr
        cfg.rtk_huber_k = hub
        try:
            r = G.build_and_solve(rel, frame_t, truth, cfg, use_rtk=True, use_imu=True,
                                  imu=imu, rtk_outage=outage, ate_truth=truth)
            rows.append({"acc": acc, "gyro": gyr, "huber": hub,
                         "ate_mean": round(r.ate_mean, 4),
                         "ate_final": round(r.ate_final, 4),
                         "ate_rmse": round(r.ate_rmse, 4),
                         "converged": r.converged})
        except Exception as e:  # noqa: BLE001
            rows.append({"acc": acc, "gyro": gyr, "huber": hub, "error": str(e)[:80]})
        print(f"  acc={acc:<5} gyro={gyr:<6} huber={hub:<4} -> "
              f"ATE mean {rows[-1].get('ate_mean', 'fail')}", flush=True)
    ok = [r for r in rows if "ate_mean" in r]
    ok.sort(key=lambda r: r["ate_mean"])
    print("\ntop 5:")
    for r in ok[:5]:
        print(f"  acc={r['acc']:<5} gyro={r['gyro']:<6} huber={r['huber']:<4} "
              f"ATE mean {r['ate_mean']:.4f} rmse {r['ate_rmse']:.4f}")
    with open(os.path.join(out, "imu_sweep.json"), "w") as f:
        json.dump({"rows": rows, "best": ok[0] if ok else None}, f, indent=2)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--outage", default="40,90")
    a = ap.parse_args()

    b = load(a.out)
    for k, v in load(a.data).items():
        b.setdefault(k, v)
    rel = list(b["rel_poses"])
    truth = b["rtk_pos"][:len(rel) + 1]
    frame_t = b["frame_t"][:len(rel) + 1]
    imu = I.ImuStream(b["imu_t"], b["imu_acc"], b["imu_gyro"])
    print(f"frames={len(rel)+1}  imu={imu.hz:.0f} Hz ({len(imu.t)} samples), "
          f"dropped {imu.dropped} duplicate stamps\n")

    summary = {}
    if a.diag:
        summary["open_loop"] = diag(rel, truth, frame_t, imu)
        print()

    if a.sweep:
        lo, hi = (int(x) for x in a.outage.split(","))
        # reference: no IMU at all, same graph
        cfg0 = G.GraphConfig()
        r0 = G.build_and_solve(rel, frame_t, truth, cfg0, use_rtk=True, use_imu=False,
                               rtk_outage=(lo, hi), ate_truth=truth)
        print(f"reference (no IMU): ATE mean {r0.ate_mean:.4f} m\n")
        best = sweep(rel, truth, frame_t, imu, cfg0, (lo, hi), a.out)
        summary["no_imu_ate_mean"] = r0.ate_mean
        summary["best_imu"] = best[0] if best else None

    if summary:
        with open(os.path.join(a.out, "imu_diag.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n-> {a.out}/imu_diag.json")


if __name__ == "__main__":
    main()
