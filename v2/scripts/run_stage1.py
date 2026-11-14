#!/usr/bin/env python3
"""Stage 1: registration + factor graph, with an ablation table.

On the same LIO front end:

    dead_reckon          cumulative product of the LIO increments
    graph_lio            pose graph, LIO between-factors only
    graph_lio_rtk        + RTK translation prior with a Huber kernel
    graph_full           + IMU pre-integration
    graph_imu_only       IMU + LIO, RTK factor disabled
    longgap_*            same pair with a 90-frame RTK outage (58 % of the run)
    graph_full_loops     add loop-closure between-factors

Every row carries ATE (mean / final / RMSE), RPE over 1 and 5 keyframes, and --
when an outage window is set -- the ATE *inside* and *outside* that window.

That last split is the point.  Averaged over 154 frames of which 50 sit in the
gap, the IMU factor looks useless (0.086 vs 0.076 m) because outside the gap the
RTK prior already pins the trajectory and the IMU can only disagree with it.
Inside the gap it is the only thing holding the solution together, and the long
gap run is where that becomes visible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from navstack import graph as G  # noqa: E402
from navstack import imu as I  # noqa: E402
from navstack import kitti, lio  # noqa: E402


def load_bundle(d: str) -> dict:
    out = {}
    for k in ("pts", "rgb", "pose_rtk", "rtk_pos", "rtk_R_w2f0", "frame_t",
              "imu_t", "imu_acc", "imu_gyro"):
        p = os.path.join(d, k + ".npy")
        if os.path.exists(p):
            out[k] = np.load(p)
    with open(os.path.join(d, "meta.json")) as f:
        out["meta"] = json.load(f)
    return out


def run_registration(pts, calib, max_range, max_frames=None, verbose=True, cache=None):
    """Frame-to-frame point-to-plane registration.  Returns body-frame increments."""
    n = len(pts)
    if max_frames:
        n = min(n, max_frames)

    if cache and os.path.exists(cache):
        rel = list(np.load(cache))
        if len(rel) >= n - 1:
            if verbose:
                print(f"  reused cached registration: {n - 1} increments", flush=True)
            return rel[:n - 1], [], n
        if verbose:
            print(f"  cache short ({len(rel)} < {n - 1}), re-registering", flush=True)

    clouds = []
    t0 = time.time()
    for i in range(n):
        clouds.append(lio.build_reg_cloud(pts[i], max_range=max_range))
    if verbose:
        print(f"  clouds: {np.mean([c.points.shape[0] for c in clouds]):.0f} src pts, "
              f"{np.mean([c.centers.shape[0] for c in clouds]):.0f} nrm voxels "
              f"({time.time() - t0:.1f}s)", flush=True)

    T_imu_velo = torch.as_tensor(calib["T_imu_velo"], dtype=torch.float32, device=lio.DEV)
    T_velo_imu = torch.as_tensor(calib["T_velo_imu"], dtype=torch.float32, device=lio.DEV)

    rel, stats = [], []
    prev_T = None
    t0 = time.time()
    for i in range(1, n):
        guess = lio.constant_velocity_guess(prev_T, None)
        T_v, st = lio.icp_point_to_plane(
            clouds[i].points, clouds[i - 1].centers, clouds[i - 1].normals,
            T0=guess, max_dist=1.2, max_resid=0.30)
        # x_prev_velo = T_v x_cur_velo  ->  same motion expressed in the body frame
        T_b = T_imu_velo @ T_v @ T_velo_imu
        rel.append(T_b.detach().cpu().numpy().astype(np.float64))
        stats.append(st)
        prev_T = T_v
        if verbose and i % 40 == 0:
            print(f"  icp {i}/{n - 1}  rmse={st.rmse:.3f} inl={st.inliers}", flush=True)

    if verbose:
        rmse = np.array([s.rmse for s in stats])
        print(f"  registration done in {time.time() - t0:.1f}s, "
              f"median rmse {np.median(rmse):.3f} m", flush=True)
    if cache:
        np.save(cache, np.stack(rel))
    return rel, stats, n


# every variant's trajectory is kept: later stages (map store, relocalisation,
# navigation) all need a real pose per frame, and re-solving is 10 s each.
POSE_CACHE: dict = {}


def outage_split(poses, truth, lo, hi):
    """ATE inside and outside the RTK gap."""
    if poses is None or lo >= hi:
        return None, None
    err = np.linalg.norm(poses[:, :3, 3] - truth[:len(poses)], axis=1)
    n = min(len(err), len(truth))
    err = err[:n]
    inside = err[lo:min(hi, n)]
    outside = np.concatenate([err[:lo], err[min(hi, n):]])
    return (float(inside.mean()) if inside.size else None,
            float(outside.mean()) if outside.size else None)


def row(name, poses, truth, notes, outage=None, res=None) -> dict:
    POSE_CACHE[name] = np.asarray(poses, np.float64)
    r1 = G.rpe(poses, truth, 1)
    r5 = G.rpe(poses, truth, 5)
    err = np.linalg.norm(poses[:, :3, 3] - truth[:len(poses)], axis=1)
    d = {
        "name": name,
        "ate_mean_m": round(float(err.mean()), 4),
        "ate_final_m": round(float(err[-1]), 4),
        "ate_rmse_m": round(float(np.sqrt((err ** 2).mean())), 4),
        "rpe1_m": round(r1["trans_mean"], 4),
        "rpe5_m": round(r5["trans_mean"], 4),
        "factors": res.factors if res else {},
        "converged": res.converged if res else True,
        "iterations": res.iterations if res else 0,
        "graph_error_initial": round(res.error_initial, 2) if res else 0.0,
        "graph_error_final": round(res.error_final, 6) if res else 0.0,
        "notes": notes,
    }
    if outage:
        ins, outs = outage_split(poses, truth, *outage)
        d["ate_in_outage_m"] = None if ins is None else round(ins, 4)
        d["ate_out_of_outage_m"] = None if outs is None else round(outs, 4)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--raw", default="/home/nvidia/rtk_fusion_demo/raw")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drive", default="2011_09_26_drive_0005")
    ap.add_argument("--max-range", type=float, default=55.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--outage", default="40,90", help="simulated RTK outage, i,j")
    ap.add_argument("--longgap", default="30,120", help="long outage window")
    ap.add_argument("--degen", default="60,90",
                    help="LiDAR-degeneracy window: LIO and RTK both removed, i,j")
    ap.add_argument("--loops", default="", help="comma list i:j from scancontext")
    ap.add_argument("--skip-imu", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print("== load bundle ==", flush=True)
    b = load_bundle(a.data)
    calib = kitti.load_calib(os.path.join(a.raw, a.date))
    truth_all = b["rtk_pos"]
    frame_t_all = b["frame_t"]
    print(f"  {b['meta']['frames']} frames, {b['meta']['duration_s']:.1f} s, "
          f"{b['meta']['path_length_m']:.1f} m, IMU {b['meta']['imu_hz']:.0f} Hz", flush=True)

    print("== registration ==", flush=True)
    rel, stats, n = run_registration(b["pts"], calib, a.max_range,
                                     a.max_frames or None,
                                     cache=os.path.join(a.out, "rel_poses.npy"))
    truth = truth_all[:n]
    frame_t = frame_t_all[:n]
    cfg = G.GraphConfig()

    imu = None
    if not a.skip_imu and "imu_t" in b:
        imu = I.ImuStream(b["imu_t"], b["imu_acc"], b["imu_gyro"])
        print(f"  IMU {imu.hz:.0f} Hz, {len(imu.t)} samples, "
              f"{imu.dropped} duplicate stamps dropped", flush=True)

    # ---- frame0 attitude -------------------------------------------------- #
    # The first version initialised X(0) at identity, which assumes the IMU is
    # level.  Deriving the attitude from the accelerometer instead was tried and
    # rejected on measurement: this clip never stops (min speed 2.54 m/s, no
    # sample below 0.40 deg/s of gyro), so no window isolates gravity.  Three
    # estimators disagree by 6.3 deg -- whole-stream mean 0.72 deg, first 20
    # samples 4.29 deg, quietest 0.5 s window 7.04 deg -- and a 1 deg attitude
    # error is 1.2 m of position over 69 m, i.e. larger than the effect being
    # measured.  The frame is therefore defined the way ``kitti.frame0`` defines
    # it (z up by construction) and the residual gravity leak is left to the
    # accelerometer bias term, which is what a bias key is for.
    R0 = None
    tilt_report = {}
    if imu is not None:
        fwd = np.zeros(3)
        for k in range(1, min(6, len(rel))):
            v = rel[k][:3, 3]
            fwd += v / max(np.linalg.norm(v), 1e-9)
        for tag, samp in (("whole_stream", b["imu_acc"]),
                          ("first_20", b["imu_acc"][:20]),
                          ("quietest_window",
                           b["imu_acc"][I.quietest_window(b["imu_acc"], imu.hz)])):
            u = samp.mean(0)
            u = u / np.linalg.norm(u)
            tilt_report[tag] = round(float(np.degrees(np.arccos(np.clip(u[2], -1, 1)))), 3)
        print(f"  gravity tilt estimates (deg): {tilt_report} -> "
              f"spread {max(tilt_report.values()) - min(tilt_report.values()):.2f} deg, "
              f"frame0 kept upright", flush=True)

    lo, hi = (int(x) for x in a.outage.split(","))
    glo, ghi = (int(x) for x in a.longgap.split(","))
    results = []

    print("== baseline: dead reckoning ==", flush=True)
    dr = G.dead_reckon(rel)
    results.append(row("dead_reckon", dr, truth,
                       "cumulative product of LIO increments", outage=(lo, hi)))
    print(f"  ATE mean {results[-1]['ate_mean_m']:.3f} m "
          f"(in gap {results[-1].get('ate_in_outage_m')})", flush=True)

    print("== graph: LIO only ==", flush=True)
    r = G.build_and_solve(rel, frame_t, None, cfg, use_rtk=False, use_imu=False,
                          ate_truth=truth, init_rotation=R0)
    results.append(row("graph_lio", r.poses, truth,
                       "pose graph, LIO between factors only", outage=(lo, hi), res=r))
    print(f"  ATE mean {r.ate_mean:.3f} m", flush=True)

    print(f"== graph: LIO + RTK, gap {lo}-{hi} ==", flush=True)
    r = G.build_and_solve(rel, frame_t, truth, cfg, use_rtk=True, use_imu=False,
                          rtk_outage=(lo, hi), ate_truth=truth, init_rotation=R0)
    results.append(row("graph_lio_rtk", r.poses, truth,
                       f"RTK translation prior, Huber k={cfg.rtk_huber_k}, "
                       f"gap frames {lo}-{hi}", outage=(lo, hi), res=r))
    print(f"  ATE mean {r.ate_mean:.3f} m (in gap "
          f"{results[-1].get('ate_in_outage_m')})", flush=True)

    if imu is not None:
        print("== graph: LIO + RTK + IMU ==", flush=True)
        r = G.build_and_solve(rel, frame_t, truth, cfg, use_rtk=True, use_imu=True,
                              imu=imu, rtk_outage=(lo, hi), ate_truth=truth, init_rotation=R0)
        results.append(row("graph_full", r.poses, truth,
                           f"adds ImuFactor, acc {cfg.imu.acc_sigma} gyro "
                           f"{cfg.imu.gyro_sigma}", outage=(lo, hi), res=r))
        print(f"  ATE mean {r.ate_mean:.3f} m (in gap "
              f"{results[-1].get('ate_in_outage_m')})", flush=True)

        print("== graph: IMU + LIO, RTK disabled ==", flush=True)
        r = G.build_and_solve(rel, frame_t, truth, cfg, use_rtk=False, use_imu=True,
                              imu=imu, ate_truth=truth, init_rotation=R0)
        results.append(row("graph_imu_only", r.poses, truth,
                           "no RTK factor at all: IMU dead reckoning", res=r))
        print(f"  ATE mean {r.ate_mean:.3f} m", flush=True)

        print(f"== long gap {glo}-{ghi}: does the IMU carry the gap? ==", flush=True)
        for flag, tag, note in (
            (False, "longgap_lio_rtk", "RTK+LIO, 90-frame gap, no IMU"),
            (True, "longgap_full", "RTK+LIO+IMU, same 90-frame gap"),
        ):
            r = G.build_and_solve(rel, frame_t, truth, cfg, use_rtk=True,
                                  use_imu=flag, imu=imu, rtk_outage=(glo, ghi),
                                  ate_truth=truth, init_rotation=R0)
            results.append(row(tag, r.poses, truth, note, outage=(glo, ghi), res=r))
            print(f"  {tag}: overall {r.ate_mean:.3f} m, in gap "
                  f"{results[-1].get('ate_in_outage_m')} m", flush=True)

    if imu is not None:
        # ---- LiDAR degeneracy: the only place the IMU can win ------------- #
        # Removing the RTK factor alone does not show what an IMU is for,
        # because the LiDAR odometry already holds the gap to 0.20 m.  A tunnel,
        # fog bank or open field kills the LiDAR between-factors too; then the
        # IMU is the only relative measurement left.  This window removes both.
        dlo, dhi = (int(x) for x in a.degen.split(","))
        print(f"== LiDAR + RTK degeneracy, frames {dlo}-{dhi} ==", flush=True)
        for flag, tag, note in (
            (False, "degen_no_imu", "no LIO, no RTK in the window: graph interpolates"),
            (True, "degen_imu", "same window, IMU factor carries the motion"),
        ):
            r = G.build_and_solve(rel, frame_t, truth, cfg, use_rtk=True,
                                  use_imu=flag, imu=imu, rtk_outage=(dlo, dhi),
                                  lio_outage=(dlo, dhi), ate_truth=truth,
                                  init_rotation=R0)
            results.append(row(tag, r.poses, truth, note, outage=(dlo, dhi), res=r))
            print(f"  {tag}: overall {r.ate_mean:.3f} m, in window "
                  f"{results[-1].get('ate_in_outage_m')} m", flush=True)

    if a.loops:
        loops = []
        for tok in a.loops.split(","):
            i, j = (int(x) for x in tok.split(":"))
            loops.append((i, j, np.linalg.inv(dr[i]) @ dr[j]))
        print(f"== graph: full + {len(loops)} loop factors ==", flush=True)
        r = G.build_and_solve(rel, frame_t, truth, cfg, use_rtk=True,
                              use_imu=imu is not None, imu=imu,
                              rtk_outage=(lo, hi), loops=loops, ate_truth=truth, init_rotation=R0)
        results.append(row("graph_full_loops", r.poses, truth,
                           f"{len(loops)} loop between-factors", outage=(lo, hi), res=r))
        print(f"  ATE mean {r.ate_mean:.3f} m", flush=True)

    # ---- persist ---------------------------------------------------------- #
    # "best" is chosen among the *standard* configurations only.  The long-gap and
    # degeneracy rows are stress tests of specific failure modes; letting one of
    # them win would hand later stages a trajectory from a scenario the robot is
    # not normally in.
    standard = ("dead_reckon", "graph_lio", "graph_lio_rtk", "graph_full")
    best = min([r for r in results if r["name"] in standard],
               key=lambda x: x["ate_rmse_m"])
    payload = {
        "dataset": b["meta"],
        "frames": int(n),
        "config": {
            "lio_sigma_rot": cfg.lio_sigma_rot, "lio_sigma_xy": cfg.lio_sigma_xy,
            "lio_sigma_z": cfg.lio_sigma_z, "rtk_sigma_xy": cfg.rtk_sigma_xy,
            "rtk_sigma_z": cfg.rtk_sigma_z, "rtk_huber_k": cfg.rtk_huber_k,
            "imu_acc_sigma": cfg.imu.acc_sigma, "imu_gyro_sigma": cfg.imu.gyro_sigma,
            "imu_hz": round(imu.hz, 2) if imu is not None else None,
            "outage": [lo, hi], "longgap": [glo, ghi],
            "degen": [int(x) for x in a.degen.split(",")],
            "frame0_tilt_estimates_deg": tilt_report,
        },
        "ablation": results,
        "best": best["name"],
    }
    with open(os.path.join(a.out, "stage1_ablation.json"), "w") as f:
        json.dump(payload, f, indent=2)

    np.save(os.path.join(a.out, "dr_poses.npy"), dr)
    for name, ps in POSE_CACHE.items():
        np.save(os.path.join(a.out, f"{name}_poses.npy"), ps)
    np.save(os.path.join(a.out, "best_poses.npy"), POSE_CACHE[best["name"]])
    np.save(os.path.join(a.out, "ate_truth.npy"), truth)
    np.save(os.path.join(a.out, "ate_frame_t.npy"), frame_t)

    hdr = (f"{'variant':18s} {'ATE mean':>9s} {'ATE fin':>8s} {'ATE rmse':>9s} "
           f"{'RPE1':>7s} {'RPE5':>7s} {'in gap':>8s} {'out gap':>8s}")
    print("\n== ablation ==")
    print(hdr)
    print("-" * len(hdr))
    for r_ in results:
        ig = r_.get("ate_in_outage_m")
        og = r_.get("ate_out_of_outage_m")
        print(f"{r_['name']:18s} {r_['ate_mean_m']:9.3f} {r_['ate_final_m']:8.3f} "
              f"{r_['ate_rmse_m']:9.3f} {r_['rpe1_m']:7.3f} {r_['rpe5_m']:7.3f} "
              f"{('-' if ig is None else f'{ig:8.3f}'):>8s} "
              f"{('-' if og is None else f'{og:8.3f}'):>8s}")
    print(f"\nbest by RMSE: {best['name']}  -> {a.out}/stage1_ablation.json")


if __name__ == "__main__":
    main()
