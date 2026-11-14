#!/usr/bin/env python3
"""KITTI raw -> compact numpy bundle for the whole stack.

The bundle is deliberately plain numpy: 154 frames of 20 k points is 37 MB, it
loads in 30 ms, and every downstream stage can be run without the raw archive.

Outputs
-------
``pts.npy``           (N, P, 3) float32  velodyne frame, range limited
``rgb.npy``           (N, P, 3) uint8    colour from cameras 02+03, grey if unseen
``pose_rtk.npy``      (N, 4, 4) float64  v1-compatible Mercator pose of the velodyne frame
``rtk_pos.npy``       (N, 3)   float64   RTK translation rotated into frame0
``rtk_R_w2f0.npy``    (3, 3)             world -> frame0 rotation used above
``frame_t.npy``       (N,)     float64   unix seconds, camera 02 clock
``imu_t.npy``         (M,)     float64   unix seconds
``imu_acc.npy``       (M, 3)             specific force, body frame
``imu_gyro.npy``      (M, 3)             angular rate, body frame
``meta.json``
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import kitti  # noqa: E402


def colourise(xyz: np.ndarray, imgs, projs) -> np.ndarray:
    """Sample RGB from each colour camera in turn; first hit wins.

    One camera leaves ~84 % of a 360 deg sweep grey; cameras 02 and 03 together
    cover about 160 deg forward.  Grey (110,110,110) is kept as a *marker* so the
    renderer can dim uncoloured returns instead of painting them a made-up colour.
    """
    n = len(xyz)
    rgb = np.full((n, 3), 110, dtype=np.uint8)
    left = np.ones(n, dtype=bool)
    h = np.hstack([xyz, np.ones((n, 1))])
    for img, proj in zip(imgs, projs):
        if not left.any():
            break
        ih, iw = img.shape[:2]
        uv = (proj @ h[left].T).T
        z = uv[:, 2]
        ok = z > 1e-3
        if not np.any(ok):
            continue
        u = uv[ok, 0] / z[ok]
        v = uv[ok, 1] / z[ok]
        inside = (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
        sel = np.where(left)[0][np.where(ok)[0][inside]]
        ui = np.clip(u[inside].astype(np.int32), 0, iw - 1)
        vi = np.clip(v[inside].astype(np.int32), 0, ih - 1)
        rgb[sel] = img[vi, ui]
        left[sel] = False
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="/home/nvidia/navstack_v2/raw")
    ap.add_argument("--out", default="/home/nvidia/navstack_v2/data")
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drive", default="2011_09_26_drive_0005")
    ap.add_argument("--max-pts", type=int, default=20000)
    ap.add_argument("--range", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-colour", action="store_true", help="skip RGB sampling (3x faster)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.raw, a.date)
    calib = kitti.load_calib(base)
    drive = kitti.list_drive(os.path.join(base, a.drive + "_sync"))
    n = drive.n
    print(f"frames={n}  camera baseline={calib['stereo_baseline_m']:.3f} m", flush=True)

    # ---- timestamps: camera 02 is the reference clock -------------------- #
    stamps02 = kitti._parse_stamps(os.path.join(drive.root, "image_02", "timestamps.txt"))
    frame_t = stamps02[:n]

    # ---- IMU: prefer 100 Hz extract, fall back to 10 Hz sync ------------- #
    synced = kitti.find_oxts(os.path.join(base, a.drive + "_sync"),
                             os.path.join(base, a.drive + "_extract"))
    print(f"IMU: {len(synced)} samples @ {synced.hz:.1f} Hz", flush=True)

    # ---- RTK ------------------------------------------------------------- #
    # v1-compatible poses use the 10 Hz sync oxts so they stay numerically
    # identical to the v1 run; the 100 Hz stream has the same trajectory.
    sync_oxts = kitti.find_oxts(os.path.join(base, a.drive + "_sync"))
    m = min(n, len(sync_oxts))
    pose_rtk = kitti.rtk_poses(sync_oxts.raw[:m], calib["T_imu_velo"])
    # resample to frame count if the sync oxts is shorter than the images
    if m < n:
        pad = np.repeat(pose_rtk[-1:], n - m, axis=0)
        pose_rtk = np.concatenate([pose_rtk, pad], axis=0)
    rtk_pos, R_w2f0 = kitti.rtk_in_frame0(sync_oxts.raw[:m])
    if len(rtk_pos) < n:
        rtk_pos = np.concatenate(
            [rtk_pos, np.repeat(rtk_pos[-1:], n - len(rtk_pos), axis=0)], axis=0)

    path_len = float(np.linalg.norm(np.diff(rtk_pos, axis=0), axis=1).sum())
    print(f"RTK path length {path_len:.2f} m over {n} frames", flush=True)

    # ---- point clouds ---------------------------------------------------- #
    rng = np.random.default_rng(a.seed)
    pts = np.zeros((n, a.max_pts, 3), dtype=np.float32)
    cols = np.full((n, a.max_pts, 3), 110, dtype=np.uint8)
    grey_frac = []
    for i in range(n):
        xyz = kitti.read_velodyne(drive.velo[i])
        keep = np.linalg.norm(xyz, axis=1) < a.range
        xyz = xyz[keep]
        if len(xyz) > a.max_pts:
            xyz = xyz[rng.choice(len(xyz), a.max_pts, replace=False)]
        if not a.no_colour:
            imgs = [np.array(Image.open(drive.img02[i]).convert("RGB")),
                    np.array(Image.open(drive.img03[i]).convert("RGB"))]
            rgb = colourise(xyz, imgs, [calib["P_rect_02"], calib["P_rect_03"]])
            cols[i, :len(xyz)] = rgb
            grey_frac.append(float((rgb[:, 0] == 110).mean()))
        pts[i, :len(xyz)] = xyz
        if i % 40 == 0:
            print(f"  frame {i}/{n}  pts={len(xyz)}", flush=True)

    np.save(os.path.join(a.out, "pts.npy"), pts)
    np.save(os.path.join(a.out, "rgb.npy"), cols)
    np.save(os.path.join(a.out, "pose_rtk.npy"), pose_rtk)
    np.save(os.path.join(a.out, "rtk_pos.npy"), rtk_pos.astype(np.float64))
    np.save(os.path.join(a.out, "rtk_R_w2f0.npy"), R_w2f0)
    np.save(os.path.join(a.out, "frame_t.npy"), frame_t)
    np.save(os.path.join(a.out, "imu_t.npy"), synced.t)
    np.save(os.path.join(a.out, "imu_acc.npy"), synced.raw[:, kitti.OXT_ACC])
    np.save(os.path.join(a.out, "imu_gyro.npy"), synced.raw[:, kitti.OXT_GYRO])

    meta = {
        "frames": int(n),
        "max_points_per_frame": a.max_pts,
        "range_limit_m": a.range,
        "duration_s": float(frame_t[-1] - frame_t[0]),
        "path_length_m": path_len,
        "imu_hz": round(synced.hz, 2),
        "imu_samples": int(len(synced)),
        "stereo_baseline_m": round(float(calib["stereo_baseline_m"]), 4),
        "grey_point_fraction": round(float(np.mean(grey_frac)), 4) if grey_frac else None,
        "source": "KITTI raw 2011_09_26_drive_0005 (CC BY-NC-SA 3.0)",
        "sensors": "Velodyne HDL-64E, cameras 02+03, OXTS RT3003 GNSS/INS (RTK-grade)",
        "frame0": "IMU body frame at frame 0; x = travel direction from the first RTK baseline",
        "gravity_frame0": [0.0, 0.0, -9.80665],
        "projection": "X_img = P_rect_0{2,3} @ R_rect_00 @ Tr_velo_to_cam @ X_velo",
        "colour_note": "grey(110,110,110) = point fell in neither colour camera",
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {a.out}: {path_len:.1f} m, {meta['duration_s']:.1f} s, "
          f"IMU {synced.hz:.0f} Hz", flush=True)


if __name__ == "__main__":
    main()
