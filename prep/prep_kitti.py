#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KITTI raw drive_0005 -> compact npz bundle, running on the DGX itself.

Same logic as the workstation version, but it reads the raw archive that the DGX
downloaded locally instead of a bundle pushed over the relay link.

Projection chain (documented KITTI):
    X_img = P_rect_02 @ R_rect_00 @ Tr_velo_to_cam @ X_velo
A point is coloured only when it lands inside image 02 with positive depth;
the rest keep grey (110,110,110) so untextured regions stay visibly untextured.
"""
import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

R_EARTH = 6378137.0


def load_calib(base):
    def mat(path, key, shape):
        with open(path) as f:
            for line in f:
                if line.startswith(key + ":"):
                    v = np.array([float(x) for x in line.split(":")[1].split()])
                    return v.reshape(shape)
        raise KeyError(f"{key} not in {path}")

    R_v2c = mat(os.path.join(base, "calib_velo_to_cam.txt"), "R", (3, 3))
    T_v2c = mat(os.path.join(base, "calib_velo_to_cam.txt"), "T", (3,))
    R_i2v = mat(os.path.join(base, "calib_imu_to_velo.txt"), "R", (3, 3))
    T_i2v = mat(os.path.join(base, "calib_imu_to_velo.txt"), "T", (3,))
    R_rect = mat(os.path.join(base, "calib_cam_to_cam.txt"), "R_rect_00", (3, 3))
    P2 = mat(os.path.join(base, "calib_cam_to_cam.txt"), "P_rect_02", (3, 4))
    P3 = mat(os.path.join(base, "calib_cam_to_cam.txt"), "P_rect_03", (3, 4))

    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = R_v2c
    Tr_velo2cam[:3, 3] = T_v2c
    Tr_imu2velo = np.eye(4)
    Tr_imu2velo[:3, :3] = R_i2v
    Tr_imu2velo[:3, 3] = T_i2v
    R_rect4 = np.eye(4)
    R_rect4[:3, :3] = R_rect

    # both colour cameras: 02 front-left, 03 front-right, ~160 deg together
    return [P2 @ R_rect4 @ Tr_velo2cam, P3 @ R_rect4 @ Tr_velo2cam], \
        np.linalg.inv(Tr_imu2velo)


def euler_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def read_oxts(d):
    frames = sorted(glob.glob(os.path.join(d, "oxts", "data", "*.txt")))
    o = np.array([[float(x) for x in open(f).read().split()] for f in frames])
    return o


def rtk_poses(o, T_imu2velo):
    """RTK/INS pose of the velodyne frame in a local Mercator world."""
    lat0 = o[0, 0]
    scale = np.cos(lat0 * np.pi / 180.0)
    tx = scale * o[:, 1] * np.pi * R_EARTH / 180.0
    ty = scale * R_EARTH * np.log(np.tan((90.0 + o[:, 0]) * np.pi / 360.0))
    tz = o[:, 2]
    out = np.zeros((len(o), 4, 4))
    t0 = np.array([tx[0], ty[0], tz[0]])
    for i in range(len(o)):
        T = np.eye(4)
        T[:3, :3] = euler_to_R(o[i, 3], o[i, 4], o[i, 5])
        T[:3, 3] = np.array([tx[i], ty[i], tz[i]]) - t0
        out[i] = T @ T_imu2velo
    return out


def colourise(xyz, imgs, projs):
    """Sample RGB from each colour camera in turn; the first hit wins.

    A single camera leaves roughly 84 % of a 360 deg LiDAR sweep grey, which makes
    the fused map look monochrome for reasons that have nothing to do with the
    fusion.  Cameras 02 and 03 together cover about 160 deg of the forward view.
    """
    n = len(xyz)
    rgb = np.full((n, 3), 110, dtype=np.uint8)
    left = np.ones(n, dtype=bool)                   # still uncoloured
    h_xyz = np.hstack([xyz, np.ones((n, 1))])
    for img, proj in zip(imgs, projs):
        if not left.any():
            break
        h, w = img.shape[:2]
        uv = (proj @ h_xyz[left].T).T
        z = uv[:, 2]
        ok = z > 1e-3
        if not np.any(ok):
            continue
        u = uv[ok, 0] / z[ok]
        v = uv[ok, 1] / z[ok]
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        sel = np.where(left)[0][np.where(ok)[0][inside]]
        ui = np.clip(u[inside].astype(np.int32), 0, w - 1)
        vi = np.clip(v[inside].astype(np.int32), 0, h - 1)
        rgb[sel] = img[vi, ui]
        left[sel] = False
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/nvidia/rtk_fusion_demo/raw")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/data")
    ap.add_argument("--drive", default="2011_09_26_drive_0005")
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--max-pts", type=int, default=20000)
    ap.add_argument("--range", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.raw, a.date)
    projs, T_imu2velo = load_calib(base)
    drive = os.path.join(base, a.drive + "_sync")

    o = read_oxts(drive)
    poses = rtk_poses(o, T_imu2velo)
    velo = sorted(glob.glob(os.path.join(drive, "velodyne_points", "data", "*.bin")))
    img2 = sorted(glob.glob(os.path.join(drive, "image_02", "data", "*.png")))
    img3 = sorted(glob.glob(os.path.join(drive, "image_03", "data", "*.png")))
    n = min(len(velo), len(img2), len(img3), len(poses))
    print(f"velo={len(velo)} img02={len(img2)} img03={len(img3)} oxts={len(poses)} "
          f"-> using {n}", flush=True)

    rng = np.random.default_rng(a.seed)
    pts = np.zeros((n, a.max_pts, 3), dtype=np.float32)
    cols = np.zeros((n, a.max_pts, 3), dtype=np.uint8)
    for i in range(n):
        raw = np.fromfile(velo[i], dtype=np.float32).reshape(-1, 4)
        xyz = raw[:, :3]
        imgs = [np.array(Image.open(img2[i]).convert("RGB")),
                np.array(Image.open(img3[i]).convert("RGB"))]
        keep = np.linalg.norm(xyz, axis=1) < a.range
        xyz = xyz[keep]
        if len(xyz) > a.max_pts:
            sel = rng.choice(len(xyz), a.max_pts, replace=False)
            xyz = xyz[sel]
        rgb = colourise(xyz, imgs, projs)
        m = len(xyz)
        pts[i, :m] = xyz
        cols[i, :m] = rgb
        if i % 30 == 0:
            print(f"  frame {i}/{n} pts={m}", flush=True)

    np.save(os.path.join(a.out, "pts.npy"), pts)
    np.save(os.path.join(a.out, "rgb.npy"), cols)
    np.save(os.path.join(a.out, "pose_rtk.npy"), poses)
    meta = {
        "frames": int(n), "max_points_per_frame": a.max_pts,
        "range_limit_m": a.range, "seed": a.seed,
        "source": "KITTI raw 2011_09_26_drive_0005 (CC BY-NC-SA 3.0)",
        "sensors": "Velodyne HDL-64E, cameras 02+03 (colour), "
                   "OXTS RT3003 GNSS/INS (RTK-grade)",
        "projection": "X_img = P_rect_0{2,3} @ R_rect_00 @ Tr_velo_to_cam @ X_velo",
        "note": "grey(110,110,110) means the point projected into neither colour camera",
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    d = np.linalg.norm(poses[:, :3, 3], axis=1)
    print(f"saved to {a.out}; path length {d.max():.1f} m, duration {n*0.1:.1f} s", flush=True)


if __name__ == "__main__":
    main()
