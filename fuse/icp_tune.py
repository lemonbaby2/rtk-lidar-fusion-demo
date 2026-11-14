#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare registration-cloud strategies for the ICP odometry.

The first run drifted by 70 m over a 63 m path, which is not "drift", it is a
broken front end: the residual and the frame-to-frame motion were the same size,
so the alignment was barely constrained.  The candidate fix is to register on
random point subsets instead of voxel means (voxel averaging smears the surface
and leaves only a few thousand smoothed blobs to match), and to keep the
registration radius small where the geometry is dense.

Prints step-length statistics next to the ATE so the scale of the motion can be
checked against the RTK path directly.
"""
import time

import numpy as np
import torch

from fuse import DEV, align_to_world, icp, voxel_downsample

DATA = "/home/nvidia/rtk_fusion_demo/data"
pts = np.load(f"{DATA}/pts.npy")
pose = np.load(f"{DATA}/pose_rtk.npy")
Pts = torch.from_numpy(pts).to(DEV)
n = pts.shape[0]
z = torch.from_numpy(pose).to(DEV).float()[:, :3, 3]
z = (z - z[0]).cpu().numpy()

CONFIGS = [
    ("voxel0.6/r60/n3000", "voxel", 0.6, 60.0, 3000),
    ("voxel0.4/r30/n6000", "voxel", 0.4, 30.0, 6000),
    ("random/r30/n5000", "random", 0.0, 30.0, 5000),
    ("random/r25/n8000", "random", 0.0, 25.0, 8000),
    ("random/r40/n12000", "random", 0.0, 40.0, 12000),
]


def build(mode, vox, radius, want, seed=0):
    rng = np.random.default_rng(seed)
    clouds = []
    for i in range(n):
        p = Pts[i]
        p = p[torch.linalg.norm(p, dim=1) < radius]
        if mode == "voxel":
            p = voxel_downsample(p, vox)
            if p.shape[0] > want:
                sel = torch.linspace(0, p.shape[0] - 1, want).long()
                p = p[sel]
        else:
            if p.shape[0] > want:
                idx = rng.choice(p.shape[0], want, replace=False)
                p = p[torch.from_numpy(idx).to(p.device)]
        clouds.append(p)
    return clouds


def run(clouds, iters=40):
    T = [torch.eye(4, device=DEV, dtype=torch.float32)]
    rel = []
    fits = []
    for i in range(1, n):
        guess = rel[-1].clone() if rel else torch.eye(4, device=DEV)
        T_abs, me, md = icp(clouds[i], clouds[i - 1], guess, iters=iters)
        rel.append(T_abs)
        T.append(T[-1] @ T_abs)
        fits.append((me, md))
    Tall = align_to_world(torch.stack(T), pose)
    p = Tall[:, :3, 3].double().cpu().numpy()
    err = np.linalg.norm(p - z, axis=1)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    zstep = np.linalg.norm(np.diff(z, axis=0), axis=1)
    return err, step, zstep, np.median([f[1] for f in fits])


print(f"device={DEV} frames={n}  rtk path={np.linalg.norm(np.diff(z, axis=0), axis=1).sum():.1f} m")
print(f"{'config':22s} {'ATE mean':>9s} {'ATE final':>10s} {'step mean':>10s} "
      f"{'rtk step':>9s} {'med nn':>7s} {'sec':>5s}")
for name, mode, vox, radius, want in CONFIGS:
    t0 = time.time()
    clouds = build(mode, vox, radius, want)
    err, step, zstep, med = run(clouds)
    print(f"{name:22s} {err.mean():9.2f} {err[-1]:10.2f} {step.mean():10.3f} "
          f"{zstep.mean():9.3f} {med:7.3f} {time.time()-t0:5.1f}", flush=True)
