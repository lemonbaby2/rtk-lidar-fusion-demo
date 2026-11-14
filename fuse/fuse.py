#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTK + LiDAR + camera fusion demo on the DGX.

Pipeline (all heavy steps on GPU via torch):
  1. frame-to-frame ICP  -> a real odometry chain (no RTK used here)
  2. translation pose graph: odometry relative constraints + RTK absolute
     observations, solved as sparse linear least squares
  3. a deliberate RTK dropout window, to show what the constraint is worth
  4. global coloured point cloud rebuilt with the fused poses

Why the pose graph is translation-only: the RTK/INS solution mainly corrects
position drift; orientation is carried by the odometry chain and only mildly
pulled toward the INS yaw.  We state that instead of pretending to solve full
SE(3) without a real LIO front end.

Run with the venv interpreter that already has torch (read-only use):
  /home/nvidia/sop-vision/.venv/bin/python fuse.py
"""
import json
import os
import time

import numpy as np
import torch

DATA = os.environ.get("FUSE_DATA", "/home/nvidia/rtk_fusion_demo/data")
OUT = os.environ.get("FUSE_OUT", "/home/nvidia/rtk_fusion_demo/out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ICP_PTS = 8000          # points per frame used for registration
ICP_RADIUS = 25.0       # only register on the near field, where geometry is dense
ICP_ITERS = 40
DROP_START, DROP_END = 40, 90   # frames where RTK is treated as unavailable


def log(*a):
    print(*a, flush=True)


def voxel_downsample(x, voxel=0.6):
    """Deterministic voxel-grid downsample; returns one representative per cell."""
    if len(x) == 0:
        return x
    key = torch.floor(x / voxel).to(torch.int64)
    _, inv, cnt = torch.unique(key, dim=0, return_inverse=True, return_counts=True)
    # mean inside each voxel keeps the surface centred
    out = torch.zeros((cnt.numel(), 3), dtype=x.dtype, device=x.device)
    out.index_add_(0, inv, x)
    return out / cnt.unsqueeze(1).to(x.dtype)


def best_fit_transform(src, dst):
    """Closed-form rigid transform mapping src onto dst (Kabsch)."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=src.device, dtype=src.dtype))
    R = Vt.T @ D @ U.T
    t = cd - R @ cs
    T = torch.eye(4, device=src.device, dtype=src.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def icp(src, dst, T0, iters=ICP_ITERS):
    """Point-to-point ICP with an adaptive correspondence-distance gate.

    torch.quantile is refused on large CUDA tensors, so the gate uses a scaled
    mean instead: it adapts to the residual scale without a sort.
    """
    T = T0.clone()
    prev = None
    for _ in range(iters):
        s = (T[:3, :3] @ src.T).T + T[:3, 3]
        d = torch.cdist(s, dst)                 # (Ns, Nd)
        dist, idx = d.min(dim=1)
        thr = torch.clamp(dist.mean() * 1.5, max=2.0)
        gate = dist < thr
        if gate.sum() < 50:
            break
        Tn = best_fit_transform(s[gate], dst[idx[gate]])
        T = Tn @ T
        err = dist[gate].mean().item()
        if prev is not None and abs(prev - err) < 1e-5:
            break
        prev = err
    s = (T[:3, :3] @ src.T).T + T[:3, 3]
    err = torch.cdist(s, dst).min(dim=1).values
    return T, float(err.mean().item()), float(err.median().item())


def align_to_world(T_odo, pose_rtk):
    """Put the ICP chain into the RTK world frame.

    T_odo expresses each pose in the coordinate frame of frame 0, whose x axis is
    whatever direction the vehicle happened to be facing.  The RTK positions live
    in an ENU-like world where x is east.  Comparing the two without a rotation
    makes a perfectly good odometry look catastrophically wrong, and it also
    mixes two frames inside one pose graph.  Rotating the chain by the RTK
    orientation of frame 0 fixes both, and it uses information an INS always
    provides (initial heading) rather than a fitted alignment.
    """
    R0 = np.eye(4, dtype=np.float64)
    R0[:3, :3] = pose_rtk[0][:3, :3]
    return (torch.from_numpy(R0).to(T_odo.device, T_odo.dtype) @ T_odo)


def main():
    t_all = time.time()
    os.makedirs(OUT, exist_ok=True)
    log(f"device: {DEV} | {torch.cuda.get_device_name(0) if DEV=='cuda' else 'cpu'}")

    pts = np.load(f"{DATA}/pts.npy")            # (N, P, 3) float32, frame 0 origin
    rgb = np.load(f"{DATA}/rgb.npy")            # (N, P, 3) uint8
    pose_rtk = np.load(f"{DATA}/pose_rtk.npy")  # (N, 4, 4) RTK/INS ground truth
    meta = json.load(open(f"{DATA}/meta.json"))
    n, P = pts.shape[0], pts.shape[1]
    log(f"frames={n} points/frame={P} trajectory={meta.get('source','')}")

    Pts = torch.from_numpy(pts).to(DEV)
    # Registration clouds are random subsets of the near field.  Voxel means were
    # tried first and smeared the surfaces into a few thousand blobs, which made
    # the alignment far weaker than the raw geometry allows (see icp_tune.py):
    # median nearest-neighbour 0.35 m for voxel means against 0.13 m here.
    rng = np.random.default_rng(0)
    small = []
    for i in range(n):
        p = Pts[i]
        p = p[torch.linalg.norm(p, dim=1) < ICP_RADIUS]
        if p.shape[0] > ICP_PTS:
            sel = rng.choice(p.shape[0], ICP_PTS, replace=False)
            p = p[torch.from_numpy(sel).to(p.device)]
        small.append(p)
    log(f"registration clouds: mean {np.mean([len(s) for s in small]):.0f} pts/frame "
        f"within {ICP_RADIUS:.0f} m")

    # ---------- 1. odometry via ICP, constant-velocity initialisation ----------
    log("ICP odometry ...")
    T_odo = [torch.eye(4, device=DEV, dtype=torch.float32)]
    rel = []
    fitness = []
    t0 = time.time()
    for i in range(1, n):
        guess = rel[-1].clone() if rel else torch.eye(4, device=DEV)
        # T_abs maps points of frame i into frame i-1, so the world pose
        # composes on the left: T_world_i = T_world_{i-1} @ T_abs
        T_abs, mean_err, med_err = icp(small[i], small[i - 1], guess)
        rel.append(T_abs)
        T_odo.append(T_odo[-1] @ T_abs)
        fitness.append((mean_err, med_err))
        if i % 20 == 0:
            log(f"  {i}/{n}  mean_nn={mean_err:.3f} med_nn={med_err:.3f}  "
                f"{(time.time()-t0):.1f}s")
    T_odo = torch.stack(T_odo)
    T_odo = align_to_world(T_odo, pose_rtk)
    torch.cuda.synchronize()
    log(f"ICP done in {time.time()-t0:.1f}s")

    # ---------- 2. translation pose graph ----------
    p_odo = T_odo[:, :3, 3].double().cpu().numpy()
    z_rtk = torch.from_numpy(pose_rtk).to(DEV).float()[:, :3, 3].double().cpu().numpy()
    z_rtk = z_rtk - z_rtk[0]

    W_ODO, W_RTK = 1.0, 8.0
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for i in range(n - 1):
        d = p_odo[i + 1] - p_odo[i]
        for k in range(3):
            rows += [r, r]; cols += [3 * i + k, 3 * (i + 1) + k]; vals += [-W_ODO, W_ODO]
            rhs.append(W_ODO * d[k]); r += 1
    for i in range(n):
        if DROP_START <= i < DROP_END:
            continue                      # simulated RTK outage
        for k in range(3):
            rows.append(r); cols.append(3 * i + k); vals.append(W_RTK)
            rhs.append(W_RTK * z_rtk[i, k]); r += 1
    A = np.zeros((r, 3 * n)); A[rows, cols] = vals
    b = np.array(rhs)
    p_fused, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_fused = p_fused.reshape(n, 3)

    err_odo = np.linalg.norm(p_odo - z_rtk, axis=1)
    err_fused = np.linalg.norm(p_fused - z_rtk, axis=1)
    log(f"ATE(translation) odometry: mean {err_odo.mean():.2f} m  final {err_odo[-1]:.2f} m")
    log(f"ATE(translation) fused   : mean {err_fused.mean():.2f} m  final {err_fused[-1]:.2f} m")
    log(f"improvement: {(1-err_fused.mean()/max(err_odo.mean(),1e-9))*100:.1f}%")

    # ---------- 3. rebuild the coloured map with fused poses ----------
    log("rebuilding global coloured cloud ...")
    world = []
    world_rgb = []
    R_rtk = torch.from_numpy(pose_rtk).to(DEV).float()
    for i in range(n):
        # keep the odometry rotation, use the fused translation
        T = T_odo[i].clone()
        T[:3, 3] = torch.tensor(p_fused[i], device=DEV, dtype=T.dtype)
        R = R_rtk[i, :3, :3] @ R_rtk[0, :3, :3].T
        T[:3, :3] = R
        x = Pts[i]
        keep = torch.linalg.norm(x, dim=1) < 60.0
        x = x[keep]
        c = torch.from_numpy(rgb[i]).to(DEV)[keep]
        world.append((T[:3, :3] @ x.T).T + T[:3, 3])
        world_rgb.append(c)
    counts = np.array([int(w.shape[0]) for w in world], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    W = torch.cat(world)
    C = torch.cat(world_rgb)
    log(f"global points: {W.shape[0]} (offsets allow per-frame accumulation)")

    # one voxel pass: mean position and mean colour per 0.15 m cell.  The mean
    # keeps the surface centred, and doing it in a single pass avoids hashing
    # three million keys twice.
    vox = 0.15
    key = torch.floor(W / vox).to(torch.int64)
    uniq, inv = torch.unique(key, dim=0, return_inverse=True)
    cnt = torch.zeros(uniq.shape[0], dtype=torch.float32, device=DEV)
    cnt.index_add_(0, inv, torch.ones(W.shape[0], dtype=torch.float32, device=DEV))
    pos = torch.zeros((uniq.shape[0], 3), dtype=torch.float32, device=DEV)
    pos.index_add_(0, inv, W)
    pos = pos / cnt.unsqueeze(1)
    acc = torch.zeros((uniq.shape[0], 3), dtype=torch.float32, device=DEV)
    acc.index_add_(0, inv, C.float())
    Cds = (acc / cnt.unsqueeze(1)).clamp(0, 255).to(torch.uint8)
    log(f"map after voxel({vox} m): {uniq.shape[0]} points")

    rot_rtk = (R_rtk[:, :3, :3].cpu().numpy()).astype(np.float32)
    np.savez_compressed(
        f"{OUT}/fusion_result.npz",
        world=W.cpu().numpy().astype(np.float32),
        world_rgb=C.cpu().numpy().astype(np.uint8),
        map_xyz=pos.cpu().numpy().astype(np.float32),
        map_rgb=Cds.cpu().numpy(),
        map_key=uniq.cpu().numpy(),
        frame_offsets=offsets, frame_counts=counts, rot_rtk=rot_rtk,
        p_odo=p_odo, p_fused=p_fused, p_rtk=z_rtk,
        err_odo=err_odo, err_fused=err_fused,
        fitness=np.array(fitness),
        drop_window=np.array([DROP_START, DROP_END]),
    )
    summary = {
        "frames": int(n), "points_per_frame": int(P),
        "global_points": int(W.shape[0]), "map_points_voxel15cm": int(uniq.shape[0]),
        "path_length_m": float(np.linalg.norm(np.diff(z_rtk, axis=0), axis=1).sum()),
        "ate_odo_mean_m": float(err_odo.mean()), "ate_odo_final_m": float(err_odo[-1]),
        "ate_fused_mean_m": float(err_fused.mean()), "ate_fused_final_m": float(err_fused[-1]),
        "improvement_pct": float((1 - err_fused.mean() / max(err_odo.mean(), 1e-9)) * 100),
        "rtk_dropout_frames": [DROP_START, DROP_END],
        "icp_median_nn_m": float(np.median([f[1] for f in fitness])),
        "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
        "odometry": "frame-to-frame point-to-point ICP, constant-velocity init, no RTK",
        "fusion": "translation pose graph, odometry relative + RTK absolute, lstsq",
        "rotation_source": "RTK/INS relative orientation for the map rebuild; "
                           "the pose graph itself solves translation only",
        "alignment": "odometry chain rotated into the RTK world frame with the "
                     "frame-0 INS orientation before any error is computed",
        "gpu_seconds": round(time.time() - t_all, 1),
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))
    log(f"total {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
