#!/usr/bin/env python3
"""Render the result video: bird's-eye map growth beside the camera view.

Three things have to be visible at once for the video to be evidence rather than
decoration:

* **the map growing** -- points appear as the vehicle sees them, and the ghost
  trail behind a moving car is visible when dynamic masking is off,
* **the dynamic mask** -- the camera panel shows the same car the BEV panel marks
  in red, so the reader can check the projection by eye,
* **the error** -- the RTK reference and the estimate are drawn together, with the
  current ATE in the HUD.

Implementation note: decoding 154 Velodyne scans plus running Mask R-CNN per
frame would dominate the runtime, so both are done once and cached
(``out/world_cloud.npz``, ``out/dyn_points.npy``).  The world cloud stores a
per-point *first frame index*, which lets the renderer show "the map as it existed
at frame i" with one boolean mask instead of rebuilding the cloud 154 times.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from navstack import bundle as B  # noqa: E402
from navstack import kitti, tiles as T, vision  # noqa: E402

BEV_W, BEV_H = 880, 660
IMG_W, IMG_H = 880, 660
HUD_H = 78
FPS = 15


def build_world_cloud(poses, pts, rgb, leaf=0.35, stride=1):
    """One merged world cloud plus, per point, the frame it first appeared in."""
    P, C, F = [], [], []
    for i in range(0, len(pts), stride):
        keep = np.linalg.norm(pts[i], axis=1) < 70.0
        p = pts[i][keep]
        c = rgb[i][keep] if rgb is not None else np.full((len(p), 3), 110, np.uint8)
        R, t = poses[i][:3, :3], poses[i][:3, 3]
        P.append(p @ R.T + t[None, :])
        C.append(c)
        F.append(np.full(len(p), i, np.int32))
    P = np.concatenate(P).astype(np.float32)
    C = np.concatenate(C).astype(np.uint8)
    F = np.concatenate(F)
    # voxel dedupe keeps the nearest-in-time copy, which is what "grows" visually
    key, _ = T.voxel_index(P, leaf)
    order = np.argsort(F)
    key, P, C, F = key[order], P[order], C[order], F[order]
    _, first = np.unique(key, return_index=True)
    return P[first], C[first], F[first]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--raw", default="/home/nvidia/rtk_fusion_demo/raw")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--media", default="/home/nvidia/rtk_fusion_demo/navstack_v2/media")
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drive", default="2011_09_26_drive_0005")
    ap.add_argument("--poses", default="best")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-mask", action="store_true",
                    help="skip Mask R-CNN (fast preview, no red points)")
    a = ap.parse_args()
    os.makedirs(a.media, exist_ok=True)

    b = B.load_bundle(a.data)
    poses = B.poses_from(a.out, prefer=(a.poses, "graph_full", "best", "dr"))
    truth = b["rtk_pos"]
    n = min(len(poses), len(b["pts"]), len(truth))
    if a.limit:
        n = min(n, a.limit)
    calib = kitti.load_calib(os.path.join(a.raw, a.date))
    drv = kitti.list_drive(os.path.join(a.raw, a.date, a.drive + "_sync"))

    # ---- world cloud (cached) -------------------------------------------- #
    cache = os.path.join(a.out, "world_cloud.npz")
    if os.path.exists(cache):
        with np.load(cache) as z:
            W, WC, WF = z["pts"], z["rgb"], z["frame"]
        print(f"  cached world cloud: {len(W)} pts", flush=True)
    else:
        print("== merging world cloud ==", flush=True)
        W, WC, WF = build_world_cloud(poses[:n], b["pts"][:n], b.get("rgb"), stride=1)
        np.savez(cache, pts=W, rgb=WC, frame=WF)
        print(f"  {len(W)} pts", flush=True)

    # ---- per-point dynamic flags (cached) -------------------------------- #
    dpath = os.path.join(a.out, "dyn_points.npy")
    if os.path.exists(dpath):
        DYN = np.load(dpath)
        print(f"  cached dynamic flags: {DYN.shape}", flush=True)
    elif a.no_mask:
        DYN = np.zeros((n, b["pts"].shape[1]), bool)
    else:
        print("== dynamic masks (Mask R-CNN) ==", flush=True)
        masker = vision.DynamicMasker()
        P02 = calib["P_rect_02"]
        DYN = np.zeros((n, b["pts"].shape[1]), bool)
        for i in range(n):
            img = np.array(Image.open(drv.img02[i]).convert("RGB"))
            m, _ = masker.mask(img, name=drv.img02[i])
            hit = masker.point_mask(b["pts"][i], P02, m, dilate=9) \
                if hasattr(masker, "point_mask") else masker.mask_points(b["pts"][i], P02, m, dilate=9)
            DYN[i, :len(hit)] = hit
            if i % 25 == 0:
                print(f"  {i}/{n}", flush=True)
        np.save(dpath, DYN)

    # world cloud counterpart of the dynamic flags, matched by nearest frame
    dyn_world = np.zeros(len(W), bool)
    if DYN.any():
        R = poses[:n, :3, :3]
        for i in range(n):
            sel = WF == i
            if not sel.any() or not DYN[i].any():
                continue
            p = b["pts"][i]
            keep = np.linalg.norm(p, axis=1) < 70.0
            pd = p[keep][DYN[i][:len(p)][keep]]
            if len(pd) == 0:
                continue
            pw = pd @ R[i].T + poses[i][:3, 3][None, :]
            # nearest-neighbour match into the deduped cloud, within one voxel
            d = ((W[sel][None, :, :2] - pw[:, None, :2]) ** 2).sum(-1)
            j = d.argmin(1)
            idx = np.where(sel)[0][j]
            dyn_world[idx] = True

    # ---- BEV framing ------------------------------------------------------ #
    lo = np.minimum(W[:, :2].min(0), truth[:, :2].min(0)) - 6
    hi = np.maximum(W[:, :2].max(0), truth[:, :2].max(0)) + 6
    span = np.maximum(hi - lo, 1e-3)
    scale = min((BEV_W - 20) / span[0], (BEV_H - 20) / span[1])
    cx, cy = (lo + hi) / 2

    def to_px(xy: np.ndarray) -> np.ndarray:
        u = ((xy[:, 0] - cx) * scale + BEV_W / 2).astype(np.int32)
        v = (BEV_H / 2 - (xy[:, 1] - cy) * scale).astype(np.int32)
        return np.stack([u, v], 1)

    px = to_px(W[:, :2])
    valid = (px[:, 0] >= 0) & (px[:, 0] < BEV_W) & (px[:, 1] >= 0) & (px[:, 1] < BEV_H)
    tr_px = to_px(truth[:n, :2])
    est_px = to_px(poses[:n, :3, 3][:, :2])

    # depth-sorted draw order: far first, so near returns overwrite
    order = np.argsort(W[:, 2])
    W, WC, WF, dyn_world, px, valid = (W[order], WC[order], WF[order],
                                       dyn_world[order], px[order], valid[order])

    vw = cv2.VideoWriter(os.path.join(a.media, "_raw.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (BEV_W + IMG_W, HUD_H + max(BEV_H, IMG_H)))
    print(f"== rendering {n} frames ==", flush=True)
    ate = np.linalg.norm(poses[:n, :3, 3] - truth[:n], axis=1)

    for i in range(n):
        bev = np.full((BEV_H, BEV_W, 3), 16, np.uint8)
        vis = valid & (WF <= i)
        p = px[vis]
        col = WC[vis][:, ::-1].astype(np.int16)          # RGB -> BGR
        # dim uncoloured returns: grey(110) is the "no camera saw this" marker
        grey = np.abs(WC[vis].astype(np.int16) - 110).max(1) < 3
        col[grey] = (col[grey] * 0.55).astype(np.int16) + np.array([40, 55, 40], np.int16)
        dyn = dyn_world[vis]
        col[dyn] = (30, 40, 245)                          # red in BGR
        bev[p[:, 1], p[:, 0]] = np.clip(col, 0, 255).astype(np.uint8)

        # trajectories: RTK green, estimate blue, current pose yellow
        cv2.polylines(bev, [tr_px[:i + 1].reshape(-1, 1, 2)], False, (90, 200, 90), 2, cv2.LINE_AA)
        cv2.polylines(bev, [est_px[:i + 1].reshape(-1, 1, 2)], False, (235, 170, 60), 2, cv2.LINE_AA)
        cv2.circle(bev, tuple(est_px[i]), 5, (60, 235, 255), -1, cv2.LINE_AA)
        cv2.rectangle(bev, (0, 0), (BEV_W - 1, BEV_H - 1), (40, 52, 66), 1)

        # camera panel
        img = np.array(Image.open(drv.img02[i]).convert("RGB"))[:, :, ::-1]
        keep = np.linalg.norm(b["pts"][i], axis=1) < 70.0
        if DYN[i].any():
            m = np.zeros(len(b["pts"][i]), bool)
            m[:len(DYN[i])] = DYN[i]
            hit = m[keep]
            if hit.any():
                uv = calib["P_rect_02"] @ np.hstack(
                    [b["pts"][i][keep][hit], np.ones((hit.sum(), 1))]).T
                z = uv[2]
                ok = z > 1e-3
                u = (uv[0][ok] / z[ok]).astype(int)
                v = (uv[1][ok] / z[ok]).astype(int)
                ok2 = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
                img[v[ok2], u[ok2]] = (40, 40, 255)
        pad = np.zeros((IMG_H, IMG_W, 3), np.uint8)
        s = min(IMG_W / img.shape[1], IMG_H / img.shape[0])
        r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                       interpolation=cv2.INTER_AREA)
        y0 = (IMG_H - r.shape[0]) // 2
        x0 = (IMG_W - r.shape[1]) // 2
        pad[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        cv2.rectangle(pad, (0, 0), (IMG_W - 1, IMG_H - 1), (40, 52, 66), 1)

        hud = np.full((HUD_H, BEV_W + IMG_W, 3), 21, np.uint8)
        cv2.putText(hud, f"frame {i:3d}/{n - 1}   t = {(b['frame_t'][i] - b['frame_t'][0]):5.2f} s",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 237, 243), 1, cv2.LINE_AA)
        cv2.putText(hud, f"ATE {ate[i]:.3f} m", (14, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 200, 90), 1, cv2.LINE_AA)
        cv2.putText(hud, f"map {int(vis.sum()):,} pts", (250, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 152, 168), 1, cv2.LINE_AA)
        # the estimate is drawn with BGR (235,170,60) = blue, and the legend used
        # to call it orange; a legend that disagrees with the pixels is worse than
        # no legend, so the text follows the colour, not the other way round
        cv2.putText(hud, "green = RTK truth   blue = fused estimate   "
                         "yellow = current pose   red = masked dynamic returns",
                    (14, 22 + 34 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (110, 125, 145), 1, cv2.LINE_AA)

        frame = np.vstack([hud, np.hstack([bev, pad])])
        vw.write(frame)
        if i % 30 == 0:
            print(f"  {i}/{n}", flush=True)
    vw.release()

    raw = os.path.join(a.media, "_raw.mp4")
    final = os.path.join(a.media, "navstack_v2.mp4")
    try:
        # -nostdin and stdin=DEVNULL are not optional: ffmpeg opens the terminal
        # to poll for interactive keys, and over an SSH session with a pty that
        # read raises SIGTTIN, which stops the process (measured: state T,
        # wchan do_signal_stop, python parked in do_wait for 10+ minutes).
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                        "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", final],
                       check=True, stdin=subprocess.DEVNULL)
        os.remove(raw)
        print(f"  h264 -> {final} ({os.path.getsize(final) / 1e6:.1f} MB)", flush=True)
    except Exception as e:
        print(f"  ffmpeg unavailable ({e}); keeping {raw}", flush=True)
        final = raw

    # poster from a frame in the middle of the run
    mid = n // 2
    post = np.array(Image.open(drv.img02[mid]).convert("RGB"))
    Image.fromarray(post).save(os.path.join(a.media, "poster.png"))
    print(f"saved {final}")


if __name__ == "__main__":
    main()
