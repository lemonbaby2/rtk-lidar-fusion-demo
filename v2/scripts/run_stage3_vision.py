#!/usr/bin/env python3
"""Stage 3: the vision layer, measured three ways.

Every candidate model is scored on the question it is supposed to answer, and
nothing else:

``dynamic mask``
    does removing masked returns improve the *registration residual* on frame
    pairs that contain a moving vehicle?  (This is the only vision component that
    touches geometry, so it is the only one that can move the ATE.)
``semantic cost``
    what fraction of the local grid gets a usable label, and how long does it
    take per frame?
``visual descriptor``
    leave-one-out place-recognition recall on the camera stream, to be used
    later as a tie-breaking vote on top of the LiDAR retrieval.

Plus one negative result: ``stereo_vs_lidar`` shows the calibrated stereo pair
already agrees with LiDAR to within the beam spacing, which is why monocular
depth completion is not part of this stack.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from navstack import bundle as B  # noqa: E402
from navstack import kitti, levels, lio, vision  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--raw", default="/home/nvidia/rtk_fusion_demo/raw")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--media", default="/home/nvidia/rtk_fusion_demo/navstack_v2/media")
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drive", default="2011_09_26_drive_0005")
    ap.add_argument("--poses", default="best")
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--pairs", type=int, default=24, help="ICP pairs to compare")
    ap.add_argument("--stereo-frames", type=int, default=6)
    ap.add_argument("--tolerance", type=int, default=15,
                    help="frames counted as the right place; must exceed the "
                         "10-frame temporal exclusion or recall is 0 by construction")
    ap.add_argument("--save-masks", type=int, default=6)
    ap.add_argument("--force-masks", action="store_true",
                    help="re-run Mask R-CNN even if a mask cache exists")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.media, exist_ok=True)
    b = B.load_bundle(a.data)
    poses = B.poses_from(a.out, prefer=(a.poses, "graph_full", "best", "dr"))
    calib = kitti.load_calib(os.path.join(a.raw, a.date))
    drv = kitti.list_drive(os.path.join(a.raw, a.date, a.drive + "_sync"))
    n = min(len(poses), len(b["pts"]), drv.n)
    if a.limit:
        n = min(n, a.limit)
    print(f"== vision over {n} frames ==", flush=True)

    masker = vision.DynamicMasker(score_thr=0.55)
    P02 = calib["P_rect_02"]
    P03 = calib["P_rect_03"]

    dyn_frac_pts = np.zeros(n, np.float32)
    dyn_frac_img = np.zeros(n, np.float32)
    instances = np.zeros(n, np.int32)
    ms_list = np.zeros(n, np.float32)
    per_img = np.zeros((n, 3), np.float32)      # idx01, idx02, idx03
    per_mask = np.zeros((n, 15), np.float32)    # A,B,C,D,E,F,G

    t0 = time.perf_counter()
    mcache = os.path.join(a.out, "stage3_mask_cache.npz")
    reuse = (not a.force_masks) and os.path.exists(mcache)
    if reuse:
        with np.load(mcache) as z:
            dyn_frac_pts = z["dyn_frac_pts"]
            dyn_frac_img = z["dyn_frac_img"]
            instances = z["instances"]
            ms_list = z["ms_list"]
            t_mask = float(z["t_mask"])
        n = min(len(dyn_frac_pts), n)
        print(f"  reused cached masks for {n} frames (saves the network pass)",
              flush=True)
    else:
        for i in range(n):
            img2 = np.array(Image.open(drv.img02[i]).convert("RGB"))
            m2, st2 = masker.mask(img2, name=drv.img02[i])
            hit = masker.mask_points(b["pts"][i], P02, m2, dilate=9)
            dyn_frac_img[i] = st2.mask_fraction
            dyn_frac_pts[i] = float(hit.mean()) if hit.size else 0.0
            instances[i] = st2.instances
            ms_list[i] = st2.ms

            if i < a.save_masks:
                ov = img2.copy()
                ov[m2] = (0.35 * ov[m2] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
                side = np.concatenate([img2, ov], axis=1)
                Image.fromarray(side).save(os.path.join(a.media, f"mask_{i:04d}.png"))
            if i % 20 == 0:
                print(f"  {i}/{n}  inst={st2.instances:2d} "
                      f"img={st2.mask_fraction * 100:5.2f}%  "
                      f"pts={dyn_frac_pts[i] * 100:5.2f}%  {st2.ms:.0f} ms", flush=True)
        t_mask = time.perf_counter() - t0
        np.savez(mcache, dyn_frac_pts=dyn_frac_pts, dyn_frac_img=dyn_frac_img,
                 instances=instances, ms_list=ms_list, t_mask=t_mask)

    # ------------------------------------------------- registration effect --
    print(f"== registration: {a.pairs} pairs, with vs without dynamic ==", flush=True)
    rng = np.random.default_rng(11)
    # prefer pairs that actually contain dynamic points, otherwise the test
    # measures nothing: a pair of empty streets has the same residual either way
    cand = [i for i in range(1, n) if dyn_frac_pts[i] > 0 or dyn_frac_pts[i - 1] > 0]
    if len(cand) < 4:
        cand = list(range(1, n))
    pick = rng.choice(cand, size=min(a.pairs, len(cand)), replace=False)
    rmses = {"all": [], "masked": []}
    inl = {"all": [], "masked": []}
    T_imu_velo = torch.as_tensor(calib["T_imu_velo"], dtype=torch.float32, device=lio.DEV)
    T_velo_imu = torch.as_tensor(calib["T_velo_imu"], dtype=torch.float32, device=lio.DEV)
    for i in sorted(pick):
        j = i - 1
        keep_i = None
        m2 = None
        img_i = np.array(Image.open(drv.img02[i]).convert("RGB"))
        m_i, _ = masker.mask(img_i, name=f"pair{i}")
        hit_i = masker.mask_points(b["pts"][i], P02, m_i, dilate=9)
        img_j = np.array(Image.open(drv.img02[j]).convert("RGB"))
        m_j, _ = masker.mask(img_j, name=f"pair{j}")
        hit_j = masker.mask_points(b["pts"][j], P02, m_j, dilate=9)
        for tag, use_mask in (("all", False), ("masked", True)):
            src_p = b["pts"][i][~hit_i] if use_mask else b["pts"][i]
            tgt_p = b["pts"][j][~hit_j] if use_mask else b["pts"][j]
            src = lio.build_reg_cloud(src_p, max_range=55.0)
            tgt = lio.build_reg_cloud(tgt_p, max_range=55.0)
            _, stt = lio.icp_point_to_plane(src.points, tgt.centers, tgt.normals,
                                            T0=None, iters=30, max_dist=1.2,
                                            max_resid=0.30)
            rmses[tag].append(stt.rmse)
            inl[tag].append(stt.inliers)
    ra = np.asarray(rmses["all"])
    rm = np.asarray(rmses["masked"])
    reg = {
        "pairs": int(len(ra)),
        "rmse_all_median": round(float(np.median(ra)), 4),
        "rmse_masked_median": round(float(np.median(rm)), 4),
        "rmse_improve_pct": round(float(100 * (np.median(ra) - np.median(rm))
                                        / max(np.median(ra), 1e-9)), 2),
        "inliers_all_median": float(np.median(inl["all"])),
        "inliers_masked_median": float(np.median(inl["masked"])),
        "rmse_all": [round(float(x), 4) for x in ra],
        "rmse_masked": [round(float(x), 4) for x in rm],
    }
    print(f"  median rmse {reg['rmse_all_median']:.4f} -> "
          f"{reg['rmse_masked_median']:.4f} m ({reg['rmse_improve_pct']:+.1f}%)",
          flush=True)

    # ------------------------------------------------------------ stereo ----
    print("== stereo vs lidar ==", flush=True)
    stereo = []
    for i in rng.choice(n, size=min(a.stereo_frames, n), replace=False):
        i = int(i)
        il = np.array(Image.open(drv.img02[i]).convert("RGB"))
        ir = np.array(Image.open(drv.img03[i]).convert("RGB"))
        depth, dstat = vision.stereo_depth(il, ir, P02, P03)
        cmp_ = vision.stereo_vs_lidar(b["pts"][i], depth, P02)
        stereo.append({"frame": i, **dstat, **cmp_})
        print(f"  frame {i:3d}: {cmp_.get('n', 0):6d} px, "
              f"median |err| {cmp_.get('median_abs_m')} m, "
              f"p90 {cmp_.get('p90_abs_m')} m, "
              f"{cmp_.get('within_1m_pct')}% within 1 m", flush=True)

    # ------------------------------------------------------- descriptors ----
    # Both variants are evaluated.  The colour histogram is the intuitive choice
    # and it fails here (every straight residential frame looks the same), so it
    # is reported next to the variant that works instead of being dropped -- the
    # negative result is the useful part.
    #
    # The tolerance must exceed the exclusion window.  With exclusion 10 and
    # tolerance 5 every query's correct answers are all excluded, so recall is 0
    # by construction -- which is exactly the bug the first run reported.
    print("== visual descriptors ==", flush=True)
    desc_metrics = {}
    descs = {}
    excl, marg = 10, a.tolerance
    for method in ("tiny", "hsv"):
        D = vision.ImageDescriptor(method=method)
        desc = np.zeros((n, D.dim), np.float32)
        for i in range(n):
            img = np.array(Image.open(drv.img02[i]).convert("RGB"))
            desc[i] = D.describe(img)
        descs[method] = desc
        hits1 = hits5 = 0
        spread = []
        for i in range(n):
            d = 1.0 - desc @ desc[i]
            d[np.abs(np.arange(n) - i) <= excl] = np.inf
            order = np.argsort(d)[:5]
            spread.append(float(d[order[-1]] - d[order[0]]))
            if abs(int(order[0]) - i) <= marg:
                hits1 += 1
            if any(abs(int(k) - i) <= marg for k in order):
                hits5 += 1
        # a near-zero spread means the ranking is decided by numerical noise, so a
        # low recall would say nothing about the descriptor's discriminative power
        sub = desc[::3]
        near = [float((1.0 - sub @ sub[k]).min()) for k in range(0, len(sub), 7)]
        desc_metrics[method] = {
            "dim": int(desc.shape[1]),
            "recall_at_1": round(hits1 / n, 4),
            "recall_at_5": round(hits5 / n, 4),
            "median_5th_minus_1st_distance": round(float(np.median(spread)), 5),
            "median_nearest_distance": round(float(np.median(near)), 5),
        }
        print(f"  {method:4s} dim {desc.shape[1]:4d}: recall@1 "
              f"{desc_metrics[method]['recall_at_1']:.3f}  recall@5 "
              f"{desc_metrics[method]['recall_at_5']:.3f}  "
              f"5th-1st spread {desc_metrics[method]['median_5th_minus_1st_distance']:.5f}",
              flush=True)

    # ------------------------------------------------------ semantic cost ----
    print("== semantic cost layer ==", flush=True)
    sem = []
    for i in rng.choice(n, size=min(4, n), replace=False):
        i = int(i)
        g = levels.ground_height(b["pts"][i])
        img = np.array(Image.open(drv.img02[i]).convert("RGB"))
        m, _ = masker.mask(img, name=f"sem{i}")
        hit = masker.mask_points(b["pts"][i], P02, m, dilate=9)
        _, s_all = vision.semantic_cost(b["pts"][i], g, None)
        _, s_dyn = vision.semantic_cost(b["pts"][i], g, hit)
        sem.append({"frame": i, "ground_z": round(g, 3), "without_dynamic": s_all,
                    "with_dynamic": s_dyn})
        print(f"  frame {i:3d}: free {s_dyn['free_pct']:5.2f}%  static "
              f"{s_dyn['static_obstacle_pct']:5.2f}%  dynamic "
              f"{s_dyn['dynamic_obstacle_pct']:5.2f}%  ({s_dyn['ms']:.0f} ms)",
              flush=True)

    np.save(os.path.join(a.out, "dyn_frac_pts.npy"), dyn_frac_pts)
    np.save(os.path.join(a.out, "desc_tiny.npy"), descs["tiny"])
    np.save(os.path.join(a.out, "dyn_desc.npy"), descs["tiny"])

    payload = {
        "frames": int(n),
        "dynamic_mask": {
            "model": "torchvision maskrcnn_resnet50_fpn_v2 (COCO), score>=0.55",
            "movable_classes": list(vision.MOVABLE_CLASSES.values()),
            "frames_with_instance": int((instances > 0).sum()),
            "instance_detection_rate": round(float((instances > 0).mean()), 4),
            "instances_total": int(instances.sum()),
            "instances_per_frame_mean": round(float(instances.mean()), 3),
            "mask_pixel_frac_mean": round(float(dyn_frac_img.mean()), 5),
            "lidar_point_frac_mean": round(float(dyn_frac_pts.mean()), 5),
            "lidar_point_frac_p95": round(float(np.percentile(dyn_frac_pts, 95)), 5),
            "ms_per_frame_mean": round(float(ms_list.mean()), 1),
            "ms_total": round(t_mask, 1),
            "registration": reg,
        },
        "stereo_vs_lidar": stereo,
        "descriptor": desc_metrics,
        "descriptor_protocol": {"exclusion_frames": excl,
                               "tolerance_frames": marg,
                               "note": "leave-one-out with a temporal exclusion "
                                       "window: without it every query return. "
                                       "itself and recall is 100 % by construction"},
        "semantic_cost": sem,
    }
    with open(os.path.join(a.out, "stage3_vision.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {a.out}/stage3_vision.json")


if __name__ == "__main__":
    main()
