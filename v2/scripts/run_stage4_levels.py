#!/usr/bin/env python3
"""Stage 4: floor segmentation and per-level loading.

The real KITTI sequence is a single storey, so on this data the honest answer is
"one level" -- and reporting that is part of the result: the segmenter must not
invent a floor out of a 1.3 m kerb.

To show that the mechanism works, a **synthetic duplex** is built from the same
map: every keyframe is duplicated onto a floor 3.6 m higher and rotated 180
degrees.  That is a deliberate construction, labelled as such everywhere it
appears, and it is the only way to exercise floor splitting without a real
multi-storey dataset.  The measurement that matters is not "2 levels were found
in fake data", it is **how much of the map a level-aware loader avoids touching**
when the robot is on one floor.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import bundle as B  # noqa: E402
from navstack import levels, schema as S, store as ST  # noqa: E402

FLOOR_GAP_M = 3.6


def build_duplex(src_map: str, dst_map: str, gap: float = FLOOR_GAP_M) -> int:
    """Copy a single-storey map onto a second, rotated floor above it."""
    if os.path.isdir(dst_map):
        shutil.rmtree(dst_map)
    src = ST.MapStore(src_map).open()
    dst = ST.MapStore(dst_map)
    dst.create(map_id="kitti_duplex_synthetic", window=12,
               source="synthetic 2-floor map: drive 0005 duplicated at +%.1f m" % gap)
    # rotate the upper floor 180 deg about the map centre so the two floors are
    # not copy-paste identical: an identical copy would make the level classifier
    # look better than it is, because any leakage between floors would be
    # invisible
    centre = np.mean([kf.pose[:3, 3] for kf in src.keyframes], axis=0)
    Rz = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    kid = 0
    for kf in src.keyframes:
        with np.load(os.path.join(src.dir_kf, f"{kf.id:06d}.npz")) as z:
            pts, rgb = z["pts"], z["rgb"]
        kf0 = S.Keyframe(id=kid, t=kf.t, pose=kf.pose.copy(), n_points=kf.n_points,
                         level=0, chunks=[], seq=kf.seq)
        dst.add_keyframe(kf0, pts, rgb)
        kid += 1
    for kf in src.keyframes:
        with np.load(os.path.join(src.dir_kf, f"{kf.id:06d}.npz")) as z:
            pts, rgb = z["pts"], z["rgb"]
        T = np.eye(4)
        T[:3, :3] = Rz
        p = kf.pose[:3, 3] - centre
        p = Rz @ p + centre
        p[2] += gap
        T[:3, 3] = p
        pts_up = (pts - kf.pose[:3, 3][None, :]) @ Rz.T + p[None, :]
        kf1 = S.Keyframe(id=kid, t=kf.t, pose=T, n_points=kf.n_points,
                         level=1, chunks=[], seq=kf.seq + 10000)
        dst.add_keyframe(kf1, pts_up.astype(np.float32), rgb)
        kid += 1
    dst.commit()
    return kid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/nvidia/rtk_fusion_demo/navstack_v2/data")
    ap.add_argument("--out", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out")
    ap.add_argument("--map", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--duplex", default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/duplex.navmap")
    ap.add_argument("--gap", type=float, default=FLOOR_GAP_M)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    b = B.load_bundle(a.data)
    result = {"floor_gap_m_used_for_synthetic": a.gap}

    # ------------------------------------------------------------- real map --
    print("== real map: how many floors? ==", flush=True)
    st = ST.MapStore(a.map).open()
    lv = levels.estimate_levels(st)
    baked = levels.bake(st, lod="mid")
    grid, mm = levels.minimap(st, res_m=1.0)
    st.commit()
    result["real"] = {
        "keyframes": len(st.keyframes),
        "levels_detected": len(lv),
        "level_detail": [{k: v for k, v in d.items() if k not in ("pts", "rgb")}
                         for d in baked.values()],
        "minimap": mm,
        "conclusion": ("single storey: the segmenter correctly refuses to split a "
                       "continuous ground profile" if len(lv) == 1 else
                       "unexpected split on single-storey data"),
    }
    print(f"  levels={len(lv)}, minimap {mm['shape']} "
          f"({mm['coverage_m2']} m2 observed)", flush=True)

    # ------------------------------------------------------ synthetic duplex --
    print("== synthetic duplex ==", flush=True)
    if a.rebuild or not os.path.isdir(a.duplex):
        n = build_duplex(a.map, a.duplex, a.gap)
        print(f"  built duplex map with {n} keyframes", flush=True)
    st2 = ST.MapStore(a.duplex).open()
    lv2 = levels.estimate_levels(st2)
    baked2 = levels.bake(st2, lod="mid")
    st2.commit()

    # level-aware loading vs loading everything
    loads = {}
    for tag, lvl in (("L0", 0), ("L1", 1), ("all", None)):
        pts, _, info = st2.load_all(lod="mid", level=lvl)
        loads[tag] = {"points": int(len(pts)), "ms": info["ms"],
                      "keyframes": info["keyframes"]}
        print(f"  load {tag:3s}: {info['points']:7d} pts, {info['ms']:6.1f} ms, "
              f"{info['keyframes']:3d} keyframes", flush=True)
    correct = 0
    for kf in st2.keyframes:
        want = kf.seq // 10000          # lower floor = 0, upper = 1 by construction
        correct += int(kf.level == want)
    result["synthetic_duplex"] = {
        "keyframes": len(st2.keyframes),
        "levels_detected": len(lv2),
        "level_detail": [{k: v for k, v in d.items() if k not in ("pts", "rgb")}
                         for d in baked2.values()],
        "assignment_accuracy": round(correct / max(len(st2.keyframes), 1), 4),
        "loads": loads,
        "level_load_saving_vs_all_pct": round(
            100 * (1 - max(loads["L0"]["points"], loads["L1"]["points"])
                   / max(loads["all"]["points"], 1)), 2),
        "note": "both floors are correct-by-construction; accuracy is a sanity "
                "check on the ground-height clustering, not an ML score",
    }
    print(f"  levels={len(lv2)}, assignment accuracy "
          f"{result['synthetic_duplex']['assignment_accuracy']:.3f}, "
          f"level load saves "
          f"{result['synthetic_duplex']['level_load_saving_vs_all_pct']:.1f}% of points",
          flush=True)

    # ------------------------------------------------------- global minimap --
    print("== global minimap ==", flush=True)
    for res in (1.0, 0.5):
        g, m = levels.minimap(st, res_m=res)
        print(f"  {res} m: {m['shape']} cells, {m['occupied_cells']} occupied, "
              f"{m['coverage_m2']} m2 observed", flush=True)
        result.setdefault("minimap_resolutions", {})[str(res)] = m

    with open(os.path.join(a.out, "stage4_levels.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {a.out}/stage4_levels.json")


if __name__ == "__main__":
    main()
