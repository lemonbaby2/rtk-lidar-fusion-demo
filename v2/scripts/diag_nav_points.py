#!/usr/bin/env python3
"""What is putting returns inside the cells the vehicle drove through?

The ground hypothesis is dead: only 1.1 % of the old band sits below 0.2 m above
the measured ground, and anchoring the band to local ground made occupancy
*finer*, not coarser.  Yet 74 % of ego cells carry >= 3 returns at 0.25 m, so
something is genuinely returning points within a quarter metre of the sensor's
own track.  This probe stops speculating and lists them:

* for the start cell and its 8 neighbours: how many returns, from which keyframe,
  at what range from that keyframe's origin, at what height above ground;
* the global picture: distribution of (range from the recording origin) for all
  returns that fall within 0.4 m of any ego position.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navstack import nav, store as ST  # noqa: E402


def kf_clouds(st: ST.MapStore):
    out = []
    for kf in st.keyframes:
        with np.load(os.path.join(st.dir_kf, f"{kf.id:06d}.npz")) as z:
            out.append((kf.pose, z["pts"].astype(np.float32)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map",
                    default="/home/nvidia/rtk_fusion_demo/navstack_v2/out/drive0005.navmap")
    ap.add_argument("--res", type=float, default=0.25)
    a = ap.parse_args()

    st = ST.MapStore(a.map).open()
    kfs = kf_clouds(st)
    poses = np.stack([kf.pose[:3, 3] for kf in st.keyframes])
    P = np.concatenate([p for _, p in kfs])
    G = nav.build_ground_field(P)
    cfg = nav.LayerCfg(z_min=0.20, z_max=2.20)
    band = np.isfinite(G.dz(P)) & (G.dz(P) >= cfg.z_min) & (G.dz(P) <= cfg.z_max)

    # grid origin the same way build_layer_from_banded does it
    org = np.minimum(P[:, :2].min(0), poses[:, :2].min(0)) - 5 * a.res
    def cell_of(xy):
        return np.floor((np.atleast_2d(xy)[:, :2] - org) / a.res).astype(np.int64)

    ego = cell_of(poses)
    print(f"== {len(P)} points, {band.sum()} in band "
          f"({100*band.mean():.1f} %) ==", flush=True)

    # ---- who is in the start cell and its neighbours? ---------------------- #
    target = ego[0]
    ij = cell_of(P)
    print(f"\n== cells around the start ego cell {tuple(target)} "
          f"(0.25 m) ==", flush=True)
    for di, dj in [(0, 0)] + nav._NB:
        c = (target[0] + di, target[1] + dj)
        sel = (ij[:, 0] == c[0]) & (ij[:, 1] == c[1]) & band
        k = int(sel.sum())
        if k == 0:
            print(f"  cell {c} offset ({di:+d},{dj:+d}): 0 in-band returns")
            continue
        p = P[sel]
        d = np.linalg.norm(p[:, :2] - poses[0][None, :2], axis=1)
        dz = G.dz(p)
        # which keyframe does each return belong to?
        acc, off = [], 0
        for fi, (_, q) in enumerate(kfs):
            acc.append(np.full(len(q), fi))
        fidx = np.concatenate(acc)[sel]
        print(f"  cell {c} offset ({di:+d},{dj:+d}): {k:5d} in-band returns | "
              f"range from ego {d.min():.2f}-{d.max():.2f} m (med {np.median(d):.2f}) | "
              f"dz {dz.min():+.2f}..{dz.max():+.2f} m | "
              f"z {p[:,2].min():+.2f}..{p[:,2].max():+.2f} | "
              f"frames {len(np.unique(fidx))}")

    # ---- what is within 0.4 m of the track? -------------------------------- #
    print("\n== returns within 0.4 m of any ego position ==", flush=True)
    near = np.zeros(len(P), bool)
    track = poses[:, :2]
    for k in range(0, len(P), 200000):
        p = P[k:k + 200000, :2]
        d = np.linalg.norm(p[:, None, :] - track[None, ::4, :], axis=2).min(1)
        near[k:k + 200000] = d < 0.4
    print(f"  {near.sum()} points ({100*near.mean():.3f} % of all, "
          f"{100*near.sum()/max(band.sum(),1):.2f} % of the band)")
    if near.any():
        d_all = []
        for k in range(0, len(P), 200000):
            p = P[k:k + 200000, :2]
            d_all.append(np.linalg.norm(p[:, None, :] - track[None, ::4, :],
                                        axis=2).min(1))
        dmin = np.concatenate(d_all)[near]
        dz = G.dz(P)[near]
        hist = np.histogram(dmin, bins=[0, .05, .1, .2, .3, .4, .6, 1, 2, 5, 1e9])[0]
        edges = [0, .05, .1, .2, .3, .4, .6, 1, 2, 5, 1e9]
        print("  distance to the track:")
        for k in range(len(hist)):
            print(f"    [{edges[k]:>6.2f},{edges[k+1]:>8.1f}) : {hist[k]:8d}")
        print(f"  dz of those: median {np.median(dz):+.2f} m, "
              f"p5 {np.percentile(dz,5):+.2f}, p95 {np.percentile(dz,95):+.2f}")
        print(f"  frac with dz in band: "
              f"{100*np.mean((dz>=0.20)&(dz<=2.20)):.1f} %")

    # ---- compare: how many in-band returns per cell on vs off the track? --- #
    hits = np.zeros((int(ij[:, 0].max()) + 2, int(ij[:, 1].max()) + 2), np.int32)
    np.add.at(hits, (ij[band, 0], ij[band, 1]), 1)
    on = hits[ego[:, 0], ego[:, 1]]
    print(f"\n== hit count on ego cells ==")
    print(f"  mean {on.mean():.1f} median {np.median(on):.0f} "
          f"max {on.max()} -> a 0.25 m cell at the ego position receives "
          f"{np.median(on):.0f} returns" , flush=True)
    print(f"  global median over occupied cells: "
          f"{np.median(hits[hits>=3]):.0f}")


if __name__ == "__main__":
    main()
