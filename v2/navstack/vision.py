"""Vision layer: what actually earns its place next to LiDAR + RTK + Nav2.

The question this module answers, with measurements rather than opinion, is
"which vision model should the stack add?".  Three candidates are implemented and
each one is scored on what it improves, because the honest ranking is not the
intuitive one:

1. **Dynamic-object masking** (``DynamicMasker``) -- instance masks for the
   movable classes of COCO, projected onto the LiDAR returns so those points can
   be excluded from the registration and the map.  This is the only one that
   improves the *geometry*, and it improves it in both directions: fewer moving
   returns corrupt the pose estimate, and the map stops containing ghost cars.

2. **Semantic cost layer** (``semantic_cost``) -- per-cell labels (free /
   obstacle / dynamic / unknown) for the planner.  Cheap, useful for Nav2 costmap
   plugins, but it changes *where* the robot may go, not *where it thinks it is*.

3. **Visual place descriptor** (``ImageDescriptor``) -- a compact global image
   descriptor used as a second modality in the relocalisation vote.  It helps
   exactly where Scan Context is weakest (long, self-similar corridors) and is
   useless where there is no texture.

Deliberately **not** implemented: monocular depth completion and online VLM
localisation.  ``stereo_vs_lidar`` is included to show why the first is
unnecessary -- this platform already carries a calibrated stereo pair, whose
depth agrees with LiDAR to within the LiDAR's own beam spacing, so a learned
monocular completion model would be inventing information that is already
measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

# COCO category ids that can move on their own.  The set is deliberately narrow:
# a traffic cone is dynamic in a story, static for a 15 s map, and masking it
# would only remove useful geometry.
MOVABLE_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    6: "bus", 8: "truck",
}


# --------------------------------------------------------------------------- #
# 1. dynamic masks
# --------------------------------------------------------------------------- #
@dataclass
class MaskStats:
    name: str
    instances: int
    classes: List[str]
    mask_fraction: float            # of image pixels
    ms: float


class DynamicMasker:
    """Mask R-CNN instance masks, cached per model instance."""

    def __init__(self, device: Optional[str] = None, score_thr: float = 0.55,
                 min_area_frac: float = 3e-4, half: bool = False):
        if torch is None:
            raise RuntimeError("torch is required for DynamicMasker")
        import torchvision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (torchvision.models.detection
                      .maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
                      .to(self.device))
        if half and self.device == "cuda":
            self.model.half()
        self.half = bool(half and self.device == "cuda")
        self.score_thr = score_thr
        self.min_area_frac = min_area_frac
        self.classes = MOVABLE_CLASSES
        self.log: List[MaskStats] = []

    @torch.no_grad() if torch is not None else (lambda f: f)
    def mask(self, image_rgb: np.ndarray, name: str = ""
             ) -> Tuple[np.ndarray, MaskStats]:
        """Boolean (H,W) mask of dynamic pixels, plus its statistics."""
        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]
        x = torch.as_tensor(image_rgb, device=self.device).permute(2, 0, 1)
        x = x.float().div(255.0)
        if self.half:
            x = x.half()
        out = self.model([x])[0]
        keep = np.zeros((h, w), bool)
        names: List[str] = []
        n_inst = 0
        for lab, sc, mk in zip(out["labels"].tolist(), out["scores"].tolist(),
                               out["masks"]):
            if sc < self.score_thr or lab not in self.classes:
                continue
            m = (mk[0] > 0.5).to(torch.uint8).cpu().numpy().astype(bool)
            if m.sum() < self.min_area_frac * h * w:
                continue
            keep |= m
            n_inst += 1
            names.append(f"{self.classes[lab]}:{sc:.2f}")
        st = MaskStats(name=name, instances=n_inst, classes=names,
                       mask_fraction=round(float(keep.mean()), 5),
                       ms=round((time.perf_counter() - t0) * 1e3, 1))
        self.log.append(st)
        return keep, st

    # ------------------------------------------------------------------ use
    def mask_points(self, pts: np.ndarray, P: np.ndarray, mask: np.ndarray,
                    dilate: int = 9) -> np.ndarray:
        """Boolean per LiDAR point: does it land inside the dynamic mask?

        The mask is dilated by a few pixels because a car's LiDAR returns cluster
        on its body panels, and the network's mask boundary sits inside the
        silhouette -- an un-dilated projection leaves a rim of the very returns
        that cause the ghost trail.
        """
        if len(pts) == 0:
            return np.zeros(0, bool)
        h, w = mask.shape
        m = mask
        if dilate > 1 and cv2 is not None:
            k = np.ones((dilate, dilate), np.uint8)
            m = cv2.dilate(mask.astype(np.uint8), k, iterations=1).astype(bool)
        hg = np.concatenate([pts, np.ones((len(pts), 1))], 1)
        uv = (P @ hg.T).T
        z = uv[:, 2]
        ok = z > 1e-3
        u = np.zeros(len(pts), int)
        v = np.zeros(len(pts), int)
        u[ok] = np.round(uv[ok, 0] / z[ok]).astype(int)
        v[ok] = np.round(uv[ok, 1] / z[ok]).astype(int)
        inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        hit = np.zeros(len(pts), bool)
        hit[inside] = m[v[inside], u[inside]]
        return hit


# --------------------------------------------------------------------------- #
# 2. stereo depth vs LiDAR -- the "do we need monocular depth?" measurement
# --------------------------------------------------------------------------- #
@dataclass
class StereoCfg:
    num_disp: int = 128
    block: int = 5
    min_disp: int = 1
    max_depth_m: float = 60.0


def stereo_depth(img_l: np.ndarray, img_r: np.ndarray, P_l: np.ndarray,
                 P_r: np.ndarray,
                 cfg: Optional[StereoCfg] = None
                 ) -> Tuple[np.ndarray, Dict[str, float]]:
    """SGBM disparity -> metric depth.  Returns depth (H,W) float32 and stats.

    The baseline must come from the **difference** of the two projection
    matrices, not from ``P_l[0,3]``.  In KITTI, ``P_rect_02[0,3] = 44.86`` is
    camera 02's offset from the cam0 reference (-f * 0.0621 m), while the 02-03
    stereo baseline of 0.537 m is encoded in ``P_r[0,3] - P_l[0,3] = -f * 0.537``.
    Using ``P_l[0,3]/f`` gives a 0.062 m baseline and therefore depths 8.6x too
    large -- which is exactly the 9.6 m median error the first version reported.
    """
    cfg = cfg or StereoCfg()
    if cv2 is None:
        return np.zeros(img_l.shape[:2], np.float32), {}
    t0 = time.perf_counter()
    gl = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=cfg.min_disp, numDisparities=cfg.num_disp,
        blockSize=cfg.block, P1=8 * cfg.block ** 2, P2=32 * cfg.block ** 2,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    f = float(P_l[0, 0])
    # disparity = x_left - x_right = f * B / Z, and the translation terms are
    # -f * B for the pair, so B = |P_r[0,3] - P_l[0,3]| / f
    b = abs(float(P_r[0, 3]) - float(P_l[0, 3])) / f
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disp > cfg.min_disp, f * b / np.maximum(disp, 1e-3), 0.0)
    valid = (depth > 0.5) & (depth < cfg.max_depth_m)
    return depth.astype(np.float32), {
        "baseline_m": round(b, 4), "focal_px": round(f, 1),
        "valid_frac": round(float(valid.mean()), 4),
        "ms": round((time.perf_counter() - t0) * 1e3, 1),
    }


def stereo_vs_lidar(pts: np.ndarray, depth: np.ndarray, P: np.ndarray,
                    max_depth: float = 50.0) -> Dict[str, float]:
    """Median |stereo - lidar| over pixels where both have a value."""
    h, w = depth.shape
    hg = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    uv = (P @ hg.T).T
    z = uv[:, 2]
    ok = (z > 1.0) & (z < max_depth)
    if not ok.any():
        return {"n": 0}
    u = np.round(uv[ok, 0] / z[ok]).astype(int)
    v = np.round(uv[ok, 1] / z[ok]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[ok][inside]
    d = depth[v, u]
    good = d > 0.5
    if good.sum() < 30:
        return {"n": int(good.sum())}
    err = np.abs(d[good] - z[good])
    rel = err / z[good]
    return {
        "n": int(good.sum()),
        "median_abs_m": round(float(np.median(err)), 3),
        "p90_abs_m": round(float(np.percentile(err, 90)), 3),
        "median_rel_pct": round(float(np.median(rel)) * 100, 2),
        "within_1m_pct": round(float((err < 1.0).mean()) * 100, 2),
    }


# --------------------------------------------------------------------------- #
# 3. semantic cost layer
# --------------------------------------------------------------------------- #
FREE, STATIC_OBS, DYN_OBS, UNKNOWN_C = 0, 1, 2, 3


def semantic_cost(pts: np.ndarray, ground_z: float, dynamic: Optional[np.ndarray] = None,
                  res: float = 0.25, h_lo: float = 0.15, h_hi: float = 2.0,
                  min_hits: int = 2, report_radius: float = 40.0
                  ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Per-cell semantic label grid.  Returns the grid and its statistics.

    ``ground_z`` is the local ground datum from ``levels.ground_height``; heights
    are measured relative to it, which is what makes the same thresholds work on a
    sloped street and on a flat floor.

    Percentages are reported **inside a disc of ``report_radius``**, not over the
    grid's bounding box.  A single 360 deg scan spans ~160 m across, so a
    bbox-relative share spends 95 % of its denominator on cells the sensor could
    not see, and the number stops meaning anything: the first version reported
    "2.1 % free" for a street that is nearly all drivable.
    """
    t0 = time.perf_counter()
    if len(pts) == 0:
        return np.zeros((1, 1), np.int8), {}
    x0, y0 = pts[:, 0].min() - res, pts[:, 1].min() - res
    ij = np.floor((pts[:, :2] - np.array([x0, y0])) / res).astype(int)
    w = int(ij[:, 0].max()) + 3
    h = int(ij[:, 1].max()) + 3
    dz = pts[:, 2] - ground_z
    is_obs = (dz >= h_lo) & (dz <= h_hi)
    is_dyn = np.zeros(len(pts), bool) if dynamic is None else dynamic

    def acc(mask):
        g = np.zeros((w, h), np.int32)
        if mask.any():
            np.add.at(g, (ij[mask, 0], ij[mask, 1]), 1)
        return g

    occ = acc(is_obs & ~is_dyn)
    dyn = acc(is_obs & is_dyn)
    ground = acc(~is_obs)
    grid = np.full((w, h), UNKNOWN_C, np.int8)
    grid[ground > 0] = FREE
    grid[occ >= min_hits] = STATIC_OBS
    grid[dyn >= min_hits] = DYN_OBS

    # report inside the disc centred on the sensor (origin of a body-frame scan)
    cx = np.arange(w)[:, None] * res + x0 + res / 2
    cy = np.arange(h)[None, :] * res + y0 + res / 2
    disc = (cx ** 2 + cy ** 2) <= report_radius ** 2
    tot = float(disc.sum())
    stats = {
        "res_m": res, "shape": [w, h],
        "report_radius_m": report_radius,
        "cells_in_disc": int(tot),
        "free_pct": round(float(((grid == FREE) & disc).sum()) / tot * 100, 2),
        "static_obstacle_pct": round(float(((grid == STATIC_OBS) & disc).sum()) / tot * 100, 2),
        "dynamic_obstacle_pct": round(float(((grid == DYN_OBS) & disc).sum()) / tot * 100, 2),
        "unknown_pct": round(float(((grid == UNKNOWN_C) & disc).sum()) / tot * 100, 2),
        "observed_pct": round(100.0 - float(((grid == UNKNOWN_C) & disc).sum()) / tot * 100, 2),
        "z_band_m": [h_lo, h_hi],
        "ms": round((time.perf_counter() - t0) * 1e3, 1),
    }
    return grid, stats


# --------------------------------------------------------------------------- #
# 4. visual place descriptor
# --------------------------------------------------------------------------- #
@dataclass
class ImageDescriptor:
    """Compact global image descriptor, two variants, deliberately score-based.

    ``hsv``
        Joint hue/saturation/value histogram plus an orientation histogram of the
        upper half, 144 numbers.
    ``tiny``
        Grayscale image resampled to 32x16, mean removed, L2 normalised.  The
        classic "tiny image" baseline: 512 numbers that encode layout.

    Measured on drive 0005 (leave-one-out, 10-frame temporal exclusion, 15-frame
    tolerance): **hsv recall@1 0.948, tiny 0.747**.  That reverses the usual
    intuition that colour histograms are too weak for place recognition, and it
    is worth stating because the first version of this module asserted the
    opposite.  The reason it holds here is that a residential street's *layout*
    repeats every few tens of metres -- driveways, parked cars, kerbs -- while its
    colour balance varies with what happens to be in the upper half of the frame.
    The layout descriptor is therefore the one that aliases, not the colour one.

    Neither is a stand-in for a learned descriptor; they are used only as a
    tie-breaking *second modality* on top of LiDAR retrieval, never alone.
    """

    size: int = 64
    method: str = "tiny"
    hsv_bins: Tuple[int, int, int] = (8, 4, 4)
    grad_bins: int = 16

    @property
    def dim(self) -> int:
        if self.method == "tiny":
            return 32 * 16
        return self.hsv_bins[0] * self.hsv_bins[1] * self.hsv_bins[2] + self.grad_bins

    def describe(self, img_rgb: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return np.zeros(1, np.float32)
        return self._tiny(img_rgb) if self.method == "tiny" else self._hsv(img_rgb)

    def _tiny(self, img_rgb: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        t = cv2.resize(g, (32, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        t -= t.mean()
        n = np.linalg.norm(t)
        return (t / n if n > 1e-6 else t).ravel()

    def _hsv(self, img_rgb: np.ndarray) -> np.ndarray:
        im = cv2.resize(img_rgb, (self.size * 2, self.size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
        hb = cv2.calcHist([hsv], [0, 1, 2], None, list(self.hsv_bins),
                          [0, 180, 0, 256, 0, 256]).ravel()
        parts = [hb / max(hb.sum(), 1e-6)]
        top = im[: self.size // 2]
        gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
        hist, _ = np.histogram(ang, bins=self.grad_bins, range=(0, 1), weights=mag)
        parts.append(hist / max(hist.sum(), 1e-6))
        v = np.concatenate(parts).astype(np.float32)
        return v / max(np.linalg.norm(v), 1e-9)

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))
