"""KITTI raw reader.

Every convention the rest of the stack depends on is fixed here, once, so no
downstream module ever re-derives it.

Conventions
-----------
``frame0``
    The IMU body frame at the first keyframe: x forward, y left, z up.  Every
    pose this stack emits is ``T_frame0_body``.

``frame0`` origin
    The first RTK/INS sample, and the heading is defined by the direction the
    vehicle travels over the first ``head_n`` samples.  The rotation from the
    Mercator world into ``frame0`` is therefore computed *before* any error is
    measured.  The v1 demo computed ATE in the Mercator world and then rotated
    the odometry chain afterwards with the first INS attitude; that worked but
    it hid a 90 degree frame mismatch behind a post-hoc alignment, and one
    intermediate run reported 31.76 m of error for a chain that was in fact
    fine.  Doing the rotation at ingest removes that whole failure mode.

Gravity
    ``(0, 0, -9.80665)`` expressed in ``frame0``.  The OXTS IMU reports
    specific force, so a stationary sample reads about ``+9.81`` on z, which
    means the pre-integration uses ``MakeSharedU`` and *not* ``MakeSharedD``.

RTK ground truth
    ``pose_rtk.npy`` keeps the v1 layout (Mercator positions, Euler attitude,
    post-multiplied by ``Tr_imu_to_velo``) so the v1 results stay comparable.
    ``rtk_in_frame0()`` rotates just the translations into ``frame0``.
"""

from __future__ import annotations

import datetime as _dt
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

R_EARTH = 6378137.0
GRAVITY = 9.80665

# OXTS column indices (KITTI raw dataformat.txt, 30 columns).
OXT_LAT, OXT_LON, OXT_ALT = 0, 1, 2
OXT_ROLL, OXT_PITCH, OXT_YAW = 3, 4, 5
OXT_ACC = slice(11, 14)      # ax, ay, az  [m/s^2]  specific force
OXT_GYRO = slice(17, 20)     # wx, wy, wz  [rad/s]
OXT_POS_ACC = 23


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _calib_vector(path: str, key: str, shape) -> np.ndarray:
    with open(path) as f:
        for line in f:
            if line.startswith(key + ":"):
                vals = np.array([float(x) for x in line.split(":", 1)[1].split()])
                return vals.reshape(shape) if shape else vals
    raise KeyError(f"{key} not in {path}")


def load_calib(base: str) -> Dict[str, np.ndarray]:
    """Return every transform the stack needs, as 4x4 homogeneous matrices."""
    v2c = np.eye(4)
    v2c[:3, :3] = _calib_vector(os.path.join(base, "calib_velo_to_cam.txt"), "R", (3, 3))
    v2c[:3, 3] = _calib_vector(os.path.join(base, "calib_velo_to_cam.txt"), "T", (3,))

    i2v = np.eye(4)
    i2v[:3, :3] = _calib_vector(os.path.join(base, "calib_imu_to_velo.txt"), "R", (3, 3))
    i2v[:3, 3] = _calib_vector(os.path.join(base, "calib_imu_to_velo.txt"), "T", (3,))

    rect = np.eye(4)
    rect[:3, :3] = _calib_vector(os.path.join(base, "calib_cam_to_cam.txt"), "R_rect_00", (3, 3))
    p2 = _calib_vector(os.path.join(base, "calib_cam_to_cam.txt"), "P_rect_02", (3, 4))
    p3 = _calib_vector(os.path.join(base, "calib_cam_to_cam.txt"), "P_rect_03", (3, 4))
    # baseline between the two colour cameras, used by the vision module to
    # sanity-check that both images really are the same scene
    baseline = abs(float(
        _calib_vector(os.path.join(base, "calib_cam_to_cam.txt"), "P_rect_03", (3, 4))[0, 3]
        - p2[0, 3]
    )) / float(p2[0, 0])

    return {
        "T_cam_velo": v2c,                       # velodyne -> camera 0 (unrectified)
        "T_velo_cam": np.linalg.inv(v2c),
        "T_velo_imu": i2v,                       # imu -> velodyne
        "T_imu_velo": np.linalg.inv(i2v),
        "R_rect": rect,
        "P_rect_02": p2 @ rect @ v2c,            # full projection chain velo -> image 02
        "P_rect_03": p3 @ rect @ v2c,
        "stereo_baseline_m": baseline,
    }


# --------------------------------------------------------------------------- #
# rotations
# --------------------------------------------------------------------------- #
def euler_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def Rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# OXTS / IMU
# --------------------------------------------------------------------------- #
def _parse_stamps(path: str) -> np.ndarray:
    """KITTI timestamps are nanosecond precision; datetime only does microseconds."""
    out: List[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            head, _, frac = line.partition(".")
            frac = (frac + "000000")[:6]
            t = _dt.datetime.strptime(f"{head}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
            out.append(t.replace(tzinfo=_dt.timezone.utc).timestamp())
    return np.asarray(out, dtype=np.float64)


@dataclass
class OxtsStream:
    raw: np.ndarray            # (M, 30)
    t: np.ndarray              # (M,) unix seconds
    hz: float

    def __len__(self) -> int:
        return int(self.raw.shape[0])


def find_oxts(drive_sync: str, drive_extract: str | None = None) -> OxtsStream:
    """Prefer the 100 Hz ``_extract`` stream, fall back to the 10 Hz ``_sync`` one.

    The sync archive ships OXTS decimated to camera rate (10 Hz).  Pre-integrating
    a single measurement per 0.11 s step still works but throws away exactly the
    motion the accelerometer is there to capture, so the 100 Hz stream is used
    whenever it has been unpacked.
    """
    for cand in (drive_extract, drive_sync):
        if not cand:
            continue
        d = os.path.join(cand, "oxts")
        if not os.path.isdir(os.path.join(d, "data")):
            continue
        files = sorted(glob.glob(os.path.join(d, "data", "*.txt")))
        if not files:
            continue
        # np.fromstring(..., sep=) was removed in numpy 2.x
        raw = np.array([[float(x) for x in open(f).read().split()] for f in files],
                       dtype=np.float64)
        t = _parse_stamps(os.path.join(d, "timestamps.txt"))
        if len(t) != len(raw):
            n = min(len(t), len(raw))
            t, raw = t[:n], raw[:n]
        dt = np.median(np.diff(t)) if len(t) > 1 else 0.11
        return OxtsStream(raw=raw, t=t, hz=float(1.0 / dt))
    raise FileNotFoundError(f"no oxts/data under {drive_sync} or {drive_extract}")


# --------------------------------------------------------------------------- #
# RTK / INS poses
# --------------------------------------------------------------------------- #
def mercator(raw: np.ndarray, scale_lat0: float | None = None) -> np.ndarray:
    """KITTI devkit Mercator projection: x east, y north, z altitude."""
    lat0 = float(raw[0, OXT_LAT]) if scale_lat0 is None else scale_lat0
    scale = np.cos(lat0 * np.pi / 180.0)
    tx = scale * raw[:, OXT_LON] * np.pi * R_EARTH / 180.0
    ty = scale * R_EARTH * np.log(np.tan((90.0 + raw[:, OXT_LAT]) * np.pi / 360.0))
    return np.stack([tx, ty, raw[:, OXT_ALT]], axis=1)


def rtk_poses(raw: np.ndarray, T_imu2velo: np.ndarray) -> np.ndarray:
    """v1-compatible: velodyne pose in the Mercator world, (N, 4, 4)."""
    xyz = mercator(raw)
    xyz = xyz - xyz[0]
    out = np.zeros((len(raw), 4, 4))
    for i in range(len(raw)):
        T = np.eye(4)
        T[:3, :3] = euler_to_R(raw[i, OXT_ROLL], raw[i, OXT_PITCH], raw[i, OXT_YAW])
        T[:3, 3] = xyz[i]
        out[i] = T @ T_imu2velo
    return out


def rtk_in_frame0(raw: np.ndarray, head_n: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """RTK translations rotated into ``frame0``.  Returns ``(pos (N,3), R_w2f0 (3,3))``.

    ``frame0`` x points the way the vehicle actually drives over the first
    ``head_n`` samples, so the trajectory starts along +x with zero yaw.  A
    straight-line head is a poor heading estimate when the vehicle is stopped at
    the start, so the samples are taken at the far end of a short window and the
    window is grown until the baseline is at least 0.5 m.
    """
    xyz = mercator(raw)
    xyz = xyz - xyz[0]
    n = max(2, min(head_n, len(xyz)))
    while n < len(xyz):
        d = xyz[n] - xyz[0]
        if np.linalg.norm(d[:2]) >= 0.5:
            break
        n += 1
    d = xyz[n] - xyz[0]
    heading = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d[:2]) > 1e-6 else 0.0
    R_w2f0 = Rz(-heading)
    return xyz @ R_w2f0.T, R_w2f0


# --------------------------------------------------------------------------- #
# drive directory
# --------------------------------------------------------------------------- #
@dataclass
class Drive:
    root: str
    velo: List[str]
    img02: List[str]
    img03: List[str]
    n: int


def list_drive(drive_sync: str) -> Drive:
    velo = sorted(glob.glob(os.path.join(drive_sync, "velodyne_points", "data", "*.bin")))
    img02 = sorted(glob.glob(os.path.join(drive_sync, "image_02", "data", "*.png")))
    img03 = sorted(glob.glob(os.path.join(drive_sync, "image_03", "data", "*.png")))
    n = min(len(velo), len(img02))
    return Drive(drive_sync, velo[:n], img02[:n], img03[:n], n)


def read_velodyne(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
