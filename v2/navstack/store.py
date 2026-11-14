"""Map store: incremental, crash-safe, and loadable by region.

This module is theme 1 and theme 2 of the navigation-stack plan.

Theme 1 -- real-time persistence while mapping
----------------------------------------------
The failure mode this is built against is "the robot mapped for 20 minutes and
the process died, so the map is gone".

* **Keyframes are the source of truth, and each one is written atomically**:
  ``tmp/<id>.npz`` then ``os.replace``.  ``replace`` is atomic within a
  filesystem, so a reader never sees a half-written npz and killing the process
  costs at most one temp file.
* **A journal, appended and fsynced per keyframe.**  The manifest is the
  authoritative index and rewriting it per keyframe is O(n^2); the journal is one
  JSON line per keyframe, so recovery is "replay the lines the manifest does not
  know about".  ``recover()`` does exactly that.
* **A cursor**, so a restart resumes where the front end stopped instead of at
  frame 0.

**Chunks are derived, not primary.**  The first version appended every new
keyframe into its chunk file, which meant rewriting a growing file: measured
47 ms for the first keyframe and 1146 ms for the 150th, an O(n^2) that made
real-time capture impossible.  Chunk files are now buffered in memory and flushed
once per ``commit()`` (or every ``flush_every`` keyframes).  This is safe
precisely because keyframes are already durable and chunks are pure index:
``rebuild_chunks()`` reconstructs them from the keyframes, so a crash between
commits loses no information, only the index.

Theme 2 -- on-demand loading
----------------------------
LOD is **baked at flush time**, one file per (chunk, level), not recomputed on
every load.  Theme 2's point is that a local planner should touch a few chunks
instead of the whole map; recomputing a voxel grid for the same chunk on every
query would hide that saving behind decimation cost.  ``load_region()`` therefore
does I/O and a disc cut, nothing else, and reports the miss rate rather than
assuming it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import schema as S
from . import tiles as T


class MapStore:
    """Writer/reader for a ``.navmap`` directory."""

    def __init__(self, root: str, chunk_size: float = S.CHUNK_SIZE_M,
                 buffer_m: float = S.CHUNK_BUFFER_M, tmp_dir: Optional[str] = None,
                 flush_every: int = 40, dedupe_m: float = 0.10):
        self.root = root
        self.chunk_size = float(chunk_size)
        self.buffer_m = float(buffer_m)
        self.flush_every = int(flush_every)
        self.dedupe_m = float(dedupe_m)
        self.dir_chunks = os.path.join(root, "chunks")
        self.dir_kf = os.path.join(root, "keyframes")
        self.dir_tmp = tmp_dir or os.path.join(root, "_tmp")
        self.path_manifest = os.path.join(root, "manifest.json")
        self.path_journal = os.path.join(root, "journal.jsonl")
        self.path_cursor = os.path.join(root, "cursor.json")
        self.cache = T.ChunkCache()

        self.meta = S.MapMeta(chunk_size_m=self.chunk_size,
                              chunk_buffer_m=self.buffer_m)
        self.levels: List[S.Level] = [S.Level(0, 0.0, 0.0, 0.0)]
        self.keyframes: List[S.Keyframe] = []
        self.level_chunks: Dict[int, set] = {}
        self.window: List[S.Keyframe] = []
        self.write_log: List[dict] = []
        self._chunk_buf: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self._chunk_level: Dict[str, int] = {}
        self._window_size = 12
        self.flush_log: List[dict] = []

    # ------------------------------------------------------------------ setup
    def create(self, map_id: str, window: int = 12, source: str = "") -> None:
        for d in (self.root, self.dir_chunks, self.dir_kf, self.dir_tmp):
            os.makedirs(d, exist_ok=True)
        for f in os.listdir(self.dir_tmp):
            try:
                os.remove(os.path.join(self.dir_tmp, f))
            except OSError:
                pass
        self.meta = S.MapMeta(map_id=map_id, chunk_size_m=self.chunk_size,
                              chunk_buffer_m=self.buffer_m, source=source)
        self.levels = []
        self.keyframes = []
        self.level_chunks = {}
        self.window = []
        self.write_log = []
        self.flush_log = []
        self._chunk_buf = {}
        self._chunk_level = {}
        self._window_size = int(window)
        open(self.path_journal, "w").close()
        self._write_manifest()
        self.set_cursor({"frame": 0, "t": 0.0, "keyframes": 0, "done": False})

    # ------------------------------------------------------- theme 1: writing
    def _atomic_npz(self, path: str, **arrays) -> int:
        tmp = os.path.join(self.dir_tmp, os.path.basename(path) + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return os.path.getsize(path)

    def _fsync_dir(self, d: str) -> None:
        try:
            fd = os.open(d, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def add_keyframe(self, kf: S.Keyframe, pts: np.ndarray,
                     rgb: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Persist one keyframe; buffer its chunk contributions.  O(1)."""
        t0 = time.perf_counter()
        kf.chunks = S.chunks_for_points(pts, self.chunk_size)
        kf.n_points = int(len(pts))
        cols = rgb if rgb is not None else np.full((len(pts), 3), 110, np.uint8)

        nbytes = self._atomic_npz(
            os.path.join(self.dir_kf, f"{kf.id:06d}.npz"),
            pts=pts.astype(np.float32), rgb=cols.astype(np.uint8),
            pose=kf.pose.astype(np.float64),
            meta=np.array([kf.id, kf.seq, kf.t, kf.level, kf.dynamic_frac],
                          dtype=np.float64))
        t_kf = (time.perf_counter() - t0) * 1e3

        # chunk contributions are buffered, not written: see the module docstring
        ci = np.floor(pts[:, 0] / self.chunk_size).astype(np.int64)
        cj = np.floor(pts[:, 1] / self.chunk_size).astype(np.int64)
        for key in kf.chunks:
            i, j = (int(x) for x in key.replace("+", "").split("_"))
            sel = (ci == i) & (cj == j)
            if sel.sum() < 12:
                continue
            self._chunk_buf.setdefault(key, []).append(
                (pts[sel].astype(np.float32), cols[sel].astype(np.uint8)))
            self._chunk_level[key] = kf.level
            self.level_chunks.setdefault(kf.level, set()).add(key)

        self.keyframes.append(kf)
        self.window.append(kf)
        while len(self.window) > self._window_size:
            self.window.pop(0)

        self._append_journal({
            "id": kf.id, "seq": kf.seq, "t": kf.t, "level": kf.level,
            "pose": kf.pose.round(6).tolist(), "n_points": kf.n_points,
            "chunks": kf.chunks, "dynamic_frac": round(kf.dynamic_frac, 5),
            "bytes": int(nbytes)})
        self.meta.n_keyframes = len(self.keyframes)
        self.meta.total_points += int(len(pts))
        rec = {"id": kf.id, "t_keyframe_ms": round(t_kf, 2),
               "buffered_chunks": len(kf.chunks), "bytes": int(nbytes)}
        self.write_log.append(rec)
        return rec

    def flush_chunks(self) -> dict:
        """Merge buffered contributions and write one file per (chunk, LOD)."""
        if not self._chunk_buf:
            return {"chunks": 0, "ms": 0.0, "files": 0, "bytes": 0}
        t0 = time.perf_counter()
        files = 0
        nbytes = 0
        for key, parts in self._chunk_buf.items():
            P = np.concatenate([p for p, _ in parts])
            C = np.concatenate([c for _, c in parts])
            # a map should not carry five copies of the same wall: dedupe at a
            # fixed 0.10 m before baking the LODs, so the three files are nested
            # representations of the same surface rather than of the raw scans
            P, C = T.decimate(P, self.dedupe_m, C)
            for lod, leaf in T.LOD_LEAF.items():
                p2, c2 = T.decimate(P, leaf, C)
                path = os.path.join(self.dir_chunks, f"c_{key}_{lod}.npz")
                nbytes += self._atomic_npz(path, pts=p2.astype(np.float32),
                                           rgb=c2.astype(np.uint8))
                files += 1
        self._fsync_dir(self.dir_chunks)
        dt = (time.perf_counter() - t0) * 1e3
        n = len(self._chunk_buf)
        self._chunk_buf = {}
        rec = {"chunks": n, "files": files, "ms": round(dt, 1), "bytes": int(nbytes)}
        self.flush_log.append(rec)
        return rec

    def _append_journal(self, rec: dict) -> None:
        with open(self.path_journal, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def commit(self) -> dict:
        """Flush chunks, fold the journal into the manifest, fsync the root."""
        fl = self.flush_chunks()
        self._write_manifest()
        self._fsync_dir(self.root)
        return fl

    def _write_manifest(self) -> None:
        all_pts = [kf.pose[:3, 3] for kf in self.keyframes]
        if all_pts:
            arr = np.stack(all_pts)
            self.meta.bbox_min = arr.min(0).round(3).tolist()
            self.meta.bbox_max = arr.max(0).round(3).tolist()
        self.meta.level_count = max(1, len(self.levels))
        self.meta.n_keyframes = len(self.keyframes)
        doc = {
            "schema_version": S.SCHEMA_VERSION,
            "meta": self.meta.as_dict(),
            "levels": [{"id": l.id, "z_min": round(l.z_min, 3),
                        "z_max": round(l.z_max, 3), "z_ref": round(l.z_ref, 3),
                        "keyframes": l.keyframes, "n_points": l.n_points}
                       for l in self.levels],
            "chunks": {str(k): len(v) for k, v in sorted(self.level_chunks.items())},
            "chunk_keys": sorted(self.level_chunks.get(0, set()) |
                                 set().union(*self.level_chunks.values())
                                 if self.level_chunks else set()),
            "keyframes": [{"id": k.id, "seq": k.seq, "t": k.t,
                           "pose": k.pose.tolist(), "level": k.level,
                           "n_points": k.n_points, "chunks": k.chunks,
                           "dynamic_frac": k.dynamic_frac}
                          for k in self.keyframes],
            "journal_lines": len(self.keyframes),
        }
        tmp = self.path_manifest + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path_manifest)

    def set_cursor(self, state: dict) -> None:
        tmp = self.path_cursor + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path_cursor)

    def get_cursor(self) -> dict:
        if not os.path.exists(self.path_cursor):
            return {"frame": 0, "t": 0.0, "keyframes": 0, "done": False}
        with open(self.path_cursor) as f:
            return json.load(f)

    # ------------------------------------------------------- theme 2: loading
    def _chunk_path(self, key: str, lod: str) -> str:
        return os.path.join(self.dir_chunks, f"c_{key}_{lod}.npz")

    def load_region(self, center_xy: np.ndarray, radius: float = 30.0,
                    lod: str = "fine", level: Optional[int] = None,
                    use_buffer: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Chunks intersecting ``radius`` plus a load buffer, at one baked LOD."""
        t0 = time.perf_counter()
        r = radius + (self.buffer_m if use_buffer else 0.0)
        keys = S.keys_in_disc(np.asarray(center_xy, float), r, self.chunk_size)
        if level is not None:
            keys = [k for k in keys if k in self.level_chunks.get(int(level), set())]
        parts, loaded, hit, miss, t_io = [], 0, 0, 0, 0.0
        for k in keys:
            ck = (k, lod)
            v = self.cache.get(ck)
            if v is None:
                path = self._chunk_path(k, lod)
                if not os.path.exists(path):
                    miss += 1
                    continue
                ti = time.perf_counter()
                with np.load(path) as z:
                    v = (z["pts"], z["rgb"])
                t_io += (time.perf_counter() - ti) * 1e3
                self.cache.put(ck, v)
            else:
                hit += 1
            parts.append(v)
            loaded += 1
        if parts:
            pts = np.concatenate([p for p, _ in parts])
            rgb = np.concatenate([c for _, c in parts])
        else:
            pts = np.zeros((0, 3), np.float32)
            rgb = np.zeros((0, 3), np.uint8)
        c = np.asarray(center_xy, float)
        if len(pts):
            keep = (pts[:, 0] - c[0]) ** 2 + (pts[:, 1] - c[1]) ** 2 <= radius * radius
            pts, rgb = pts[keep], rgb[keep]
        return pts, rgb, {
            "chunks_requested": len(keys), "chunks_loaded": loaded,
            "cache_hits": hit, "chunks_missing": miss,
            "points": int(len(pts)), "lod": lod,
            "leaf_m": T.LOD_LEAF.get(lod, S.DEFAULT_LOD_M),
            "radius_m": radius, "buffer_m": self.buffer_m if use_buffer else 0.0,
            "ms": round((time.perf_counter() - t0) * 1e3, 2),
            "io_ms": round(t_io, 2), "level": level}

    def load_all(self, lod: str = "mid", level: Optional[int] = None,
                 from_keyframes: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Whole map, for benchmarks and the web export."""
        t0 = time.perf_counter()
        if not from_keyframes:
            parts = []
            for k, lv in self._chunk_level.items():
                if level is not None and lv != int(level):
                    continue
                p = self._chunk_path(k, lod)
                if os.path.exists(p):
                    with np.load(p) as z:
                        parts.append((z["pts"], z["rgb"]))
            pts = np.concatenate([p for p, _ in parts]) if parts else np.zeros((0, 3), np.float32)
            rgb = np.concatenate([c for _, c in parts]) if parts else np.zeros((0, 3), np.uint8)
            return pts, rgb, {"keyframes": -1, "points": int(len(pts)), "lod": lod,
                              "leaf_m": T.LOD_LEAF.get(lod, 0.2),
                              "ms": round((time.perf_counter() - t0) * 1e3, 2)}
        leaf = T.LOD_LEAF.get(lod, S.DEFAULT_LOD_M)
        clouds, n = [], 0
        for kf in self.keyframes:
            if level is not None and kf.level != int(level):
                continue
            with np.load(os.path.join(self.dir_kf, f"{kf.id:06d}.npz")) as z:
                clouds.append((z["pts"], z["rgb"]))
            n += 1
        pts, rgb = T.merge_with_limit(clouds, leaf)
        return pts, rgb, {"keyframes": n, "points": int(len(pts)), "lod": lod,
                          "leaf_m": leaf,
                          "ms": round((time.perf_counter() - t0) * 1e3, 2)}

    def rebuild_chunks(self) -> dict:
        """Reconstruct every chunk from the keyframes.  Proves chunks are derived."""
        t0 = time.perf_counter()
        for f in os.listdir(self.dir_chunks):
            if f.endswith(".npz"):
                os.remove(os.path.join(self.dir_chunks, f))
        self._chunk_buf = {}
        self.level_chunks = {}
        self._chunk_level = {}
        for kf in self.keyframes:
            with np.load(os.path.join(self.dir_kf, f"{kf.id:06d}.npz")) as z:
                pts, cols = z["pts"], z["rgb"]
            ci = np.floor(pts[:, 0] / self.chunk_size).astype(np.int64)
            cj = np.floor(pts[:, 1] / self.chunk_size).astype(np.int64)
            for key in kf.chunks:
                i, j = (int(x) for x in key.replace("+", "").split("_"))
                sel = (ci == i) & (cj == j)
                if sel.sum() < 12:
                    continue
                self._chunk_buf.setdefault(key, []).append(
                    (pts[sel].astype(np.float32), cols[sel].astype(np.uint8)))
                self._chunk_level[key] = kf.level
                self.level_chunks.setdefault(kf.level, set()).add(key)
        rec = self.flush_chunks()
        rec["total_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
        return rec

    # ------------------------------------------------------------- reopening
    def open(self) -> "MapStore":
        if not os.path.exists(self.path_manifest):
            raise FileNotFoundError(f"no manifest in {self.root}")
        with open(self.path_manifest) as f:
            doc = json.load(f)
        ver = int(doc.get("schema_version", -1))
        if ver != S.SCHEMA_VERSION:
            raise RuntimeError(
                f"map schema {ver} != reader {S.SCHEMA_VERSION}; this reader "
                f"refuses to open a map it may mis-index")
        m = doc["meta"]
        self.meta = S.MapMeta(**{k: v for k, v in m.items()
                                 if k in S.MapMeta.__dataclass_fields__})
        self.levels = [S.Level(**{**lv, "keyframes": lv.get("keyframes", []),
                                  "n_points": lv.get("n_points", 0)})
                       for lv in doc.get("levels", [])]
        self.level_chunks = {}
        for kf in doc["keyframes"]:
            self.keyframes.append(S.Keyframe(
                id=kf["id"], t=kf["t"], pose=np.asarray(kf["pose"], float),
                n_points=kf["n_points"], level=kf["level"],
                chunks=kf["chunks"], seq=kf["seq"],
                dynamic_frac=kf.get("dynamic_frac", 0.0)))
        for kf in self.keyframes:
            for k in kf.chunks:
                self.level_chunks.setdefault(kf.level, set()).add(k)
        for lv, keys in self.level_chunks.items():
            for k in keys:
                self._chunk_level[k] = lv
        self.window = self.keyframes[-self._window_size:]
        return self

    def recover(self) -> dict:
        """Replay journal lines the manifest does not know about."""
        if not os.path.exists(self.path_journal):
            return {"recovered": 0}
        known = {kf.id for kf in self.keyframes}
        n = 0
        with open(self.path_journal) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["id"] in known:
                    continue
                path = os.path.join(self.dir_kf, f"{rec['id']:06d}.npz")
                if not os.path.exists(path):
                    continue
                self.keyframes.append(S.Keyframe(
                    id=rec["id"], t=rec["t"], pose=np.asarray(rec["pose"], float),
                    n_points=rec["n_points"], level=rec["level"],
                    chunks=rec["chunks"], seq=rec["seq"],
                    dynamic_frac=rec.get("dynamic_frac", 0.0)))
                known.add(rec["id"])
                n += 1
        self.keyframes.sort(key=lambda k: k.seq)
        for kf in self.keyframes:
            for k in kf.chunks:
                self.level_chunks.setdefault(kf.level, set()).add(k)
                self._chunk_level[k] = kf.level
        self._write_manifest()
        return {"recovered": n, "keyframes": len(self.keyframes)}

    # --------------------------------------------------------------- stats
    def disk_bytes(self) -> int:
        n = 0
        for base, _, files in os.walk(self.root):
            for f in files:
                n += os.path.getsize(os.path.join(base, f))
        return n

    def write_stats(self) -> dict:
        if not self.write_log:
            return {}
        tk = np.array([w["t_keyframe_ms"] for w in self.write_log])
        out = {
            "keyframes": len(self.write_log),
            "keyframe_ms_mean": round(float(tk.mean()), 2),
            "keyframe_ms_p95": round(float(np.percentile(tk, 95)), 2),
            "keyframe_ms_max": round(float(tk.max()), 2),
            "keyframe_bytes_total": int(sum(w["bytes"] for w in self.write_log)),
            "disk_total": int(self.disk_bytes()),
        }
        if self.flush_log:
            fm = np.array([f["ms"] for f in self.flush_log])
            out.update({
                "flush_calls": len(self.flush_log),
                "flush_ms_mean": round(float(fm.mean()), 1),
                "flush_ms_total": round(float(fm.sum()), 1),
                "flush_files": int(sum(f["files"] for f in self.flush_log)),
            })
        return out
