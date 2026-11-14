#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel HTTP range downloader, tuned for high-latency S3 access.

A single connection from this host to the KITTI bucket settles around 55 KB/s,
which would need three hours for the 646 MB archive.  Six parallel range
requests measured roughly 350 KB/s, so throughput here is limited per stream and
not by the link, and more streams pay off.

Range tasks are queued and pulled by a fixed pool of workers, so a slow chunk
does not hold up the others.  Any prefix already present in OUT is kept and the
remaining bytes are fetched, which makes the run resumable.

  python3 parallel_fetch.py URL OUT [STREAMS] [CHUNK_MB]
"""
import os
import queue
import sys
import threading
import time
import urllib.request

URL = sys.argv[1]
OUT = sys.argv[2]
STREAMS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
CHUNK = int(float(sys.argv[4]) * 1e6) if len(sys.argv) > 4 else 2_000_000

lock = threading.Lock()
stats = {"bytes": 0, "ok": 0, "fail": []}


def total_size():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_range(start, end, path, tag):
    """Download [start, end] inclusive into path, resuming a partial part file."""
    want = end - start + 1
    for attempt in range(1, 15):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= want:
            return True
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start+have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(path, "ab") as fh:
                while True:
                    b = r.read(1 << 15)
                    if not b:
                        break
                    fh.write(b)
                    with lock:
                        stats["bytes"] += len(b)
            if os.path.getsize(path) >= want:
                return True
        except Exception as e:
            with lock:
                print(f"  [{tag}] attempt {attempt}: {type(e).__name__} {e}", flush=True)
            time.sleep(1 + attempt)
    return os.path.exists(path) and os.path.getsize(path) >= want


def worker(q, tag):
    while True:
        try:
            k, s, e = q.get_nowait()
        except queue.Empty:
            return
        p = f"{OUT}.part{k:04d}"
        ok = fetch_range(s, e, p, tag)
        with lock:
            if ok:
                stats["ok"] += 1
            else:
                stats["fail"].append(k)
        q.task_done()


def reporter(stop, rest, t0):
    while not stop.is_set():
        time.sleep(5)
        with lock:
            d = stats["bytes"]
            ok = stats["ok"]
            bad = len(stats["fail"])
        print(f"  {d/1e6:7.1f} / {rest/1e6:.1f} MB  "
              f"{d/1e6/max(time.time()-t0,1e-9):.2f} MB/s  "
              f"chunks ok={ok} fail={bad}", flush=True)


def main():
    total = total_size()
    head = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"url={URL}\ntotal={total/1e6:.1f} MB  existing prefix={head/1e6:.1f} MB",
          flush=True)
    if head >= total:
        print("already complete", flush=True)
        return

    q = queue.Queue()
    k = 0
    s = head
    while s < total:
        e = min(total - 1, s + CHUNK - 1)
        q.put((k, s, e))
        k += 1
        s = e + 1
    nchunks = k
    rest = total - head
    print(f"{nchunks} chunks of ~{CHUNK/1e6:.0f} MB across {STREAMS} workers", flush=True)

    t0 = time.time()
    stop = threading.Event()
    threading.Thread(target=reporter, args=(stop, rest, t0), daemon=True).start()

    pool = [threading.Thread(target=worker, args=(q, f"w{i}"), daemon=True)
            for i in range(STREAMS)]
    for t in pool:
        t.start()
        time.sleep(0.1)
    for t in pool:
        t.join()
    stop.set()

    if stats["fail"] or stats["ok"] != nchunks:
        sys.exit(f"chunks failed: {sorted(stats['fail'])[:10]} "
                 f"(ok {stats['ok']}/{nchunks})")

    print("concatenating", flush=True)
    with open(OUT, "ab") as out:
        for k in range(nchunks):
            p = f"{OUT}.part{k:04d}"
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            os.remove(p)
    size = os.path.getsize(OUT)
    el = time.time() - t0
    print(f"final size {size} (expected {total}) "
          f"{'OK' if size == total else 'MISMATCH'}", flush=True)
    print(f"elapsed {el:.0f}s  avg {rest/1e6/max(el,1e-9):.2f} MB/s", flush=True)


if __name__ == "__main__":
    main()
