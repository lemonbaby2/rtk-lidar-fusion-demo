#!/usr/bin/env bash
# Download KITTI raw drive_0005 on the DGX itself.
#
# The link from this workstation runs over a Tailscale DERP relay at roughly
# 65 KB/s, so pushing the 46 MB preprocessed bundle took twelve minutes.  The
# DGX reaches the KITTI S3 bucket at 6 MB/s, so the raw archive comes down here
# in about two minutes and the preprocessing runs locally instead.
set -euo pipefail

RAW=/home/nvidia/rtk_fusion_demo/raw
B=https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data
mkdir -p "$RAW"
cd "$RAW"

fetch() {
  local url="$1" out="$2"
  if [ -s "$out" ]; then echo "have $out"; return; fi
  echo "fetching $out"
  curl -sS --retry 5 --retry-delay 3 -C - -o "$out" "$url"
  ls -l "$out"
}

fetch "$B/2011_09_26_calib.zip" 2011_09_26_calib.zip
fetch "$B/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip" 2011_09_26_drive_0005_sync.zip

echo "unzipping"
unzip -oq 2011_09_26_calib.zip
unzip -oq 2011_09_26_drive_0005_sync.zip

echo "layout:"
ls -1 "$RAW"/2011_09_26 | head
D="$RAW/2011_09_26/2011_09_26_drive_0005_sync"
echo "velodyne: $(ls -1 "$D/velodyne_points/data" 2>/dev/null | wc -l)"
echo "image_02: $(ls -1 "$D/image_02/data" 2>/dev/null | wc -l)"
echo "oxts:     $(ls -1 "$D/oxts/data" 2>/dev/null | wc -l)"
du -sh "$RAW"
