"""
Download Sen1Floods11 — the primary labeled flood training dataset.

WHY THIS DATASET:
  - 4,831 chips of 512×512 px Sentinel-1 GRD tiles (VV + VH, already in dB)
  - Hand-labeled binary flood masks for every chip
  - Covers Sri Lanka (2017 Matara flood event) + 10 other global events
  - 14 GB total — much faster than downloading raw Sentinel products
  - No preprocessing needed: tiles are ready for model training

WHAT YOU GET:
  *_S1.tif   — 2-band GeoTIFF (band 1=VV, band 2=VH), Float32, dB
  *_QC.tif   — 1-band label mask: -1=NoData, 0=Not Water, 1=Water (Flood)
  *_S2.tif   — 13-band Sentinel-2 optical (optional, not needed for SAR model)

DOWNLOAD OPTIONS:
  Option A (recommended): gsutil  — fast, parallel, resumable
  Option B: Python google-cloud-storage — no gsutil install needed

Usage:
    # Option A — install gsutil first (part of Google Cloud SDK)
    pip install gsutil
    python 00_download_sen1floods11.py --method gsutil

    # Option B — pure Python
    pip install google-cloud-storage
    python 00_download_sen1floods11.py --method python

    # Download Sri Lanka chips only (fastest, ~200 MB)
    python 00_download_sen1floods11.py --sri-lanka-only
"""

import argparse
import subprocess
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "sen1floods11"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GCS_BUCKET = "gs://sen1floods11"

# Sen1Floods11 uses event codes. Sri Lanka 2017 = "Sri-Lanka"
# Full list: Bolivia, Colombia, Ghana, India, Myanmar, Nigeria,
#            Pakistan, Paraguay, Peru, Senegal, Somalia, Spain, Sri-Lanka
SRI_LANKA_PREFIX = "Sri-Lanka"


def download_gsutil(sri_lanka_only: bool):
    """
    Use gsutil for fast parallel download with auto-resume on failure.

    WHY gsutil:
      rsync is resumable — if it crashes, re-running skips already-downloaded files.
      -m flag uses parallel threads (much faster than sequential).
    """
    if sri_lanka_only:
        # Only S1 + QC tiles for Sri Lanka (~200 MB, fastest option)
        for layer in ("S1", "QC"):
            cmd = [
                "gsutil", "-m", "rsync", "-r",
                f"{GCS_BUCKET}/v1.1/data/flood_events/HandLabeled/{layer}/",
                str(OUT_DIR / "HandLabeled" / layer),
            ]
            # Filter to Sri Lanka chips only
            print(f"Downloading Sri Lanka {layer} tiles...")
            # gsutil doesn't support filename filtering in rsync directly,
            # so we use cp with wildcard
            cmd = [
                "gsutil", "-m", "cp",
                f"{GCS_BUCKET}/v1.1/data/flood_events/HandLabeled/{layer}/{SRI_LANKA_PREFIX}*",
                str(OUT_DIR / "HandLabeled" / layer) + "/",
            ]
            (OUT_DIR / "HandLabeled" / layer).mkdir(parents=True, exist_ok=True)
            print(f"  Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
    else:
        # Full dataset (~14 GB) — S1 + QC only (skip S2 to save space)
        for layer in ("S1", "QC"):
            dest = OUT_DIR / "HandLabeled" / layer
            dest.mkdir(parents=True, exist_ok=True)
            cmd = [
                "gsutil", "-m", "rsync", "-r",
                f"{GCS_BUCKET}/v1.1/data/flood_events/HandLabeled/{layer}/",
                str(dest),
            ]
            print(f"\nDownloading full {layer} dataset...")
            print(f"  Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

    # Also download the metadata GeoJSON
    meta_cmd = [
        "gsutil", "cp",
        f"{GCS_BUCKET}/v1.1/Sen1Floods11_Metadata.geojson",
        str(OUT_DIR / "Sen1Floods11_Metadata.geojson"),
    ]
    subprocess.run(meta_cmd, check=False)
    print("\nMetadata downloaded.")


def download_python(sri_lanka_only: bool):
    """
    Pure Python download using google-cloud-storage.
    Slower than gsutil but no extra CLI install needed.
    The bucket is public — no credentials required.
    """
    try:
        from google.cloud import storage
    except ImportError:
        print("Run: pip install google-cloud-storage")
        return

    # Anonymous client for public bucket
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket("sen1floods11")

    layers = ["S1", "QC"]
    for layer in layers:
        prefix = f"v1.1/data/flood_events/HandLabeled/{layer}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if sri_lanka_only:
            blobs = [b for b in blobs if SRI_LANKA_PREFIX in b.name]

        dest_dir = OUT_DIR / "HandLabeled" / layer
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDownloading {len(blobs)} {layer} files...")
        for i, blob in enumerate(blobs, 1):
            filename = Path(blob.name).name
            dest = dest_dir / filename
            if dest.exists():
                print(f"  [{i}/{len(blobs)}] skip {filename}")
                continue
            print(f"  [{i}/{len(blobs)}] {filename} ({blob.size / 1e6:.1f} MB)")
            blob.download_to_filename(str(dest))

    # Metadata
    meta_blob = bucket.blob("v1.1/Sen1Floods11_Metadata.geojson")
    meta_blob.download_to_filename(str(OUT_DIR / "Sen1Floods11_Metadata.geojson"))
    print("\nMetadata downloaded.")


def print_summary():
    s1_files = list((OUT_DIR / "HandLabeled" / "S1").glob("*.tif"))
    qc_files = list((OUT_DIR / "HandLabeled" / "QC").glob("*.tif"))
    sri_lanka = [f for f in s1_files if SRI_LANKA_PREFIX in f.name]

    print(f"\n{'='*50}")
    print(f"  Sen1Floods11 Download Summary")
    print(f"{'='*50}")
    print(f"  S1 tiles downloaded : {len(s1_files)}")
    print(f"  QC masks downloaded : {len(qc_files)}")
    print(f"  Sri Lanka chips     : {len(sri_lanka)}")
    print(f"  Location            : {OUT_DIR}")
    print(f"\n  Next: run 04_tile_dataset.py or go straight to training.")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gsutil", "python"], default="python")
    parser.add_argument("--sri-lanka-only", action="store_true",
                        help="Download only Sri Lanka chips (~200 MB instead of 14 GB)")
    args = parser.parse_args()

    print(f"Downloading Sen1Floods11 {'(Sri Lanka only)' if args.sri_lanka_only else '(full dataset)'}")
    print(f"Output directory: {OUT_DIR}\n")

    if args.method == "gsutil":
        download_gsutil(args.sri_lanka_only)
    else:
        download_python(args.sri_lanka_only)

    print_summary()


if __name__ == "__main__":
    main()
