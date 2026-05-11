"""
Download Sen1Floods11 Sri Lanka chips — S1Hand (VV+VH) + LabelHand (flood mask).

HOW IT WORKS:
  1. Lists all Sri Lanka chip IDs from the STAC catalog (label collection)
  2. Builds direct GCS HTTPS URLs for S1Hand and LabelHand TIF files
  3. Downloads each file (~1.8 MB S1, ~0.5 MB label per chip)

FILES:
  *_S1Hand.tif    — 2-band Sentinel-1 GRD (band1=VV, band2=VH, Float32, dB)
  *_LabelHand.tif — hand-labeled flood mask (0=not water, 1=flood water, -1=nodata)

Sri Lanka has 42 labeled chips (~100 MB total) — downloads in ~2-3 minutes.

Usage:
    python 00_download_sen1floods11.py
    python 00_download_sen1floods11.py --all-countries   # full 14 GB dataset
"""

import argparse
from pathlib import Path

import requests
from tqdm import tqdm
from google.cloud import storage

GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled"

OUT_DIR   = Path(__file__).parent.parent / "data" / "sen1floods11" / "HandLabeled"
S1_DIR    = OUT_DIR / "S1Hand"
LABEL_DIR = OUT_DIR / "LabelHand"


def list_chip_ids(country_prefix: str = "Sri-Lanka") -> list[str]:
    """
    Walk the STAC catalog to find all chip IDs for a country.
    Returns list of chip IDs like ['Sri-Lanka_101973', 'Sri-Lanka_14484', ...]
    """
    client = storage.Client.create_anonymous_client()
    prefix = f"v1.1/catalog/sen1floods11_hand_labeled_label/{country_prefix}"
    blobs  = client.list_blobs("sen1floods11", prefix=prefix)
    ids = []
    for b in blobs:
        if b.name.endswith("_label.json"):
            # e.g. .../Sri-Lanka_101973_label/Sri-Lanka_101973_label.json
            chip_id = b.name.split("/")[-1].replace("_label.json", "")
            ids.append(chip_id)
    return sorted(ids)


def download_file(url: str, dest: Path) -> bool:
    """Stream-download a single file. Returns True if downloaded, False if skipped."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=30)
    if r.status_code != 200:
        print(f"  SKIP (HTTP {r.status_code}): {url}")
        return False
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            f.write(chunk)
    return True


def download_chips(chip_ids: list[str]):
    """Download S1Hand + LabelHand TIFs for each chip ID."""
    print(f"\nDownloading {len(chip_ids)} chips (S1Hand + LabelHand)...")

    downloaded, skipped = 0, 0
    for chip_id in tqdm(chip_ids, unit="chip"):
        s1_url    = f"{GCS_BASE}/S1Hand/{chip_id}_S1Hand.tif"
        label_url = f"{GCS_BASE}/LabelHand/{chip_id}_LabelHand.tif"

        s1_ok    = download_file(s1_url,    S1_DIR    / f"{chip_id}_S1Hand.tif")
        label_ok = download_file(label_url, LABEL_DIR / f"{chip_id}_LabelHand.tif")

        if s1_ok or label_ok:
            downloaded += 1
        else:
            skipped += 1

    return downloaded, skipped


def print_summary():
    s1_files    = list(S1_DIR.glob("*.tif"))    if S1_DIR.exists()    else []
    label_files = list(LABEL_DIR.glob("*.tif")) if LABEL_DIR.exists() else []
    s1_mb    = sum(f.stat().st_size for f in s1_files)    / 1e6
    label_mb = sum(f.stat().st_size for f in label_files) / 1e6

    sri_lanka_s1 = [f for f in s1_files if "Sri-Lanka" in f.name]

    print(f"\n{'='*50}")
    print(f"  Sen1Floods11 Download Complete")
    print(f"{'='*50}")
    print(f"  S1Hand tiles    : {len(s1_files)}  ({s1_mb:.0f} MB)")
    print(f"  LabelHand masks : {len(label_files)}  ({label_mb:.0f} MB)")
    print(f"  Sri Lanka chips : {len(sri_lanka_s1)}")
    print(f"  Saved to        : {OUT_DIR}")
    print(f"\n  S1Hand:    band1=VV, band2=VH (Float32, dB)")
    print(f"  LabelHand: 0=not water, 1=flood, -1=no data")
    print(f"\n  Ready for model training!")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-countries", action="store_true",
                        help="Download all countries (~14 GB). Default: Sri Lanka only (~100 MB).")
    args = parser.parse_args()

    if args.all_countries:
        countries = [
            "Bolivia", "Colombia", "Ghana", "India", "Myanmar",
            "Nigeria", "Pakistan", "Paraguay", "Peru", "Senegal",
            "Somalia", "Spain", "Sri-Lanka"
        ]
    else:
        countries = ["Sri-Lanka"]

    all_chips = []
    for country in countries:
        chips = list_chip_ids(country)
        print(f"  {country}: {len(chips)} chips")
        all_chips.extend(chips)

    print(f"\nTotal chips: {len(all_chips)}")
    downloaded, skipped = download_chips(all_chips)
    print(f"Downloaded: {downloaded}  |  Skipped (already exist): {skipped}")
    print_summary()


if __name__ == "__main__":
    main()
