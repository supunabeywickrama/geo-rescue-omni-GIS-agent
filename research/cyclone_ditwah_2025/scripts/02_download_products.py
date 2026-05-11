"""
Download products listed by 01_search_products.py using Copernicus CDSE OAuth.

WHY THIS APPROACH:
  - CDSE (dataspace.copernicus.eu) provides free direct download of full products
  - Authentication uses a short-lived Bearer token (expires in 10 min)
  - This script auto-refreshes the token every 9 minutes
  - Downloads are streamed in chunks so large files don't exhaust RAM
  - Already-downloaded zips are skipped (safe to re-run after interruption)

EXPECTED SIZES (after fixing search to GRD only):
  - Sentinel-1 GRD IW: ~300-800 MB per product
  - Sentinel-2 L2A:    ~600-1000 MB per product
  - Total (~45 products): ~15-25 GB

Usage:
    python 02_download_products.py

Requires:
    CDSE_USERNAME and CDSE_PASSWORD in .env
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / ".env")

CDSE_USERNAME = os.getenv("CDSE_USERNAME")
CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1/Products"

RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_token() -> str:
    """
    Fetch a Bearer token from CDSE identity server.

    WHY needed:
      The download endpoint rejects requests without a valid Bearer token.
      Tokens expire after 10 minutes — this function is called at startup
      and again every 9 minutes during the download loop.
    """
    if not CDSE_USERNAME or not CDSE_PASSWORD:
        raise EnvironmentError(
            "Set CDSE_USERNAME and CDSE_PASSWORD in research/cyclone_ditwah_2025/.env"
        )
    r = requests.post(TOKEN_URL, data={
        "client_id": "cdse-public",
        "username":   CDSE_USERNAME,
        "password":   CDSE_PASSWORD,
        "grant_type": "password",
    }, timeout=20)
    r.raise_for_status()
    print("[auth] Token acquired (valid 10 min).")
    return r.json()["access_token"]


def download_product(product_id: str, name: str, token: str) -> Path:
    """
    Stream-download one product zip into DOWNLOAD_DIR.

    WHY streaming:
      Products can be 300 MB – 1 GB. Streaming with iter_content writes
      chunks to disk immediately instead of buffering the full file in RAM.

    WHY skip if exists:
      Safe to re-run after interruption — already-downloaded files are skipped.
    """
    out_path = DOWNLOAD_DIR / f"{name}.zip"
    if out_path.exists():
        print(f"  [skip] {name[:60]} already downloaded.")
        return out_path

    url = f"{DOWNLOAD_BASE}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers, stream=True, timeout=120)
    if r.status_code == 401:
        raise PermissionError("Token expired mid-download — re-run to resume.")
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with out_path.open("wb") as f, tqdm(
        desc=name[:50], total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  → saved {out_path.stat().st_size / 1e6:.1f} MB")
    return out_path


def main():
    s1_path = RAW_DIR / "sentinel1_products.json"
    s2_path = RAW_DIR / "sentinel2_products.json"

    if not s1_path.exists():
        print("Run 01_search_products.py first.")
        return

    with s1_path.open() as f: s1 = json.load(f)
    with s2_path.open() as f: s2 = json.load(f)

    # S1 first (most important — SAR works through cyclone clouds)
    all_products = s1 + s2
    total_mb = sum(p.get("ContentLength", 0) for p in all_products) / 1e6

    print(f"\nProducts to download : {len(all_products)}")
    print(f"  Sentinel-1 GRD     : {len(s1)}")
    print(f"  Sentinel-2 L2A     : {len(s2)}")
    print(f"  Estimated total    : ~{total_mb:.0f} MB")
    print(f"\nDownloads go to     : {DOWNLOAD_DIR}")
    print("Already-downloaded files will be skipped.\n")

    token = get_token()
    token_time = time.time()

    for i, p in enumerate(all_products, 1):
        # Refresh token every 9 min (expires at 10 min)
        if time.time() - token_time > 540:
            token = get_token()
            token_time = time.time()

        phase = p.get("_phase", "?")
        name  = p["Name"]
        print(f"\n[{i}/{len(all_products)}] [{phase}] {name[:70]}")
        try:
            download_product(p["Id"], name, token)
        except Exception as e:
            print(f"  ERROR: {e} — continuing with next product.")

    print("\nAll downloads complete. Next: run 03_preprocess.py")


if __name__ == "__main__":
    main()
