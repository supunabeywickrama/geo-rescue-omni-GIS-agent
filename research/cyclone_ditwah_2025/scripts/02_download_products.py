"""
Download products listed by 01_search_products.py using Copernicus CDSE OAuth.

Usage:
    python 02_download_products.py

Requires:
    CDSE_USERNAME and CDSE_PASSWORD in .env  (Copernicus Data Space account)
    Run: pip install python-dotenv requests tqdm
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

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "downloads"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_token() -> str:
    """Fetch a short-lived Bearer token from CDSE."""
    if not CDSE_USERNAME or not CDSE_PASSWORD:
        raise EnvironmentError(
            "Set CDSE_USERNAME and CDSE_PASSWORD in research/cyclone_ditwah_2025/.env"
        )
    r = requests.post(TOKEN_URL, data={
        "client_id": "cdse-public",
        "username": CDSE_USERNAME,
        "password": CDSE_PASSWORD,
        "grant_type": "password",
    }, timeout=20)
    r.raise_for_status()
    token = r.json()["access_token"]
    print("[auth] Token acquired.")
    return token


def download_product(product_id: str, name: str, token: str) -> Path:
    """Download one product zip to DOWNLOAD_DIR. Returns path."""
    out_path = DOWNLOAD_DIR / f"{name}.zip"
    if out_path.exists():
        print(f"  [skip] {name} already downloaded.")
        return out_path

    url = f"{DOWNLOAD_BASE}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers, stream=True, timeout=60)
    if r.status_code == 401:
        raise PermissionError("Token expired — re-run to get a fresh one.")
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with out_path.open("wb") as f, tqdm(
        desc=name[:40], total=total, unit="B", unit_scale=True
    ) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    return out_path


def main():
    s1_path = RAW_DIR / "sentinel1_products.json"
    s2_path = RAW_DIR / "sentinel2_products.json"

    if not s1_path.exists():
        print("Run 01_search_products.py first.")
        return

    with s1_path.open() as f: s1_products = json.load(f)
    with s2_path.open() as f: s2_products = json.load(f)

    all_products = s1_products + s2_products
    print(f"Total products to download: {len(all_products)}")
    print("  (Sentinel-1 SAR works through clouds — prioritised for cyclone phase)")

    token = get_token()
    token_time = time.time()

    for i, p in enumerate(all_products):
        # Refresh token every 9 minutes (tokens expire in 10 min)
        if time.time() - token_time > 540:
            token = get_token()
            token_time = time.time()

        pid  = p["Id"]
        name = p["Name"]
        phase = p.get("_phase", "unknown")
        print(f"\n[{i+1}/{len(all_products)}] [{phase}] {name}")
        try:
            path = download_product(pid, name, token)
            print(f"  → {path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nAll downloads complete. Next: run 03_preprocess.py")


if __name__ == "__main__":
    main()
