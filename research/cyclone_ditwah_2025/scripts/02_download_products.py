"""
Download Sentinel-1 GRD products from Copernicus CDSE.

Safe to re-run after any crash — already-downloaded files are skipped.
S2 / Sentinel-2 products are ignored even if present in the JSON.

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


def get_token(retries: int = 5) -> str:
    """
    Fetch a Bearer token from CDSE with retry + backoff.
    Retries up to 5 times on timeout or network error.
    """
    if not CDSE_USERNAME or not CDSE_PASSWORD:
        raise EnvironmentError(
            "Set CDSE_USERNAME and CDSE_PASSWORD in .env"
        )
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(TOKEN_URL, data={
                "client_id":  "cdse-public",
                "username":   CDSE_USERNAME,
                "password":   CDSE_PASSWORD,
                "grant_type": "password",
            }, timeout=30)
            r.raise_for_status()
            print(f"[auth] Token acquired (attempt {attempt}).")
            return r.json()["access_token"]
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            wait = attempt * 15
            print(f"[auth] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                print(f"[auth] Retrying in {wait}s ...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    "Cannot reach CDSE auth server after multiple retries. "
                    "Check your internet connection and re-run."
                ) from e


def download_product(product_id: str, name: str, token: str) -> Path:
    """
    Stream-download one zip. Returns path.
    Skips if file already exists (resume-safe).
    """
    out_path = DOWNLOAD_DIR / f"{name}.zip"
    if out_path.exists():
        print(f"  [skip] {name[:70]}")
        return out_path

    url = f"{DOWNLOAD_BASE}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(url, headers=headers, stream=True, timeout=120)
    if r.status_code == 401:
        raise PermissionError("Token rejected — will refresh and retry.")
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with out_path.open("wb") as f, tqdm(
        desc=name[:55], total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  → saved {out_path.stat().st_size / 1e6:.1f} MB")
    return out_path


def main():
    s1_path = RAW_DIR / "sentinel1_products.json"
    if not s1_path.exists():
        print("Run 01_search_products.py first.")
        return

    with s1_path.open() as f:
        all_products = json.load(f)

    # Only Sentinel-1 — skip any S2 products that snuck in from an old search
    s1_products = [
        p for p in all_products
        if "S1" in p["Name"] or "SENTINEL-1" in p.get("Collection", {}).get("Name", "")
        or p["Name"].startswith("S1")
    ]
    skipped_s2 = len(all_products) - len(s1_products)

    already_done = [p for p in s1_products
                    if (DOWNLOAD_DIR / f"{p['Name']}.zip").exists()]
    remaining    = [p for p in s1_products
                    if not (DOWNLOAD_DIR / f"{p['Name']}.zip").exists()]

    total_mb = sum(p.get("ContentLength", 0) for p in remaining) / 1e6

    print(f"\n{'='*55}")
    print(f"  Sentinel-1 GRD products : {len(s1_products)}")
    print(f"  Already downloaded      : {len(already_done)}")
    print(f"  Remaining               : {len(remaining)}  (~{total_mb:.0f} MB)")
    if skipped_s2:
        print(f"  Skipping S2 products    : {skipped_s2}  (not needed)")
    print(f"  Output folder           : {DOWNLOAD_DIR}")
    print(f"{'='*55}\n")

    if not remaining:
        print("All products already downloaded.")
        return

    token = get_token()
    token_time = time.time()

    failed = []
    for i, p in enumerate(remaining, 1):
        # Refresh token every 9 min
        if time.time() - token_time > 540:
            token = get_token()
            token_time = time.time()

        phase = p.get("_phase", "?")
        name  = p["Name"]
        print(f"\n[{i}/{len(remaining)}] [{phase}] {name[:70]}")

        try:
            download_product(p["Id"], name, token)
        except PermissionError:
            # Token rejected mid-download — refresh once and retry
            print("  [retry] Refreshing token and retrying ...")
            token = get_token()
            token_time = time.time()
            try:
                download_product(p["Id"], name, token)
            except Exception as e:
                print(f"  ERROR after retry: {e}")
                failed.append(name)
        except Exception as e:
            print(f"  ERROR: {e} — skipping, re-run to retry.")
            failed.append(name)

    print(f"\n{'='*55}")
    print(f"  Done.  Failed: {len(failed)}")
    if failed:
        print("  Failed products (re-run to retry):")
        for n in failed:
            print(f"    {n}")
    print(f"{'='*55}")
    print("\nNext: run 03_preprocess.py")


if __name__ == "__main__":
    main()
