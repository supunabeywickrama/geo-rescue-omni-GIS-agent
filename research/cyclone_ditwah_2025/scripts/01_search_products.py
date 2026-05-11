"""
Search Copernicus Data Space for Sentinel-1 SAR and Sentinel-2 optical products
covering Sri Lanka during Cyclone Ditwah (Nov 25 – Dec 5, 2025).

Usage:
    python 01_search_products.py

Outputs:
    data/raw/sentinel1_products.json
    data/raw/sentinel2_products.json
"""

import json
import requests
from pathlib import Path

# Sri Lanka bounding box (lon_min, lat_min, lon_max, lat_max)
AOI_WKT = "POLYGON((79.5 5.9, 82.0 5.9, 82.0 10.0, 79.5 10.0, 79.5 5.9))"

# Before / during / after cyclone
DATE_BEFORE  = ("2025-11-15T00:00:00Z", "2025-11-28T00:00:00Z")
DATE_DURING  = ("2025-11-28T00:00:00Z", "2025-12-02T00:00:00Z")
DATE_AFTER   = ("2025-12-02T00:00:00Z", "2025-12-10T00:00:00Z")

ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def search(collection: str, date_start: str, date_end: str, label: str) -> list:
    """Query OData catalogue for products matching collection + date + AOI."""
    filter_str = (
        f"Collection/Name eq '{collection}'"
        f" and ContentDate/Start gt {date_start}"
        f" and ContentDate/Start lt {date_end}"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{AOI_WKT}')"
    )
    params = {
        "$filter": filter_str,
        "$orderby": "ContentDate/Start asc",
        "$top": 50,
        "$expand": "Attributes",
    }
    print(f"\n[{label}] Searching {collection} ...")
    r = requests.get(ODATA_BASE, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("value", [])
    print(f"  Found {len(results)} products")
    return results


def main():
    all_s1, all_s2 = [], []

    for label, (start, end) in {
        "before": DATE_BEFORE,
        "during": DATE_DURING,
        "after":  DATE_AFTER,
    }.items():
        s1 = search("SENTINEL-1", start, end, f"S1 {label}")
        s2 = search("SENTINEL-2", start, end, f"S2 {label}")

        # Tag each product with its phase
        for p in s1: p["_phase"] = label
        for p in s2: p["_phase"] = label

        # Filter S2 by cloud cover < 30%
        s2_clear = []
        for p in s2:
            cc = next(
                (a["Value"] for a in p.get("Attributes", [])
                 if a["Name"] == "cloudCover"), 100
            )
            if float(cc) < 30:
                s2_clear.append(p)
        print(f"  S2 after cloud filter (<30%): {len(s2_clear)}")

        all_s1.extend(s1)
        all_s2.extend(s2_clear)

    s1_path = OUT_DIR / "sentinel1_products.json"
    s2_path = OUT_DIR / "sentinel2_products.json"

    with s1_path.open("w") as f:
        json.dump(all_s1, f, indent=2, default=str)
    with s2_path.open("w") as f:
        json.dump(all_s2, f, indent=2, default=str)

    print(f"\nSaved {len(all_s1)} S1 products → {s1_path}")
    print(f"Saved {len(all_s2)} S2 products → {s2_path}")
    print("\nNext: run 02_download_products.py")


if __name__ == "__main__":
    main()
