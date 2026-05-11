"""
Search Copernicus for Sentinel-1 GRD products covering the Colombo / Kelani River
flood zone during and after Cyclone Ditwah (Nov 30, 2025).

AOI: Colombo + Kelani River basin (79.8–80.4°E, 6.7–7.3°N)
  - Colombo city centre
  - Kelani River from highlands to sea mouth
  - Kelaniya, Kaduwela, Hanwella flood corridors

PHASES:
  before  Nov 15–27  → dry baseline, no flood
  during  Nov 28–Dec 2  → Cyclone Ditwah landfall + peak flooding
  after   Dec 3–Dec 10  → flood extent post-storm

WHY GRD NOT SLC:
  GRD = processed backscatter amplitude (~300 MB). Ready for flood mapping.
  SLC = raw complex phase data (7+ GB). Needs interferometric processing. Wrong type.

Usage:
    python 01_search_products.py

Outputs:
    data/raw/sentinel1_products.json
    data/raw/search_summary.txt
"""

import json
import requests
from pathlib import Path

# Colombo + Kelani River basin — tight AOI, avoids unnecessary island-wide data
AOI_WKT = "POLYGON((79.8 6.7, 80.4 6.7, 80.4 7.3, 79.8 7.3, 79.8 6.7))"

DATE_BEFORE = ("2025-11-15T00:00:00Z", "2025-11-28T00:00:00Z")
DATE_DURING = ("2025-11-28T00:00:00Z", "2025-12-03T00:00:00Z")
DATE_AFTER  = ("2025-12-03T00:00:00Z", "2025-12-10T00:00:00Z")

ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_PER_PHASE = 6   # 6 × 3 phases = 18 products max, ~5-6 GB total


def search_s1_grd(date_start: str, date_end: str, phase: str) -> list:
    """Search Sentinel-1 GRD IW products intersecting Colombo/Kelani AOI."""
    filter_str = (
        f"Collection/Name eq 'SENTINEL-1'"
        f" and ContentDate/Start gt {date_start}"
        f" and ContentDate/Start lt {date_end}"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{AOI_WKT}')"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD')"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'operationalMode' and att/OData.CSC.StringAttribute/Value eq 'IW')"
    )
    params = {
        "$filter": filter_str,
        "$orderby": "ContentDate/Start asc",
        "$top": MAX_PER_PHASE,
        "$expand": "Attributes",
    }
    r = requests.get(ODATA_BASE, params=params, timeout=30)
    r.raise_for_status()
    products = r.json().get("value", [])
    for p in products:
        p["_phase"] = phase
    size_mb = sum(p.get("ContentLength", 0) for p in products) / 1e6
    print(f"  [{phase}] {len(products)} products  (~{size_mb:.0f} MB)")
    return products


def main():
    print("Searching Sentinel-1 GRD — Colombo / Kelani River / Cyclone Ditwah 2025\n")

    all_products = []
    for phase, (start, end) in {
        "before": DATE_BEFORE,
        "during": DATE_DURING,
        "after":  DATE_AFTER,
    }.items():
        products = search_s1_grd(start, end, phase)
        all_products.extend(products)

    total_mb = sum(p.get("ContentLength", 0) for p in all_products) / 1e6

    with (OUT_DIR / "sentinel1_products.json").open("w") as f:
        json.dump(all_products, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  AOI     : Colombo + Kelani River basin")
    print(f"  Event   : Cyclone Ditwah — Nov 30, 2025")
    print(f"  Products: {len(all_products)} Sentinel-1 GRD IW")
    print(f"  Size    : ~{total_mb:.0f} MB total")
    print(f"  Saved   : {OUT_DIR / 'sentinel1_products.json'}")

    # Per-product list
    print(f"\n  {'Phase':<8} {'Date':<12} {'Name'}")
    print(f"  {'-'*8} {'-'*12} {'-'*50}")
    for p in all_products:
        date = p.get("ContentDate", {}).get("Start", "")[:10]
        name = p["Name"][:55]
        print(f"  {p['_phase']:<8} {date:<12} {name}")
    print(f"{'='*55}")
    print("\nNext: run 02_download_products.py")


if __name__ == "__main__":
    main()
