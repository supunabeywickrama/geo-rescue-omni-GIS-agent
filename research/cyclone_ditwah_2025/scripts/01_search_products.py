"""
Search Copernicus Data Space for Sentinel-1 GRD and Sentinel-2 products
covering Sri Lanka during Cyclone Ditwah (Nov 25 – Dec 5, 2025).

KEY FILTERS APPLIED:
  - Sentinel-1: GRD only (NOT SLC). GRD = ~300 MB. SLC = 7+ GB and needs
    complex interferometric processing — wrong for flood detection.
  - Sentinel-1: IW mode (Interferometric Wide Swath, 250 km coverage)
  - Sentinel-2: cloud cover < 30%
  - Max 8 products per phase per sensor — enough for training, avoids TB downloads

Usage:
    python 01_search_products.py

Outputs:
    data/raw/sentinel1_products.json
    data/raw/sentinel2_products.json
    data/raw/search_summary.txt
"""

import json
import requests
from pathlib import Path

# Sri Lanka bounding box — focused on coastal/flood-prone areas
AOI_WKT = "POLYGON((79.5 5.9, 82.0 5.9, 82.0 10.0, 79.5 10.0, 79.5 5.9))"

DATE_BEFORE = ("2025-11-15T00:00:00Z", "2025-11-28T00:00:00Z")
DATE_DURING = ("2025-11-28T00:00:00Z", "2025-12-02T00:00:00Z")
DATE_AFTER  = ("2025-12-02T00:00:00Z", "2025-12-10T00:00:00Z")

ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cap per phase — 8 S1 + 5 S2 per phase = ~45 products total, ~15 GB max
MAX_S1_PER_PHASE = 8
MAX_S2_PER_PHASE = 5


def search_s1_grd(date_start: str, date_end: str, label: str) -> list:
    """
    Search Sentinel-1 GRD IW products only.

    WHY GRD?
      - GRD (Ground Range Detected): pre-processed amplitude image, ~300 MB zip
      - SLC (Single Look Complex): raw complex data, 5-10 GB, needs phase processing
      - For flood mapping we only need backscatter intensity → GRD is correct
      - IW mode gives 250 km swath width covering all of Sri Lanka in one pass
    """
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
        "$top": MAX_S1_PER_PHASE,
        "$expand": "Attributes",
    }
    print(f"\n[S1 GRD IW | {label}] Searching ...")
    r = requests.get(ODATA_BASE, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("value", [])
    for p in results:
        p["_phase"] = label
    size_mb = sum(p.get("ContentLength", 0) for p in results) / 1e6
    print(f"  Found {len(results)} products  (~{size_mb:.0f} MB total)")
    return results


def search_s2(date_start: str, date_end: str, label: str) -> list:
    """
    Search Sentinel-2 L2A products with cloud cover < 30%.

    WHY cloud filter?
      - During cyclone (during phase) S2 will be 100% cloud → skip it
      - Before/after phases: only download cloud-free scenes for clean baseline
    """
    filter_str = (
        f"Collection/Name eq 'SENTINEL-2'"
        f" and ContentDate/Start gt {date_start}"
        f" and ContentDate/Start lt {date_end}"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{AOI_WKT}')"
        f" and Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt 30.00)"
        f" and Attributes/OData.CSC.StringAttribute/any("
        f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
    )
    params = {
        "$filter": filter_str,
        "$orderby": "ContentDate/Start asc",
        "$top": MAX_S2_PER_PHASE,
        "$expand": "Attributes",
    }
    print(f"\n[S2 L2A <30% cloud | {label}] Searching ...")
    r = requests.get(ODATA_BASE, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("value", [])
    for p in results:
        p["_phase"] = label
    size_mb = sum(p.get("ContentLength", 0) for p in results) / 1e6
    print(f"  Found {len(results)} products  (~{size_mb:.0f} MB total)")
    return results


def main():
    all_s1, all_s2 = [], []
    summary_lines = ["=== Cyclone Ditwah 2025 — Product Search Summary ===\n"]

    for label, (start, end) in {
        "before": DATE_BEFORE,
        "during": DATE_DURING,
        "after":  DATE_AFTER,
    }.items():
        s1 = search_s1_grd(start, end, label)
        s2 = search_s2(start, end, label)
        all_s1.extend(s1)
        all_s2.extend(s2)

        s1_mb = sum(p.get("ContentLength", 0) for p in s1) / 1e6
        s2_mb = sum(p.get("ContentLength", 0) for p in s2) / 1e6
        summary_lines.append(
            f"[{label}]  S1 GRD: {len(s1)} products ({s1_mb:.0f} MB) | "
            f"S2 L2A: {len(s2)} products ({s2_mb:.0f} MB)"
        )

    total_mb = sum(p.get("ContentLength", 0) for p in all_s1 + all_s2) / 1e6
    summary_lines.append(f"\nTotal: {len(all_s1)} S1 + {len(all_s2)} S2 "
                         f"= {len(all_s1)+len(all_s2)} products  (~{total_mb:.0f} MB)")

    # Save results
    with (OUT_DIR / "sentinel1_products.json").open("w") as f:
        json.dump(all_s1, f, indent=2, default=str)
    with (OUT_DIR / "sentinel2_products.json").open("w") as f:
        json.dump(all_s2, f, indent=2, default=str)

    summary_text = "\n".join(summary_lines)
    (OUT_DIR / "search_summary.txt").write_text(summary_text)

    print(f"\n{'='*55}")
    print(summary_text)
    print(f"{'='*55}")
    print("\nNext: run 02_download_products.py")


if __name__ == "__main__":
    main()
