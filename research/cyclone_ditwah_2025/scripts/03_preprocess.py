"""
Preprocess downloaded Sentinel products:
  - Sentinel-1 GRD: extract VV+VH bands → GeoTIFF
  - Sentinel-2 L2A: extract B02, B03, B04 (RGB) + B08 (NIR) → GeoTIFF

Usage:
    python 03_preprocess.py

Output:
    data/processed/<product_name>_VV.tif   (S1)
    data/processed/<product_name>_VH.tif   (S1)
    data/processed/<product_name>_RGB.tif  (S2)
    data/processed/<product_name>_NIR.tif  (S2)
"""

import zipfile
import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw" / "downloads"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def extract_s1(zip_path: Path):
    """Extract VV and VH intensity bands from a Sentinel-1 GRD .zip."""
    print(f"\n[S1] {zip_path.stem}")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        for pol in ("VV", "VH"):
            tiff_candidates = [n for n in names
                               if f"-{pol.lower()}-" in n.lower() and n.endswith(".tiff")]
            if not tiff_candidates:
                print(f"  WARNING: no {pol} band found, skipping.")
                continue

            # Extract first matching file
            tiff_name = tiff_candidates[0]
            extracted = z.extract(tiff_name, RAW_DIR)
            src_path  = RAW_DIR / tiff_name

            out_path = PROCESSED_DIR / f"{zip_path.stem}_{pol}.tif"
            with rasterio.open(src_path) as src:
                profile = src.profile.copy()
                profile.update(driver="GTiff", compress="lzw")
                data = src.read(1).astype(np.float32)
                # Convert DN to dB: 10 * log10(DN²) — avoids log(0)
                db = 10 * np.log10(np.maximum(data ** 2, 1e-10))
                profile["dtype"] = "float32"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(db, 1)
            src_path.unlink(missing_ok=True)
            print(f"  → {out_path.name}")


def extract_s2(zip_path: Path):
    """Extract RGB + NIR from a Sentinel-2 L2A .zip (10m resolution bands)."""
    print(f"\n[S2] {zip_path.stem}")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        band_map = {"B02": "B", "B03": "G", "B04": "R", "B08": "NIR"}
        extracted_bands = {}

        for band_code, label in band_map.items():
            # 10m bands live in R10m folder for L2A
            candidates = [n for n in names
                          if f"_{band_code}_10m.jp2" in n or f"_{band_code}.jp2" in n]
            if not candidates:
                print(f"  WARNING: {band_code} not found.")
                continue
            ext = z.extract(candidates[0], RAW_DIR)
            extracted_bands[label] = RAW_DIR / candidates[0]

        # Save RGB composite
        if all(k in extracted_bands for k in ("R", "G", "B")):
            out_rgb = PROCESSED_DIR / f"{zip_path.stem}_RGB.tif"
            sources = [rasterio.open(extracted_bands[k]) for k in ("R", "G", "B")]
            profile = sources[0].profile.copy()
            profile.update(count=3, driver="GTiff", compress="lzw")
            with rasterio.open(out_rgb, "w", **profile) as dst:
                for i, src in enumerate(sources, 1):
                    dst.write(src.read(1), i)
            for s in sources: s.close()
            print(f"  → {out_rgb.name}")

        # Save NIR band
        if "NIR" in extracted_bands:
            out_nir = PROCESSED_DIR / f"{zip_path.stem}_NIR.tif"
            with rasterio.open(extracted_bands["NIR"]) as src:
                profile = src.profile.copy()
                profile.update(driver="GTiff", compress="lzw")
                with rasterio.open(out_nir, "w", **profile) as dst:
                    dst.write(src.read(1), 1)
            print(f"  → {out_nir.name}")

        # Cleanup extracted jp2s
        for p in extracted_bands.values():
            p.unlink(missing_ok=True)


def main():
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        print("No zips found in data/raw/downloads. Run 02_download_products.py first.")
        return

    for z in zips:
        name = z.stem.upper()
        if "S1" in name or "SENTINEL-1" in name:
            extract_s1(z)
        elif "S2" in name or "SENTINEL-2" in name:
            extract_s2(z)
        else:
            print(f"[skip] Unknown product type: {z.name}")

    print("\nPreprocessing complete. Next: run 04_tile_dataset.py")


if __name__ == "__main__":
    main()
