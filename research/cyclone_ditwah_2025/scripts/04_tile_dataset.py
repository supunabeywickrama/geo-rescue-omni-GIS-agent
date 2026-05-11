"""
Cut processed Sentinel-1 GeoTIFFs into 256x256 tiles for model training.
S2 files are skipped — only S1 VV/VH bands are needed for SAR flood detection.

Phase detection uses the 8-digit date embedded in the S1 filename (YYYYMMDD):
  before : 20251115 – 20251127
  during : 20251128 – 20251202
  after  : 20251203 – 20251210

Usage:
    python 04_tile_dataset.py

Output:
    data/tiles/S1/<phase>/<tile>.tif
    data/tiles/manifest.csv
"""

import csv
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
TILES_DIR     = Path(__file__).parent.parent / "data" / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 256
STRIDE    = 256


def detect_phase(filename: str) -> str:
    """Extract YYYYMMDD from filename, compare to phase date ranges."""
    match = re.search(r"(\d{8})T\d{6}", filename)   # e.g. 20251130T012345
    if not match:
        match = re.search(r"(\d{8})", filename)      # fallback: any 8-digit run
    if not match:
        return "unknown"

    date = int(match.group(1))
    if 20251115 <= date <= 20251127:
        return "before"
    if 20251128 <= date <= 20251202:
        return "during"
    if 20251203 <= date <= 20251210:
        return "after"
    return "unknown"


def tile_file(tif_path: Path, manifest_rows: list):
    # Skip S2 entirely — not useful for SAR flood model
    if "_VV" not in tif_path.stem and "_VH" not in tif_path.stem:
        print(f"  [skip S2] {tif_path.name}")
        return

    phase   = detect_phase(tif_path.stem)
    out_dir = TILES_DIR / "S1" / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        w, h    = src.width, src.height
        profile = src.profile.copy()
        profile.update(width=TILE_SIZE, height=TILE_SIZE, driver="GTiff")

        tile_idx = 0
        for row in range(0, h - TILE_SIZE + 1, STRIDE):
            for col in range(0, w - TILE_SIZE + 1, STRIDE):
                window = Window(col, row, TILE_SIZE, TILE_SIZE)
                data   = src.read(window=window)

                # Skip tiles that are >60% NoData
                if np.count_nonzero(np.isfinite(data) & (data != 0)) < 0.4 * data.size:
                    continue

                transform = src.window_transform(window)
                profile.update(transform=transform, count=data.shape[0])

                tile_name = f"{tif_path.stem}_{tile_idx:05d}.tif"
                tile_path = out_dir / tile_name
                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(data)

                manifest_rows.append({
                    "tile":   str(tile_path.relative_to(TILES_DIR.parent)),
                    "source": "S1",
                    "phase":  phase,
                    "parent": tif_path.name,
                    "row":    row,
                    "col":    col,
                })
                tile_idx += 1

    print(f"  {tif_path.name} → {tile_idx} tiles  [{phase}]")


def main():
    tifs = sorted(PROCESSED_DIR.glob("*.tif"))
    if not tifs:
        print("No processed TIFFs found. Run 03_preprocess.py first.")
        return

    print(f"Processing {len(tifs)} TIFFs (S2 will be skipped) ...\n")
    manifest_rows = []
    for tif in tifs:
        tile_file(tif, manifest_rows)

    manifest_path = TILES_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["tile", "source", "phase", "parent", "row", "col"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary
    from collections import Counter
    phases = Counter(r["phase"] for r in manifest_rows)
    print(f"\n{'='*45}")
    print(f"  S1 tiles written : {len(manifest_rows)}")
    for phase, n in sorted(phases.items()):
        print(f"    {phase:<10} {n}")
    print(f"  Manifest → {manifest_path}")
    print(f"{'='*45}")
    print("Next: run 06_auto_label.py")


if __name__ == "__main__":
    main()
