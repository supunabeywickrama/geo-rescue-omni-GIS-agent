"""
Cut processed GeoTIFFs into 256×256 tiles for model training.

Usage:
    python 04_tile_dataset.py

Output:
    data/tiles/<source>/<phase>/<tile_index>.tif
    data/tiles/manifest.csv   — lists every tile with phase + source
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
TILES_DIR     = Path(__file__).parent.parent / "data" / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZE = 256
STRIDE    = 256   # set < TILE_SIZE for overlap

PHASE_KEYWORDS = {
    "before": ["2025111", "2025112", "2025113"],   # Nov 15–27
    "during": ["2025112[89]", "20251130", "20251201"],
    "after":  ["20251202", "20251203", "20251204", "20251205"],
}


def detect_phase(filename: str) -> str:
    for phase, prefixes in PHASE_KEYWORDS.items():
        for p in prefixes:
            if p in filename:
                return phase
    return "unknown"


def tile_file(tif_path: Path, manifest_rows: list):
    phase  = detect_phase(tif_path.stem)
    source = "S1" if "_VV" in tif_path.stem or "_VH" in tif_path.stem else "S2"
    out_dir = TILES_DIR / source / phase
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        width, height = src.width, src.height
        profile = src.profile.copy()
        profile.update(width=TILE_SIZE, height=TILE_SIZE, driver="GTiff")

        tile_idx = 0
        for row in range(0, height - TILE_SIZE + 1, STRIDE):
            for col in range(0, width - TILE_SIZE + 1, STRIDE):
                window = Window(col, row, TILE_SIZE, TILE_SIZE)
                data   = src.read(window=window)

                # Skip near-empty tiles (>60% NoData/zero)
                if np.count_nonzero(data) < 0.4 * data.size:
                    continue

                transform = src.window_transform(window)
                profile.update(transform=transform, count=data.shape[0])

                tile_name = f"{tif_path.stem}_{tile_idx:05d}.tif"
                tile_path = out_dir / tile_name
                with rasterio.open(tile_path, "w", **profile) as dst:
                    dst.write(data)

                manifest_rows.append({
                    "tile":   str(tile_path.relative_to(TILES_DIR.parent)),
                    "source": source,
                    "phase":  phase,
                    "parent": tif_path.name,
                    "row":    row,
                    "col":    col,
                })
                tile_idx += 1

    print(f"  {tif_path.name} → {tile_idx} tiles [{phase}]")


def main():
    tifs = sorted(PROCESSED_DIR.glob("*.tif"))
    if not tifs:
        print("No processed TIFFs found. Run 03_preprocess.py first.")
        return

    manifest_rows = []
    for tif in tifs:
        tile_file(tif, manifest_rows)

    manifest_path = TILES_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tile", "source", "phase", "parent", "row", "col"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n{len(manifest_rows)} tiles written.")
    print(f"Manifest → {manifest_path}")
    print("Next: manually label flood tiles in data/labels/, then run 05_verify_dataset.py")


if __name__ == "__main__":
    main()
