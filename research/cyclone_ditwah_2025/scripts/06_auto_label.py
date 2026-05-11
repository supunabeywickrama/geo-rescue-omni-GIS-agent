"""
Auto-label S1 tiles using SAR backscatter thresholding (Otsu method).

HOW IT WORKS:
  Flood water absorbs radar signal → VV backscatter drops sharply.
  Pixels below a threshold (typically -15 to -20 dB) = flood water.
  Otsu's method finds the optimal threshold automatically per tile.

  before tiles → label 0 everywhere (no flood baseline)
  after/during tiles → Otsu threshold on VV band → 0=land, 1=flood

Output:
  data/labels/S1/<phase>/<tile>_label.tif   (0=no flood, 1=flood)

Usage:
    python 06_auto_label.py
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from skimage.filters import threshold_otsu

TILES_DIR  = Path(__file__).parent.parent / "data" / "tiles"
LABELS_DIR = Path(__file__).parent.parent / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def label_before_tile(tile_path: Path, label_path: Path):
    """Before-phase tile: all pixels = 0 (no flood)."""
    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(label_path, "w", **profile) as dst:
            dst.write(np.zeros((1, src.height, src.width), dtype=np.uint8))


def label_flood_tile(tile_path: Path, label_path: Path):
    """
    After/during tile: apply Otsu threshold on VV band (band 1).
    Pixels below threshold = flood water (1). Above = land (0).
    Falls back to fixed -17 dB threshold if Otsu fails.
    """
    with rasterio.open(tile_path) as src:
        vv = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=255)

        # Mask NoData
        valid = np.isfinite(vv) & (vv != 0)
        if valid.sum() < 100:
            # Not enough valid pixels — skip
            return

        vv_valid = vv[valid]

        try:
            thresh = threshold_otsu(vv_valid)
            # Otsu on SAR sometimes picks too high — cap at -12 dB
            thresh = min(thresh, -12.0)
        except Exception:
            thresh = -17.0   # standard SAR water threshold

        mask = np.where(valid & (vv < thresh), 1, 0).astype(np.uint8)

        label_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(label_path, "w", **profile) as dst:
            dst.write(mask[np.newaxis, :, :])


def main():
    manifest_path = TILES_DIR / "manifest.csv"
    if not manifest_path.exists():
        print("Run 04_tile_dataset.py first.")
        return

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))

    # Only S1 tiles, skip unknown phase
    rows = [r for r in rows if r["source"] == "S1" and r["phase"] != "unknown"]
    print(f"Labeling {len(rows)} S1 tiles ...\n")

    counts = {"before": 0, "during": 0, "after": 0, "skipped": 0}

    for i, row in enumerate(rows):
        tile_path  = TILES_DIR.parent / row["tile"]
        label_path = LABELS_DIR / row["tile"].replace("tiles/", "labels/").replace(".tif", "_label.tif")

        if label_path.exists():
            continue

        phase = row["phase"]
        try:
            if phase == "before":
                label_before_tile(tile_path, label_path)
                counts["before"] += 1
            else:
                label_flood_tile(tile_path, label_path)
                counts[phase] += 1
        except Exception as e:
            counts["skipped"] += 1
            continue

        if i % 5000 == 0:
            print(f"  {i}/{len(rows)} ...")

    print(f"\n{'='*45}")
    print(f"  Labels written:")
    for k, v in counts.items():
        print(f"    {k:<10} {v}")
    print(f"  Output → {LABELS_DIR}")
    print(f"{'='*45}")
    print("Next: run 07_train_model.py")


if __name__ == "__main__":
    main()
