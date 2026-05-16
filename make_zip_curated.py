"""
Create a curated training zip with balanced VV tiles (25k flood + 25k no-flood).
Output: C:/Users/LENOVO/cyclone_ditwah_training.zip
"""
import csv
import random
import zipfile
from pathlib import Path

SEED = 42
MAX_PER_CLASS = 25000
OUT = Path(r"C:\Users\LENOVO\cyclone_ditwah_training.zip")

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "research" / "cyclone_ditwah_2025" / "data"
TILES_DIR  = DATA_DIR / "tiles"
LABELS_DIR = DATA_DIR / "labels"
SCRIPTS_DIR = ROOT / "research" / "cyclone_ditwah_2025" / "scripts"
MANIFEST   = TILES_DIR / "manifest.csv"

random.seed(SEED)

# ── Read manifest ──────────────────────────────────────────────────────────────
with MANIFEST.open() as f:
    rows = list(csv.DictReader(f))

flood_pairs    = []
no_flood_pairs = []

for row in rows:
    if row["source"] != "S1" or row["phase"] == "unknown":
        continue
    tile_rel = Path(row["tile"])  # tiles\S1\before\xxx.tif  (Windows backslash OK)
    if "_VH_" in tile_rel.name:
        continue

    tile_path  = DATA_DIR / tile_rel
    label_path = LABELS_DIR / tile_rel.parent / (tile_rel.stem + "_label.tif")

    if not tile_path.exists() or not label_path.exists():
        continue

    if row["phase"] in ("during", "after"):
        flood_pairs.append((tile_path, label_path, tile_rel))
    elif row["phase"] == "before":
        no_flood_pairs.append((tile_path, label_path, tile_rel))

n = min(len(flood_pairs), len(no_flood_pairs), MAX_PER_CLASS)
flood_sel    = random.sample(flood_pairs, n)
no_flood_sel = random.sample(no_flood_pairs, n)
selected     = flood_sel + no_flood_sel

print(f"Flood pairs available  : {len(flood_pairs)}")
print(f"No-flood pairs available: {len(no_flood_pairs)}")
print(f"Selected               : {n} flood + {n} no-flood = {len(selected)} total pairs")

if n == 0:
    print("ERROR: no valid pairs found. Check that tile and label files exist.")
    raise SystemExit(1)

# ── Build zip ─────────────────────────────────────────────────────────────────
total = len(selected) * 2  # tiles + labels
written = 0
print(f"\nWriting to {OUT} ...")

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:

    for tile_path, label_path, tile_rel in selected:
        # Forward-slash arc names for Linux compatibility
        tile_arc  = str(tile_rel).replace("\\", "/")
        # label lives at data/labels/tiles/S1/<phase>/<stem>_label.tif
        # tile_rel = tiles/S1/<phase>/<stem>.tif  →  prepend "data/labels/"
        label_arc = "data/labels/" + str(tile_rel.parent / (tile_rel.stem + "_label.tif")).replace("\\", "/")

        zf.write(tile_path,  tile_arc)
        written += 1
        zf.write(label_path, label_arc)
        written += 1

        if written % 4000 == 0:
            pct = written * 100 // total
            print(f"  {written}/{total} files ({pct}%)")

    # Manifest
    zf.write(MANIFEST, "data/tiles/manifest.csv")

    # Scripts
    for script in sorted(SCRIPTS_DIR.glob("*.py")):
        zf.write(script, f"scripts/{script.name}")

    # Handoff doc
    handoff = ROOT / "research" / "cyclone_ditwah_2025" / "TRAINING_HANDOFF.md"
    if handoff.exists():
        zf.write(handoff, "TRAINING_HANDOFF.md")
    colab_guide = ROOT / "COLAB_TRAINING_GUIDE.md"
    if colab_guide.exists():
        zf.write(colab_guide, "COLAB_TRAINING_GUIDE.md")

size_gb = OUT.stat().st_size / 1e9
print(f"\nDone!")
print(f"Zip size : {size_gb:.2f} GB")
print(f"Path     : {OUT}")
