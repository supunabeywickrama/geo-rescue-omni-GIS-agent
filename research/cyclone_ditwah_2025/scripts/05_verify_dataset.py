"""
Verify dataset completeness and print a summary before training.

Usage:
    python 05_verify_dataset.py
"""

import csv
from collections import Counter
from pathlib import Path

TILES_DIR = Path(__file__).parent.parent / "data" / "tiles"


def main():
    manifest_path = TILES_DIR / "manifest.csv"
    if not manifest_path.exists():
        print("manifest.csv not found. Run 04_tile_dataset.py first.")
        return

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'='*50}")
    print(f"  Dataset Summary — Cyclone Ditwah 2025")
    print(f"{'='*50}")
    print(f"  Total tiles : {len(rows)}")

    phase_counts  = Counter(r["phase"]  for r in rows)
    source_counts = Counter(r["source"] for r in rows)

    print(f"\n  By phase:")
    for phase, count in sorted(phase_counts.items()):
        print(f"    {phase:<10} {count:>5} tiles")

    print(f"\n  By source:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source:<10} {count:>5} tiles")

    # Check class balance (before vs during+after)
    flood_phases = {"during", "after"}
    flood   = sum(1 for r in rows if r["phase"] in flood_phases)
    no_flood = len(rows) - flood
    ratio = flood / no_flood if no_flood > 0 else float("inf")

    print(f"\n  Class balance (approx):")
    print(f"    Flood    (during+after) : {flood}")
    print(f"    No-flood (before)       : {no_flood}")
    print(f"    Ratio                   : {ratio:.2f}  (target ≈ 1.0)")

    if ratio > 2 or ratio < 0.5:
        print("  WARNING: Dataset is imbalanced. Consider:")
        print("    - Oversampling minority class")
        print("    - Weighted loss in training")
        print("    - Collecting more data for the smaller class")
    else:
        print("  Balance looks good.")

    print(f"\n{'='*50}")
    print("  Ready for training. Good luck!")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
