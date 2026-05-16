# Model Training Handoff — Cyclone Ditwah Flood Detection

> For the team member responsible for training the U-Net flood segmentation model.
> All data has been collected, preprocessed, tiled, and auto-labeled. Your job starts at Step 3.

---

## What Has Already Been Done (your input)

| Step | Done by | Output |
|------|---------|--------|
| Download Sentinel-1 SAR (Colombo/Kelani, Nov–Dec 2025) | Supun | `data/raw/downloads/` |
| Extract VV/VH bands → GeoTIFF | Supun | `data/processed/` (42 GB) |
| Cut into 256×256 training tiles | Supun | `data/tiles/S1/` (36.6 GB) |
| Auto-label flood masks (Otsu SAR threshold) | Supun | `data/labels/` (1.2 GB) |

**You only need tiles + labels + scripts. You do NOT need raw downloads or processed GeoTIFFs.**

---

## Files to Share With You

### Transfer via Google Drive / USB / Cloud Storage

```
📦 Share these two folders (~38 GB total):

data/tiles/S1/
├── before/     53,312 tiles   (no-flood baseline)
├── during/     25,480 tiles   (cyclone active)
└── after/      53,312 tiles   (post-cyclone flooding)

data/labels/
├── S1/before/  53,312 masks   (all zeros — no flood)
├── S1/during/  25,480 masks   (Otsu flood masks)
└── S1/after/   53,312 masks   (Otsu flood masks)

data/tiles/manifest.csv          (index of all tiles with phase/location)
```

### From Git (already in the repo — no transfer needed)

```
research/cyclone_ditwah_2025/
├── scripts/
│   ├── 07_train_model.py          ← TRAIN the model
│   └── 08_predict_to_geojson.py   ← RUN inference + export GeoJSON
└── TRAINING_HANDOFF.md            ← this file
```

---

## Tile Format

Each tile is a **256 × 256 GeoTIFF** with 2 bands:
- Band 1 = **VV** (vertical-vertical polarisation, Float32, dB scale)
- Band 2 = **VH** (vertical-horizontal polarisation, Float32, dB scale)

Value range: roughly **−30 dB to 0 dB**
- Open water (flood) → very dark, typically **< −17 dB**
- Urban / vegetation → brighter, typically **−10 to −5 dB**

Each label mask is a **256 × 256 single-band GeoTIFF**:
- `0` = no flood (land / road clear)
- `1` = flood water detected
- `255` = no data

---

## Dataset Summary

| Phase | Tiles | Labels | Class |
|-------|-------|--------|-------|
| before | 53,312 | all 0 | no-flood |
| during | 25,480 | Otsu mask | flood |
| after | 53,312 | Otsu mask | flood |
| **Total** | **132,104** | **132,104** | — |

**Class imbalance:** ~0.39 (more no-flood than flood).
The training script handles this with balanced sampling (equal flood/no-flood batches).

---

## Setup

```bash
pip install torch torchvision rasterio numpy scikit-image
```

For AMD MI300X GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

Place the shared data so your directory looks like:
```
research/cyclone_ditwah_2025/
├── data/
│   ├── tiles/S1/before/  ← tiles here
│   ├── tiles/S1/during/
│   ├── tiles/S1/after/
│   ├── tiles/manifest.csv
│   └── labels/S1/...     ← labels here
└── scripts/
    ├── 07_train_model.py
    └── 08_predict_to_geojson.py
```

---

## Step 1 — Train the Model

```bash
cd research/cyclone_ditwah_2025
python scripts/07_train_model.py
```

**What it does:**
- Loads VV+VH tile pairs with their flood masks
- Balances flood vs no-flood samples (20,000 pairs by default)
- Trains a U-Net with BCE + Dice loss for 30 epochs
- Saves best model to `data/models/flood_unet.pth`

**To tune:**

| Parameter | Location in script | Default | Suggestion |
|-----------|-------------------|---------|------------|
| `MAX_TILES` | line ~36 | 20,000 | Increase to 50,000+ for better accuracy |
| `EPOCHS` | line ~37 | 30 | 50–100 for full training |
| `BATCH_SIZE` | line ~34 | 16 | 32 on MI300X |
| `LR` | line ~38 | 1e-4 | Try 3e-4 with warmup |

Expected training time: ~2–4 hours on MI300X at 20k tiles / 30 epochs.

---

## Step 2 — Run Inference + Export GeoJSON

After training, run the model on a new Sentinel-1 image and export the flood polygon:

```bash
python scripts/08_predict_to_geojson.py \
    --input data/processed/S1A_IW_GRDH_..._VV.tif \
    --out flood_polygon.geojson
```

**What it does:**
- Slides the model across the full image in 256×256 windows
- Outputs a binary flood mask
- Polygonizes the mask into a GeoJSON FeatureCollection
- Automatically copies the GeoJSON to `ml_serving/data/processed/live_flood_polygon.geojson`
  so the road routing API picks it up immediately

**After running, Supun triggers the routing update:**
```bash
curl -X POST http://localhost:9000/gis/run-cycle
curl http://localhost:9000/gis/safe-route
```

---

## What the Output Connects To

```
flood_polygon.geojson (your model output)
         │
         ▼
ml_serving/gis_pipeline/flood_overlay.py
  → identifies which Colombo roads are blocked
         │
         ▼
ml_serving/gis_pipeline/routing.py
  → computes safest route avoiding blocked roads
         │
         ▼
GET /gis/safe-route  →  Map UI
```

---

## Deliver Back

When training is complete, share back:

```
data/models/flood_unet.pth    ← trained model weights
```

And optionally a short note on:
- Final validation loss
- Approximate flood pixel accuracy on a held-out after-phase tile
- Any hyperparameter changes made
