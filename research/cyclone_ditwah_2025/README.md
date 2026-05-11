# Cyclone Ditwah 2025 — Flood Model Training Data

> Research folder for training a Sentinel-1 SAR flood detection model for Sri Lanka.
> Model output (flood mask GeoJSON) feeds directly into the GeoRescue road routing pipeline.

---

## Project Goal

```
Satellite image (Sentinel-1 SAR)
         │
         ▼
  Flood detection model
  (trained on Sen1Floods11 + Cyclone Ditwah)
         │
         ▼
  Flood mask → GeoJSON polygon
         │
         ▼
  flood_overlay.py  →  routing.py  →  Safe route GeoJSON
         │
         ▼
  GET /gis/safe-route  →  Map UI
```

The model replaces the current Open-Meteo weather approximation with real satellite-based flood detection.

---

## Data Sources

### Source 1 — Sen1Floods11 (primary, already labeled)

**Use this first.** It already contains Sri Lanka.

| Property | Value |
|----------|-------|
| Sri Lanka event | May 2017, Matara district floods |
| Total dataset | 4,831 chips across 11 global flood events |
| Tile size | 512 × 512 px, 10 m resolution |
| S1 bands | VV + VH (Float32, dB units) — ready to use |
| Labels | Binary flood mask: 0 = not water, 1 = flood water |
| Total size | ~14 GB (S1 + QC layers only) |
| Sri Lanka only | ~200 MB |
| Download | Google Cloud Storage (public, free) |

### Source 2 — Cyclone Ditwah 2025 (supplementary, Sri Lanka specific)

More recent Sri Lanka data to improve model accuracy for the current region.
Requires manual or automatic labeling (SAR backscatter change detection).

| Phase | Dates | What it adds |
|-------|-------|--------------|
| Before | Nov 15–27 | No-flood baseline for Colombo/coastal Sri Lanka |
| During | Nov 28–Dec 1 | Active cyclone flooding (SAR only — clouds block optical) |
| After | Dec 2–10 | Peak flood extent |

---

## Folder Structure

```
cyclone_ditwah_2025/
├── scripts/
│   ├── 00_download_sen1floods11.py  # Download primary labeled dataset
│   ├── 01_search_products.py        # Search Copernicus for Ditwah products
│   ├── 02_download_products.py      # Download Ditwah Sentinel-1 GRD zips
│   ├── 03_preprocess.py             # Extract bands → GeoTIFF
│   ├── 04_tile_dataset.py           # Cut into 256×256 tiles + manifest.csv
│   └── 05_verify_dataset.py         # Class balance check
├── data/
│   ├── sen1floods11/                # Sen1Floods11 tiles + labels (gitignored)
│   │   └── HandLabeled/
│   │       ├── S1/   ← *_S1.tif  (VV+VH bands)
│   │       └── QC/   ← *_QC.tif  (flood mask labels)
│   ├── raw/                         # Ditwah raw downloads
│   │   └── downloads/
│   ├── processed/                   # Extracted Ditwah GeoTIFFs
│   ├── tiles/                       # 256×256 training tiles + manifest.csv
│   └── labels/                      # Ditwah flood masks (auto or manual)
├── notebooks/                       # EDA and visualisation
├── .env                             # Credentials (not committed)
├── requirements.txt
└── README.md
```

---

## Run Order

### Phase 1 — Sen1Floods11 (do this first)

```bash
pip install -r requirements.txt

# Sri Lanka chips only (~200 MB, fastest start)
python scripts/00_download_sen1floods11.py --sri-lanka-only

# Or full dataset (14 GB, all 11 global events — better model)
python scripts/00_download_sen1floods11.py
```

Data lands in `data/sen1floods11/HandLabeled/S1/` and `.../QC/`.
These are ready for training immediately — no preprocessing needed.

### Phase 2 — Cyclone Ditwah 2025 (adds Sri Lanka 2025 context)

```bash
# Search catalogue (no auth, instant)
python scripts/01_search_products.py

# Download GRD products (~15-25 GB, ~1-1.5 hours)
python scripts/02_download_products.py

# Extract bands to GeoTIFF
python scripts/03_preprocess.py

# Cut into 256×256 tiles
python scripts/04_tile_dataset.py

# Verify class balance
python scripts/05_verify_dataset.py
```

---

## Understanding the Scripts

### `00_download_sen1floods11.py`

Downloads pre-made, pre-labeled Sentinel-1 flood chips from a public Google Cloud Storage bucket.

- `*_S1.tif` — 2-band GeoTIFF (VV, VH), already in dB, 512×512 px
- `*_QC.tif` — binary flood mask label (-1=NoData, 0=Not Water, 1=Flood)
- Public bucket — no credentials needed
- `--sri-lanka-only` flag downloads only Sri Lanka chips (~200 MB) for a fast start

### `01_search_products.py`

Queries the Copernicus OData catalogue. No authentication needed for search.

Key filters:
- `productType = GRD` — Ground Range Detected, ~300 MB per file. **Not SLC** (7+ GB, wrong type).
- `operationalMode = IW` — 250 km swath, covers all Sri Lanka in one pass.
- `cloudCover < 30%` — Sentinel-2 only. During cyclone, S2 is skipped (100% cloud).
- Max 8 S1 + 5 S2 per phase — caps total at ~25 GB.

### `02_download_products.py`

Downloads using CDSE Bearer token authentication. Tokens expire in 10 minutes — script auto-refreshes every 9 minutes. Files are streamed in chunks (no RAM issues). Already-downloaded files are skipped (safe to resume after crash).

### `03_preprocess.py`

- **S1 GRD:** Extracts VV and VH bands, converts DN → dB scale (`10 × log10(DN²)`)
- **S2 L2A:** Extracts 10 m bands → RGB composite + NIR GeoTIFF

### `04_tile_dataset.py`

Slides a 256×256 window over each GeoTIFF, saves non-empty tiles, writes `manifest.csv` with phase label (before/during/after) per tile.

### `05_verify_dataset.py`

Checks flood vs no-flood tile ratio. Warns if imbalanced (ratio > 2× triggers weighted loss recommendation).

---

## Setup

```bash
pip install -r requirements.txt
```

Edit `.env`:
```
CDSE_USERNAME=your_email@example.com
CDSE_PASSWORD=your_password
```

Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu). Only needed for Cyclone Ditwah download — not for Sen1Floods11.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Token not found` on download | Missing Bearer token | Set CDSE_USERNAME + CDSE_PASSWORD in `.env` |
| Download is 7+ GB per file | Old script grabbed SLC products | Fixed — now GRD only (~300 MB) |
| `401 Unauthorized` mid-download | Token expired | Script auto-refreshes; re-run if it still fails |
| S2 images are all black | Cloud cover during cyclone | Expected — S2 not searched for "during" phase |
| `google-cloud-storage` not found | Missing package | `pip install google-cloud-storage` |

---

## Labelling Ditwah Data

Sen1Floods11 tiles are already labeled. Cyclone Ditwah tiles need labels.

**Automatic method (recommended):**
Compare VV backscatter between before and during phases. Water causes a strong drop in backscatter (typically > 3 dB). Pixels below a threshold in the during-phase image = flood.

**Manual method:** Use QGIS — load the VV GeoTIFF, draw flood polygons, export as raster mask.

**Reference:** [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) labeling methodology paper.

---

## Notes

- All `data/` directories are gitignored — satellite files are never committed
- Ensure **50 GB free disk space** before downloading full Ditwah dataset
- Sen1Floods11 is the faster path — start there, add Ditwah for Sri Lanka specificity
