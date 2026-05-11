# Cyclone Ditwah 2025 — Training Data Collection

> Research folder for collecting Sentinel-1/2 satellite imagery of Cyclone Ditwah (Sri Lanka, Nov 30, 2025) to fine-tune the GeoRescue flood detection model.

---

## Folder Structure

```
cyclone_ditwah_2025/
├── scripts/
│   ├── 01_search_products.py    # Search Copernicus catalogue (no auth needed)
│   ├── 02_download_products.py  # Download products via CDSE OAuth
│   ├── 03_preprocess.py         # Extract bands → GeoTIFF
│   ├── 04_tile_dataset.py       # Cut into 256×256 tiles
│   └── 05_verify_dataset.py     # Check class balance before training
├── data/
│   ├── raw/                     # Downloaded .zip products + search results JSON
│   │   └── downloads/           # Raw Sentinel product zips
│   ├── processed/               # Extracted GeoTIFFs (VV, VH, RGB, NIR)
│   ├── tiles/                   # 256×256 training tiles + manifest.csv
│   └── labels/                  # Flood / no-flood annotation masks
├── notebooks/                   # EDA and visualisation notebooks
├── .env                         # Credentials (not committed)
├── requirements.txt
└── README.md
```

---

## Why This Event

Cyclone Ditwah made landfall near Sri Lanka on **November 30, 2025**, causing significant coastal and inland flooding. This event provides:

- High-quality before / during / after satellite contrast
- Large flood extent across Colombo and southern coastal areas
- Real SAR backscatter change signal (ideal for supervised training)

---

## Data Strategy

| Phase | Date Range | Sensors | Purpose |
|-------|-----------|---------|---------|
| Before | Nov 15 – Nov 27 | Sentinel-1 GRD + Sentinel-2 | Clean baseline (no-flood class) |
| During | Nov 28 – Dec 1 | **Sentinel-1 GRD only** | Active flooding (clouds block optical) |
| After | Dec 2 – Dec 10 | Sentinel-1 GRD + Sentinel-2 | Peak flood extent (clouds clearing) |

### Why Sentinel-1 SAR?
Cyclones produce 100% cloud cover. Sentinel-2 optical images will be black during the storm. **Sentinel-1 SAR radar penetrates cloud and rain** and returns a clear flood signal (water appears dark in VV/VH bands).

### Bands collected

| Sensor | Bands | Use |
|--------|-------|-----|
| Sentinel-1 GRD | VV, VH (converted to dB) | Primary flood detection |
| Sentinel-2 L2A | B02, B03, B04 (RGB) | Visual reference, pre/post |
| Sentinel-2 L2A | B08 (NIR) | NDWI water index |

---

## Understanding the Scripts

### `01_search_products.py` — Catalogue search (no auth, runs instantly)

Queries the Copernicus OData catalogue API to find matching satellite products.
No account needed for searching — only for downloading.

**Key filters applied:**

| Filter | Why |
|--------|-----|
| `productType = GRD` | **Critical.** GRD files are ~300 MB. SLC files are 7+ GB and contain raw complex phase data that needs interferometric processing — wrong for flood mapping. |
| `operationalMode = IW` | IW (Interferometric Wide) mode gives 250 km swath — wide enough to cover all of Sri Lanka in one pass. |
| `cloudCover < 30%` | For Sentinel-2 only. During the cyclone phase, S2 is skipped entirely because cloud cover is 100%. |
| Max 8 S1 + 5 S2 per phase | Caps the download at ~25 GB total. 181 SLC products (what the old script found) = ~1.4 TB — not practical. |

Output: `data/raw/sentinel1_products.json`, `data/raw/sentinel2_products.json`, `data/raw/search_summary.txt`

---

### `02_download_products.py` — Authenticated download

Downloads the products found by script 01 using CDSE OAuth Bearer tokens.

**Why you need a token:**
The download endpoint (`download.dataspace.copernicus.eu`) returns `"Token not found"` without authentication. The script fetches a token at startup and refreshes it every 9 minutes (tokens expire at 10 minutes).

**Why streaming:**
Files are 300 MB–1 GB. Streaming writes chunks directly to disk instead of buffering in RAM — prevents memory crashes on large downloads.

**Safe to resume:**
Already-downloaded `.zip` files are skipped. If the script crashes at product 9, re-running picks up from product 10.

Expected download time at ~5 MB/s:
- ~45 products × ~400 MB average = ~18 GB → roughly **1–1.5 hours**

---

### `03_preprocess.py` — Extract bands to GeoTIFF

Unpacks the downloaded `.zip` archives and extracts the useful bands:

- **Sentinel-1 GRD:** extracts VV and VH TIFF files, converts from raw DN to dB scale (`10 × log10(DN²)`) — this is the standard for SAR flood detection
- **Sentinel-2 L2A:** extracts 10m bands (B02=Blue, B03=Green, B04=Red, B08=NIR) and saves as RGB composite + separate NIR file

Output: `data/processed/*.tif`

---

### `04_tile_dataset.py` — Cut into 256×256 training tiles

Slides a 256×256 window over each GeoTIFF and saves each window as a separate tile.

- Skips tiles that are >60% empty/NoData (ocean edges, no-data borders)
- Tags each tile with its phase (before / during / after)
- Writes `data/tiles/manifest.csv` listing every tile with its coordinates and phase label

Output: `data/tiles/S1/before/`, `data/tiles/S1/during/`, etc.

---

### `05_verify_dataset.py` — Check class balance

Reads `manifest.csv` and prints:
- Total tile count per phase and sensor
- Flood (during+after) vs no-flood (before) ratio
- Warning if the dataset is badly imbalanced (ratio > 2× or < 0.5×)

A balanced dataset prevents the model from just predicting "no flood" for everything.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Edit `.env` with your Copernicus Data Space account:

```
CDSE_USERNAME=your_email@example.com
CDSE_PASSWORD=your_password
```

Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu) — no payment, active immediately after email verification.

> Your existing `SENTINEL_CLIENT_ID` / `SENTINEL_CLIENT_SECRET` are for **Sentinel Hub** (sentinelhub.com) — a different service. They won't work for direct product download here.

---

## Run Order

```bash
# Step 1 — Search (no auth, instant)
python scripts/01_search_products.py

# Step 2 — Download (~1–1.5 hours, ~18 GB)
python scripts/02_download_products.py

# Step 3 — Extract bands to GeoTIFF
python scripts/03_preprocess.py

# Step 4 — Cut into 256×256 tiles
python scripts/04_tile_dataset.py

# Step 5 — Verify class balance
python scripts/05_verify_dataset.py
```

---

## Expected Output

After running all scripts:

```
data/tiles/
├── S1/
│   ├── before/    ← no-flood class  (~300-500 MB)
│   ├── during/    ← flood class     (~300-500 MB)
│   └── after/     ← flood class     (~300-500 MB)
├── S2/
│   ├── before/
│   └── after/
└── manifest.csv
```

Target: **~2,000–5,000 tiles** split roughly 50/50 flood vs no-flood.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Token not found` | Missing Bearer token | Make sure CDSE_USERNAME and CDSE_PASSWORD are set in `.env` |
| Download is 7+ GB per file | Old script grabbed SLC products | Re-run `01_search_products.py` (now fixed to GRD only) |
| `401 Unauthorized` mid-download | Token expired | Script auto-refreshes every 9 min; if it still fails, re-run |
| S2 images are black | Cloud cover during cyclone | Expected — S2 is not searched for the "during" phase |

---

## Labelling

Tiles in `data/labels/` should contain binary flood masks (0 = no flood, 1 = flood).

Options:
- **Automatic** — threshold SAR backscatter change (before VV vs during VV). Dark areas in during-phase = water = flood.
- **Manual** — use QGIS or Label Studio to draw masks
- **Reference dataset** — [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) has hand-labeled flood masks for Sentinel-1

---

## Notes

- All `data/` directories are gitignored — do not commit raw satellite files
- Ensure at least **50 GB free disk space** before starting downloads
- Products are downloaded S1-first since SAR is the primary training signal
