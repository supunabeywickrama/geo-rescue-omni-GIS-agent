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
| Before | Nov 15 – Nov 27 | Sentinel-1 + Sentinel-2 | Clean baseline (no-flood class) |
| During | Nov 28 – Dec 1 | **Sentinel-1 only** | Active flooding (clouds block optical) |
| After | Dec 2 – Dec 10 | Sentinel-1 + Sentinel-2 | Peak flood extent (clouds clearing) |

### Why Sentinel-1 SAR?
Cyclones produce 100% cloud cover. Sentinel-2 optical images will be black during the storm. **Sentinel-1 SAR radar penetrates cloud and rain** and returns a clear flood signal (water appears dark in VV/VH bands).

### Bands collected

| Sensor | Bands | Use |
|--------|-------|-----|
| Sentinel-1 GRD | VV, VH (converted to dB) | Primary flood detection |
| Sentinel-2 L2A | B02, B03, B04 (RGB) | Visual reference, pre/post |
| Sentinel-2 L2A | B08 (NIR) | NDWI water index |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

Edit `.env`:

```
CDSE_USERNAME=your_email@example.com
CDSE_PASSWORD=your_password
```

Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).
Your existing Sentinel Hub credentials (`SENTINEL_CLIENT_ID`) are for a different service and won't work for direct download here.

---

## Run Order

```bash
# Step 1 — Search catalogue (no auth, runs immediately)
python scripts/01_search_products.py

# Step 2 — Download products (needs CDSE credentials in .env)
python scripts/02_download_products.py

# Step 3 — Extract bands to GeoTIFF
python scripts/03_preprocess.py

# Step 4 — Cut into 256×256 tiles
python scripts/04_tile_dataset.py

# Step 5 — Verify class balance before training
python scripts/05_verify_dataset.py
```

---

## Expected Output

After running all scripts:

```
data/tiles/
├── S1/
│   ├── before/    ← no-flood class
│   ├── during/    ← flood class
│   └── after/     ← flood class
├── S2/
│   ├── before/
│   └── after/
└── manifest.csv   ← tile path, phase, source, coordinates
```

Target: **~2,000–5,000 tiles** split roughly 50/50 flood vs no-flood.

---

## Labelling

Tiles in `data/labels/` should contain binary flood masks (0 = no flood, 1 = flood).

Options for labelling:
- **Automatic** — threshold SAR backscatter change (before vs during)
- **Manual** — use QGIS or Label Studio to draw masks
- **Reference** — use [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) labels if AOI overlaps

---

## Notes

- All data directories (`data/`) are gitignored — do not commit raw satellite files
- Token expiry: CDSE tokens last 10 minutes; `02_download_products.py` auto-refreshes every 9 minutes
- Large `.SAFE` archives can be 800 MB–4 GB each — ensure enough disk space before downloading
