# Cyclone Ditwah 2025 — Flood Training Data (Colombo / Kelani River)

> Sentinel-1 SAR imagery of the Colombo and Kelani River flood zone during and after
> Cyclone Ditwah (Nov 30, 2025). Used to train the GeoRescue flood detection model.

---

## What This Is

Cyclone Ditwah made landfall near Sri Lanka on **November 30, 2025**, causing severe
flooding along the **Kelani River** and across **Colombo** and surrounding areas
(Kelaniya, Kaduwela, Hanwella).

This folder collects **Sentinel-1 SAR radar images** of that specific event to train
a flood detection model. The model output (flood polygon GeoJSON) feeds directly into
the road routing pipeline.

```
Sentinel-1 SAR image (Colombo area)
         │
         ▼
   Flood detection model (trained here)
         │
         ▼
   Flood mask → GeoJSON polygon
         │
         ▼
   flood_overlay.py → routing.py → Safe route
         │
         ▼
   GET /gis/safe-route → Map UI
```

---

## Area of Interest

**Colombo + Kelani River basin**
- Longitude: 79.8° – 80.4°E
- Latitude: 6.7° – 7.3°N
- Covers: Colombo city, Kelani River mouth to Hanwella, Kelaniya, Kaduwela

---

## Data — Sentinel-1 GRD

| Phase | Dates | Purpose |
|-------|-------|---------|
| **Before** | Nov 15 – Nov 27, 2025 | Dry baseline — roads clear, no flooding |
| **During** | Nov 28 – Dec 2, 2025 | Cyclone landfall — peak flooding |
| **After** | Dec 3 – Dec 10, 2025 | Flood extent post-storm |

**Why Sentinel-1 SAR?**
Cyclones produce 100% cloud cover — optical satellites (Sentinel-2) see nothing.
Sentinel-1 radar penetrates cloud and rain. Flooded areas appear dark in VV/VH bands.

**Why GRD and not SLC?**
GRD = processed amplitude image, ~300 MB per product. Ready for flood mapping.
SLC = raw complex phase data, 7+ GB per product. Wrong type — needs interferometric processing.

---

## Folder Structure

```
cyclone_ditwah_2025/
├── scripts/
│   ├── 01_search_products.py   # Search Copernicus catalogue (no auth)
│   ├── 02_download_products.py # Download via CDSE OAuth token
│   ├── 03_preprocess.py        # Extract VV/VH bands → GeoTIFF
│   ├── 04_tile_dataset.py      # Cut into 256×256 tiles + manifest.csv
│   └── 05_verify_dataset.py    # Check class balance before training
├── data/                       # All gitignored — never committed
│   ├── raw/
│   │   ├── sentinel1_products.json  # Search results
│   │   ├── search_summary.txt
│   │   └── downloads/               # Downloaded .zip products
│   ├── processed/              # Extracted VV/VH GeoTIFFs
│   ├── tiles/                  # 256×256 training tiles + manifest.csv
│   └── labels/                 # Flood masks (auto or manual)
├── .env                        # Credentials — never committed
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Add your Copernicus Data Space credentials to `.env`:

```
CDSE_USERNAME=your_email@example.com
CDSE_PASSWORD=your_password
```

Register free at **dataspace.copernicus.eu** (no payment, active immediately).

---

## Run Order

```bash
# 1. Search catalogue — finds Sentinel-1 GRD over Colombo/Kelani for Nov-Dec 2025
python scripts/01_search_products.py

# 2. Download products (~5-6 GB, ~20-30 min at 5 MB/s)
python scripts/02_download_products.py

# 3. Extract VV and VH bands → GeoTIFF (converts DN to dB scale)
python scripts/03_preprocess.py

# 4. Cut GeoTIFFs into 256×256 tiles, write manifest.csv
python scripts/04_tile_dataset.py

# 5. Check flood vs no-flood tile balance before training
python scripts/05_verify_dataset.py
```

---

## Script Details

### `01_search_products.py`
Queries the free Copernicus OData catalogue. No login needed.
Filters: GRD product type, IW mode, AOI intersects Colombo/Kelani River basin.
Saves product list and sizes to `data/raw/sentinel1_products.json`.

### `02_download_products.py`
Downloads using a CDSE Bearer token (expires every 10 min, auto-refreshed every 9 min).
Files streamed to disk in chunks — no RAM issues. Already-downloaded files skipped on re-run.

### `03_preprocess.py`
Extracts VV and VH bands from each `.zip`, converts raw DN values to dB scale
(`10 × log10(DN²)`). Saves as GeoTIFF. Flooded areas will appear significantly darker
in the during/after phase compared to before.

### `04_tile_dataset.py`
Slides a 256×256 window across each GeoTIFF. Skips tiles that are >60% empty.
Tags each tile with its phase (before / during / after).
Writes `data/tiles/manifest.csv` with path, phase, and coordinates per tile.

### `05_verify_dataset.py`
Reads manifest.csv and prints tile counts per phase.
Checks flood (during+after) vs no-flood (before) ratio.
Warns if imbalanced — imbalanced datasets cause models to predict "no flood" for everything.

---

## Labelling

Tiles need binary flood masks (0 = road clear, 1 = flooded).

**Recommended — automatic SAR change detection:**
Compare VV backscatter between before and during phases.
Flood water causes a strong drop (typically >3 dB darker).
Pixels below a threshold in the during-phase image = flood.

**Manual:** Load VV GeoTIFF in QGIS, draw flood polygons, export as raster mask.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Token not found` | Missing auth header | Set `CDSE_USERNAME` + `CDSE_PASSWORD` in `.env` |
| Files are 7+ GB | Old script fetched SLC — now fixed | Re-run `01_search_products.py` |
| `401 Unauthorized` mid-download | Token expired | Script auto-refreshes; re-run if still fails |
| 0 products found | Date range has no passes | Widen date range in `01_search_products.py` |

---

## Notes

- All `data/` is gitignored — satellite files are never committed to git
- Ensure **20 GB free disk space** before downloading
- Sentinel-1 revisit time over Sri Lanka is ~6 days — expect 2–3 passes per phase
