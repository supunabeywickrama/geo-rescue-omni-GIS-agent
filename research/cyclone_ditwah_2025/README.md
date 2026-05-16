# Cyclone Ditwah 2025 — Flood Training Data (Colombo / Kelani River)

> Sentinel-1 SAR imagery of the Colombo and Kelani River flood zone during and after
> Cyclone Ditwah (Nov 30, 2025). Used to fine-tune the Qwen2.5-VL-7B flood damage
> assessment model for the GeoRescue multi-agent disaster response system.

---

## What This Is

Cyclone Ditwah made landfall near Sri Lanka on **November 30, 2025**, causing severe
flooding along the **Kelani River** and across **Colombo** and surrounding areas
(Kelaniya, Kaduwela, Hanwella).

This folder collects **Sentinel-1 SAR radar images** of that specific event to
fine-tune Qwen2.5-VL-7B via QLoRA on AMD Instinct MI300X. The fine-tuned model
performs satellite image damage classification and returns structured JSON with
severity, flood fraction, and damage-zone GeoJSON polygons.

```
Sentinel-1 SAR tile (256×256, VV+VH dB)
         │
         ▼
   Qwen2.5-VL-7B (fine-tuned, checkpoints/final/)
         │
         ▼
   { severity, flood_fraction, damage_description, affected_zone }
         │
         ▼
   flood_overlay.py → routing.py → Safe evacuation route
         │
         ▼
   GET /gis/safe-route → Streamlit + Folium map UI
```

---

## Area of Interest

**Colombo + Kelani River basin**
- Longitude: 79.8° – 80.4°E
- Latitude: 6.7° – 7.3°N
- Covers: Colombo city, Kelani River mouth to Hanwella, Kelaniya, Kaduwela

---

## Dataset

| Phase | Dates | Tiles | Purpose |
|-------|-------|-------|---------|
| **Before** | Nov 15 – Nov 27, 2025 | 53,312 | Dry baseline — no flood |
| **During** | Nov 28 – Dec 2, 2025 | 25,480 | Cyclone active — peak flooding |
| **After** | Dec 3 – Dec 10, 2025 | 53,312 | Post-storm flood extent |

**Training split actually used (balanced):**
- 25,000 flood (during 8,047 + after 16,953) + 25,000 no-flood (before) = **50,000 pairs**
- Train / Val: 45,000 / 5,000

**Why Sentinel-1 SAR?**
Cyclones produce 100% cloud cover — optical satellites (Sentinel-2) see nothing.
Sentinel-1 radar penetrates cloud and rain. Flooded areas appear dark in VV/VH bands.

**Labels:** Binary Otsu flood masks (0 = land, 1 = flood). Flood fraction from mask
→ severity mapping: `<5% → none`, `<15% → low`, `<35% → medium`, `<60% → high`, `≥60% → critical`.

---

## Folder Structure

```
cyclone_ditwah_2025/
├── scripts/
│   ├── 01_search_products.py     # Search Copernicus catalogue
│   ├── 02_download_products.py   # Download via CDSE OAuth token
│   ├── 03_preprocess.py          # Extract VV/VH bands → GeoTIFF
│   ├── 04_tile_dataset.py        # Cut into 256×256 tiles + manifest.csv
│   ├── 05_verify_dataset.py      # Check class balance
│   ├── 06_label_tiles.py         # Generate Otsu flood masks
│   ├── 07_train_model.py         # U-Net flood segmentation (preprocessing helper)
│   ├── 08_predict_to_geojson.py  # U-Net inference → GeoJSON
│   └── 09_finetune_qwen2vl.py   # Qwen2.5-VL-7B QLoRA fine-tuning (main model)
├── checkpoints/                  # Training output — gitignored
│   ├── epoch_01/                 # LoRA adapter after epoch 1
│   ├── epoch_02/                 # LoRA adapter after epoch 2
│   ├── epoch_03/                 # LoRA adapter after epoch 3 (last saved)
│   └── final/                    # Merged model — ready for inference
├── data/                         # All gitignored — never committed
│   ├── raw/                      # Downloaded .zip products
│   ├── processed/                # Extracted VV/VH GeoTIFFs
│   ├── tiles/                    # 256×256 tiles + manifest.csv
│   └── labels/                   # Otsu flood masks
├── TRAINING_HANDOFF.md           # Step-by-step server training guide
├── .env                          # Credentials — never committed
├── requirements.txt
└── README.md
```

---

## Model Training

### Architecture
- **Base model:** `Qwen/Qwen2.5-VL-7B-Instruct`
- **Method:** QLoRA (4-bit quantization) via PEFT
- **LoRA config:** r=16, alpha=32, target=q/k/v/o projections, dropout=0.05
- **Trainable params:** 10,092,544 / 8,302,259,200 (0.12%)
- **Hardware:** AMD Instinct MI300X VF — 205.8 GB VRAM
- **Batch:** BATCH_SIZE=16, GRAD_ACCUM=1 → effective batch=16
- **Epochs:** 5 (trained 3, merged from epoch_03)
- **LR:** 2e-4 with CosineAnnealingLR

### SAR → RGB conversion for VLM
```
R = VV normalized  (primary flood indicator)
G = VH normalized
B = VV/VH ratio    (flood water has distinctively low ratio)
```

### Run training
```bash
pip install "transformers==4.49.0" "peft==0.12.0" bitsandbytes accelerate tqdm rasterio
python scripts/09_finetune_qwen2vl.py
```

### Load fine-tuned model for inference
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "checkpoints/final",
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("checkpoints/final")
```

---

## Data Pipeline (reproduce from scratch)

```bash
# 1. Search catalogue — finds Sentinel-1 GRD over Colombo/Kelani
python scripts/01_search_products.py

# 2. Download products (~5-6 GB, ~20-30 min at 5 MB/s)
python scripts/02_download_products.py

# 3. Extract VV and VH bands → GeoTIFF (converts DN to dB scale)
python scripts/03_preprocess.py

# 4. Cut GeoTIFFs into 256×256 tiles, write manifest.csv
python scripts/04_tile_dataset.py

# 5. Check flood vs no-flood tile balance
python scripts/05_verify_dataset.py

# 6. Generate Otsu flood masks
python scripts/06_label_tiles.py

# 7. Fine-tune Qwen2.5-VL-7B
python scripts/09_finetune_qwen2vl.py
```

Add credentials to `.env`:
```
CDSE_USERNAME=your_email@example.com
CDSE_PASSWORD=your_password
```
Register free at **dataspace.copernicus.eu**.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Valid pairs found: 0` | Windows backslash paths in manifest | Script normalizes automatically — ensure updated `09` script |
| `No GPU detected` | CPU-only PyTorch installed | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |
| `transformers moe.py ValueError` | transformers ≥ 4.52 breaks on ROCm PyTorch | Pin `transformers==4.49.0` |
| `Token not found` | Missing auth header | Set `CDSE_USERNAME` + `CDSE_PASSWORD` in `.env` |
| `401 Unauthorized` mid-download | Token expired | Script auto-refreshes; re-run if still fails |
| `No space left on device` | D: drive full during zip | Write zip to C: drive |

---

## Notes

- All `data/` and `checkpoints/` are gitignored — never committed
- Ensure **20 GB free disk space** before downloading raw products
- Sentinel-1 revisit time over Sri Lanka: ~6 days (2–3 passes per phase)
- `07_train_model.py` trains a U-Net segmentation model — not the paper's main model.
  The paper model is Qwen2.5-VL fine-tuned via `09_finetune_qwen2vl.py`
