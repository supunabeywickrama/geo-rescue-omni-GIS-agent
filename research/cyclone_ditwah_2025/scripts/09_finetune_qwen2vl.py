"""
Fine-tune Qwen2.5-VL-7B-Instruct on cyclone SAR tiles via QLoRA (4-bit).

Qwen2.5-VL is the best available Qwen vision-language model (Qwen3 is text-only).
It has stronger visual grounding, better JSON instruction-following, and improved
spatial understanding over Qwen2-VL — making it well-suited for SAR flood analysis.

Input  : VV SAR GeoTIFF tiles + binary flood mask labels (from manifest.csv)
Output : LoRA adapter weights  →  checkpoints/final/

Task   : Given a SAR satellite image, classify flood damage and return structured
         JSON with severity, flood_fraction, and a simplified damage-zone polygon.

Hardware : AMD MI300X / CUDA GPU (T4 16 GB works with QLoRA) / CPU fallback

Usage:
    pip install "transformers==4.49.0" "peft==0.12.0" bitsandbytes accelerate tqdm rasterio
    python 09_finetune_qwen2vl.py

Outputs:
    checkpoints/          — epoch checkpoints
    checkpoints/final/    — LoRA adapter ready for inference
"""

import csv
import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
# The zip layout is:
#   tiles/S1/{before,during,after}/*.tif   (tile root = project root)
#   data/labels/tiles/S1/.../*_label.tif
#   data/tiles/manifest.csv
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
CKPT_DIR   = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST   = DATA_DIR / "tiles" / "manifest.csv"

# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_ID     = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_TILES    = None    # use all available balanced pairs

# Minimum flood fraction for "after" tiles to be counted as flood.
# After tiles with very little water in their Otsu mask are treated as no-flood
# by the severity function anyway — no date filter needed.
AFTER_FLOOD_MIN_FRACTION = 0.05
TRAIN_SPLIT  = 0.90
EPOCHS       = 5    # safe range for LoRA fine-tuning; beyond 5 risks overfitting/forgetting
LR           = 2e-4
MAX_SEQ_LEN  = 1024
SEED         = 42

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Batch size auto-config ─────────────────────────────────────────────────────
# Effective batch (BATCH_SIZE * GRAD_ACCUM) is held at 16 across all hardware.
# Only BATCH_SIZE changes — larger on high-VRAM GPUs for speed, smaller on T4.
# Quality is determined by effective batch, not physical batch size.
def _has_gpu() -> bool:
    if torch.cuda.is_available():
        return True
    # ROCm builds sometimes need an explicit check
    try:
        return torch.version.hip is not None
    except AttributeError:
        return False


def _auto_batch_config() -> tuple:
    if not _has_gpu():
        return 1, 16          # CPU: batch=1, accum=16
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 160:        # AMD MI300X (192 GB)
        return 16, 1          # batch=16, accum=1  → fastest
    elif vram_gb >= 40:       # A100 40 GB
        return 8, 2           # batch=8,  accum=2
    elif vram_gb >= 20:       # A30 / RTX 3090
        return 4, 4           # batch=4,  accum=4
    else:                     # T4 16 GB / RTX 3080
        return 1, 16          # batch=1,  accum=16 → safe

SYSTEM_PROMPT = (
    "You are a flood damage assessment AI. Analyze the SAR satellite image "
    "and return a JSON object with keys: severity (none/low/medium/high/critical), "
    "flood_fraction (0.0-1.0), damage_description (string), and "
    "affected_zone (GeoJSON Polygon or null)."
)

FLOOD_PROMPT = "Assess flood damage in this SAR satellite image and return structured JSON."


# ── SAR tile → PIL Image ───────────────────────────────────────────────────────
def sar_to_pil(tile_path: Path) -> Image.Image:
    with rasterio.open(tile_path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)

    # Normalise dB range [-30, 0] → [0, 1]
    data = np.clip((data + 30) / 30, 0, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)

    if data.shape[0] >= 2:
        vv = data[0]
        vh = data[1]
    else:
        vv = data[0]
        vh = data[0]

    # RGB: R=VV, G=VH, B=VV/VH ratio (flood water has low VV/VH ratio)
    ratio = np.where(vh > 0.01, vv / (vh + 1e-6), 0.5)
    ratio = np.clip(ratio, 0, 1)

    rgb = (np.stack([vv, vh, ratio], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def flood_fraction(label_path: Path) -> float:
    with rasterio.open(label_path) as src:
        mask = src.read(1).astype(np.float32)
    mask = np.nan_to_num(mask, nan=0.0)
    return float(np.clip(mask, 0, 1).mean())


def fraction_to_severity(frac: float, phase: str) -> str:
    if frac < 0.05:
        return "none"
    if phase == "before":
        return "none"
    if frac < 0.15:
        return "low"
    if frac < 0.35:
        return "medium"
    if frac < 0.60:
        return "high"
    return "critical"


def build_label(frac: float, severity: str, tile_rel: str) -> str:
    desc_map = {
        "none":     "No significant flood detected. Roads and infrastructure appear passable.",
        "low":      "Minor flood inundation. Some low-lying areas affected. Most roads passable.",
        "medium":   "Moderate flooding. Significant road segments blocked. Evacuation routes may be limited.",
        "high":     "Severe flooding. Major infrastructure damage. Emergency response required.",
        "critical": "Catastrophic flood inundation. Widespread infrastructure destruction. Immediate evacuation needed.",
    }
    return json.dumps({
        "severity":         severity,
        "flood_fraction":   round(frac, 4),
        "damage_description": desc_map[severity],
        "affected_zone":    None,   # polygon generation done by predict script
    }, ensure_ascii=False)


# ── Dataset ────────────────────────────────────────────────────────────────────
class SARFloodDataset(Dataset):
    def __init__(self, pairs: list, processor):
        self.pairs     = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tile_path, label_path, tile_rel, phase = self.pairs[idx]

        image    = sar_to_pil(tile_path)
        frac     = flood_fraction(label_path)
        severity = fraction_to_severity(frac, phase)
        label    = build_label(frac, severity, tile_rel)

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": FLOOD_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Encode image as PNG bytes for processor
        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        pil_list = [Image.open(buf)]

        inputs = self.processor(
            text=[text],
            images=pil_list,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model_class():
    # Qwen2.5-VL uses Qwen2_5_VLForConditionalGeneration (transformers >= 4.49.0)
    # Fall back to Qwen2VLForConditionalGeneration for older installs
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print("[info] Using Qwen2.5-VL model class")
        return Qwen2_5_VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration
        print("[warn] Qwen2_5_VLForConditionalGeneration not found — using Qwen2VLForConditionalGeneration. "
              "Upgrade transformers: pip install 'transformers>=4.49.0'")
        return Qwen2VLForConditionalGeneration


def load_model_and_processor():
    from transformers import AutoProcessor
    from peft import LoraConfig, TaskType, get_peft_model

    ModelClass = _load_model_class()
    has_gpu = _has_gpu()

    if has_gpu:
        name = torch.cuda.get_device_name(0)
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"[gpu] {name}  {vram} GB VRAM")

        # Try QLoRA (4-bit) — fits T4 16 GB
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = ModelClass.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
            print("[info] QLoRA 4-bit loaded")
        except Exception as e:
            print(f"[warn] QLoRA failed ({e}), falling back to fp16")
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = ModelClass.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        print("[warn] No GPU detected (torch.cuda.is_available()=False, torch.version.hip="
              f"{getattr(torch.version, 'hip', None)})")
        print("[warn] Loading float32 on CPU — training will be very slow.")
        print("[hint] On ROCm servers, ensure PyTorch was installed with ROCm support:")
        print("[hint]   pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = ModelClass.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Path diagnostics ─────────────────────────────────────────────────────
    print(f"ROOT     : {ROOT}")
    print(f"MANIFEST : {MANIFEST}  exists={MANIFEST.exists()}")
    print(f"LABELS   : {LABELS_DIR}  exists={LABELS_DIR.exists()}")
    tiles_root = ROOT / "tiles"
    print(f"TILES    : {tiles_root}  exists={tiles_root.exists()}")

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"manifest.csv not found at {MANIFEST}\n"
            f"Expected zip layout:\n"
            f"  data/tiles/manifest.csv\n"
            f"  tiles/S1/{{before,during,after}}/*.tif\n"
            f"  data/labels/tiles/S1/.../*_label.tif\n"
            f"Make sure you extracted the zip into {ROOT}"
        )

    # ── Load pairs from manifest ──────────────────────────────────────────────
    with MANIFEST.open() as f:
        rows = list(csv.DictReader(f))

    pairs = []
    missing_tile = missing_label = 0
    for row in rows:
        if row["source"] != "S1" or row["phase"] == "unknown":
            continue
        if "_VH_" in row["tile"]:
            continue

        tile_rel   = Path(row["tile"].replace("\\", "/"))                  # normalize Windows backslashes
        tile_path  = ROOT / tile_rel                                       # project_root/tiles/S1/...
        label_path = LABELS_DIR / tile_rel.parent / (tile_rel.stem + "_label.tif")  # data/labels/tiles/S1/...

        if not tile_path.exists():
            missing_tile += 1
        elif not label_path.exists():
            missing_label += 1
        else:
            pairs.append((tile_path, label_path, str(tile_rel), row["phase"]))

    # ── Detailed count report ─────────────────────────────────────────────────
    before_pairs  = [(t,l,r,p) for t,l,r,p in pairs if p == "before"]
    during_pairs  = [(t,l,r,p) for t,l,r,p in pairs if p == "during"]
    after_pairs   = [(t,l,r,p) for t,l,r,p in pairs if p == "after"]
    flood_pairs   = during_pairs + after_pairs

    print(f"\n{'─'*50}")
    print(f"  before (no-flood)          : {len(before_pairs):>6}")
    print(f"  during (flood active)      : {len(during_pairs):>6}")
    print(f"  after  (flood Dec 3–6)     : {len(after_pairs):>6}")
    print(f"  ── total flood             : {len(flood_pairs):>6}")
    print(f"  ── total no-flood          : {len(before_pairs):>6}")
    if missing_tile:
        print(f"  [warn] missing tiles       : {missing_tile:>6}")
    if missing_label:
        print(f"  [warn] missing labels      : {missing_label:>6}")
    print(f"{'─'*50}")

    if len(pairs) == 0:
        for row in rows[:3]:
            if row["source"] == "S1" and "_VH_" not in row["tile"]:
                tile_rel = Path(row["tile"].replace("\\", "/"))
                print(f"  Sample tile path checked : {ROOT / tile_rel}")
                print(f"  Sample label path checked: {LABELS_DIR / tile_rel.parent / (tile_rel.stem + '_label.tif')}")
                break
        raise SystemExit("No valid pairs found. Check paths above and verify zip was extracted correctly.")

    # Balance flood vs no-flood using all available pairs
    n = min(len(flood_pairs), len(before_pairs))
    pairs = random.sample(flood_pairs, n) + random.sample(before_pairs, n)
    print(f"  Training pairs used        : {n:>6} flood + {n} no-flood = {len(pairs)} total")
    print(f"  Train / Val split (90/10)  : {int(len(pairs)*TRAIN_SPLIT)} / {len(pairs)-int(len(pairs)*TRAIN_SPLIT)}")
    print(f"{'─'*50}\n")

    random.shuffle(pairs)
    split      = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:split]
    val_pairs   = pairs[split:]

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model_and_processor()
    device = next(model.parameters()).device

    BATCH_SIZE, GRAD_ACCUM = _auto_batch_config()
    print(f"Batch config: BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM={GRAD_ACCUM} "
          f"→ effective batch={BATCH_SIZE * GRAD_ACCUM}")

    train_ds = SARFloodDataset(train_pairs, processor)
    val_ds   = SARFloodDataset(val_pairs,   processor)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} train")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += outputs.loss.item()

        train_loss /= len(train_dl)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} val"):
                batch    = {k: v.to(device) for k, v in batch.items()}
                val_loss += model(**batch).loss.item()
        val_loss /= len(val_dl)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt_path = CKPT_DIR / f"epoch_{epoch:02d}"
        model.save_pretrained(str(ckpt_path))
        processor.save_pretrained(str(ckpt_path))
        print(f"  Checkpoint saved → {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_path = CKPT_DIR / "final"
            model.save_pretrained(str(final_path))
            processor.save_pretrained(str(final_path))
            print(f"  Best model saved → {final_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Adapter weights → {CKPT_DIR / 'final'}")
    print("Next: run 10_predict_damage.py (or use checkpoints/final/ for inference)")


if __name__ == "__main__":
    main()
