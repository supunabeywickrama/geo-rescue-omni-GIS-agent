"""
Train a U-Net flood segmentation model on S1 VV/VH tiles.

Input  : 2-band GeoTIFF (band1=VV, band2=VH, Float32, dB)
Output : binary flood mask (0=land, 1=flood)

Architecture : U-Net with EfficientNet-B0 encoder (pretrained on ImageNet)
Loss         : BCE + Dice (handles class imbalance)
Hardware     : AMD MI300X / any CUDA GPU / CPU fallback

Usage:
    pip install segmentation-models-pytorch torch torchvision
    python 07_train_model.py

Output:
    data/models/flood_unet.pth
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent / "data"
TILES_DIR  = BASE_DIR / "tiles"
LABELS_DIR = BASE_DIR / "labels"
MODEL_DIR  = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "flood_unet.pth"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS      = 75
LR          = 3e-4
MAX_TILES   = 50000   # cap for faster iteration; set None to use all
TRAIN_SPLIT = 0.85
SEED        = 42


# ── Dataset ────────────────────────────────────────────────────────────────────
class FloodDataset(Dataset):
    def __init__(self, pairs: list):
        self.pairs = pairs   # [(tile_path, label_path), ...]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tile_path, label_path = self.pairs[idx]

        with rasterio.open(tile_path) as src:
            img = src.read().astype(np.float32)   # (2, H, W)  VV + VH

        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.float32)  # (H, W)

        # Normalise VV/VH from dB range [-30, 0] → [0, 1]
        img = np.clip((img + 30) / 30, 0, 1)

        # Replace NaN / inf
        img  = np.nan_to_num(img,  nan=0.0, posinf=1.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0)
        mask = np.clip(mask, 0, 1)

        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0)


# ── Loss: BCE + Dice ───────────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    def forward(self, pred, target):
        bce  = nn.functional.binary_cross_entropy_with_logits(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        inter = (pred_sigmoid * target).sum(dim=(2, 3))
        dice  = 1 - (2 * inter + 1) / (pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1)
        return bce + dice.mean()


# ── Simple U-Net (no extra deps) ───────────────────────────────────────────────
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.out  = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifest and find tile-label pairs
    manifest_path = TILES_DIR / "manifest.csv"
    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))

    pairs = []
    for row in rows:
        if row["source"] != "S1" or row["phase"] == "unknown":
            continue
        # Only VV tiles (skip VH — VV is the primary flood indicator)
        if "_VH_" in row["tile"]:
            continue
        tile_path  = TILES_DIR.parent / row["tile"]
        label_path = (LABELS_DIR / row["tile"]
                      .replace("tiles/", "labels/")
                      .replace(".tif", "_label.tif"))
        if tile_path.exists() and label_path.exists():
            pairs.append((tile_path, label_path))

    print(f"Valid tile-label pairs: {len(pairs)}")

    # Cap and shuffle
    if MAX_TILES and len(pairs) > MAX_TILES:
        # Balance flood vs no-flood
        manifest_map = {TILES_DIR.parent / r["tile"]: r["phase"] for r in rows}
        flood    = [(t, l) for t, l in pairs if manifest_map.get(t) in ("during", "after")]
        no_flood = [(t, l) for t, l in pairs if manifest_map.get(t) == "before"]
        n = min(len(flood), len(no_flood), MAX_TILES // 2)
        pairs = random.sample(flood, n) + random.sample(no_flood, n)
        print(f"Balanced sample: {n} flood + {n} no-flood = {len(pairs)} pairs")

    random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_SPLIT)
    train_ds = FloodDataset(pairs[:split])
    val_ds   = FloodDataset(pairs[split:])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = UNet(in_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = BCEDiceLoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(val_dl)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ saved best model → {MODEL_PATH}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print("Next: run 08_predict_to_geojson.py")


if __name__ == "__main__":
    main()
