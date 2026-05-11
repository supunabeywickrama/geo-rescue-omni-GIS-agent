"""
Run the trained flood model on a Sentinel-1 GeoTIFF and export a flood polygon GeoJSON.
The output GeoJSON can be dropped directly into the GeoRescue routing pipeline.

Usage:
    python 08_predict_to_geojson.py --input data/processed/S1A_..._VV.tif
    python 08_predict_to_geojson.py --input data/processed/S1A_..._VV.tif --out my_flood.geojson

Output:
    flood_polygon.geojson  (or --out path)
    Also copies to ml_serving/data/processed/live_flood_polygon.geojson
    so the routing pipeline picks it up immediately.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(__file__).parent.parent / "data" / "models" / "flood_unet.pth"
GIS_OUT_PATH = Path(__file__).parents[3] / "ml_serving" / "data" / "processed" / "live_flood_polygon.geojson"

TILE_SIZE = 256


# ── U-Net (must match 07_train_model.py) ─────────────────────────────────────
import torch.nn as nn

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


def load_model(device):
    model = UNet(in_channels=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[model] Loaded from {MODEL_PATH}")
    return model


def predict_full_image(model, vv_path: Path, device) -> tuple:
    """
    Slide 256×256 window over the full VV GeoTIFF.
    Returns (flood_mask_array, rasterio_transform, crs).
    """
    # Find matching VH band (same product, different suffix)
    vh_path = Path(str(vv_path).replace("_VV.tif", "_VH.tif"))
    if not vh_path.exists():
        raise FileNotFoundError(f"VH band not found: {vh_path}")

    with rasterio.open(vv_path) as vv_src:
        vv      = vv_src.read(1).astype(np.float32)
        profile = vv_src.profile.copy()
        transform = vv_src.transform
        crs     = vv_src.crs

    with rasterio.open(vh_path) as vh_src:
        vh = vh_src.read(1).astype(np.float32)

    H, W = vv.shape
    flood_mask  = np.zeros((H, W), dtype=np.float32)
    count_mask  = np.zeros((H, W), dtype=np.float32)

    # Sliding window prediction
    for row in range(0, H - TILE_SIZE + 1, TILE_SIZE):
        for col in range(0, W - TILE_SIZE + 1, TILE_SIZE):
            vv_tile = vv[row:row+TILE_SIZE, col:col+TILE_SIZE]
            vh_tile = vh[row:row+TILE_SIZE, col:col+TILE_SIZE]

            # Normalise to [0,1]
            img = np.stack([vv_tile, vh_tile], axis=0)
            img = np.clip((img + 30) / 30, 0, 1)
            img = np.nan_to_num(img, nan=0.0)

            tensor = torch.from_numpy(img).unsqueeze(0).to(device)  # (1,2,256,256)
            with torch.no_grad():
                pred = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

            flood_mask[row:row+TILE_SIZE, col:col+TILE_SIZE] += pred
            count_mask[row:row+TILE_SIZE, col:col+TILE_SIZE] += 1

    # Average overlapping predictions
    count_mask[count_mask == 0] = 1
    flood_mask /= count_mask

    # Threshold at 0.5
    binary = (flood_mask >= 0.5).astype(np.uint8)
    return binary, transform, crs


def mask_to_geojson(binary: np.ndarray, transform, crs, out_path: Path):
    """Polygonize binary raster → GeoJSON FeatureCollection."""
    polygons = []
    for geom, val in shapes(binary, transform=transform):
        if val == 1:   # flood pixels only
            polygons.append(shape(geom))

    if not polygons:
        print("WARNING: No flood pixels detected. Check input image quality.")
        flood_union = None
    else:
        flood_union = unary_union(polygons)

    # Build GeoJSON
    features = []
    if flood_union:
        geoms = [flood_union] if flood_union.geom_type != "GeometryCollection" else flood_union.geoms
        for geom in (geoms if hasattr(geoms, '__iter__') and not hasattr(geoms, 'exterior') else [flood_union]):
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"source": "flood_unet", "type": "flood_zone"},
            })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(geojson, f)

    area_km2 = sum(g.area * 111**2 for g in [flood_union] if flood_union) if flood_union else 0
    print(f"[geojson] {len(features)} flood polygon(s)")
    print(f"[geojson] Estimated area: ~{area_km2:.1f} km²")
    print(f"[geojson] Saved → {out_path}")
    return geojson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to *_VV.tif GeoTIFF")
    parser.add_argument("--out",   default="flood_polygon.geojson")
    args = parser.parse_args()

    vv_path  = Path(args.input)
    out_path = Path(args.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(device)

    print(f"[predict] Running on {vv_path.name} ...")
    binary, transform, crs = predict_full_image(model, vv_path, device)

    flood_pct = binary.mean() * 100
    print(f"[predict] Flood pixels: {flood_pct:.1f}%")

    geojson = mask_to_geojson(binary, transform, crs, out_path)

    # Copy to GIS pipeline so routing picks it up immediately
    if GIS_OUT_PATH.parent.exists():
        with GIS_OUT_PATH.open("w") as f:
            json.dump(geojson, f)
        print(f"[pipeline] Updated → {GIS_OUT_PATH}")
        print("[pipeline] Call POST /gis/run-cycle to recompute safe route.")


if __name__ == "__main__":
    main()
