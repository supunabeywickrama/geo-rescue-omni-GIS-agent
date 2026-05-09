# GeoRescue AI — Omni GIS Agent

> AMD Hackathon project — AI-powered disaster response platform combining satellite vision, real-time flood analysis, and safe route planning for Colombo, Sri Lanka.

---

## What It Does

GeoRescue AI ingests live weather data and satellite imagery, runs them through a fine-tuned vision model, and outputs actionable GeoJSON for field responders:

1. **Live flood zone detection** — pulls precipitation from Open-Meteo, converts to a flood polygon with severity level (low / moderate / high / extreme)
2. **Road impact analysis** — overlays the flood zone on the Colombo OSM road network to identify blocked segments
3. **Safe route planning** — computes the shortest path avoiding blocked roads using NetworkX
4. **Satellite image analysis** — Qwen2-VL-7B fine-tuned with LoRA, analyzes aerial/satellite images and returns affected zones as GeoJSON

---

## Architecture

```
                        ┌──────────────────────────────────┐
                        │         FastAPI Server           │
                        │       (port 9000)                │
                        └────────────┬─────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
     ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
     │  Vision Module  │   │  GIS Pipeline   │   │  Llama-3 Server │
     │  Qwen2-VL-7B   │   │  (flood + route)│   │  (text Q&A)     │
     │  + LoRA final3  │   │                 │   │                 │
     └─────────────────┘   └────────┬────────┘   └─────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
             Open-Meteo API    OSM Roads        GraphML
             (live weather)    (GeoJSON)        (road graph)
```

---

## API Endpoints

### Vision

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze-image` | Upload a satellite/aerial image → returns severity + GeoJSON affected zones |
| `GET`  | `/health` | Service health + GPU status |

### GIS / Flood Intelligence

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/gis/status` | Latest cycle summary (severity, affected road count, route length) |
| `GET`  | `/gis/flood-polygon` | Live flood zone as GeoJSON |
| `GET`  | `/gis/blocked-roads` | Flood-impacted road segments as GeoJSON |
| `GET`  | `/gis/safe-route` | Optimal safe route avoiding blocked roads as GeoJSON |
| `POST` | `/gis/run-cycle` | Trigger a fresh live-weather analysis cycle |

---

## Project Structure

```
geo-rescue-omni-GIS-agent/
├── final3/                        # Fine-tuned LoRA adapter (Qwen2-VL-7B)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  # not tracked (large binary)
│   └── tokenizer.json / chat_template.jinja / ...
│
└── ml_serving/                    # Main serving package
    ├── api/
    │   ├── app.py                 # FastAPI app + lifespan (model warmup)
    │   ├── routes.py              # Vision endpoints
    │   ├── gis_routes.py          # GIS endpoints
    │   └── schemas.py             # Pydantic models
    ├── qwen_vl/
    │   ├── model_loader.py        # Loads base model + merges LoRA adapter
    │   ├── inference.py           # Vision inference pipeline
    │   ├── image_processor.py
    │   └── geojson_generator.py
    ├── gis_pipeline/
    │   ├── live_flood_feed.py     # Open-Meteo → flood polygon
    │   ├── flood_overlay.py       # Spatial overlay → blocked roads
    │   ├── routing.py             # NetworkX safe route planning
    │   └── pipeline.py            # Full cycle orchestrator
    ├── data_pipeline/             # Sentinel-2 satellite data collector
    ├── llama_server/              # Llama-3 text inference server
    ├── training/                  # LoRA fine-tuning code
    ├── data/processed/            # GeoJSON outputs + road graph (gitignored except graphml)
    ├── docker/Dockerfile
    └── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
cd ml_serving
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env   # or edit ml_serving/.env directly
# Set SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET (Copernicus access)
# Optionally set ADAPTER_PATH if final3/ is in a different location
```

### 3. Run the API server

```bash
cd ml_serving
uvicorn api.app:app --host 0.0.0.0 --port 9000
```

On startup the server:
- Downloads and loads `Qwen/Qwen2-VL-7B-Instruct`
- Merges the `final3/` LoRA adapter into the base model
- Registers all vision + GIS endpoints

### 4. Trigger a live flood cycle

```bash
curl -X POST http://localhost:9000/gis/run-cycle
```

Then poll the outputs:

```bash
curl http://localhost:9000/gis/status
curl http://localhost:9000/gis/flood-polygon
curl http://localhost:9000/gis/safe-route
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| Adapter type | LoRA (PEFT) |
| LoRA rank | r=16, alpha=32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Task | Disaster image analysis → GeoJSON zone extraction |
| Adapter location | `final3/` (weights excluded from git, config tracked) |

---

## Team

| Member | Contribution |
|--------|-------------|
| Supun | ML serving infrastructure, LoRA adapter integration, GIS pipeline API |
| Minindu | GIS flood analysis pipeline, road routing, GeoJSON outputs |
| Member 3 | LoRA fine-tuning (Colab), training data, `final3/` adapter |
| Member 4 | Frontend / UI |

---

## Hardware Target

AMD MI300X GPU — `device_map="auto"` with `torch_dtype="auto"` (bf16).
