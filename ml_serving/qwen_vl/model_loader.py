"""Qwen2-VL model loader with singleton pattern for memory efficiency."""

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# Adapter lives at <repo_root>/final3; override with ADAPTER_PATH env var.
_DEFAULT_ADAPTER = Path(__file__).resolve().parents[2] / "final3"
ADAPTER_PATH = Path(os.getenv("ADAPTER_PATH", str(_DEFAULT_ADAPTER)))

_model = None
_processor = None


def load_model():
    """Load base Qwen2-VL-7B and apply the fine-tuned LoRA adapter from final3/."""
    global _model, _processor

    if _model is None:
        print(f"[GeoRescue] Loading base model {MODEL_NAME}...")
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )

        if ADAPTER_PATH.exists():
            print(f"[GeoRescue] Applying LoRA adapter from {ADAPTER_PATH}...")
            _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
            _model = _model.merge_and_unload()  # fuse weights for faster inference
        else:
            print(f"[GeoRescue] Adapter not found at {ADAPTER_PATH}, using base model.")
            _model = base

        # Load processor from adapter dir so custom tokenizer/chat_template is used.
        processor_source = str(ADAPTER_PATH) if ADAPTER_PATH.exists() else MODEL_NAME
        _processor = AutoProcessor.from_pretrained(processor_source)

        print(f"[GeoRescue] Model ready on {_model.device}")

    return _model, _processor


def get_gpu_info():
    """Return current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return {"allocated_gb": round(allocated, 2), "reserved_gb": round(reserved, 2)}
    return {"error": "No GPU available"}
