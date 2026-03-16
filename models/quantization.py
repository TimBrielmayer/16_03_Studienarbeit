# models/quantization.py

from typing import Dict, Any

import torch


def get_quantization_kwargs(quantization: str) -> Dict[str, Any]:
    """
    Gibt die passenden kwargs fuer transformers.from_pretrained zurueck.
    """
    q = quantization.lower().strip()

    if q == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("4bit Quantisierung benoetigt CUDA (GPU).")
        return {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        }

    if q == "8bit":
        if not torch.cuda.is_available():
            raise RuntimeError("8bit Quantisierung benoetigt CUDA (GPU).")
        return {
            "load_in_8bit": True,
        }

    if q == "fp16":
        # Fuer GPUs; auf CPU wird fp16 ggf. nicht unterstuetzt.
        return {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }

    raise ValueError(f"Unbekannte Quantisierung: {quantization}")
