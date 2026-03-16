# models/quantization.py

from typing import Dict, Any

import torch


def get_quantization_kwargs(quantization: str) -> Dict[str, Any]:
    """
    Gibt die passenden kwargs fuer transformers.from_pretrained zurueck.
    Nutzt BitsAndBytesConfig fuer 4bit/8bit (kompatibel mit neueren Transformers-Versionen).
    """
    q = quantization.lower().strip()

    if q in {"4bit", "8bit"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{q} Quantisierung benoetigt CUDA (GPU).")
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("BitsAndBytesConfig nicht verfuegbar. Bitte transformers/bitsandbytes aktualisieren.") from exc

        if q == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        return {"quantization_config": bnb_config}

    if q == "fp16":
        # Fuer GPUs; auf CPU wird fp16 ggf. nicht unterstuetzt.
        return {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }

    raise ValueError(f"Unbekannte Quantisierung: {quantization}")
