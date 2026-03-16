# models/model_loader.py

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.quantization import get_quantization_kwargs
from models.layer_manipulation import get_layer_count


class ModelLoader:
    def __init__(self, model_name: str, quantization: str = "4bit"):
        self.model_name = model_name
        self.quantization = quantization

    def load(self) -> Tuple[torch.nn.Module, object, int]:
        """Laedt Modell + Tokenizer und liefert (model, tokenizer, layer_count)."""
        print(f"\n🔍 Lade Modell: {self.model_name}")

        quant_kwargs = get_quantization_kwargs(self.quantization)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Fuer Causal-LMs ist links-Padding meist stabiler bei Batch-Inferenz
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        # Modell
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            **quant_kwargs,
        )

        model.eval()

        # Layer-Anzahl ermitteln
        layer_count = get_layer_count(model)

        return model, tokenizer, layer_count
