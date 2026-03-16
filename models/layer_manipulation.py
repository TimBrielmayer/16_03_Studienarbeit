# models/layer_manipulation.py

from contextlib import contextmanager
from typing import Tuple

import torch


def _get_by_path(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _set_by_path(obj, path: str, value) -> None:
    parts = path.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def find_layers_path(model) -> Tuple[str, torch.nn.ModuleList]:
    """
    Versucht, die Transformer-Layerliste in einem HF-Modell zu finden.
    Gibt (path, layers) zurueck.
    """
    candidates = [
        "model.layers",                # Llama/Mistral
        "model.model.layers",          # manchmal doppelt gekapselt
        "model.decoder.layers",        # OPT
        "model.model.decoder.layers",  # OPT (alt)
        "transformer.h",               # GPT-2/Neo/J
        "transformer.blocks",          # MPT
        "gpt_neox.layers",             # GPT-NeoX
    ]

    for path in candidates:
        layers = _get_by_path(model, path)
        if isinstance(layers, torch.nn.ModuleList):
            return path, layers

    raise ValueError(
        "Konnte die Transformer-Layerliste nicht finden. "
        "Bitte Modellstruktur pruefen und candidates in layer_manipulation.py erweitern."
    )


def get_layer_count(model) -> int:
    _, layers = find_layers_path(model)
    return len(layers)


class _BypassLayer(torch.nn.Module):
    """
    Laesst den Layer bestehen, gibt aber die Eingabe als Output zurueck.
    Das Original-Layer wird weiterhin aufgerufen, damit Cache/Shapes stabil bleiben.
    """

    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, *args, **kwargs):
        outputs = self.layer(hidden_states, *args, **kwargs)

        # Erwartet wird in Decoder-Layern typischerweise ein Tuple
        if isinstance(outputs, tuple):
            return (hidden_states,) + outputs[1:]

        # Falls ein ModelOutput verwendet wird, versuche hidden_states zu ersetzen
        if hasattr(outputs, "hidden_states"):
            outputs.hidden_states = hidden_states
        return outputs


@contextmanager
def temporarily_remove_layer(model, layer_index: int):
    """
    Setzt den Layer an Position layer_index temporär auf eine Bypass-Variante,
    um Stabilitaet (Generation/Cache) zu wahren.
    """
    _, layers = find_layers_path(model)

    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"Layer-Index {layer_index} ausserhalb [0, {len(layers)-1}]")

    orig_layer = layers[layer_index]
    layers[layer_index] = _BypassLayer(orig_layer)

    try:
        yield
    finally:
        layers[layer_index] = orig_layer
