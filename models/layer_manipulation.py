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


@contextmanager
def temporarily_remove_layer(model, layer_index: int):
    """
    Entfernt temporär einen Layer und stellt danach den Originalzustand wieder her.
    """
    path, layers = find_layers_path(model)
    orig_layers = layers
    orig_count = len(layers)
    orig_cfg = None
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        orig_cfg = model.config.num_hidden_layers

    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"Layer-Index {layer_index} ausserhalb [0, {len(layers)-1}]")

    new_layers = torch.nn.ModuleList([layer for i, layer in enumerate(layers) if i != layer_index])

    _set_by_path(model, path, new_layers)
    if orig_cfg is not None:
        model.config.num_hidden_layers = len(new_layers)

    try:
        yield
    finally:
        _set_by_path(model, path, orig_layers)
        if orig_cfg is not None:
            model.config.num_hidden_layers = orig_cfg
