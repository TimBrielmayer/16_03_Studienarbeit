# benchmarks/run_benchmark.py

from typing import Dict, Any, List

import os
import inspect
import importlib

from models.layer_manipulation import temporarily_remove_layer
from benchmarks.lm_eval_wrapper import evaluate_with_lm_eval


class BenchmarkRunner:
    def __init__(
        self,
        model,
        tokenizer,
        layer_count: int,
        tasks: List[str],
        num_fewshot: int = 0,
        limit: int = 50,
        batch_size: int = 1,
        model_name: str = "",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_count = layer_count
        self.tasks = tasks
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.batch_size = batch_size
        self.model_name = model_name
        self._debug_printed = False

    def _eval(self) -> Dict[str, Any]:
        return evaluate_with_lm_eval(
            model=self.model,
            tokenizer=self.tokenizer,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            limit=self.limit,
            batch_size=self.batch_size,
        )

    def run_all(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        print("\n📈 Baseline Evaluation...")
        baseline = self._eval()
        results["baseline"] = {
            "metrics": baseline.get("results", {}),
            "config": baseline.get("config", {}),
        }

        # Layer-Entfernung
        debug = os.getenv("LAYER_DEBUG", "").lower() in {"1", "true", "yes"}

        if debug and not self._debug_printed:
            try:
                mod = importlib.import_module(temporarily_remove_layer.__module__)
                mod_file = getattr(mod, "__file__", None)
            except Exception as exc:  # pragma: no cover
                mod_file = f"<unbekannt: {exc}>"

            print(f"[DEBUG] temporarily_remove_layer: {temporarily_remove_layer}")
            print(f"[DEBUG] module file: {mod_file}")
            try:
                print(f"[DEBUG] isgeneratorfunction: {inspect.isgeneratorfunction(temporarily_remove_layer)}")
            except Exception as exc:  # pragma: no cover
                print(f"[DEBUG] isgeneratorfunction: <error {exc}>")

            self._debug_printed = True

        for layer_idx in range(self.layer_count):
            key = f"layer_{layer_idx}"
            print(f"\n🧪 Eval ohne Layer {layer_idx}...")
            try:
                cm = temporarily_remove_layer(self.model, layer_idx)
                if not hasattr(cm, "__enter__"):
                    raise TypeError(
                        "temporarily_remove_layer liefert keinen Context-Manager. "
                        "Vermutlich ist @contextmanager nicht aktiv oder alte Datei geladen."
                    )
                if debug:
                    print(f"[DEBUG] ctx type: {type(cm)} has __enter__: {hasattr(cm, '__enter__')}")
                with cm:
                    out = self._eval()
                results[key] = {
                    "metrics": out.get("results", {}),
                    "config": out.get("config", {}),
                }
            except Exception as exc:
                results[key] = {
                    "error": str(exc),
                }
                print(f"⚠️ Fehler bei {key}: {exc}")

        return results
