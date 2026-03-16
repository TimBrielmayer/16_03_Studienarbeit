# benchmarks/run_benchmark.py

from typing import Dict, Any, List

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
        for layer_idx in range(self.layer_count):
            key = f"layer_{layer_idx}"
            print(f"\n🧪 Eval ohne Layer {layer_idx}...")
            try:
                with temporarily_remove_layer(self.model, layer_idx):
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
