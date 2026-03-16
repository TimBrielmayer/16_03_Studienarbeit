# experiments/experiment_manager.py

import os
from typing import Dict, Any, List

from models.model_loader import ModelLoader
from benchmark.run_benchmark import BenchmarkRunner


class ExperimentManager:
    """
    Verantwortlich für:
      - Modell laden
      - BenchmarkRunner ausführen (Baseline + Layer-Experimente)
      - Ergebnisse zurückgeben

    Dient als zentrale Orchestrierungsschicht und macht das gesamte
    Framework leicht erweiterbar auf mehrere Modelle.
    """

    def __init__(
        self,
        model_name: str,
        quantization: str,
        tasks: List[str],
        num_fewshot: int = 0,
        limit: int = 50,
        batch_size: int = 1
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.tasks = tasks
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.batch_size = batch_size

        # Ausgabeverzeichnis
        os.makedirs("experiments", exist_ok=True)

    # ------------------------------------------------------------
    # Hauptablauf für ein einzelnes Modell
    # ------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        print("\n==============================")
        print(f"🚀 Starte Experiment für Modell: {self.model_name}")
        print("==============================\n")

        # 1) Modell laden
        loader = ModelLoader(self.model_name, quantization=self.quantization)
        model, tokenizer, layer_count = loader.load()

        print("\n==============================")
        print(f"📊 Modell geladen — hat {layer_count} Layer")
        print("==============================\n")

        # 2) Benchmark Runner erstellen
        runner = BenchmarkRunner(
            model=model,
            tokenizer=tokenizer,
            layer_count=layer_count,
            tasks=self.tasks,
            num_fewshot=self.num_fewshot,
            limit=self.limit,
            batch_size=self.batch_size,
            model_name=self.model_name
        )

        # 3) Benchmark durchführen
        results = runner.run_all()

        print("\n==============================")
        print("🏁 Experiment abgeschlossen!")
        print("==============================\n")

        return results

    # ------------------------------------------------------------
    # Hilfe: erlaubt externen Zugriff auf die Parameter
    # ------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "tasks": self.tasks,
            "num_fewshot": self.num_fewshot,
            "limit": self.limit,
            "batch_size": self.batch_size,
        }
