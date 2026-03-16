# main.py

"""
Haupt-Einstiegspunkt für das Layer-Entfernungs-Benchmarking auf Decoder-Only LLMs
(Initial: Mistral). Entwickelt für Google Colab T4 und VSCode Remote.

Funktionen:
- Lädt und quantisiert das Modell (über ExperimentManager)
- Führt Baseline + pro-Layer Benchmarks (lm-eval-harness)
- Speichert Ergebnisse (JSON via BenchmarkRunner)
- Optional: Erzeugt Visualisierungen (Heatmap + Balkendiagramme)

Nutzung (Beispiele):
    python main.py \
        --model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --quantization 4bit \
        --tasks mmlu,hellaswag,arc_easy,truthfulqa \
        --limit 50 \
        --batch_size 1 \
        --save_plots

Hinweise:
- Stelle sicher, dass lm-eval installiert ist:  pip install -U lm-eval
- Für 4-bit Quantisierung: bitsandbytes + transformers >= 4.36
"""

import argparse
import json
import os
from datetime import datetime
from typing import List

# Optional: Headless-Backend, falls keine GUI vorhanden
os.environ.setdefault("MPLBACKEND", "Agg")

from experiments.experiment_manager import ExperimentManager
from visualization.plot_results import ResultPlotter


DEFAULT_TASKS = ["mmlu", "hellaswag", "arc_easy", "truthfulqa"]


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Layer-Entfernungs-Benchmarking (Decoder-Only)")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                        help="HuggingFace Model-ID oder lokaler Pfad")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "fp16"],
                        help="Quantisierungsmodus")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS),
                        help="Kommagetrennte Liste von Tasks (z. B. mmlu,hellaswag,arc_easy,truthfulqa)")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-Shot Beispiele pro Task")
    parser.add_argument("--limit", type=int, default=50, help="Limit pro Task (None/-1 = volle Größe)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batchgröße für Likelihood")

    # Visualisierung
    parser.add_argument("--save_plots", action="store_true", help="Plotte und speichere Grafiken")
    parser.add_argument("--plots_dir", type=str, default="experiments/plots", help="Zielordner für Plots")
    parser.add_argument("--metric", type=str, default="acc", help="Metrik für Plots (z. B. acc, acc_norm, ppl)")
    parser.add_argument("--skip_plots", action="store_true", help="Visualisierung überspringen")

    # Reproduzierbarkeit
    parser.add_argument("--seed", type=int, default=123, help="Zufallsseed")

    args = parser.parse_args()

    # Tasks parsen
    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    args.tasks = tasks if tasks else DEFAULT_TASKS

    # Limit verarbeiten
    if args.limit is not None and args.limit < 0:
        args.limit = None

    return args


def save_latest_copy(results: dict, model_name: str) -> str:
    """Speichert eine zusätzliche Kopie der Ergebnisse für einfachen Zugriff.
    Gibt den Pfad zurück.
    """
    os.makedirs("experiments", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model_name.replace("/", "_")
    out_path = os.path.join("experiments", f"latest_results_{safe_model}_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Zusätzliche Kopie gespeichert unter: {out_path}")
    return out_path


def print_console_summary(results: dict, metric: str):
    """Gibt eine kurze Zusammenfassung pro Task auf der Konsole aus."""
    baseline = results.get("baseline", {}).get("metrics", {})
    layer_keys = sorted([k for k in results.keys() if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))

    print("\n===== Zusammenfassung (Delta zur Baseline) =====")
    for task, base_metrics in baseline.items():
        base_val = base_metrics.get(metric)
        if base_val is None:
            continue
        print(f"\nTask: {task}")
        print(f"  Baseline {metric}: {base_val:.4f}")
        for lk in layer_keys[:10]:  # Konsole knapp halten: erste 10 Layer
            lm = results[lk]["metrics"].get(task, {})
            val = lm.get(metric)
            if val is None:
                continue
            delta = val - base_val
            print(f"  {lk}: {val:.4f}  (Δ {delta:+.4f})")


def main():
    args = parse_args()

    print("\n==============================")
    print("🔧 Konfiguration")
    print("==============================")
    print(f"Model:        {args.model_name}")
    print(f"Quantization: {args.quantization}")
    print(f"Tasks:        {args.tasks}")
    print(f"Few-shot:     {args.num_fewshot}")
    print(f"Limit:        {args.limit}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Plots:        {'ON' if args.save_plots and not args.skip_plots else 'OFF'} (metric={args.metric})\n")

    manager = ExperimentManager(
        model_name=args.model_name,
        quantization=args.quantization,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    results = manager.run()

    # Konsolen-Kurzbericht
    print_console_summary(results, metric=args.metric)

    # Zusätzliche Kopie speichern (leicht auffindbar)
    latest_path = save_latest_copy(results, model_name=args.model_name)

    # Visualisierung
    if not args.skip_plots:
        plotter = ResultPlotter(results)
        if args.save_plots:
            os.makedirs(args.plots_dir, exist_ok=True)
            heatmap_path = os.path.join(args.plots_dir, f"delta_heatmap_{args.metric}.png")
            plotter.plot_delta_heatmap(metric=args.metric, save_path=heatmap_path)
            print(f"📊 Heatmap gespeichert: {heatmap_path}")

            plotter.plot_bar_per_task(metric=args.metric, save_dir=args.plots_dir)
            print(f"📊 Balkendiagramme gespeichert in: {args.plots_dir}")
        else:
            # Anzeige ohne Speicherung
            plotter.plot_delta_heatmap(metric=args.metric, save_path=None)
            plotter.plot_bar_per_task(metric=args.metric, save_dir=None)

    print("\n✅ Fertig. Ergebnisse und (optional) Plots sind in 'experiments/' abgelegt.")


if __name__ == "__main__":
    main()
