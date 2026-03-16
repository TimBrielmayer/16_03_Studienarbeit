# visualization/plot_results.py

from typing import Dict, Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _layer_keys(results: Dict[str, Any]) -> List[str]:
    return sorted(
        [k for k in results.keys() if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1])
    )


class ResultPlotter:
    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def _collect_delta_df(self, metric: str) -> pd.DataFrame:
        baseline = self.results.get("baseline", {}).get("metrics", {})
        layer_keys = _layer_keys(self.results)

        tasks = list(baseline.keys())
        if not tasks:
            return pd.DataFrame()

        data = []
        index = []
        for lk in layer_keys:
            row = []
            for task in tasks:
                base_val = baseline.get(task, {}).get(metric)
                cur_val = self.results.get(lk, {}).get("metrics", {}).get(task, {}).get(metric)
                if base_val is None or cur_val is None:
                    row.append(float("nan"))
                else:
                    row.append(cur_val - base_val)
            data.append(row)
            index.append(lk)

        return pd.DataFrame(data, index=index, columns=tasks)

    def plot_delta_heatmap(self, metric: str = "acc", save_path: str = None) -> None:
        df = self._collect_delta_df(metric)
        if df.empty:
            print("Keine Daten fuer Heatmap vorhanden.")
            return

        plt.figure(figsize=(max(8, len(df.columns) * 1.2), max(4, len(df) * 0.3)))
        sns.heatmap(df, cmap="coolwarm", center=0.0, annot=False)
        plt.title(f"Delta zur Baseline ({metric})")
        plt.xlabel("Task")
        plt.ylabel("Layer")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close()
        else:
            plt.show()

    def plot_bar_per_task(self, metric: str = "acc", save_dir: str = None) -> None:
        df = self._collect_delta_df(metric)
        if df.empty:
            print("Keine Daten fuer Balkendiagramme vorhanden.")
            return

        for task in df.columns:
            plt.figure(figsize=(10, 4))
            df[task].plot(kind="bar")
            plt.title(f"Delta zur Baseline ({metric}) - {task}")
            plt.xlabel("Layer")
            plt.ylabel("Delta")
            plt.tight_layout()

            if save_dir:
                out = f"{save_dir}/delta_{task}_{metric}.png"
                plt.savefig(out, dpi=200)
                plt.close()
            else:
                plt.show()
