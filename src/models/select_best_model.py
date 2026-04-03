from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils.helpers import (
    BEST_MODEL_SUMMARY_PATH,
    MODEL_COMPARISON_PATH,
    copy_as_best_model,
    ensure_dirs,
    load_json,
    metrics_path_for,
)
from models.training_utils import build_naive_baseline

MODEL_NAMES = ["logistic", "random_forest", "xgboost"]

def main():
    ensure_dirs()

    rows = []
    for model_name in MODEL_NAMES:
        metrics = load_json(metrics_path_for(model_name))
        rows.append({
            "model_name": model_name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "log_loss": metrics["log_loss"],
            "training_seconds": metrics["training_seconds"],
            "draw_recall": metrics["classification_report"]["1"]["recall"],
            "draw_f1": metrics["classification_report"]["1"]["f1-score"],
        })

    baseline_metrics = build_naive_baseline()
    rows.append({
        "model_name": "naive_baseline",
        "accuracy": baseline_metrics["accuracy"],
        "balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "log_loss": baseline_metrics["log_loss"],
        "training_seconds": 0.0,
        "draw_recall": baseline_metrics["classification_report"]["1"]["recall"],
        "draw_f1": baseline_metrics["classification_report"]["1"]["f1-score"],
    })

    comparison_df = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "log_loss"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)

    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model_name"]

    if best_model_name == "naive_baseline":
        raise RuntimeError("Naive baseline was selected as best. Check training outputs.")

    summary = {
        "selection_rule": [
            "highest balanced accuracy",
            "highest accuracy",
            "lowest log loss"
        ],
        "best_model_name": best_model_name,
        "comparison_table_path": str(MODEL_COMPARISON_PATH),
        "chosen_metrics": {
            "accuracy": float(best_row["accuracy"]),
            "balanced_accuracy": float(best_row["balanced_accuracy"]),
            "log_loss": float(best_row["log_loss"]),
            "training_seconds": float(best_row["training_seconds"]),
            "draw_recall": float(best_row["draw_recall"]),
            "draw_f1": float(best_row["draw_f1"]),
        }
    }

    copy_as_best_model(best_model_name, summary)

    print("=== MODEL SELECTION COMPLETE ===")
    print(comparison_df.to_string(index=False))
    print(f"\nSelected best model: {best_model_name}")
    print(f"Saved comparison table to: {MODEL_COMPARISON_PATH}")
    print(f"Saved best-model summary to: {BEST_MODEL_SUMMARY_PATH}")

if __name__ == "__main__":
    main()
