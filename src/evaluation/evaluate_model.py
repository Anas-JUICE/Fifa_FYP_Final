from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils.helpers import BEST_MODEL_SUMMARY_PATH, load_artifact, save_json, load_json

def main():
    artifact = load_artifact("best")
    metrics = artifact["metrics"]

    evaluation_report = {
        "best_model_name": artifact["model_name"],
        "selection_summary": load_json(BEST_MODEL_SUMMARY_PATH),
        "metrics": metrics
    }

    save_json(evaluation_report, ROOT / "results" / "evaluation_report.json")

    print("=== BEST MODEL EVALUATION ===")
    print(f"Best Model        : {artifact['model_name']}")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Log Loss          : {metrics['log_loss']:.4f}")
    print(f"Draw Recall       : {metrics['classification_report']['1']['recall']:.4f}")
    print(f"Draw F1           : {metrics['classification_report']['1']['f1-score']:.4f}")
    print("Saved detailed evaluation report to: results/evaluation_report.json")

if __name__ == "__main__":
    main()
