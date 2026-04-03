from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "src/data/build_team_profiles.py",
    "src/models/train_logistic.py",
    "src/models/train_random_forest.py",
    "src/models/train_xgboost.py",
    "src/models/select_best_model.py",
    "src/evaluation/evaluate_model.py",
    "src/simulation/tournament_simulation.py",
]

def run_script(script_path: str):
    full_path = ROOT / script_path
    print(f"\n>>> Running: {script_path}")
    completed = subprocess.run([sys.executable, str(full_path)], cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(f"Pipeline stopped because {script_path} failed.")

def main():
    for script in SCRIPTS:
        run_script(script)
    print("\n=== FULL PIPELINE COMPLETE ===")
    print("Check models/, results/, and reports/figures/ for outputs.")

if __name__ == "__main__":
    main()
