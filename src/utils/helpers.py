from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "final_training_dataset_2006plus.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TEAM_PROFILES_PATH = PROCESSED_DIR / "team_profiles.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_SUMMARY_PATH = RESULTS_DIR / "best_model_summary.json"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"

NUMERIC_FEATURES = [
    "elo_A", "elo_B", "elo_diff",
    "A_wins_last5", "A_draws_last5", "A_loss_last5",
    "A_gf_avg_last5", "A_ga_avg_last5", "A_gd_avg_last5",
    "B_wins_last5", "B_draws_last5", "B_loss_last5",
    "B_gf_avg_last5", "B_ga_avg_last5", "B_gd_avg_last5",
    "match_importance"
]

CATEGORICAL_FEATURES = [
    "tournament",
    "neutral"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL_MAP = {
    0: "Team A Win",
    1: "Draw",
    2: "Team B Win"
}

LABEL_NAMES = ["Team A Win", "Draw", "Team B Win"]


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "Put final_training_dataset_2006plus.csv inside data/raw/"
        )
    df = pd.read_csv(RAW_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def split_timewise(df: pd.DataFrame, test_ratio: float = 0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def build_team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profiles = {}

    for row in df.sort_values("date").itertuples(index=False):
        profiles[row.team_A] = {
            "team": row.team_A,
            "elo": float(row.elo_A),
            "wins_last5": float(row.A_wins_last5),
            "draws_last5": float(row.A_draws_last5),
            "loss_last5": float(row.A_loss_last5),
            "gf_avg_last5": float(row.A_gf_avg_last5),
            "ga_avg_last5": float(row.A_ga_avg_last5),
            "gd_avg_last5": float(row.A_gd_avg_last5),
            "last_seen_date": str(row.date.date())
        }

        profiles[row.team_B] = {
            "team": row.team_B,
            "elo": float(row.elo_B),
            "wins_last5": float(row.B_wins_last5),
            "draws_last5": float(row.B_draws_last5),
            "loss_last5": float(row.B_loss_last5),
            "gf_avg_last5": float(row.B_gf_avg_last5),
            "ga_avg_last5": float(row.B_ga_avg_last5),
            "gd_avg_last5": float(row.B_gd_avg_last5),
            "last_seen_date": str(row.date.date())
        }

    profiles_df = pd.DataFrame(list(profiles.values())).sort_values(
        ["elo", "team"], ascending=[False, True]
    ).reset_index(drop=True)

    return profiles_df


def save_team_profiles(df: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs()
    profiles_df = build_team_profiles(df)
    profiles_df.to_csv(TEAM_PROFILES_PATH, index=False)
    return profiles_df


def load_team_profiles() -> pd.DataFrame:
    if TEAM_PROFILES_PATH.exists():
        return pd.read_csv(TEAM_PROFILES_PATH)

    df = load_dataset()
    return save_team_profiles(df)


@lru_cache(maxsize=1)
def get_profiles_dict() -> Dict[str, pd.Series]:
    profiles_df = load_team_profiles()
    return {row["team"]: row for _, row in profiles_df.iterrows()}


def build_match_features(
    team_a: str,
    team_b: str,
    profiles: Optional[Dict[str, pd.Series]] = None,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
    match_importance: float = 5.0
) -> pd.DataFrame:
    if profiles is None:
        profiles = get_profiles_dict()

    if team_a not in profiles:
        raise ValueError(f"Unknown team_A: {team_a}")
    if team_b not in profiles:
        raise ValueError(f"Unknown team_B: {team_b}")

    a = profiles[team_a]
    b = profiles[team_b]

    row = {
        "elo_A": float(a["elo"]),
        "elo_B": float(b["elo"]),
        "elo_diff": float(a["elo"]) - float(b["elo"]),

        "A_wins_last5": float(a["wins_last5"]),
        "A_draws_last5": float(a["draws_last5"]),
        "A_loss_last5": float(a["loss_last5"]),
        "A_gf_avg_last5": float(a["gf_avg_last5"]),
        "A_ga_avg_last5": float(a["ga_avg_last5"]),
        "A_gd_avg_last5": float(a["gd_avg_last5"]),

        "B_wins_last5": float(b["wins_last5"]),
        "B_draws_last5": float(b["draws_last5"]),
        "B_loss_last5": float(b["loss_last5"]),
        "B_gf_avg_last5": float(b["gf_avg_last5"]),
        "B_ga_avg_last5": float(b["ga_avg_last5"]),
        "B_gd_avg_last5": float(b["gd_avg_last5"]),

        "match_importance": float(match_importance),
        "tournament": str(tournament),
        "neutral": str(bool(neutral))
    }

    return pd.DataFrame([row], columns=ALL_FEATURES)


def artifact_path_for(model_name: str) -> Path:
    return MODELS_DIR / f"{model_name}_model.joblib"


def metrics_path_for(model_name: str) -> Path:
    return RESULTS_DIR / f"metrics_{model_name}.json"


def confusion_matrix_figure_for(model_name: str) -> Path:
    return FIGURES_DIR / f"confusion_matrix_{model_name}.png"


def feature_importance_figure_for(model_name: str) -> Path:
    return FIGURES_DIR / f"feature_importance_{model_name}.png"


def top_features_csv_for(model_name: str) -> Path:
    return RESULTS_DIR / f"top_features_{model_name}.csv"


@lru_cache(maxsize=8)
def load_artifact(model_name: str = "best"):
    path = BEST_MODEL_PATH if model_name == "best" else artifact_path_for(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    return joblib.load(path)


def get_feature_names_from_pipeline(pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def predict_match_proba(
    team_a: str,
    team_b: str,
    model_name: str = "best",
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
    match_importance: float = 5.0
) -> dict:
    artifact = load_artifact(model_name=model_name)
    profiles = get_profiles_dict()
    X = build_match_features(
        team_a=team_a,
        team_b=team_b,
        profiles=profiles,
        tournament=tournament,
        neutral=neutral,
        match_importance=match_importance
    )
    probs = artifact["pipeline"].predict_proba(X)[0]
    predicted_class = int(probs.argmax())

    return {
        "model_name": artifact.get("model_name", model_name),
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win": float(probs[0]),
        "draw": float(probs[1]),
        "team_b_win": float(probs[2]),
        "predicted_label": predicted_class,
        "predicted_outcome": LABEL_MAP[predicted_class]
    }


def copy_as_best_model(model_name: str, summary: dict) -> None:
    src = artifact_path_for(model_name)
    if not src.exists():
        raise FileNotFoundError(f"Cannot copy best model. Missing file: {src}")
    shutil.copyfile(src, BEST_MODEL_PATH)
    save_json(summary, BEST_MODEL_SUMMARY_PATH)
