from __future__ import annotations

import time
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from utils.helpers import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FIGURES_DIR,
    LABEL_NAMES,
    LABEL_MAP,
    NUMERIC_FEATURES,
    artifact_path_for,
    confusion_matrix_figure_for,
    feature_importance_figure_for,
    get_feature_names_from_pipeline,
    load_dataset,
    metrics_path_for,
    save_json,
    save_team_profiles,
    split_timewise,
    top_features_csv_for,
    ensure_dirs,
)

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

def get_model_by_name(model_name: str):
    if model_name == "logistic":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )

    if model_name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=2
        )

    raise ValueError(f"Unsupported model_name: {model_name}")

def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    }

def save_confusion_matrix_figure(cm, model_name: str):
    path = confusion_matrix_figure_for(model_name)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
    plt.colorbar()
    tick_marks = range(3)
    plt.xticks(tick_marks, ["A Win", "Draw", "B Win"])
    plt.yticks(tick_marks, ["A Win", "Draw", "B Win"])

    threshold = np.max(cm) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def extract_feature_importance(pipeline, model_name: str) -> pd.DataFrame:
    feature_names = get_feature_names_from_pipeline(pipeline)
    model = pipeline.named_steps["model"]

    if model_name == "logistic":
        importances = np.mean(np.abs(model.coef_), axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    fi_df.to_csv(top_features_csv_for(model_name), index=False)

    top_df = fi_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"Top 15 Features - {model_name.replace('_', ' ').title()}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(feature_importance_figure_for(model_name), dpi=200, bbox_inches="tight")
    plt.close()

    return fi_df

def train_and_save_model(model_name: str):
    ensure_dirs()

    df = load_dataset()
    save_team_profiles(df)

    train_df, test_df = split_timewise(df, test_ratio=0.2)

    X_train = train_df[ALL_FEATURES].copy()
    y_train = train_df["label"].copy()
    X_test = test_df[ALL_FEATURES].copy()
    y_test = test_df["label"].copy()

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", get_model_by_name(model_name))
    ])

    start = time.time()
    
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["tournament"] = X_train["tournament"].astype(str)
    X_test["tournament"] = X_test["tournament"].astype(str)

    X_train["neutral"] = X_train["neutral"].astype(str)
    X_test["neutral"] = X_test["neutral"].astype(str)

    pipeline.fit(X_train, y_train)
    training_seconds = time.time() - start

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    metrics = {
        "model_name": model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_date_min": str(train_df["date"].min().date()),
        "train_date_max": str(train_df["date"].max().date()),
        "test_date_min": str(test_df["date"].min().date()),
        "test_date_max": str(test_df["date"].max().date()),
        "training_seconds": float(training_seconds),
    }
    metrics.update(evaluate_predictions(y_test, y_pred, y_prob))

    artifact = {
        "model_name": model_name,
        "pipeline": pipeline,
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "label_map": LABEL_MAP,
        "metrics": metrics
    }

    joblib.dump(artifact, artifact_path_for(model_name))
    save_json(metrics, metrics_path_for(model_name))
    save_confusion_matrix_figure(np.array(metrics["confusion_matrix"]), model_name)
    fi_df = extract_feature_importance(pipeline, model_name)

    print(f"=== TRAINING COMPLETE: {model_name.upper()} ===")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Log Loss         : {metrics['log_loss']:.4f}")
    print(f"Training Seconds : {metrics['training_seconds']:.2f}")
    print("Top features:")
    print(fi_df.head(10).to_string(index=False))

def build_naive_baseline():
    df = load_dataset()
    train_df, test_df = split_timewise(df, test_ratio=0.2)

    X_train = train_df[ALL_FEATURES].copy()
    y_train = train_df["label"].copy()
    X_test = test_df[ALL_FEATURES].copy()
    y_test = test_df["label"].copy()

    baseline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", DummyClassifier(strategy="prior"))
    ])
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    y_prob = baseline.predict_proba(X_test)

    metrics = {
        "model_name": "naive_baseline",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    metrics.update(evaluate_predictions(y_test, y_pred, y_prob))
    save_json(metrics, ROOT / "results" / "metrics_naive_baseline.json")
    return metrics
