"""
Crop yield regression model: training, evaluation, and persistence.
Uses a Random Forest regressor with leave-one-season-out cross-validation.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "ndvi_mean", "ndvi_max", "rainfall_mm", "temp_mean_c", "temp_max_c",
    "humidity_pct", "growing_days", "fertilizer_kg", "soil_ph",
    "soil_nitrogen", "irrigation", "farm_size_ha", "crop_encoded",
]
TARGET_COL = "yield_kg_per_ha"
MODEL_PATH = Path("assets/yield_model.pkl")
ENCODER_PATH = Path("assets/crop_encoder.pkl")
METRICS_PATH = Path("assets/metrics.json")


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df = df.copy()
    df["crop_encoded"] = le.fit_transform(df["crop"])
    return df, le


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train(df: pd.DataFrame) -> tuple[Pipeline, LabelEncoder, dict]:
    """
    Train the yield regression model.

    Returns
    -------
    tuple[Pipeline, LabelEncoder, dict]
        Fitted pipeline, label encoder, and evaluation metrics.
    """
    df, le = encode_features(df)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Leave-one-season-out evaluation
    seasons = df["season"].unique()
    loo_maes = []
    for season in seasons:
        mask_test = df["season"] == season
        X_tr, X_te = X[~mask_test], X[mask_test]
        y_tr, y_te = y[~mask_test], y[mask_test]
        p = build_pipeline()
        p.fit(X_tr, y_tr)
        loo_maes.append(mean_absolute_error(y_te, p.predict(X_te)))

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    fi = dict(zip(FEATURE_COLS, pipeline.named_steps["reg"].feature_importances_.tolist()))
    metrics = {
        "r2_train": round(float(r2_score(y, y_pred)), 4),
        "mae_train": round(float(mean_absolute_error(y, y_pred)), 2),
        "rmse_train": round(float(np.sqrt(mean_squared_error(y, y_pred))), 2),
        "loo_mae_mean": round(float(np.mean(loo_maes)), 2),
        "loo_mae_std": round(float(np.std(loo_maes)), 2),
        "feature_importance": fi,
    }
    return pipeline, le, metrics


def predict_yield(pipeline: Pipeline, le: LabelEncoder, features: dict) -> dict:
    """Predict crop yield for a single farm record."""
    data = {**features}
    data["crop_encoded"] = int(le.transform([data.pop("crop")])[0])
    X = pd.DataFrame([data])[FEATURE_COLS]
    pred = float(pipeline.predict(X)[0])
    return {"yield_kg_per_ha": round(max(0, pred), 1)}


def save_model(pipeline: Pipeline, le: LabelEncoder, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def load_model() -> tuple[Pipeline, LabelEncoder]:
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


if __name__ == "__main__":
    from data_generator import generate_crop_dataset
    df = generate_crop_dataset()
    pipeline, le, metrics = train(df)
    save_model(pipeline, le, metrics)
    print(json.dumps(metrics, indent=2))
