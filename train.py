
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split, cross_val_score

from ml_core import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_FILE = "laptop_data.csv"
MODEL_FILE = "model.pkl"
METRICS_FILE = "metrics.json"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            f"Place laptop_data.csv in the working directory."
        )
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    required_cols = {
        "Company", "TypeName", "Inches", "ScreenResolution", "Cpu",
        "Ram", "Memory", "Gpu", "OpSys", "Weight", "Price"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def main():
    logger.info("Loading dataset from %s", DATA_FILE)
    df = load_data(DATA_FILE)
    logger.info("Loaded %d rows", len(df))

    x = df.drop(columns=["Price"])
    y = np.log1p(df["Price"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    logger.info("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring="r2")
    logger.info("CV R2 scores: %s", np.round(cv_scores, 4))
    logger.info("CV R2 mean: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())

    logger.info("Fitting final model on full training set...")
    pipeline.fit(x_train, y_train)

    logger.info("Evaluating on held-out test set...")
    y_pred_log = pipeline.predict(x_test)

    r2 = r2_score(y_test, y_pred_log)

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    logger.info("Test R2:    %.4f", r2)
    logger.info("Test RMSE:  %.2f", rmse)
    logger.info("Test MAPE:  %.2f%%", mape * 100)

    metrics = {
        "cv_r2_scores": cv_scores.tolist(),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "test_r2": float(r2),
        "test_rmse": float(rmse),
        "test_mape": float(mape),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_FILE)

    joblib.dump(pipeline, MODEL_FILE)
    logger.info("Saved trained pipeline to %s", MODEL_FILE)


if __name__ == "__main__":
    main()
