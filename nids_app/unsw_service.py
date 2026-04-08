from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from .config import (
    UNSW_LABEL_ENCODERS_PATH,
    UNSW_METRICS_PATH,
    UNSW_MODEL_PATH,
    UNSW_SCALER_PATH,
    UNSW_TARGET_ENCODER_PATH,
)
from .model_service import PredictionResult

UNSW_CATEGORICAL_COLUMNS = ["proto", "service", "state"]


def unsw_available() -> bool:
    return all(
        path.exists()
        for path in [
            UNSW_MODEL_PATH,
            UNSW_SCALER_PATH,
            UNSW_LABEL_ENCODERS_PATH,
            UNSW_TARGET_ENCODER_PATH,
            UNSW_METRICS_PATH,
        ]
    )


@lru_cache(maxsize=1)
def load_unsw_metadata() -> dict:
    if not UNSW_METRICS_PATH.exists():
        return {"dataset": "UNSW-NB15", "feature_columns": [], "available": False}
    return json.loads(UNSW_METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_unsw_artifacts():
    if not unsw_available():
        raise FileNotFoundError(
            "UNSW-NB15 artifacts are not available yet. Run python unsw_nb15/train_unsw_nb15.py after placing the dataset files."
        )
    from tensorflow import keras

    model = keras.models.load_model(UNSW_MODEL_PATH)
    scaler = joblib.load(UNSW_SCALER_PATH)
    label_encoders = joblib.load(UNSW_LABEL_ENCODERS_PATH)
    target_encoder = joblib.load(UNSW_TARGET_ENCODER_PATH)
    metadata = load_unsw_metadata()
    return model, scaler, label_encoders, target_encoder, metadata


def get_unsw_feature_columns() -> List[str]:
    return list(load_unsw_metadata().get("feature_columns", []))


def prepare_unsw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = get_unsw_feature_columns()
    if not feature_columns:
        raise ValueError("UNSW-NB15 feature metadata is not available yet.")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing UNSW-NB15 columns: {', '.join(missing[:10])}")

    prepared = df[feature_columns].copy()
    _, _, label_encoders, _, _ = load_unsw_artifacts()

    for column in UNSW_CATEGORICAL_COLUMNS:
        if column not in prepared.columns or column not in label_encoders:
            continue
        prepared[column] = prepared[column].astype(str).str.strip()
        allowed = set(label_encoders[column].classes_)
        invalid_values = sorted(set(prepared[column]) - allowed)
        if invalid_values:
            joined = ", ".join(invalid_values[:5])
            raise ValueError(f"Unsupported UNSW values in {column}: {joined}")
        prepared[column] = label_encoders[column].transform(prepared[column])

    prepared = prepared.apply(pd.to_numeric, errors="coerce")
    if prepared.isna().any().any():
        invalid_columns = prepared.columns[prepared.isna().any()].tolist()
        raise ValueError(f"Invalid UNSW numeric values in columns: {', '.join(invalid_columns[:10])}")
    return prepared


def predict_unsw_records(records: List[Dict[str, Any]]) -> List[PredictionResult]:
    raw_df = pd.DataFrame(records)
    prepared = prepare_unsw_dataframe(raw_df)
    model, scaler, _, target_encoder, _ = load_unsw_artifacts()
    scaled = scaler.transform(prepared)
    probabilities = model.predict(scaled, verbose=0)
    class_names = [str(name) for name in target_encoder.classes_]

    results: List[PredictionResult] = []
    for original_record, row_proba in zip(records, probabilities):
        predicted_index = int(np.argmax(row_proba))
        predicted_label = class_names[predicted_index]
        results.append(
            PredictionResult(
                predicted_label=predicted_label,
                confidence=float(np.max(row_proba) * 100),
                probabilities={class_name: float(prob * 100) for class_name, prob in zip(class_names, row_proba)},
                features=original_record,
            )
        )
    return results
