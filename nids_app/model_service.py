from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from .config import LABEL_ENCODERS_PATH, MODEL_PATH, MODEL_WEIGHTS_PATH, SCALER_PATH, TARGET_ENCODER_PATH
from .constants import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


@dataclass
class PredictionResult:
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, Any]


@lru_cache(maxsize=1)
def load_artifacts():
    model = _load_model()
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)
    return model, scaler, label_encoders, target_encoder


def _build_model():
    inputs = keras.Input(shape=(41,))
    x = layers.Dense(128, activation="relu", name="dense")(inputs)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu", name="dense_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax", name="dense_3")(x)
    return keras.Model(inputs, outputs)


def _load_model():
    if MODEL_WEIGHTS_PATH.exists():
        model = _build_model()
        dummy = np.zeros((1, 41))
        model.predict(dummy, verbose=0)
        data = np.load(MODEL_WEIGHTS_PATH)
        weight_list = [data[key] for key in sorted(data.files, key=lambda value: int(value.replace("arr_", "")))]
        model.set_weights(weight_list)
        return model
    return keras.models.load_model(MODEL_PATH)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    prepared = df[FEATURE_COLUMNS].copy()
    _, _, label_encoders, _ = load_artifacts()

    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype(str).str.strip()
        allowed = set(label_encoders[column].classes_)
        invalid_values = sorted(set(prepared[column]) - allowed)
        if invalid_values:
            joined = ", ".join(invalid_values[:5])
            raise ValueError(f"Unsupported values in {column}: {joined}")
        prepared[column] = label_encoders[column].transform(prepared[column])

    prepared = prepared.apply(pd.to_numeric, errors="coerce")
    if prepared.isna().any().any():
        invalid_columns = prepared.columns[prepared.isna().any()].tolist()
        raise ValueError(f"Invalid numeric values in columns: {', '.join(invalid_columns)}")
    return prepared


def predict_kdd_records(records: List[Dict[str, Any]]) -> List[PredictionResult]:
    raw_df = pd.DataFrame(records)
    prepared = _prepare_dataframe(raw_df)
    model, scaler, _, target_encoder = load_artifacts()
    scaled = scaler.transform(prepared)
    probabilities = model.predict(scaled, verbose=0)
    class_names = list(target_encoder.classes_)

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


def get_available_models() -> list[dict]:
    models = [{"key": "kdd", "label": "KDD 41-Feature Model", "available": True}]
    try:
        from .unsw_service import get_unsw_feature_columns, unsw_available

        models.append(
            {
                "key": "unsw",
                "label": f"UNSW-NB15 Model ({len(get_unsw_feature_columns()) or 49} features)",
                "available": bool(unsw_available()),
            }
        )
    except Exception:
        models.append({"key": "unsw", "label": "UNSW-NB15 Model (49 features)", "available": False})
    return models


def predict_records(records: List[Dict[str, Any]], model_name: str = "kdd") -> List[PredictionResult]:
    if model_name == "unsw":
        from .unsw_service import predict_unsw_records

        return predict_unsw_records(records)
    return predict_kdd_records(records)
