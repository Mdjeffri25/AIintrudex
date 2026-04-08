from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv inside unsw_nb15/data/"
        )
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    drop_candidates = [column for column in ["id", "attack_cat"] if column in train_df.columns]
    if drop_candidates:
        train_df = train_df.drop(columns=drop_candidates)
        test_df = test_df.drop(columns=drop_candidates)

    target_column = "label"
    if target_column not in train_df.columns:
        raise ValueError("UNSW-NB15 expected target column `label` was not found.")

    categorical_columns = train_df.select_dtypes(include=["object"]).columns.tolist()
    categorical_columns = [column for column in categorical_columns if column != target_column]

    label_encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        combined = pd.concat([train_df[column], test_df[column]], axis=0).astype(str)
        encoder.fit(combined)
        train_df[column] = encoder.transform(train_df[column].astype(str))
        test_df[column] = encoder.transform(test_df[column].astype(str))
        label_encoders[column] = encoder

    target_encoder = LabelEncoder()
    target_encoder.fit(pd.concat([train_df[target_column], test_df[target_column]], axis=0))
    train_df[target_column] = target_encoder.transform(train_df[target_column])
    test_df[target_column] = target_encoder.transform(test_df[target_column])

    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler, label_encoders, target_encoder, list(x_train.columns)


def build_model(input_dim: int, num_classes: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    train_df, test_df = load_dataset()
    x_train, x_test, y_train, y_test, scaler, label_encoders, target_encoder, feature_columns = preprocess(train_df, test_df)

    model = build_model(input_dim=x_train.shape[1], num_classes=len(np.unique(y_train)))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(y_test, predictions, target_names=[str(v) for v in target_encoder.classes_], output_dict=True)
    confusion = confusion_matrix(y_test, predictions).tolist()

    model.save(ARTIFACTS_DIR / "unsw_model.keras")
    joblib.dump(scaler, ARTIFACTS_DIR / "unsw_scaler.pkl")
    joblib.dump(label_encoders, ARTIFACTS_DIR / "unsw_label_encoders.pkl")
    joblib.dump(target_encoder, ARTIFACTS_DIR / "unsw_target_encoder.pkl")

    metrics = {
        "dataset": "UNSW-NB15",
        "dataset_columns": 49,
        "input_features": len(feature_columns),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_loss": float(loss),
        "keras_accuracy": float(accuracy),
        "confusion_matrix": confusion,
        "classification_report": report,
        "feature_columns": feature_columns,
        "epochs_ran": len(history.history["loss"]),
    }
    (ARTIFACTS_DIR / "unsw_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("UNSW-NB15 training complete.")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
