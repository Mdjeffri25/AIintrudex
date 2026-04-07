import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_APPDATA = Path(os.environ.get("LOCALAPPDATA", BASE_DIR))
DATA_DIR = LOCAL_APPDATA / "nids_2026_runtime"
DB_PATH = DATA_DIR / "nids_app.db"
MODEL_PATH = BASE_DIR / "dl_model.h5"
MODEL_WEIGHTS_PATH = BASE_DIR / "model_weights.npz"
SCALER_PATH = BASE_DIR / "scaler.pkl"
LABEL_ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"
TARGET_ENCODER_PATH = BASE_DIR / "target_encoder.pkl"

API_HOST = "127.0.0.1"
API_PORT = 8000

UNSW_BASE_DIR = BASE_DIR / "unsw_nb15"
UNSW_ARTIFACTS_DIR = UNSW_BASE_DIR / "artifacts"
UNSW_MODEL_PATH = UNSW_ARTIFACTS_DIR / "unsw_model.keras"
UNSW_SCALER_PATH = UNSW_ARTIFACTS_DIR / "unsw_scaler.pkl"
UNSW_LABEL_ENCODERS_PATH = UNSW_ARTIFACTS_DIR / "unsw_label_encoders.pkl"
UNSW_TARGET_ENCODER_PATH = UNSW_ARTIFACTS_DIR / "unsw_target_encoder.pkl"
UNSW_METRICS_PATH = UNSW_ARTIFACTS_DIR / "unsw_metrics.json"
