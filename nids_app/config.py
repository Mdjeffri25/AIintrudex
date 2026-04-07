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
