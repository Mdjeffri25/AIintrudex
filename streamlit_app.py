from __future__ import annotations

import threading
import time

import requests

from app_streamlit_pro import main
from nids_app.api_server import app as backend_app
from nids_app.database import init_db


def _run_backend():
    init_db()
    backend_app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)


def ensure_backend_running():
    try:
        response = requests.get("http://127.0.0.1:8000/api/health", timeout=1)
        if response.ok:
            return
    except Exception:
        pass

    thread = threading.Thread(target=_run_backend, daemon=True)
    thread.start()

    for _ in range(15):
        try:
            response = requests.get("http://127.0.0.1:8000/api/health", timeout=1)
            if response.ok:
                return
        except Exception:
            time.sleep(0.4)


if __name__ == "__main__":
    ensure_backend_running()
    main()
