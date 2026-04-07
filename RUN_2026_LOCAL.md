# Run Guide

This upgraded local version adds:

- separate user accounts
- history database
- CSV batch prediction
- live network monitoring prototype
- local agent-style alert explanation

## Files

- `run_backend.py` -> Flask backend API
- `app_streamlit_2026.py` -> Streamlit frontend
- `nids_app/` -> database, model, auth, live monitor, and agent services

## First Run

Open two terminals in the project folder.

### Terminal 1

```powershell
python run_backend.py
```

Backend will run at:

```text
http://127.0.0.1:8000
```

### Terminal 2

```powershell
streamlit run app_streamlit_2026.py
```

Frontend will open in the browser.

## How To Use

1. Register a new user.
2. Login with that user.
3. Use `Single Prediction` for one record.
4. Use `CSV Upload` with `sample_nids_input.csv`.
5. Use `Live Monitor` to capture a short real traffic window.
6. Open `History` to see saved results.

## Important Notes

- The database is stored in:

```text
C:\Users\jeffry\AppData\Local\nids_2026_runtime\nids_app.db
```

- Live monitoring uses packet capture on Windows. If it fails on another machine, install Npcap and run the terminal with enough packet-capture permission.

- The current live monitor is a practical prototype. It captures live packets and scores the window with a heuristic engine, while the CSV/manual prediction path uses the trained deep learning model directly.

- The local agent layer is currently deterministic and free. It explains severity, reasons, and recommended actions without relying on paid cloud AI.
