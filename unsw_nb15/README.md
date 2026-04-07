# UNSW-NB15 Training Pipeline

This folder adds a modern dataset pipeline alongside the current KDD-style project.

Use it to:

- train a second intrusion detection model on UNSW-NB15
- compare KDD-style and UNSW-based approaches
- support your report statement that both datasets are considered

## Expected Input

Place the UNSW-NB15 CSV files in:

`unsw_nb15/data/`

Typical files:

- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

## Run

```powershell
python unsw_nb15/train_unsw_nb15.py
```

## Output

Saved files will be written to:

`unsw_nb15/artifacts/`

- `unsw_model.keras`
- `unsw_scaler.pkl`
- `unsw_label_encoders.pkl`
- `unsw_target_encoder.pkl`
- `unsw_metrics.json`

## Notes

- This pipeline is separate from the current KDD-based app.
- Keep the current app working, then later add a model selector if needed.

