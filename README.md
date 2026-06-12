---
title: Laptop Price Predictor
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---

# Laptop Price Predictor

Predicts laptop prices from specifications (brand, CPU, GPU, RAM, storage,
screen, etc.) using a Linear Regression model trained on a 1,303-laptop
dataset.

## Results

| Metric | Value |
|---|---|
| CV R² (mean, 5-fold) | 0.827 |
| Test R² | 0.851 |
| Test RMSE | ₹18,366 |
| Test MAPE | 19.0% |

## Project Structure

```
.
├── ml_core.py       # feature engineering + pipeline definition (shared)
├── train.py         # trains model, saves model.pkl + metrics.json
├── app.py           # Streamlit web app for live predictions
├── laptop_data.csv  # dataset
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train.py
```

This produces `model.pkl` and `metrics.json`.

## Run the app

```bash
streamlit run app.py
```

## Approach

1. **Feature engineering** — parses raw spec strings (e.g. `"16GB"`,
   `"256GB SSD + 1TB HDD"`, `"Intel Core i7 2.8GHz"`) into structured
   numeric/categorical features: RAM (GB), weight (kg), storage by type
   (SSD/HDD/Flash/Hybrid), GPU brand & model number, CPU series, OS
   category, screen PPI, touchscreen/IPS flags.
2. **Preprocessing** — `ColumnTransformer` with separate pipelines for
   categorical (one-hot), binary, GPU model (ordinal), and numeric
   (KNN-imputed, Yeo-Johnson power transform, robust-scaled) features.
3. **Model** — Linear Regression on `log1p(Price)`, evaluated with
   5-fold cross-validation and a held-out test set.

## Notes

- Target is log-transformed (`log1p`) to handle the right-skewed price
  distribution; predictions are converted back with `expm1`.
- Feature engineering handles single-row inputs (e.g. no secondary
  storage, no `+` in the `Memory` field) without errors — required for
  live predictions via the Streamlit app.
