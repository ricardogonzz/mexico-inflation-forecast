# Mexico CPI Forecast 2030
### Predicting Mexico's Inflation Trajectory with Recurrent Neural Networks

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange?logo=tensorflow)](https://tensorflow.org)
![Published](https://img.shields.io/badge/Published-CIMCIA%202025%20UNAM-green)

> **Paper:** *"El Futuro del Poder Adquisitivo en México: Proyección del IPC y la Canasta Básica hacia 2030 mediante Redes Neuronales Recurrentes"*  
> Presented and approved at **CIMCIA 2025 — UNAM International Congress**  
> Authors: Ricardo González Muñoz · Carlos A. Baltazar Vilchis · Ariadna Martínez Chaparro

---

## Overview

Inflation is one of the most pressing economic challenges in Mexico, directly impacting the purchasing power of millions of households. This project uses 42 years of Consumer Price Index (CPI) data — from 1982 to 2024 — to train and compare four machine learning models for long-range forecasting:

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **GRU** ✅ | **1.6603** | **1.4011** | **1.29%** |
| LSTM | 2.4159 | 1.9766 | 1.86% |
| Prophet | 3.7347 | 3.0065 | 2.74% |
| XGBoost | 10.6289 | 9.2276 | 7.87% |

The **GRU model** won with a MAPE of just 1.29%, meaning roughly 99% accuracy on unseen data. Its projection: **IPC ≈ 159.1 by December 2030**, which translates to a monthly basic food basket cost of **$1,538–$2,062 MXN per person**.

---

## Repository Structure

```
mexico-inflation-forecast/
├── src/
│   ├── gru_model.py          # Best model — GRU (MAPE 1.29%)
│   ├── lstm_model.py         # Runner-up — LSTM (MAPE 1.86%)
│   ├── prophet_model.py      # Baseline — Prophet (MAPE 2.74%)
│   └── xgboost_model.py      # Baseline — XGBoost (MAPE 7.87%)
├── data/
│   ├── ipc_historico_1982_2024.csv     # Source: INEGI
│   ├── inflacion_mensual.csv
│   └── inflacion_acumulada_anual.csv
├── results/
│   ├── gru_forecast_2030.png
│   ├── lstm_forecast_2030.png
│   ├── prophet_forecast_2030.png
│   └── xgboost_forecast_2030.png
└── requirements.txt
```

---

## Key Results

The GRU model projects a persistent inflationary trend through 2030:

- **IPC projection for Dec 2030:** ~159.1
- **Basic food basket cost per person (2030):**
  - Market scenario: ~$1,538 MXN/month
  - Supermarket scenario: ~$2,062 MXN/month

Recurrent architectures (GRU and LSTM) significantly outperformed both Prophet and XGBoost on this task, confirming that temporal memory mechanisms are well-suited for long-range economic forecasting.

---

## Setup

```bash
git clone https://github.com/pollowtf/mexico-inflation-forecast
cd mexico-inflation-forecast
pip install -r requirements.txt
```

Run any model from the `src/` directory:

```bash
python src/gru_model.py
```

---

## Data Sources

- **IPC Historical Series (1982–2024):** [INEGI — INPC](https://www.inegi.org.mx/temas/inpc/)
- **Basic Food Basket Prices (2025):** [SEDECO CDMX](https://www.sedeco.cdmx.gob.mx/servicios/servicio/seguimiento-de-precios-de-la-canasta-basica)

---

## Methodology

Each model required a different preprocessing strategy:

- **GRU / LSTM:** Series normalized to [0,1] with `MinMaxScaler`. Sequences of 24 months used as input windows. Early stopping with `patience=15` to prevent overfitting.
- **XGBoost:** Applied first-differencing to remove trend (XGBoost cannot extrapolate trends natively). Lag features from the previous 24 months used as input.
- **Prophet:** Used built-in cross-validation with a 30-year initial training window and 2-year evaluation horizons.

Models were evaluated on a held-out test set (last 20% of the series, chronologically), using RMSE, MAE, and MAPE as metrics.

---

## Citation

If you use this work, please cite:

```
González Muñoz, R., Baltazar Vilchis, C. A., & Martínez Chaparro, A. (2025).
El Futuro del Poder Adquisitivo en México: Proyección del IPC y la Canasta Básica
hacia 2030 mediante Redes Neuronales Recurrentes.
Presented at CIMCIA 2025, UNAM.
```

---

## Author

**Ricardo González Muñoz**  
LIA Student — UAEMéx Centro Universitario Atlacomulco  
[LinkedIn](https://linkedin.com/in/ricardogonzz/) · rgonzalezm015@alumno.uaemex.mx
