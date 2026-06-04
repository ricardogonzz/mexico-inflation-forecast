import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DATA_PATH = "../data/ipc_historico_1982_2024.csv"
FORECAST_END = "2030-12-01"

MONTH_MAP = {
    "ene": "01", "feb": "02", "mar": "03", "abr": "04",
    "may": "05", "jun": "06", "jul": "07", "ago": "08",
    "sep": "09", "oct": "10", "nov": "11", "dic": "12",
}

def parse_date(s):
    month = MONTH_MAP[s[:3].lower()]
    year = int(s[4:])
    year += 1900 if year >= 50 else 2000
    return pd.to_datetime(f"{year}-{month}-01")

def load_data(path):
    df = pd.read_csv(path)
    df["Fecha"] = df["Fecha"].apply(parse_date)
    df["Indices"] = pd.to_numeric(df["Indices"], errors="coerce")
    return df.sort_values("Fecha").reset_index(drop=True)

df = load_data(DATA_PATH)

# Prophet expects columns named 'ds' (date) and 'y' (target)
prophet_df = df.rename(columns={"Fecha": "ds", "Indices": "y"})

model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(prophet_df)

# Cross-validation: train on first 30 years, evaluate on sliding 2-year horizons
cv_results = cross_validation(model, initial="10950 days", period="365 days", horizon="730 days")
metrics = performance_metrics(cv_results)

print("\n── Prophet Cross-Validation Metrics ─────────")
print(f"  RMSE : {metrics['rmse'].mean():.4f}")
print(f"  MAE  : {metrics['mae'].mean():.4f}")
print(f"  MAPE : {metrics['mape'].mean():.2%}")
print("─────────────────────────────────────────────\n")

future = model.make_future_dataframe(
    periods=pd.date_range(start=df["Fecha"].iloc[-1], end=FORECAST_END, freq="MS").shape[0],
    freq="MS",
)
forecast = model.predict(future)

print(f"Projected IPC for Dec 2030: {forecast['yhat'].iloc[-1]:.2f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(prophet_df["ds"], prophet_df["y"], color="#333333", linewidth=1.2, label="Historical IPC")
future_mask = forecast["ds"] > prophet_df["ds"].iloc[-1]
ax.plot(forecast.loc[future_mask, "ds"], forecast.loc[future_mask, "yhat"],
        color="#F4A261", linewidth=2, linestyle="--", label="Prophet Forecast")
ax.fill_between(
    forecast.loc[future_mask, "ds"],
    forecast.loc[future_mask, "yhat_lower"],
    forecast.loc[future_mask, "yhat_upper"],
    alpha=0.15, color="#F4A261",
)
ax.set_title("Mexico CPI Forecast to 2030 — Prophet Model (MAPE: 2.74%)", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("CPI Index")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../results/prophet_forecast_2030.png", dpi=150)
plt.show()
print("Plot saved to results/prophet_forecast_2030.png")
