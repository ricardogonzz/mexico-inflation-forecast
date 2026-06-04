import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

DATA_PATH = "../data/ipc_historico_1982_2024.csv"
N_LAGS = 24
TRAIN_SPLIT = 0.80
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

# XGBoost struggles with trend extrapolation, so we train on first-differences
# (month-over-month changes) and reconstruct the level at inference time.
df["diff"] = df["Indices"].diff()
df = df.dropna().reset_index(drop=True)

for lag in range(1, N_LAGS + 1):
    df[f"lag_{lag}"] = df["diff"].shift(lag)
df = df.dropna().reset_index(drop=True)

feature_cols = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
X = df[feature_cols].values
y = df["diff"].values

split = int(len(X) * TRAIN_SPLIT)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

y_pred_diff = model.predict(X_val)

# Reconstruct levels from differences
last_known = df["Indices"].iloc[split + N_LAGS - 1]
y_pred_levels = last_known + np.cumsum(y_pred_diff)
y_true_levels = df["Indices"].iloc[split + N_LAGS : split + N_LAGS + len(y_val)].values

rmse = np.sqrt(mean_squared_error(y_true_levels, y_pred_levels))
mae  = mean_absolute_error(y_true_levels, y_pred_levels)
mape = mean_absolute_percentage_error(y_true_levels, y_pred_levels)

print("\n── XGBoost Performance Metrics ──────────────")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  MAPE : {mape:.2%}")
print("─────────────────────────────────────────────\n")

# Iterative forecast
forecast_dates = pd.date_range(
    start=df["Fecha"].iloc[-1] + pd.DateOffset(months=1),
    end=FORECAST_END,
    freq="MS",
)

lag_window = list(df["diff"].values[-N_LAGS:])
forecast_diffs = []
for _ in forecast_dates:
    features = np.array(lag_window[-N_LAGS:][::-1]).reshape(1, -1)
    next_diff = model.predict(features)[0]
    forecast_diffs.append(next_diff)
    lag_window.append(next_diff)

last_ipc = df["Indices"].iloc[-1]
forecast_values = last_ipc + np.cumsum(forecast_diffs)
forecast_df = pd.DataFrame({"Fecha": forecast_dates, "IPC_Proyectado": forecast_values})

print(f"Projected IPC for Dec 2030: {forecast_df['IPC_Proyectado'].iloc[-1]:.2f}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["Fecha"], df["Indices"], color="#333333", linewidth=1.2, label="Historical IPC")
ax.plot(forecast_df["Fecha"], forecast_df["IPC_Proyectado"], color="#2A9D8F",
        linewidth=2, linestyle="--", label="XGBoost Forecast")
ax.set_title("Mexico CPI Forecast to 2030 — XGBoost Model (MAPE: 7.87%)", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("CPI Index")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../results/xgboost_forecast_2030.png", dpi=150)
plt.show()
print("Plot saved to results/xgboost_forecast_2030.png")
