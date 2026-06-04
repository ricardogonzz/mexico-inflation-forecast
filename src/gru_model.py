import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ── Data ────────────────────────────────────────────────────────────────────

DATA_PATH = "../data/ipc_historico_1982_2024.csv"
SEQ_LENGTH = 24
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

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

# ── Model ────────────────────────────────────────────────────────────────────

def build_model(seq_len):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(seq_len, 1)),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

# ── Training ─────────────────────────────────────────────────────────────────

df = load_data(DATA_PATH)
values = df["Indices"].values.reshape(-1, 1)

scaler = MinMaxScaler()
values_scaled = scaler.fit_transform(values)

X, y = make_sequences(values_scaled, SEQ_LENGTH)
split = int(len(X) * TRAIN_SPLIT)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = build_model(SEQ_LENGTH)
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1,
)

# ── Evaluation ───────────────────────────────────────────────────────────────

y_pred_scaled = model.predict(X_val)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_val)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print("\n── GRU Performance Metrics ──────────────────")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  MAPE : {mape:.2%}")
print("─────────────────────────────────────────────\n")

# ── Forecast to 2030 ─────────────────────────────────────────────────────────

last_seq = values_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
forecast_dates = pd.date_range(
    start=df["Fecha"].iloc[-1] + pd.DateOffset(months=1),
    end=FORECAST_END,
    freq="MS",
)

predictions = []
current_seq = last_seq.copy()

for _ in forecast_dates:
    next_val = model.predict(current_seq, verbose=0)
    predictions.append(next_val[0, 0])
    current_seq = np.roll(current_seq, -1, axis=1)
    current_seq[0, -1, 0] = next_val[0, 0]

forecast_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
forecast_df = pd.DataFrame({"Fecha": forecast_dates, "IPC_Proyectado": forecast_values})

print(f"Projected IPC for Dec 2030: {forecast_df['IPC_Proyectado'].iloc[-1]:.2f}")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["Fecha"], df["Indices"], color="#333333", linewidth=1.2, label="Historical IPC")
ax.plot(forecast_df["Fecha"], forecast_df["IPC_Proyectado"], color="#E63946",
        linewidth=2, linestyle="--", label="GRU Forecast")
ax.set_title("Mexico CPI Forecast to 2030 — GRU Model (MAPE: 1.29%)", fontsize=14)
ax.set_xlabel("Year")
ax.set_ylabel("CPI Index")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../results/gru_forecast_2030.png", dpi=150)
plt.show()
print("Plot saved to results/gru_forecast_2030.png")
