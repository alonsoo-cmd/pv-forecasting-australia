# ======================================================
# ===================== INFERENCE ======================
# ======================================================

import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

# Reutilizamos el Dataset de training
from train import TimeSeriesDataset

# Modelo (elige uno solo aquí)
from models.LSTM import LSTM_two_layers

# Plots (ajusta imports si tus nombres difieren)
from utils.plots import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)


# -----------------------------
# Metrics
# -----------------------------
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mase(y_true, y_pred, y_train, m=24):
    """
    MASE correcto:
    - denominador: MAE del naive estacional en TRAIN (y_train)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    if len(y_train) <= m:
        raise ValueError(f"y_train demasiado corto para m={m}")

    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return np.mean(np.abs(y_true - y_pred)) / (scale + 1e-8)


def main():
    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    CONFIG_PATH = "config/timeseries.yaml"
    CKPT_PATH = "best_model.pth"
    DATA_DIR = Path("data/Processed")

    INFERENCE_XLSX = DATA_DIR / "inference.xlsx"
    TRAIN_XLSX = DATA_DIR / "train.xlsx"

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --------------------------------------------------
    # Load config
    # --------------------------------------------------
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)["model"]

    length = cfg["length"]
    lag = cfg["lag"]
    output_window = cfg["output_window"]
    batch_size = cfg["batch_size"]

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    train_cols = checkpoint["feature_columns"]
    input_size = checkpoint["input_size"]

    # --------------------------------------------------
    # Load inference data
    # --------------------------------------------------
    if not INFERENCE_XLSX.exists():
        raise FileNotFoundError(f"No existe {INFERENCE_XLSX}")

    df_inf = pd.read_excel(INFERENCE_XLSX, index_col=0)

    if "Energy" not in df_inf.columns:
        raise ValueError(
            "inference.xlsx debe contener columna 'Energy' para métricas. "
            "Si solo quieres predicciones sin métricas, dímelo y te lo adapto."
        )

    y_inf = df_inf["Energy"].to_numpy(dtype=np.float32)
    X_inf_df = df_inf.drop(columns=["Energy"])

    # --------------------------------------------------
    # Feature alignment (robusto)
    # - elimina extra
    # - añade faltantes con 0
    # - respeta orden del training
    # --------------------------------------------------
    X_inf_df = X_inf_df.reindex(columns=train_cols, fill_value=0.0)

    if X_inf_df.shape[1] != input_size:
        raise ValueError(f"Feature mismatch: {X_inf_df.shape[1]} vs {input_size}")

    X_inf = X_inf_df.to_numpy(dtype=np.float32)

    # --------------------------------------------------
    # Load train_y for MASE scaling
    # --------------------------------------------------
    if not TRAIN_XLSX.exists():
        raise FileNotFoundError(
            f"No existe {TRAIN_XLSX}. Necesito train.xlsx para escalar MASE correctamente."
        )

    df_train = pd.read_excel(TRAIN_XLSX, index_col=0)
    if "Energy" not in df_train.columns:
        raise ValueError("train.xlsx debe contener columna 'Energy'")

    train_y = df_train["Energy"].to_numpy(dtype=np.float32)

    # --------------------------------------------------
    # Build model (solo LSTM)
    # --------------------------------------------------
    model = LSTM_two_layers(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        output_size=cfg["output_size"],  # normalmente 1
        dropout=cfg["dropout"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Dataset & DataLoader
    # --------------------------------------------------
    ds = TimeSeriesDataset(
        X_inf,
        y_inf,
        length=length,
        lag=lag,
        output_window=output_window,
        stride=24,
    )

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------
    # Inference loop
    # --------------------------------------------------
    preds, targets = [], []

    with torch.no_grad():
        for x, t in dl:
            x = x.to(device)
            p = model(x).cpu().numpy()
            preds.append(p)
            targets.append(t.numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    # --------------------------------------------------
    # Convert log -> real (kWh)
    # (asumiendo que Energy está en log1p)
    # --------------------------------------------------
    preds_real = np.expm1(preds)
    targets_real = np.expm1(targets)
    train_y_real = np.expm1(train_y)

    # --------------------------------------------------
    # Metrics (horizon 0)
    # --------------------------------------------------
    y_true_h0 = targets_real[:, 0]
    y_pred_h0 = preds_real[:, 0]

    rmse_h0 = rmse(y_true_h0, y_pred_h0)
    mase_h0 = mase(y_true_h0, y_pred_h0, train_y_real, m=24)

    huber = nn.HuberLoss(delta=1.0)
    huber_h0 = huber(
        torch.tensor(y_pred_h0, dtype=torch.float32),
        torch.tensor(y_true_h0, dtype=torch.float32),
    ).item()

    print("\n================ METRICS (INFERENCE) ================")
    print(f"RMSE(h0)  : {rmse_h0:.4f} kWh")
    print(f"MASE(h0)  : {mase_h0:.4f}")
    print(f"Huber(h0) : {huber_h0:.4f}")
    print("=====================================================\n")

    print("Preds_real min/max:", float(preds_real.min()), float(preds_real.max()))
    print("Targets_real min/max:", float(targets_real.min()), float(targets_real.max()))

    # --------------------------------------------------
    # Save predictions (opcional)
    # --------------------------------------------------
    out_path = Path("preds_inference_horizons.csv")
    pd.DataFrame(preds_real).to_csv(out_path, index=False)
    print(f"Predicciones guardadas en: {out_path}")

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plot_continuous_horizon0(targets_real, preds_real)
    plot_one_day(targets_real, preds_real, day_idx=0)
    plot_scatter_real_vs_pred(targets_real, preds_real)


if __name__ == "__main__":
    main()
