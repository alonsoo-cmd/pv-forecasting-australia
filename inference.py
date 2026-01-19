import torch
import numpy as np
import yaml
import pandas as pd
import torch.nn as nn

from torch.utils.data import DataLoader
from train import TimeSeriesDataset

from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN
from models.Transformer import TransformerForecast

from utils.plots import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)
from utils.metrics import rmse, mase


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    with open("config/timeseries.yaml") as f:
        cfg = yaml.safe_load(f)["model"]

    # --------------------------------------------------
    # Load checkpoint (PyTorch 2.6+ compatible)
    # --------------------------------------------------
    checkpoint = torch.load(
        "best_model.pth",
        map_location=device,
        weights_only=False,
    )

    model_name = checkpoint["model_name"]
    train_cols = checkpoint["feature_columns"]
    input_size = checkpoint["input_size"]

    # --------------------------------------------------
    # Load inference data as DataFrame
    # --------------------------------------------------
    df = pd.read_excel("data/Processed/inference.xlsx", index_col=0)

    y = df["Energy"].to_numpy(dtype=np.float32)
    X_df = df.drop(columns=["Energy"])

    # --------------------------------------------------
    # FEATURE ALIGNMENT (ROBUST, SIN DUMMIES)
    # --------------------------------------------------
    # - elimina columnas extra
    # - añade columnas faltantes con 0
    # - respeta el orden del training
    X_df = X_df.reindex(columns=train_cols, fill_value=0.0)

    # Safety check
    assert X_df.shape[1] == input_size, (
        f"Feature mismatch: {X_df.shape[1]} vs {input_size}"
    )

    X = X_df.to_numpy(dtype=np.float32)

    # --------------------------------------------------
    # Build model (same as training)
    # --------------------------------------------------
    model = {
        "LSTM": LSTM_two_layers,
        "GRU": GRU_two_layers,
        "LSTM_FCN": LSTM_FCN,
        "Transformer": TransformerForecast,
    }[model_name](
        input_size,
        cfg["hidden_size"],
        cfg["output_window"],
        cfg["dropout"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Dataset & DataLoader
    # --------------------------------------------------
    ds = TimeSeriesDataset(
        X,
        y,
        cfg["length"],
        cfg["lag"],
        cfg["output_window"],
        stride=24,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    preds, targets = [], []

    with torch.no_grad():
        for x, t in dl:
            preds.append(model(x.to(device)).cpu().numpy())
            targets.append(t.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # --------------------------------------------------
    # Metrics / Losses (NO TRAMPA, TODO EL DATASET)
    # --------------------------------------------------
    rmse_all = rmse(targets[:, 0], preds[:, 0])
    mase_all = mase(targets[:, 0], preds[:, 0], targets[:, 0])

    huber = nn.HuberLoss(delta=1.0)
    huber_loss = huber(
        torch.tensor(preds[:, 0]),
        torch.tensor(targets[:, 0]),
    ).item()

    print("\n================ METRICS (ALL DATA) ================")
    print(f"RMSE  : {rmse_all:.4f}")
    print(f"MASE  : {mase_all:.4f}")
    print(f"Huber : {huber_loss:.4f}")
    print("===================================================\n")

    print("Preds min/max:", preds.min(), preds.max())
    print("Targets min/max:", targets.min(), targets.max())

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plot_continuous_horizon0(targets, preds)
    plot_one_day(targets, preds, day_idx=0)
    plot_scatter_real_vs_pred(targets, preds)


if __name__ == "__main__":
    main()
