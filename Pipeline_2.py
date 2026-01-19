# ======================================================
# ===================== IMPORTS ========================
# ======================================================
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from models.LSTM import LSTM_two_layers
from utils.graph_pipeline import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)

# ======================================================
# =================== DATASET ==========================
# ======================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, length, lag, output_window, stride=1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if self.y.ndim == 2:
            self.y = self.y.squeeze(-1)

        self.length = length
        self.lag = lag
        self.output_window = output_window
        self.stride = stride

        N = len(X)
        t0_min = lag + length
        t0_max = N - output_window
        self.forecast_starts = np.arange(t0_min, t0_max + 1, stride)

    def __len__(self):
        return len(self.forecast_starts)

    def __getitem__(self, idx):
        t0 = self.forecast_starts[idx]
        x = self.X[t0 - self.lag - self.length : t0 - self.lag]
        y = self.y[t0 : t0 + self.output_window]
        return x, y


# ======================================================
# =================== HELPERS ==========================
# ======================================================
def load_split(name, base_path, y_col="Energy"):
    df = pd.read_excel(Path(base_path) / f"{name}.xlsx", index_col=0)
    y = df[y_col].to_numpy(dtype=np.float32)
    X = df.drop(columns=[y_col]).to_numpy(dtype=np.float32)
    return X, y


def training_model(model, dataloader, num_epochs, lr, device):
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] MAE: {running_loss/len(dataloader):.6f}")


# ======================================================
# =================== TRAINING =========================
# ======================================================
def training():
    # ------------------ CONFIG -------------------------
    with open("./config/timeseries.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ------------------ DATA ---------------------------
    train_x, train_y = load_split("train", "./data/Processed")

    input_size = train_x.shape[1]
    hidden_size = config["model"]["hidden_size"]
    output_size = config["model"]["output_size"]
    output_window = config["model"]["output_window"]
    dropout = config["model"]["dropout"]
    batch_size = config["model"]["batch_size"]
    epochs = config["model"]["epochs"]
    lr = config["model"]["learning_rate"]
    lag = config["model"]["lag"]
    length = config["model"]["length"]

    ds_train = TimeSeriesDataset(
        train_x, train_y, length, lag, output_window
    )

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    # ------------------ MODEL --------------------------
    model = LSTM_two_layers(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout=dropout,
    ).to(device)

    # ------------------ TRAIN --------------------------
    training_model(model, dl_train, epochs, lr, device)

    # ------------------ SAVE ---------------------------
    MODEL_PATH = "./checkpoints/lstm_two_layers.pt"
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_type": "LSTM_two_layers",
            "model_state_dict": model.state_dict(),
            "config": {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
                "dropout": dropout,
                "length": length,
                "lag": lag,
                "output_window": output_window,
            },
        },
        MODEL_PATH,
    )

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    training()
