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
import pickle

from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN
from models.Transformer import TransformerForecast

# ======================================================
# =================== DATASET ==========================
# ======================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, length, lag, output_window, stride=1):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(-1)

        self.length = length
        self.lag = lag
        self.output_window = output_window

        N = len(X)
        t0_min = lag + length
        t0_max = N - output_window
        self.starts = np.arange(t0_min, t0_max + 1, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        t0 = self.starts[idx]
        x = self.X[t0 - self.lag - self.length : t0 - self.lag]
        y = self.y[t0 : t0 + self.output_window]
        return x, y


# ======================================================
# =================== METRICS ==========================
# ======================================================
def mase_daylight(y_true, y_pred, y_train, m=24):
    """
    MASE robusto para PV (solo horas con producción)
    """
    y_train = y_train.squeeze()
    mask = (y_train > 0) & (np.roll(y_train, m) > 0)
    naive_diff = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_diff[mask[m:]])

    return np.mean(np.abs(y_true - y_pred)) / (scale + 1e-8)


# ======================================================
# =================== HELPERS ==========================
# ======================================================
def load_split(name, base_path):
    df = pd.read_excel(Path(base_path) / f"{name}.xlsx", index_col=0)
    y = df["Energy"].to_numpy(dtype=np.float32)
    X = df.drop(columns=["Energy"]).to_numpy(dtype=np.float32)
    return X, y


def train_model(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)

            if preds.shape != y.shape:
                if y.shape[-1] == 1 and y.dim() == preds.dim() + 1:
                    y = y.squeeze(-1)
                elif preds.shape[-1] == 1 and preds.dim() == y.dim() + 1:
                    preds = preds.squeeze(-1)

            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.6f}")


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x).cpu().numpy()
            y_np = y.numpy()
            if y_np.shape[-1] == 1 and y_np.ndim == out.ndim + 1:
                y_np = y_np.squeeze(-1)
            preds.append(out)
            targets.append(y.numpy())

    return np.concatenate(preds), np.concatenate(targets)


# ======================================================
# =================== TRAIN MAIN =======================
# ======================================================
def main():
    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    with open("./config/timeseries.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    DATA_PATH = "./data/Processed"

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    train_x, train_y = load_split("train", DATA_PATH)
    val_x, val_y = load_split("val", DATA_PATH)

    input_size = train_x.shape[1]

    # --------------------------------------------------
    # Dataset & Loader
    # --------------------------------------------------
    ds_train = TimeSeriesDataset(
        train_x, train_y,
        cfg["length"], cfg["lag"], cfg["output_window"]
    )
    ds_val = TimeSeriesDataset(
        val_x, val_y,
        cfg["length"], cfg["lag"], cfg["output_window"],
        stride=24
    )

    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False)

    # --------------------------------------------------
    # Load stats (para desnormalizar y MASE)
    # --------------------------------------------------
    with open(f"{DATA_PATH}/stats.pkl", "rb") as f:
        stats = pickle.load(f)

    y_mu = stats["Y"]["mu"]["Energy"]
    y_std = stats["Y"]["std"]["Energy"]

    train_y_real = train_y * y_std + y_mu

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    models = {
        "LSTM": LSTM_two_layers(
            input_size, cfg["hidden_size"], cfg["output_size"], cfg["dropout"]
        ),
        "GRU": GRU_two_layers(
            input_size, cfg["hidden_size"], cfg["output_size"], cfg["dropout"]
        ),
        "LSTM_FCN": LSTM_FCN(
            input_size, cfg["hidden_size"], cfg["output_window"], cfg["dropout"]
        ),
        "Transformer": TransformerForecast(
            input_size=input_size,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=cfg["dropout"],
            output_window=cfg["output_window"],
        ),
    }

    best_mase = np.inf
    best_model_name = None
    best_state = None

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for name, model in models.items():
        print(f"\n===== Training {name} =====")
        train_model(model, dl_train, cfg["epochs"], cfg["learning_rate"], device)

        preds, targets = evaluate(model, dl_val, device)

        preds_real = preds * y_std + y_mu
        targets_real = targets * y_std + y_mu

        m = mase_daylight(
            targets_real[:, 0],
            preds_real[:, 0],
            train_y_real,
        )

        print(f"{name} → MASE (daylight): {m:.4f}")

        if m < best_mase:
            best_mase = m
            best_model_name = name
            best_state = model.state_dict()

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------
    MODEL_PATH = "./checkpoints/best_model.pt"
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_name": best_model_name,
        "model_state_dict": best_state,
        "config": cfg,
    }, MODEL_PATH)

    print(f"\nBest model: {best_model_name} (MASE={best_mase:.4f})")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
