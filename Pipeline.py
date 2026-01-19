# ======================================================
# ===============        IMPORTS        ================
# ======================================================
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils.graph_pipeline import plot_continuous_horizon0, plot_one_day, plot_scatter_real_vs_pred
from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.Transformer import TransformerForecast

# ======================================================
# ===============      DATASET        ==================
# ======================================================

class TimeSeriesDataset(Dataset):
    """
    Dataset para forecasting con:
    - ventana de pasado (length)
    - lag
    - horizonte futuro (output_window)
    """

    def __init__(
        self,
        X,
        y,
        length: int,
        lag: int,
        output_window: int,
        stride: int = 1,
    ):
        assert len(X) == len(y), "X e y deben tener la misma longitud"

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
# ===============      TRAINING        =================
# ======================================================

def training_model(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    criterion = nn.L1Loss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in dataloader:
            # Enviar datos al dispositivo (GPU/CPU)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Paso de optimización
            optimizer.zero_grad()
            preds = model(x_batch)
            
            # Asegúrate de que las dimensiones coincidan (frecuente en regresión)
            loss = criterion(preds.squeeze(), y_batch.squeeze()) 
            
            loss.backward()
            optimizer.step()

            # Sumamos el loss del batch (promediado por batch)
            running_loss += loss.item()

        # Promedio real: total del loss / cantidad de batches
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> MAE: {epoch_loss:.6f}")

    return epoch_losses


# ======================================================
# ===============      EVALUATE        =================
# ======================================================

def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    criterion = nn.L1Loss()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            total_loss += loss.item() * x_batch.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    mean_loss = total_loss / len(dataloader.dataset)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return mean_loss, all_preds, all_targets


# ======================================================
# ===============      MAIN          ===================
# ======================================================

CONFIG_PATH = "./config/timeseries.yaml"
DATA_PATH = "./data/Processed"
training_model_exexution = True

import numpy as np
import torch


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mase(y_true, y_pred, y_train, m=24):
    """
    Mean Absolute Scaled Error (MASE)

    y_true  : validation/test targets (N,)
    y_pred  : predictions (N,)
    y_train : training target series (1D)
    m       : seasonal period (24 for hourly PV)

    LOWER is better
    """

    # Naive seasonal forecast error (denominator)
    naive_diff = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_diff)

    return np.mean(np.abs(y_true - y_pred)) / (scale + 1e-8)

def load_split(name, base_path, y_col="Energy"):
    df = pd.read_excel(Path(base_path) / f"{name}.xlsx", index_col=0)

    y = df[y_col].to_numpy(dtype=np.float32)
    X = df.drop(columns=[y_col]).to_numpy(dtype=np.float32)

    return X, y


def training():
    # --------------------------------------------------
    # Config & device
    # --------------------------------------------------
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --------------------------------------------------
    # Load datasets (X + Y juntos)
    # --------------------------------------------------
    train_x, train_y = load_split("train", DATA_PATH)
    val_x,   val_y   = load_split("val",   DATA_PATH)
    test_x,  test_y  = load_split("test",  DATA_PATH)

    # --------------------------------------------------
    # Params
    # --------------------------------------------------
    input_size = train_x.shape[1]
    hidden_size = config["model"]["hidden_size"]
    output_window = config["model"]["output_window"]
    output_size = config["model"]["output_size"]
    dropout = config["model"]["dropout"]

    batch_size = config["model"]["batch_size"]
    num_epochs = config["model"]["epochs"]
    learning_rate = config["model"]["learning_rate"]
    lag = config["model"]["lag"]
    length = config["model"]["length"]

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------
    ds_train = TimeSeriesDataset(
        train_x, train_y, length, lag, output_window, stride=1
    )
    ds_val = TimeSeriesDataset(
        val_x, val_y, length, lag, output_window, stride=24
    )
    ds_test = TimeSeriesDataset(
        test_x, test_y, length, lag, output_window, stride=24
    )

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------
    # MODELO
    # --------------------------------------------------
    model = LSTM_two_layers(
        input_size, hidden_size, output_size, dropout
    ).to(device)

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    if training_model_exexution: 
        training_model(
            model,
            dl_train,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )

    # --------------------------------------------------
    # VALIDATE
    # --------------------------------------------------
    val_loss, val_preds, val_targets = evaluate_model(
        model, dl_val, device
    )

    # Invertir log
    val_preds_real = np.expm1(val_preds)
    val_targets_real = np.expm1(val_targets)

    mae_h0 = np.mean(np.abs(
        val_preds_real[:, 0] - val_targets_real[:, 0]
    ))

    print("Validation MAE horizon=0 (kWh):", mae_h0)

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------
    test_loss, test_preds, test_targets = evaluate_model(
        model, dl_test, device
    )

    test_preds_real = np.expm1(test_preds)
    test_targets_real = np.expm1(test_targets)

    mae_test_h0 = np.mean(np.abs(
        test_preds_real[:, 0] - test_targets_real[:, 0]
    ))

    print("Test MAE horizon=0 (kWh):", mae_test_h0)
    print("Test MAE (log-space):", test_loss)


    # --------------------------------------------------
    # GRAPH 
    # --------------------------------------------------
    

    plot_continuous_horizon0(
        val_targets_real,
        val_preds_real,
        start_idx=0,
        n_days=10,
    )

    plot_one_day(val_targets_real, val_preds_real, day_idx=20)

    plot_scatter_real_vs_pred(val_targets_real, val_preds_real)

        # --------------------------------------------------
    # Escala para MASE (usar TRAIN en escala real)
    # --------------------------------------------------
    train_y_real = np.expm1(train_y)  # 1D, en kWh (escala real)

    # --------------------------------------------------
    # VALIDATE
    # --------------------------------------------------
    val_loss, val_preds, val_targets = evaluate_model(model, dl_val, device)

    # Invertir log
    val_preds_real = np.expm1(val_preds)
    val_targets_real = np.expm1(val_targets)

    # MAE horizon=0
    mae_val_h0 = np.mean(np.abs(val_preds_real[:, 0] - val_targets_real[:, 0]))

    # MASE horizon=0
    mase_val_h0 = mase(
        y_true=val_targets_real[:, 0],
        y_pred=val_preds_real[:, 0],
        y_train=train_y_real,
        m=24
    )

    # (Opcional) MASE global (todos los horizontes)
    mase_val_all = mase(
        y_true=val_targets_real.reshape(-1),
        y_pred=val_preds_real.reshape(-1),
        y_train=train_y_real,
        m=24
    )

    print("Validation MAE horizon=0 (kWh):", mae_val_h0)
    print("Validation MASE horizon=0:", mase_val_h0)
    print("Validation MASE (all horizons):", mase_val_all)

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------
    test_loss, test_preds, test_targets = evaluate_model(model, dl_test, device)

    test_preds_real = np.expm1(test_preds)
    test_targets_real = np.expm1(test_targets)

    # MAE horizon=0
    mae_test_h0 = np.mean(np.abs(test_preds_real[:, 0] - test_targets_real[:, 0]))

    # MASE horizon=0
    mase_test_h0 = mase(
        y_true=test_targets_real[:, 0],
        y_pred=test_preds_real[:, 0],
        y_train=train_y_real,
        m=24
    )

    # (Opcional) MASE global (todos los horizontes)
    mase_test_all = mase(
        y_true=test_targets_real.reshape(-1),
        y_pred=test_preds_real.reshape(-1),
        y_train=train_y_real,
        m=24
    )

    print("Test MAE horizon=0 (kWh):", mae_test_h0)
    print("Test MAE (log-space):", test_loss)
    print("Test MASE horizon=0:", mase_test_h0)
    print("Test MASE (all horizons):", mase_test_all)



if __name__ == "__main__":
    training()
