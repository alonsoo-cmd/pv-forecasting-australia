import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN
from models.Transformer import TransformerForecast


# ======================================================
# CONFIG FLAG
# ======================================================
RUN_TRAINING = True   # True = train + test + inference | False = only inference


# ======================================================
# DATASET
# ======================================================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, length, lag, output_window, stride=1):
        assert len(X) == len(y)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if self.y.ndim == 2:
            self.y = self.y.squeeze(-1)

        self.length = length
        self.lag = lag
        self.output_window = output_window

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
# TRAINING
# ======================================================
def training_model(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - MAE: {epoch_loss:.4f}")


# ======================================================
# EVALUATE (SIN CAMBIOS)
# ======================================================
def evaluate_model(model, dataloader, device):
    model.eval()
    model.to(device)

    preds_all, targets_all = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(y_batch.numpy())

    return (
        np.concatenate(preds_all, axis=0),
        np.concatenate(targets_all, axis=0),
    )


# ======================================================
# METRICS
# ======================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mase(y_true, y_pred, y_train, m=24):
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_errors)
    return np.mean(np.abs(y_true - y_pred)) / (scale + 1e-8)


# ======================================================
# LOAD DATA
# ======================================================
def load_split(name, base_path):
    df = pd.read_excel(Path(base_path) / f"{name}.xlsx", index_col=0)
    y = df["Energy"].to_numpy(dtype=np.float32)
    X_df = df.drop(columns=["Energy"])
    return X_df.to_numpy(dtype=np.float32), y, X_df.columns.tolist()


# ======================================================
# MAIN
# ======================================================
def main():

    CONFIG_PATH = "./config/timeseries.yaml"
    DATA_PATH = "./data/Processed"

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ==================================================
    # TRAINING + VALIDATION + TEST
    # ==================================================
    if RUN_TRAINING:

        train_x, train_y, feature_columns = load_split("train", DATA_PATH)
        val_x, val_y, _ = load_split("val", DATA_PATH)
        test_x, test_y, _ = load_split("test", DATA_PATH)

        input_size = train_x.shape[1]

        ds_train = TimeSeriesDataset(
            train_x, train_y,
            config["length"], config["lag"], config["output_window"]
        )
        ds_val = TimeSeriesDataset(
            val_x, val_y,
            config["length"], config["lag"], config["output_window"],
            stride=24
        )
        ds_test = TimeSeriesDataset(
            test_x, test_y,
            config["length"], config["lag"], config["output_window"],
            stride=24
        )

        dl_train = DataLoader(ds_train, batch_size=config["batch_size"], shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=config["batch_size"], shuffle=False)
        dl_test = DataLoader(ds_test, batch_size=config["batch_size"], shuffle=False)

        models = {
            "LSTM": LSTM_two_layers(
                input_size,
                config["hidden_size"],
                config["output_size"],
                config["dropout"],
            ),
            "GRU": GRU_two_layers(
                input_size,
                config["hidden_size"],
                config["output_size"],
                config["dropout"],
            ),
            "LSTM_FCN": LSTM_FCN(
                input_size,
                config["hidden_size"],
                config["output_window"],
                config["dropout"],
            ),
            "Transformer": TransformerForecast(
                input_size=input_size,
                d_model=128,
                nhead=8,
                num_layers=4,
                dim_feedforward=256,
                dropout=config["dropout"],
                output_window=config["output_window"],
            ),
        }

        best_mase = np.inf
        best_name = None
        best_state = None

        for name, model in models.items():
            print(f"\nTraining {name}")
            training_model(
                model,
                dl_train,
                config["epochs"],
                config["learning_rate"],
                device,
            )

            val_preds, val_targets = evaluate_model(model, dl_val, device)

            val_preds_real = np.expm1(val_preds[:, 0])
            val_targets_real = np.expm1(val_targets[:, 0])

            rmse_val = rmse(val_targets_real, val_preds_real)
            mase_val = mase(
                val_targets_real,
                val_preds_real,
                np.expm1(train_y),
                m=24,
            )

            print(f"{name} | RMSE ↓ {rmse_val:.3f} | MASE ↓ {mase_val:.3f}")

            if mase_val < best_mase:
                best_mase = mase_val
                best_name = name
                best_state = model.state_dict()

        torch.save(
            {
                "model_name": best_name,
                "state_dict": best_state,
                "input_size": input_size,
                "feature_columns": feature_columns,
            },
            "best_model.pth",
        )

        print(f"\n🏆 Best model saved: {best_name}")

        # ------------------ TEST EVALUATION ------------------
        print("\n🔴 Evaluating best model on TEST set")

        best_model = {
            "LSTM": LSTM_two_layers,
            "GRU": GRU_two_layers,
            "LSTM_FCN": LSTM_FCN,
            "Transformer": TransformerForecast,
        }[best_name](
            input_size,
            config["hidden_size"],
            config["output_window"],
            config["dropout"],
        )

        best_model.load_state_dict(best_state)
        best_model.to(device)

        test_preds, test_targets = evaluate_model(best_model, dl_test, device)

        test_preds_real = np.expm1(test_preds[:, 0])
        test_targets_real = np.expm1(test_targets[:, 0])

        print("\n📊 TEST RESULTS")
        print("RMSE ↓ :", rmse(test_targets_real, test_preds_real))
        print(
            "MASE ↓ :",
            mase(test_targets_real, test_preds_real, np.expm1(train_y), m=24),
        )

    # ==================================================
    # INFERENCE
    # ==================================================
    print("\n🔵 Running inference...")

    checkpoint = torch.load("best_model.pth", map_location=device)

    model_name = checkpoint["model_name"]
    input_size = checkpoint["input_size"]
    feature_columns = checkpoint["feature_columns"]

    df_inf = pd.read_excel("data/Processed/inference.xlsx", index_col=0)
    y_inf = df_inf["Energy"].to_numpy(dtype=np.float32)
    X_inf_df = df_inf.drop(columns=["Energy"])

    X_inf_df = X_inf_df.reindex(columns=feature_columns, fill_value=0.0)
    X_inf = X_inf_df.to_numpy(dtype=np.float32)

    model = {
        "LSTM": LSTM_two_layers,
        "GRU": GRU_two_layers,
        "LSTM_FCN": LSTM_FCN,
        "Transformer": TransformerForecast,
    }[model_name](
        input_size,
        config["hidden_size"],
        config["output_window"],
        config["dropout"],
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    ds_inf = TimeSeriesDataset(
        X_inf, y_inf,
        config["length"], config["lag"], config["output_window"],
        stride=24,
    )
    dl_inf = DataLoader(ds_inf, batch_size=config["batch_size"], shuffle=False)

    inf_preds, inf_targets = evaluate_model(model, dl_inf, device)

    inf_preds_real = np.expm1(inf_preds[:, 0])
    inf_targets_real = np.expm1(inf_targets[:, 0])

    print("\n📊 INFERENCE RESULTS")
    print("RMSE ↓ :", rmse(inf_targets_real, inf_preds_real))
    print("MASE ↓ :", mase(inf_targets_real, inf_preds_real, np.expm1(y_inf), m=24))


if __name__ == "__main__":
    main()
