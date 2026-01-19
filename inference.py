# ======================================================
# ===================== IMPORTS ========================
# ======================================================
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

import Pipeline_2 as Pipeline

from models.LSTM import LSTM_two_layers
from models.GRU import GRU_two_layers
from models.LSTM_FCN import LSTM_FCN

from utils.graph_pipeline import (
    plot_continuous_horizon0,
    plot_one_day,
    plot_scatter_real_vs_pred,
)

# ======================================================
# ================= MODEL FACTORY ======================
# ======================================================
MODEL_FACTORY = {
    "LSTM": LSTM_two_layers,
    "GRU": GRU_two_layers,
    "LSTM_FCN": LSTM_FCN,
}

# ======================================================
# ================= LOAD TRAINED MODEL ================
# ======================================================
def load_trained_model(checkpoint_path, device, input_size):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    assert "model_name" in checkpoint
    assert "state_dict" in checkpoint
    assert "config" in checkpoint

    model_name = checkpoint["model_name"]
    cfg = checkpoint["config"]["model"]

    # Reconstruir parámetros del modelo
    if model_name in ["LSTM", "GRU"]:
        model_params = {
            "input_size": input_size,
            "hidden_size": cfg["hidden_size"],
            "output_size": cfg["output_size"],
            "dropout": cfg["dropout"],
        }
    elif model_name == "LSTM_FCN":
        model_params = {
            "input_size": input_size,
            "hidden_size": cfg["hidden_size"],
            "output_window": cfg["output_window"],
            "dropout": cfg["dropout"],
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = MODEL_FACTORY[model_name]
    model = model_class(**model_params)

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()

    print("\nLoaded checkpoint")
    print("  model_name:", model_name)
    print("  model_params:", model_params)

    return model, model_name, model_params,  checkpoint["config"]



# ======================================================
# =================== INFERENCE ========================
# ======================================================
def inference_model(model, dataloader, device):
    preds, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            targets.append(y.numpy())

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
    )


# ======================================================
# =================== RUN ==============================
# ======================================================
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # --------------------------------------------------
    # PATHS
    # --------------------------------------------------
    CHECKPOINT_PATH = "./checkpoints/best_model_LSTM_FCN.pt"  # ⬅️ ajusta si hace falta
    DATA_PATH = "./data/Processed"
    inf_x, inf_y = Pipeline.load_split("inference", DATA_PATH)
    input_size = inf_x.shape[1]
    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    model, model_name, model_params, cfg = load_trained_model(
        CHECKPOINT_PATH,
        device,
        input_size = input_size,
    )

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    
    
    ds_inf = Pipeline.TimeSeriesDataset(
        inf_x,
        inf_y,
        length=cfg["model"]["length"],
        lag= cfg["model"]["length"],
        output_window=model_params["output_window"],
        stride=1,
    )

    dl_inf = DataLoader(
        ds_inf,
        batch_size=64,
        shuffle=False,
    )

    # --------------------------------------------------
    # INFERENCE
    # --------------------------------------------------
    

    # --------------------------------------------------
    # INFERENCE
    # --------------------------------------------------
    preds, targets = inference_model(model, dl_inf, device)

    # --------------------------------------------------
    # DESNORMALIZAR Y (volver a kWh reales)
    # --------------------------------------------------
    import pickle

    with open(f"{DATA_PATH}/stats.pkl", "rb") as f:
        stats = pickle.load(f)

    y_mu = stats["Y"]["mu"]["Energy"]
    y_std = stats["Y"]["std"]["Energy"]

    preds_real = preds * y_std + y_mu
    targets_real = targets * y_std + y_mu

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    # Usar el scale ya calculado en TRAIN
    mase_scale = stats["mase"]["scale"]

    # MASE horizon = 0 (todas las horas)
    mase_h0 = (
        np.mean(np.abs(preds_real[:, 0] - targets_real[:, 0]))
        / mase_scale
    )

    # RMSE horizon = 0
    rmse_h0 = np.sqrt(
        np.mean((preds_real[:, 0] - targets_real[:, 0]) ** 2)
    )

    print("\n========== INFERENCE METRICS ==========")
    print(f"Model: {model_name}")
    print(f"MASE horizon=0 (all hours): {mase_h0:.4f}")
    print(f"RMSE horizon=0 (kWh):       {rmse_h0:.4f}")





    

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------
    plot_continuous_horizon0(
        targets_real,
        preds_real,
        start_idx=0,
        n_days=7,
    )

    plot_one_day(
        targets_real,
        preds_real,
        day_idx=10,
    )

    plot_scatter_real_vs_pred(
        targets_real,
        preds_real,
    )

    # --------------------------------------------------
    # SAVE OUTPUT
    # --------------------------------------------------
    df = pd.DataFrame(
        preds_real,
        columns=[f"horizon_{i}" for i in range(preds_real.shape[1])]
    )

    out_path = "./outputs/inference_predictions.xlsx"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)

    print(f"\nInference saved to {out_path}")


if __name__ == "__main__":
    run_inference()
