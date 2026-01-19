# ======================================================
# ===================== IMPORTS ========================
# ======================================================
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from models.LSTM import LSTM_two_layers
from train import TimeSeriesDataset, load_split

# ======================================================
# =================== HELPERS ==========================
# ======================================================
def load_trained_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    assert checkpoint["model_type"] == "LSTM_two_layers"

    cfg = checkpoint["config"]

    model = model_class(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        output_size=cfg["output_size"],
        dropout=cfg["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    return model, cfg


def inference_model(model, dataloader, device):
    preds = []
    model.eval()

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())

    return np.concatenate(preds, axis=0)


# ======================================================
# =================== INFERENCE ========================
# ======================================================
def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_PATH = "./checkpoints/lstm_two_layers.pt"
    DATA_PATH = "./data/Processed"

    # ------------------ LOAD MODEL ---------------------
    model, cfg = load_trained_model(
        LSTM_two_layers, MODEL_PATH, device
    )

    # ------------------ DATA ---------------------------
    inf_x, inf_y = load_split("inference", DATA_PATH)

    ds_inf = TimeSeriesDataset(
        inf_x,
        inf_y,
        length=cfg["length"],
        lag=cfg["lag"],
        output_window=cfg["output_window"],
    )

    dl_inf = DataLoader(ds_inf, batch_size=64, shuffle=False)

    # ------------------ INFERENCE ----------------------
    preds = inference_model(model, dl_inf, device)
    preds_real = np.expm1(preds)

    # ------------------ SAVE ---------------------------
    df = pd.DataFrame(
        preds_real,
        columns=[f"horizon_{i}" for i in range(preds_real.shape[1])]
    )

    out_path = "./outputs/inference_predictions.xlsx"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)

    print(f"Inference saved to {out_path}")


if __name__ == "__main__":
    run_inference()
