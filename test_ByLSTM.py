import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
from model_LSTM import LSTMModel

# ARGUMENT PARSER
def get_args():
    parser = ArgumentParser(description="Test LSTM with custom start_index")
    parser.add_argument("--csv-path", type=str, default="Vingroup_4y.csv")
    parser.add_argument("--model-path", type=str, default="trained-models/best_lstm.pt")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--start-index","-s", type=int, default=200,
                        help="Start index for 60-day window")
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    FEATURES = ["open", "high", "low", "close"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load data ----------
    df = pd.read_csv(args.csv_path)

    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f" Missing column: {col}")

    values = df[FEATURES].astype(float).values  # (N, 4)

    # ---------- Check index ----------
    start = args.start_index
    end = start + args.seq_len

    if start < 0 or end > len(values):
        raise ValueError(
            f" start_index không hợp lệ! Dataset có {len(values)} dòng."
        )

    # ---------- Scale ----------
    scaler = StandardScaler()
    scaler.fit(values)
    values_scaled = scaler.transform(values)

    # ---------- Create input ----------
    x_seq = values_scaled[start:end]          # (60, 4)
    x_seq = x_seq.reshape(1, args.seq_len, 4) # (1, 60, 4)

    x_seq = torch.tensor(x_seq, dtype=torch.float32).to(device)

    # ---------- Load model ----------
    model = LSTMModel(
        input_size=4,
        hidden_size=50,
        num_layers=2,
        dropout=0.2
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(" Model loaded successfully")

    # ---------- Predict ----------
    with torch.no_grad():
        pred_scaled = model(x_seq).cpu().numpy()

    # ---------- Inverse scale (close index = 3) ----------
    close_mean = scaler.mean_[3]
    close_std = scaler.scale_[3]
    pred_close = pred_scaled[0][0] * close_std + close_mean

    # ---------- Info ----------
    last_close = df["close"].iloc[end - 1]
    pred_date = (
        pd.to_datetime(df.iloc[end - 1]["time"]) + pd.Timedelta(days=1)
        if "time" in df.columns else "Next day"
    )

    print("\n LSTM PREDICTION RESULT")
    print(f"Start index       : {start}")
    print(f"Used range        : [{start} → {end - 1}]")
    print(f"Last close price  : {last_close:.2f}")
    print(f"Predicted close   : {pred_close:.2f}")
    print(f"Difference        : {pred_close - last_close:.2f}")
    print(f"Predicted date    : {pred_date}")
