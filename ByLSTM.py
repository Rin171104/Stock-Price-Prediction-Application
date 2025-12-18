import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from model_LSTM import LSTMModel
from torch.utils.tensorboard import SummaryWriter
import shutil
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# =============================
# Argument Parser
# =============================
def get_args():
    parser = argparse.ArgumentParser(description="LSTM Time-Series Training")
    parser.add_argument("--data_path", type=str, default="Vingroup_4y.csv")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", "-e", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log_folder", "-p", type=str, default="tensorboard")
    return parser.parse_args()


# =============================
# Create sequences
# =============================
def create_sequences(dataset, seq_len):
    x, y = [], []
    for i in range(seq_len, len(dataset)):
        x.append(dataset[i - seq_len:i, 0])
        y.append(dataset[i, 0])
    return np.array(x), np.array(y)


# =============================
# Training function
# =============================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- Load data ----------
    data = pd.read_csv(args.data_path)
    values = data[["close"]].astype(float).values

    # ---------- Split ----------
    train_size = int(len(values) * (1 - args.test_size))
    train_data = values[:train_size]
    test_data = values[train_size:]

    # ---------- Scale ----------
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # ---------- Create sequences ----------
    x_train, y_train = create_sequences(train_scaled, args.seq_len)
    x_test, y_test = create_sequences(test_scaled, args.seq_len)

    x_train = torch.tensor(x_train.reshape(-1, args.seq_len, 1), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    x_test = torch.tensor(x_test.reshape(-1, args.seq_len, 1), dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )

    # ---------- Model ----------
    model = LSTMModel(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------- TensorBoard ----------
    if os.path.isdir(args.log_folder):
        shutil.rmtree(args.log_folder)
    writer = SummaryWriter(args.log_folder)

    dummy_input = torch.zeros(1, args.seq_len, 1).to(device)
    writer.add_graph(model, dummy_input)

    print(model)

    # ---------- Training ----------
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            colour="cyan"
        )

        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # TensorBoard - step loss
            writer.add_scalar("Train/Loss_step", loss.item(), global_step)
            global_step += 1

            progress_bar.set_postfix(
                loss=epoch_loss / (progress_bar.n + 1)
            )

        # TensorBoard - epoch loss & lr
        writer.add_scalar("Train/Loss_epoch", epoch_loss / len(train_loader), epoch)
        writer.add_scalar("Train/Learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # ---------- Evaluation ----------
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(x_test.to(device)).cpu().numpy()

        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_test.numpy())

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # TensorBoard - metrics
        writer.add_scalar("Eval/MAE", mae, epoch)
        writer.add_scalar("Eval/MSE", mse, epoch)
        writer.add_scalar("Eval/R2", r2, epoch)

        print(
            f"\nEpoch {epoch+1} | "
            f"MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}"
        )

    writer.close()

    # ---------- Sample output ----------
    print("\nSample predictions:")
    for i in range(min(10, len(y_pred))):
        print(f"Predicted: {y_pred[i][0]:.4f} | Actual: {y_true[i][0]:.4f}")


# =============================
# Main
# =============================
if __name__ == "__main__":
    args = get_args()
    train(args)
