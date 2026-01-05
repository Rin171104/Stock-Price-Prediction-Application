import torch.nn as nn
import torch.optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
from model_LSTM import LSTMModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Argument Parser
def get_args():
    parser = ArgumentParser(description="LSTM Time-Series Training")
    parser.add_argument("--data-path", "-d", type=str, default="Vingroup_4y.csv")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--seq-len", "-s", type=int, default=60)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained-models", "-t", type=str, default="trained-models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    return parser.parse_args()


# Create time-series sequences
def create_sequences(dataset, seq_len):
    x, y = [], []
    for i in range(seq_len, len(dataset)):
        x.append(dataset[i - seq_len:i, :])
        y.append(dataset[i, 3])
    return np.array(x), np.array(y)


if __name__ == "__main__":
    args = get_args()

    # ---------- Load data ----------
    data = pd.read_csv(args.data_path)

    features = ["open", "high", "low", "close"]
    values = data[features].astype(float).values

    # ---------- Train / Test split ----------
    split_idx = int(len(values) * (1 - args.test_size))
    train_data = values[:split_idx]
    test_data = values[split_idx:]

    # ---------- Scale (fit only on train) ----------
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # ---------- Create sequences ----------
    x_train, y_train = create_sequences(train_scaled, args.seq_len)
    x_test, y_test = create_sequences(test_scaled, args.seq_len)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    # ---------- TensorBoard ----------
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)

    writer = SummaryWriter(args.logging)

    # ---------- Device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #----------- Model -----------
    model = LSTMModel(input_size=4,hidden_size=50,num_layers=2,dropout=0.2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------- Load checkpoint ----------
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        start_epoch = checkpoint["epoch"]
        best_mse = checkpoint["best_mse"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_mse = float("inf")

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour="cyan")
        epoch_loss = 0.0

        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.epochs} | Loss {loss.item():.4f}"
            )

        writer.add_scalar("Train/Loss", epoch_loss / len(train_loader), epoch)

        # ---------- Evaluation ----------
        model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(yb.numpy())

        preds_all = np.vstack(preds_all)
        labels_all = np.vstack(labels_all)

        # inverse scale cho CLOSE (index = 3)
        close_mean = scaler.mean_[3]
        close_std = scaler.scale_[3]

        preds_real = preds_all * close_std + close_mean
        labels_real = labels_all * close_std + close_mean

        mae = mean_absolute_error(labels_real, preds_real)
        mse = mean_squared_error(labels_real, preds_real)
        r2 = r2_score(labels_real, preds_real)

        print(f"Epoch {epoch+1}: MAE={mae:.4f}, MSE={mse:.4f}, R2={r2:.4f}")

        writer.add_scalar("Val/MAE", mae, epoch)
        writer.add_scalar("Val/MSE", mse, epoch)
        writer.add_scalar("Val/R2", r2, epoch)

        # ---------- Save last ----------
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mse": best_mse
            },
            f"{args.trained_models}/last_lstm.pt"
        )

        # ---------- Save best ----------
        if mse < best_mse:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_mse": mse
                },
                f"{args.trained_models}/best_lstm.pt"
            )
            best_mse = mse

    writer.close()
