import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math
from tqdm import tqdm

from tce_modules.model import IdealDPM, ForecastingHead
from tce_modules.dataset import ForecastingDataset

def train_model(encoder, head, train_loader, loss_fn, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        head.train()
        total_loss = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for x, y in train_iter:            
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            patches = encoder(x)
            pooled = patches.mean(dim=1)
            pred = head(pooled)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Finetune Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

def main():
    df = pd.read_csv("ETTh2.csv")
    df = df.drop(columns=[df.columns[0]])  # Drop timestamp
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.values)

    input_len = 96
    pred_len = 24
    batch_size = 32
    embed_dim = 64
    epochs = 10
    input_dim = scaled_data.shape[1]
    target_col = df.columns.get_loc("OT")

    n = len(scaled_data)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    train_data = scaled_data[:train_end]
    val_data = scaled_data[train_end:val_end]
    test_data = scaled_data[val_end:]

    train_dataset = ForecastingDataset(train_data, input_len, pred_len, target_col)
    val_dataset = ForecastingDataset(val_data, input_len, pred_len, target_col)
    test_dataset = ForecastingDataset(test_data, input_len, pred_len, target_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    encoder = IdealDPM(input_dim=input_dim, embed_dim=embed_dim)
    pretrained_weights = torch.load("masked_tce_with_uae.pt")
    dpm_state_dict = {k.replace("dpm.", ""): v for k, v in pretrained_weights.items() if k.startswith("dpm.")}
    encoder.load_state_dict(dpm_state_dict, strict=False)
    encoder.eval()

    head = ForecastingHead(embed_dim, pred_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    head = head.to(device)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    train_model(encoder, head, train_loader, loss_fn, optimizer, epochs)

    def evaluate(loader):
        head.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                patches = encoder(x)
                pooled = patches.mean(dim=1)
                pred = head(pooled)
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        rmse = math.sqrt(mse)
        mape = np.mean(np.abs((trues - preds) / np.clip(trues, 1e-8, None))) * 100

        return mse, mae, rmse, mape

    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)

    print(f"Validation - MSE: {val_metrics[0]:.4f}, MAE: {val_metrics[1]:.4f}, RMSE: {val_metrics[2]:.4f}, MAPE: {val_metrics[3]:.2f}%")
    print(f"Test - MSE: {test_metrics[0]:.4f}, MAE: {test_metrics[1]:.4f}, RMSE: {test_metrics[2]:.4f}, MAPE: {test_metrics[3]:.2f}%")

if __name__ == "__main__":
    main()

