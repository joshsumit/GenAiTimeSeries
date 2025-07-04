
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import math
from tqdm import tqdm
import torch.nn as nn


from tce_modules.dataset import ForecastingDataset
from tce_modules.model import IdealDPM, ForecastingHead, MaskedTCE

def main():
    df = pd.read_csv("ETTh2.csv")
    df = df.drop(columns=["date"])
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    scaler = StandardScaler()
    scaler.fit(train_df.values)

    train_data = scaler.transform(train_df.values)
    val_data = scaler.transform(val_df.values)
    test_data = scaler.transform(test_df.values)

    input_len = 512
    pred_len = 712
    batch_size = 32
    embed_dim = 64
    epochs = 20
    input_dim = train_data.shape[1]
    target_col = df.columns.get_loc("OT")

    train_dataset = ForecastingDataset(train_data, input_len, pred_len, target_col)
    val_dataset = ForecastingDataset(val_data, input_len, pred_len, target_col)
    test_dataset = ForecastingDataset(test_data, input_len, pred_len, target_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = IdealDPM(input_dim=input_dim, embed_dim=embed_dim).to(device)
    #encoder = MaskedTCE(input_dim=input_dim, embed_dim=embed_dim).to(device)

    head = ForecastingHead(embed_dim, pred_len).to(device)

    pretrained_weights = torch.load("masked_tce_with_uae.pt", map_location=device)
    dpm_state_dict = {k.replace("dpm.", ""): v for k, v in pretrained_weights.items() if k.startswith("dpm.")}
    encoder.load_state_dict(dpm_state_dict, strict=False)

    #encoder.load_state_dict(torch.load("masked_tce_with_uae.pt", map_location=device), strict=False)


    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': 1e-4},
        {'params': head.parameters(), 'lr': 1e-3}
    ])
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        #encoder.train()
        head.train()
        total_loss = 0
        if epoch < 10:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        else:
            encoder.train()
            for param in encoder.parameters():
                param.requires_grad = True

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            # Use only the pretrained IdealDPM encoder
            patches = encoder(x)
            pooled = patches.mean(dim=1)

            pred = head(pooled)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(loader):
        encoder.eval()
        head.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)

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
