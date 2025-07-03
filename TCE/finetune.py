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

import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def train_model(encoder, head, train_loader, loss_fn, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(epochs):
        if epoch < 10:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        else:
            encoder.train()
            for param in encoder.parameters():
                param.requires_grad = True

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
    input_len = 512
    pred_len = 96 # 720  # 96 #
    batch_size = 32
    embed_dim = 64
    epochs = 50

    df = pd.read_csv("ETTh2.csv")
    df = df.drop(columns=[df.columns[0]])  # Drop timestamp

    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    # Split before scaling
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Fit scaler only on training data
    scaler = StandardScaler()
    scaler.fit(train_df.values)

    # Transform all splits
    train_data = scaler.transform(train_df.values)
    val_data = scaler.transform(val_df.values)
    test_data = scaler.transform(test_df.values)

    input_dim = train_data.shape[1]
    target_col = df.columns.get_loc("OT")

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
    #encode.eval


    head = ForecastingHead(embed_dim, pred_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    head = head.to(device)

    #optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    
    optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': 1e-4},
    {'params': head.parameters(), 'lr': 1e-3}
    ])

    #optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)

    loss_fn = torch.nn.MSELoss()

    train_model(encoder, head, train_loader, loss_fn, optimizer, epochs)

    def evaluate(loader):
        encoder.eval()
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

