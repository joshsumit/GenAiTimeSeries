
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tce_modules.dataset import ETTh1UnlabeledDataset
from tce_modules.model import MaskedTCE

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

def main():
    df = pd.read_csv("ETTh1.csv")
    df_numeric = df.drop(columns=["date"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_numeric.values)

    input_len = 512
    batch_size = 32
    embed_dim = 64
    epochs = 10
    input_dim = scaled.shape[1]

    dataset = ETTh1UnlabeledDataset(scaled, input_len=input_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedTCE(input_dim=input_dim, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        for x in tqdm(loader):
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "masked_tce_with_uae.pt")
            print("Saved best model.")

if __name__ == "__main__":
    main()
