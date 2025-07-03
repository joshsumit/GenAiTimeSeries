import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tce_modules.model import MaskedTCE
from tce_modules.dataset import ETTh1UnlabeledDataset
from tqdm import tqdm
import time
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def main():
    df = pd.read_csv("ETTh1.csv")
    df_numeric = df.drop(columns=["date"])  # Keep 'OT' included
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_numeric.values)

    input_len = 512
    batch_size = 32
    embed_dim = 64
    epochs = 10
    input_dim = scaled.shape[1]

    dataset = ETTh1UnlabeledDataset(scaled, input_len=input_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)  #num_workers=os.cpu_count()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedTCE(input_dim=input_dim, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        start_time = time.time()

        # mixed precision training: AMP uses dynamic loss scaling to prevent underflow in gradients.
        # for x in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
        #     x = x.to(device, non_blocking=True)
        #     optimizer.zero_grad()
        #     with torch.cuda.amp.autocast():  
        #         loss = model(x)            
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     total_loss += loss.item()

        
        for x in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        avg_loss = total_loss / len(loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s - Avg Loss: {avg_loss:.4f}")

    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "masked_tce_with_uae.pt")
        print("Pretrained model saved to 'masked_tce_with_uae.pt'")

if __name__ == "__main__":
    main()

