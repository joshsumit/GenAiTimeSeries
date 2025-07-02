import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -------------------------------
# ETTh1 Dataset Loader
# -------------------------------
class ETTh1Dataset(Dataset):
    def __init__(self, data, input_len=96, pred_len=192, target_idx=7):
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_len]
        y = self.data[idx + self.input_len:idx + self.input_len + self.pred_len, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# -------------------------------
# IdealDPM Module
# -------------------------------
class IdealDPM_old(nn.Module):
    def __init__(self, input_dim, embed_dim, attn_heads=2, boundary_threshold=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=attn_heads, batch_first=True)
        self.boundary_predictor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.project = nn.Linear(embed_dim, embed_dim)
        self.boundary_threshold = boundary_threshold
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.input_proj(x)  # [B, T, embed_dim]
        attn_output, _ = self.attn(x, x, x)  # [B, T, embed_dim]
        attn_output = self.dropout(attn_output)

        boundary_scores = self.boundary_predictor(attn_output).squeeze(-1)  # [B, T]

        all_patch_embeddings = []
        for b in range(x.size(0)):
            boundaries = [0]
            for t in range(1, x.size(1) - 1):
                if boundary_scores[b, t] > self.boundary_threshold:
                    boundaries.append(t)
            boundaries.append(x.size(1))
            patch_embeddings = [attn_output[b, boundaries[i]:boundaries[i+1]].mean(dim=0) for i in range(len(boundaries)-1)]
            all_patch_embeddings.append(torch.stack(patch_embeddings))

        max_len = max(p.shape[0] for p in all_patch_embeddings)
        padded = torch.zeros(x.size(0), max_len, attn_output.size(-1), device=x.device)
        for b in range(x.size(0)):
            padded[b, :all_patch_embeddings[b].size(0)] = all_patch_embeddings[b]

        
        # Log number of patches per sample in the batch
        patch_counts = [p.shape[0] for p in all_patch_embeddings]
        print(f"[IdealDPM] Patch counts per sample in batch: {patch_counts}")

        return self.project(padded)
    


class IdealDPM(nn.Module):
    def __init__(self, input_dim, embed_dim, max_patches=8, attn_heads=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, attn_heads, batch_first=True)
        self.boundary_predictor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.project = nn.Linear(embed_dim, embed_dim)
        self.max_patches = max_patches
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)  # [B, T, D]
        attn_out, _ = self.attn(x, x, x)  # [B, T, D]
        attn_out = self.dropout(attn_out)
        boundary_scores = self.boundary_predictor(attn_out).squeeze(-1)  # [B, T]

        cbp = torch.cumsum(boundary_scores, dim=1)  # [B, T]
        cbp = cbp / (cbp[:, -1:] + 1e-6)  # Normalize to [0, 1]

        patch_ids = (cbp * self.max_patches).floor().long().clamp(max=self.max_patches - 1)  # [B, T]

        B, T, D = attn_out.shape
        patch_embeddings = torch.zeros(B, self.max_patches, D, device=x.device)
        counts = torch.zeros(B, self.max_patches, 1, device=x.device)

        patch_ids_exp = patch_ids.unsqueeze(-1).expand(-1, -1, D)
        patch_embeddings.scatter_add_(1, patch_ids_exp, attn_out)

        ones = torch.ones(B, T, 1, device=x.device)
        patch_ids_exp_count = patch_ids.unsqueeze(-1).expand(-1, -1, 1)
        counts.scatter_add_(1, patch_ids_exp_count, ones)

        patch_embeddings = patch_embeddings / counts.clamp(min=1)
        return self.project(patch_embeddings)


# -------------------------------
# UAE Module
# -------------------------------
class UAE(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(1, embed_dim), nn.Sigmoid())
        self.uncertainty_proxy = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, embeddings, mc_predictions=None):
        if mc_predictions is None:
            proxy_uncertainty = self.uncertainty_proxy(embeddings)  # [B, T, 1]
            gamma = self.gamma(proxy_uncertainty)                   # [B, T, D]
        else:
            variance = mc_predictions.var(dim=0)                    # [B, D]
            gamma = self.gamma(variance.mean(dim=1, keepdim=True)) # [B, 1]
            gamma = gamma.unsqueeze(1)                              # [B, 1, D]
        return embeddings * gamma

# -------------------------------
# TCE Module
# -------------------------------
class TCE(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.dpm = IdealDPM(input_dim=input_dim, embed_dim=embed_dim)
        self.uae = UAE(embed_dim=embed_dim)

    def forward(self, x, num_passes=30):
        patch_embeddings = self.dpm(x)
        mc_outputs = [] 
        was_training = self.training
        self.train()
        for _ in range(num_passes):
            encoded = patch_embeddings
            pooled = encoded.mean(dim=1)  # [B, D]
            mc_outputs.append(pooled.unsqueeze(0))  # [1, B, D]       
        
        if not was_training:
            self.eval() # Restore original mode

        mc_preds = torch.cat(mc_outputs, dim=0)  # [K, B, D]
        return self.uae(patch_embeddings, mc_predictions=mc_preds)

# -------------------------------
# TCETransformer Model
# -------------------------------
class TCETransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, output_len):
        super().__init__()
        self.tce = TCE(input_dim, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dropout=0.2, batch_first=True),
            num_layers=2)
        self.head = nn.Linear(embed_dim, output_len)

    def forward(self, x):
        x = self.tce(x)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


# -------------------------------
# Evaluation Metrics
# -------------------------------
def compute_metrics(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-5))).item() * 100
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
    mse = torch.mean((y_true - y_pred) ** 2).item()
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "MSE": mse}

# -------------------------------
# Training and Evaluation
# -------------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
            all_preds.append(output)
            all_targets.append(y)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(targets, preds)
    return total_loss / len(loader), metrics

# -------------------------------
# Run Training and Evaluation
# -------------------------------

input_len = 96
pred_len = 192
batch_size = 32

# dataset = ETTh1Dataset(csv_file, input_len=input_len, pred_len=pred_len)
# train_size = int(0.7 * len(dataset))
# val_size = int(0.15 * len(dataset))
# test_size = len(dataset) - train_size - val_size
# train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_set, batch_size=batch_size)
# test_loader = DataLoader(test_set, batch_size=batch_size)

from sklearn.model_selection import train_test_split

df = pd.read_csv("ETTh1.csv")
df_numeric = df.drop(columns=[df.columns[0]])



# Compute split indices based on proportions
total_len = len(df_numeric)
train_end = int(0.7 * total_len)
val_end = train_end + int(0.15 * total_len)

# Perform sequential split (no shuffling)
train_df = df_numeric.iloc[:train_end]
val_df = df_numeric.iloc[train_end:val_end]
test_df = df_numeric.iloc[val_end:]


scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df.values)
val_scaled = scaler.transform(val_df.values)
test_scaled = scaler.transform(test_df.values)
target_idx = df_numeric.columns.get_loc("OT")


train_dataset = ETTh1Dataset(train_scaled, input_len, pred_len, target_idx)
val_dataset = ETTh1Dataset(val_scaled, input_len, pred_len, target_idx)
test_dataset = ETTh1Dataset(test_scaled, input_len, pred_len, target_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


input_dim = train_dataset[0][0].shape[1]
embed_dim = 64
output_len = pred_len
model = TCETransformer(input_dim, embed_dim, output_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 30
best_val_loss = float('inf')
patience = 10
patience_counter = 0
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_metrics = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"MAE: {val_metrics['MAE']:.4f} "
          f" MSE: {val_metrics['MSE']:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_model_state)
test_loss, test_metrics = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, MAE: {test_metrics['MAE']:.4f},  "
      f" MSE: {test_metrics['MSE']:.4f}")
