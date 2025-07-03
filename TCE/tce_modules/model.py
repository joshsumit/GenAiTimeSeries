import torch
import torch.nn as nn
import torch.nn.functional as F

def init_linear_layers(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

# Attention-based patch generator with CBP
class IdealDPM(nn.Module):
    def __init__(self, input_dim, embed_dim, max_patches=8, use_cbp=True, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.max_patches = max_patches
        self.use_cbp = use_cbp
        self.project = nn.Linear(embed_dim, embed_dim)

        if use_cbp:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
            self.boundary_predictor = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid()
            )
            self.dropout = nn.Dropout(dropout)
        init_linear_layers(self)

    def forward(self, x):
        B, T, _ = x.shape
        #B, T, D = x.shape

        x = self.input_proj(x)  # [B, T, D]

        # CBP-based patching
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.dropout(attn_out)
        boundary_scores = self.boundary_predictor(attn_out).squeeze(-1)  # [B, T]
        cumulative = torch.cumsum(boundary_scores, dim=1)
        normalized = cumulative / cumulative[:, -1:].clamp(min=1e-6)
        patch_ids = torch.clamp((normalized * self.max_patches).long(), max=self.max_patches - 1)

        D = x.size(-1)
        # patch_embeddings = torch.zeros(B, self.max_patches, D, device=x.device)
        # counts = torch.zeros(B, self.max_patches, 1, device=x.device)
        patch_embeddings = torch.zeros(B, self.max_patches, D, device=x.device, dtype=x.dtype)
        counts = torch.zeros(B, self.max_patches, 1, device=x.device, dtype=x.dtype)

        patch_ids_exp = patch_ids.unsqueeze(-1).expand(-1, -1, D)
        patch_embeddings.scatter_add_(1, patch_ids_exp, x)

        ones = torch.ones(B, T, 1, device=x.device, dtype=counts.dtype)
        patch_ids_exp_count = patch_ids.unsqueeze(-1).expand(-1, -1, 1)
        counts.scatter_add_(1, patch_ids_exp_count, ones)

        patch_embeddings = patch_embeddings / counts.clamp(min=1)

        # Applying dropout after projection:
        # Regularizes the patch embeddings.
        # Prevents overfitting.
        # Encourages robustness in downstream tasks.
        return self.dropout(self.project(patch_embeddings))


# UAE Module
class UAE(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(1, embed_dim), nn.Sigmoid())
        self.uncertainty_proxy = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        init_linear_layers(self)

    def forward(self, embeddings, mc_predictions=None):
        if mc_predictions is None:
            proxy_uncertainty = self.uncertainty_proxy(embeddings)
            gamma = self.gamma(proxy_uncertainty)
        else:
            variance = mc_predictions.var(dim=0)
            mean_var = variance.mean(dim=1, keepdim=True)
            gamma = self.gamma(mean_var).unsqueeze(1)

            #gamma = self.gamma(variance.mean(dim=1, keepdim=True)).unsqueeze(1)
        return embeddings * gamma

# Masked Patch Prediction Model
class MaskedTCE(nn.Module):
    def __init__(self, input_dim, embed_dim, max_patches=8, mask_ratio=0.3, use_cbp=True):
        super().__init__()
        self.dpm = IdealDPM(input_dim, embed_dim, max_patches, use_cbp=use_cbp)
        self.mask_token = nn.Parameter(torch.randn(embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.uae = UAE(embed_dim)
        self.mask_ratio = mask_ratio
        self.max_patches = max_patches
        init_linear_layers(self)

    def forward(self, x):
        patches = self.dpm(x)
        original = patches.clone()
        B, P, D = patches.shape
        mask = torch.rand(B, P, device=patches.device) < self.mask_ratio
        patches[mask] = self.mask_token
        decoded = self.decoder(patches)
        modulated = self.uae(decoded)
        loss = F.mse_loss(modulated[mask], original[mask])
        return loss

# Forecasting Head
class ForecastingHead(nn.Module):
    def __init__(self, embed_dim, pred_len):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, pred_len)
        )
        init_linear_layers(self)

    def forward(self, x):
        return self.mlp(x)
