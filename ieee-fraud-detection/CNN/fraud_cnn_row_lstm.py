"""
Layout A — Per-row 1D-CNN feature encoder  →  Bidirectional LSTM over time.

Same forward signature as FraudLSTM / FraudCNNResAttnV2:
    forward(x, lengths) -> logits of shape (B, 1)

Pipeline:
    (B, T, F)
        ├── for each row, slide a 1D-CNN over the FEATURE axis ────► (B, T, embed_dim)
        │     (treats each transaction as a length-F mini-sequence with 1 channel)
        ├── Bidirectional LSTM over the T row embeddings ─────────► (B, T, 2*H)
        ├── masked mean pool   ───┐
        ├── masked max  pool   ───┤
        ├── static-row MLP    ────┴── concat ────────────────────► (B, pool_dim)
        └── MLP head ──────────────────────────────────────────► (B, 1)

Key idea (Layout A):
    - CNN models LOCAL FEATURE INTERACTIONS within one transaction (no time mixing).
    - LSTM models THE TEMPORAL STORY across the T transactions.
    The two responsibilities are cleanly separated.
"""

import torch
from torch import nn


# =============================================================================
# Per-row 1D-CNN feature encoder
#   maps one row of (1, F) raw features → one vector of `embed_dim`
# =============================================================================
class RowCNNEncoder(nn.Module):
    """
    Applies stacked 1D convolutions over the FEATURE axis of a single row.
    Pools along the feature axis to produce a fixed-size embedding per row.

    Input  shape: (N, 1, F)         (N = B * T after the per-batch flatten)
    Output shape: (N, embed_dim)
    """
    def __init__(self,
                 n_features: int,
                 channels: int = 64,
                 n_layers: int = 2,
                 kernel: int = 3,
                 drop: float = 0.2,
                 pool_size: int = 4):
        super().__init__()
        layers = []
        in_c = 1
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_c, channels, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(drop),
            ]
            in_c = channels
        self.cnn = nn.Sequential(*layers)

        # Reduce the feature axis to a small fixed length, then flatten.
        # AdaptiveAvgPool1d works for any n_features without recomputing dims.
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.embed_dim = channels * pool_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, F)
        h = self.cnn(x)                # (N, channels, F)
        h = self.pool(h)               # (N, channels, pool_size)
        return h.flatten(1)            # (N, channels * pool_size) = (N, embed_dim)


# =============================================================================
# Full Layout A model
# =============================================================================
class FraudCNNRowLSTM(nn.Module):
    """
    Args
    ----
    n_features         : number of raw features per timestep
    window             : sequence length T (kept for API symmetry; not used at construct time)
    cnn_channels       : channels inside the per-row CNN encoder
    cnn_layers         : how many Conv1d→BN→ReLU stacks in the row encoder
    cnn_kernel         : kernel size of the row-CNN (slides over feature axis)
    cnn_pool_size      : fixed length the row-CNN pools each row down to
    hidden_dim         : LSTM hidden size per direction
    num_layers         : number of stacked LSTM layers
    bidirectional      : use BiLSTM (recommended — matches your best LSTM baseline)
    drop               : dropout used in the row-CNN, LSTM (between layers), static tower, head
    static_hidden      : hidden width of the static-row tower
    use_max_pool       : concat masked-max pool with mean pool of the LSTM outputs
    use_static_tower   : concat an MLP-encoded last-row vector
    output_dim         : final logit dim (keep =1 for BCEWithLogitsLoss)
    """
    def __init__(self,
                 n_features: int,
                 window: int = 20,
                 cnn_channels: int = 64,
                 cnn_layers: int = 2,
                 cnn_kernel: int = 3,
                 cnn_pool_size: int = 4,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = True,
                 drop: float = 0.3,
                 static_hidden: int = 64,
                 use_max_pool: bool = True,
                 use_static_tower: bool = True,
                 output_dim: int = 1):
        super().__init__()

        self.use_max_pool     = use_max_pool
        self.use_static_tower = use_static_tower

        # --- per-row CNN encoder (Layout A's "CNN over columns") -----------
        self.row_enc = RowCNNEncoder(
            n_features=n_features,
            channels=cnn_channels,
            n_layers=cnn_layers,
            kernel=cnn_kernel,
            drop=drop,
            pool_size=cnn_pool_size,
        )
        embed_dim = self.row_enc.embed_dim

        # --- LSTM over time (Layout A's "LSTM handles time") ---------------
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=drop if num_layers > 1 else 0.0,
        )
        H_out = hidden_dim * (2 if bidirectional else 1)

        # --- Static-row tower (XGBoost-style intra-row signal) -------------
        if use_static_tower:
            self.static_mlp = nn.Sequential(
                nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(256, static_hidden), nn.ReLU(),
            )

        # --- Pool-dim arithmetic ------------------------------------------
        pool_dim = H_out                    # mean pool always present
        if use_max_pool:    pool_dim += H_out
        if use_static_tower:pool_dim += static_hidden

        # --- Head ---------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(pool_dim, 64), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, output_dim),
        )

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F), left-padded;   lengths: (B,) real-step count
        B, T, F = x.shape

        # Save last (most-recent) raw row for the static tower
        last_row = x[:, -1, :]                            # (B, F)

        # ----- Per-row CNN encoder applied to EVERY row in parallel -----
        # Merge batch and time so the CNN sees N = B*T independent rows.
        x_flat = x.reshape(B * T, 1, F)                   # (B*T, 1, F)
        embeds = self.row_enc(x_flat)                     # (B*T, embed_dim)
        embeds = embeds.view(B, T, -1)                    # (B, T, embed_dim)

        # ----- LSTM over time -----
        lstm_out, _ = self.lstm(embeds)                   # (B, T, H_out)

        # ----- Real-position mask from `lengths` (left-padded windows) ----
        idx          = torch.arange(T, device=x.device).unsqueeze(0)   # (1, T)
        pos_from_end = T - 1 - idx                                     # (1, T)
        real         = pos_from_end < lengths.unsqueeze(1)             # (B, T)
        mask_f       = real.unsqueeze(-1).float()                      # (B, T, 1)
        cnt          = mask_f.sum(dim=1).clamp(min=1.0)                # (B, 1)

        # ----- Masked Mean Pool over real timesteps -----
        pooled_parts = [(lstm_out * mask_f).sum(dim=1) / cnt]

        # ----- Masked Max Pool (padding → -inf) -----
        if self.use_max_pool:
            lstm_for_max = lstm_out.masked_fill(~real.unsqueeze(-1), float('-inf'))
            pooled_parts.append(lstm_for_max.max(dim=1).values)

        # ----- Static-row tower -----
        if self.use_static_tower:
            pooled_parts.append(self.static_mlp(last_row))

        pooled = torch.cat(pooled_parts, dim=1)                        # (B, pool_dim)
        return self.head(pooled)                                       # (B, 1)


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    B, T, F = 8, 20, 244
    m = FraudCNNRowLSTM(n_features=F, window=T)
    xb = torch.randn(B, T, F)
    lb = torch.randint(1, T + 1, (B,))
    out = m(xb, lb)
    print("output shape :", out.shape)        # → torch.Size([8, 1])
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")
