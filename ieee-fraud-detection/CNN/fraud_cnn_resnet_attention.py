"""
Layout B: 1D-CNN + ResNet1D + Self-Attention model for IEEE-CIS fraud detection.

Drop-in replacement for `FraudLSTM` in `Time_Series_LSTM_per_UID_PyTorch_v2.ipynb`.

Input shape from your existing DataLoader:
    x       : (B, T=WINDOW=20, F=N_FEATURES)   — left-padded
    lengths : (B,)                              — real-step count per sample

Output:
    logits  : (B, 1)                            — same shape as FraudLSTM

Pipeline inside the model:
    (B, T, F)
        ├── transpose ──────────────────► (B, F, T)
        ├── Conv1D stem (F → C, k=3) ───► (B, C, T)
        ├── ResBlock1D × N_RESBLOCKS ───► (B, C, T)
        ├── transpose ──────────────────► (B, T, C)
        ├── + learnable positional emb ─► (B, T, C)
        ├── Self-Attention + residual ──► (B, T, C)
        ├── FeedForward    + residual ──► (B, T, C)
        ├── masked mean pool over time ─► (B, C)
        └── MLP head ──────────────────► (B, 1)
"""

import torch
from torch import nn


# =============================================================================
# 1D Residual Block — shape preserving
# =============================================================================
class ResBlock1D(nn.Module):
    """
    Standard 1D residual block (identity skip).

    Skip BRANCH      : `identity = x`              (does not pass through conv)
    Skip CONNECTION  : `out + identity`            (element-wise sum after conv2)

    Shape preserved : (B, c, T) → (B, c, T)
    """
    def __init__(self, c: int, k: int = 3, drop: float = 0.1):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, padding=p)
        self.bn1   = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=p)
        self.bn2   = nn.BatchNorm1d(c)
        self.drop  = nn.Dropout(drop)
        self.act   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x                                   # ◄── SKIP BRANCH
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + identity                           # ◄── SKIP CONNECTION
        return self.act(out)


# =============================================================================
# Full model — Conv1D stem → ResNet stack → Self-Attention → pool → head
# =============================================================================
class FraudCNNResAttn(nn.Module):
    """
    Layout B model: 1D-CNN over time, with all features as input channels,
    refined by ResNet blocks, then re-weighted by multi-head self-attention.

    Args
    ----
    n_features    : int   — number of features per timestep (e.g. 244)
    window        : int   — sequence length (WINDOW from notebook, e.g. 20)
    c_hidden      : int   — channel width after the Conv1D stem (and inside ResNet)
    n_resblocks   : int   — how many ResBlock1D to stack
    n_heads       : int   — number of self-attention heads (must divide c_hidden)
    drop          : float — dropout used inside ResBlocks, attention, and head
    output_dim    : int   — final logit dim, keep =1 to match BCEWithLogitsLoss
    """
    def __init__(self,
                 n_features: int,
                 window: int = 20,
                 c_hidden: int = 128,
                 n_resblocks: int = 3,
                 n_heads: int = 4,
                 drop: float = 0.2,
                 output_dim: int = 1):
        super().__init__()
        assert c_hidden % n_heads == 0, "c_hidden must be divisible by n_heads"
        self.window = window

        # ------------------------------------------------------------------
        # Stage 1 — Conv1D stem
        # Mix `n_features` input channels into `c_hidden` learned channels at
        # every timestep. Kernel size 3 → captures patterns over 3 adjacent
        # transactions per channel.
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(c_hidden),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Stage 2 — ResNet1D blocks
        # Shape-preserving (no MaxPool, no stride) because WINDOW is already
        # short (20). We deepen semantics, not receptive field.
        # ------------------------------------------------------------------
        self.resblocks = nn.Sequential(*[
            ResBlock1D(c_hidden, k=3, drop=drop)
            for _ in range(n_resblocks)
        ])

        # ------------------------------------------------------------------
        # Stage 3 — Self-Attention (Transformer encoder style, single layer)
        # Treats each of the `window` timesteps as a token of dim c_hidden.
        # `key_padding_mask` ignores the left-padded positions.
        # ------------------------------------------------------------------
        # Learnable positional embedding — attention is permutation-invariant
        # without it, so the model would lose all order information.
        self.pos = nn.Parameter(torch.zeros(1, window, c_hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=c_hidden,
            num_heads=n_heads,
            dropout=drop,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(c_hidden)

        self.ff = nn.Sequential(
            nn.Linear(c_hidden, c_hidden * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(c_hidden * 2, c_hidden),
        )
        self.norm2 = nn.LayerNorm(c_hidden)

        # ------------------------------------------------------------------
        # Stage 4 — Classification head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(c_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x       : (B, T, F)  — same layout as the LSTM model expects
        # lengths : (B,)
        B, T, _ = x.shape

        # --- Conv path expects (B, C, T), so transpose feature axis to dim 1 ---
        x = x.transpose(1, 2)               # (B, F, T)
        h = self.stem(x)                    # (B, C, T)
        h = self.resblocks(h)               # (B, C, T)

        # --- Back to (B, T, C) for attention + LayerNorm ---
        h = h.transpose(1, 2)               # (B, T, C)
        h = h + self.pos                    # add learnable positional embedding

        # --- Build key padding mask from `lengths` (left-padded sequences) ---
        # `real[i, t] = True` if position t in sample i is a real timestep.
        idx          = torch.arange(T, device=h.device).unsqueeze(0)   # (1, T)
        pos_from_end = T - 1 - idx                                     # (1, T)
        real         = pos_from_end < lengths.unsqueeze(1)             # (B, T)
        key_padding_mask = ~real                                       # True = IGNORE

        # --- Self-Attention sub-block with residual + LayerNorm ---
        a, _ = self.attn(h, h, h,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        h = self.norm1(h + a)

        # --- Feed-forward sub-block with residual + LayerNorm ---
        h = self.norm2(h + self.ff(h))

        # --- Masked mean pool over the real timesteps only ---
        mask_f = real.unsqueeze(-1).float()                            # (B, T, 1)
        cnt    = mask_f.sum(dim=1).clamp(min=1.0)                      # (B, 1)
        pooled = (h * mask_f).sum(dim=1) / cnt                         # (B, C)

        return self.head(pooled)                                       # (B, 1)


# =============================================================================
# Smoke test — run when invoked directly
# =============================================================================
if __name__ == "__main__":
    B, T, F = 8, 20, 244       # mock shapes matching your IEEE-CIS pipeline
    m = FraudCNNResAttn(n_features=F, window=T)
    xb = torch.randn(B, T, F)
    lb = torch.randint(1, T + 1, (B,))
    out = m(xb, lb)
    print("output shape :", out.shape)        # → torch.Size([8, 1])
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")
