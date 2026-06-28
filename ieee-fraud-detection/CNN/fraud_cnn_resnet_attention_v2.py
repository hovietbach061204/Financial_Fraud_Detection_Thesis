"""
Layout B v2 — Conv1D + ResNet1D + STACKED Self-Attention + CLS + Mean/Max Pool + Static Tower.

Upgrades over fraud_cnn_resnet_attention.py:
  (1) Masked Mean + Max pool concatenated     → captures average + spike behavior
  (2) Learnable CLS token                     → transformer-style classification readout
  (3) Static-row tower (last transaction MLP) → recovers XGBoost-style intra-row signal
  (3) Stack of N transformer encoder blocks   → multi-hop attention reasoning

Same forward signature as v1:
    forward(x, lengths) -> logits of shape (B, 1)

Pipeline:
    (B, T, F)
        ├── transpose ───────────────────────► (B, F, T)
        ├── Conv1D stem (F → C, k=3) ────────► (B, C, T)
        ├── ResBlock1D × N_RESBLOCKS ────────► (B, C, T)
        ├── transpose + positional emb ──────► (B, T, C)
        ├── [optionally prepend CLS token] ──► (B, T+1, C)
        ├── TransformerBlock × N_ATTN_LAYERS ► (B, T+1, C)
        ├── pooling:
        │     - mean over real timesteps     → (B, C)
        │     - max  over real timesteps     → (B, C)        [optional]
        │     - CLS readout                  → (B, C)        [optional]
        │     - static MLP on last row       → (B, S)        [optional]
        ├── concatenate the parts            ► (B, pool_dim)
        └── MLP head                         ► (B, 1)
"""

import torch
from torch import nn


# =============================================================================
# 1D Residual Block — shape preserving (unchanged from v1)
# =============================================================================
class ResBlock1D(nn.Module):
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
        identity = x                                   # SKIP BRANCH
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + identity                           # SKIP CONNECTION
        return self.act(out)


# =============================================================================
# Transformer encoder block — one self-attention + FF, both with residual+LN
# =============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, c: int, n_heads: int, drop: float = 0.2):
        super().__init__()
        self.attn  = nn.MultiheadAttention(
            embed_dim=c, num_heads=n_heads, dropout=drop, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(c)
        self.ff    = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(c * 2, c),
        )
        self.norm2 = nn.LayerNorm(c)

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(h, h, h,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return h


# =============================================================================
# Full v2 model
# =============================================================================
class FraudCNNResAttnV2(nn.Module):
    """
    Args
    ----
    n_features        : number of features per timestep (e.g. 244)
    window            : sequence length T (e.g. 20)
    c_hidden          : channel width of Conv stem / ResNet / attention
    n_resblocks       : number of ResBlock1D in the CNN stack
    n_attn_layers     : number of TransformerBlock layers (was 1 in v1)        ◄── NEW
    n_heads           : self-attention heads (must divide c_hidden)
    drop              : dropout used in ResBlocks, attention, FF, head, static tower
    static_hidden     : hidden width of the static-row tower
    use_cls           : prepend a learnable CLS token, use its embedding for readout ◄── NEW
    use_max_pool      : concat masked-max pool with mean pool                      ◄── NEW
    use_static_tower  : run the last raw row through an MLP and concat it          ◄── NEW
    output_dim        : final logit dim (keep =1 for BCEWithLogitsLoss)
    """
    def __init__(self,
                 n_features: int,
                 window: int = 20,
                 c_hidden: int = 128,
                 n_resblocks: int = 3,
                 n_attn_layers: int = 2,
                 n_heads: int = 4,
                 drop: float = 0.2,
                 static_hidden: int = 64,
                 use_cls: bool = True,
                 use_max_pool: bool = True,
                 use_static_tower: bool = True,
                 output_dim: int = 1):
        super().__init__()
        assert c_hidden % n_heads == 0, "c_hidden must be divisible by n_heads"

        self.window           = window
        self.use_cls          = use_cls
        self.use_max_pool     = use_max_pool
        self.use_static_tower = use_static_tower

        # ------- Stage 1 — Conv1D stem -------
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(c_hidden),
            nn.ReLU(),
        )

        # ------- Stage 2 — ResNet1D stack -------
        self.resblocks = nn.Sequential(*[
            ResBlock1D(c_hidden, k=3, drop=drop)
            for _ in range(n_resblocks)
        ])

        # ------- Stage 3 — positional embedding (over the WINDOW positions only;
        #                  the CLS token is added on top after this)
        self.pos = nn.Parameter(torch.zeros(1, window, c_hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # ------- (Optional) CLS token -------
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, c_hidden))
            nn.init.trunc_normal_(self.cls, std=0.02)

        # ------- Stage 4 — STACK of Transformer blocks -------
        self.attn_blocks = nn.ModuleList([
            TransformerBlock(c_hidden, n_heads, drop=drop)
            for _ in range(n_attn_layers)
        ])

        # ------- (Optional) Static-row tower -------
        if use_static_tower:
            self.static_mlp = nn.Sequential(
                nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(drop),
                nn.Linear(256, static_hidden), nn.ReLU(),
            )

        # ------- Compute pool dim from the toggles -------
        pool_dim = c_hidden                       # mean is always present
        if use_max_pool:    pool_dim += c_hidden
        if use_cls:         pool_dim += c_hidden
        if use_static_tower:pool_dim += static_hidden

        # ------- Stage 5 — Head -------
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(pool_dim, 64), nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, output_dim),
        )

    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)  left-padded     lengths: (B,) real-step count
        B, T, F = x.shape

        # Keep last (most-recent) raw row aside for the static tower.
        # Left-padding writes to the RIGHT end, so x[:, -1, :] is always real.
        last_row = x[:, -1, :]                           # (B, F)

        # ----- CNN path (B, T, F) → (B, T, C) -----
        x_t = x.transpose(1, 2)                          # (B, F, T)
        h   = self.stem(x_t)                             # (B, C, T)
        h   = self.resblocks(h)                          # (B, C, T)
        h   = h.transpose(1, 2)                          # (B, T, C)
        h   = h + self.pos                               # positional encoding (over T only)

        # ----- Build real-position mask from `lengths` -----
        idx          = torch.arange(T, device=h.device).unsqueeze(0)   # (1, T)
        pos_from_end = T - 1 - idx                                     # (1, T)
        real         = pos_from_end < lengths.unsqueeze(1)             # (B, T) — True = real

        # ----- (Optional) Prepend CLS token -----
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)             # (B, 1, C)
            h   = torch.cat([cls, h], dim=1)             # (B, T+1, C)
            # CLS is always "real" so attention won't mask it
            cls_real = torch.ones(B, 1, dtype=torch.bool, device=h.device)
            real_full = torch.cat([cls_real, real], dim=1)              # (B, T+1)
            key_padding_mask = ~real_full
        else:
            key_padding_mask = ~real

        # ----- Stacked Transformer blocks -----
        for blk in self.attn_blocks:
            h = blk(h, key_padding_mask)

        # ----- Split CLS embedding from the time tokens -----
        if self.use_cls:
            cls_out = h[:, 0, :]                         # (B, C)
            seq_h   = h[:, 1:, :]                        # (B, T, C)
        else:
            seq_h = h                                    # (B, T, C)

        # ----- Pool over REAL timesteps only -----
        mask_f = real.unsqueeze(-1).float()              # (B, T, 1)
        cnt    = mask_f.sum(dim=1).clamp(min=1.0)        # (B, 1)

        pooled_parts = [(seq_h * mask_f).sum(dim=1) / cnt]   # mean

        if self.use_max_pool:
            # mask out padded positions with -inf so they never win the max
            seq_h_for_max = seq_h.masked_fill(~real.unsqueeze(-1), float('-inf'))
            pooled_parts.append(seq_h_for_max.max(dim=1).values)

        if self.use_cls:
            pooled_parts.append(cls_out)

        if self.use_static_tower:
            pooled_parts.append(self.static_mlp(last_row))

        pooled = torch.cat(pooled_parts, dim=1)          # (B, pool_dim)
        return self.head(pooled)                         # (B, 1)


class FraudCNNResAttnV2PerStep(nn.Module):
    """
    Per-timestep CNN/ResNet/Attention model.

    Input : X shape (B, T, F), left-padded
            lengths shape (B,)
    Output: logits shape (B, T)

    Train with masked BCE:
        loss = BCE(logits[M], Y[M])
    """
    def __init__(self,
                 n_features: int,
                 window: int = 20,
                 c_hidden: int = 128,
                 n_resblocks: int = 3,
                 n_attn_layers: int = 2,
                 n_heads: int = 4,
                 drop: float = 0.2):
        super().__init__()

        assert c_hidden % n_heads == 0

        self.window = window

        self.stem = nn.Sequential(
            nn.Conv1d(n_features, c_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(c_hidden),
            nn.ReLU(),
        )

        self.resblocks = nn.Sequential(*[
            ResBlock1D(c_hidden, k=3, drop=drop)
            for _ in range(n_resblocks)
        ])

        self.pos = nn.Parameter(torch.zeros(1, window, c_hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.attn_blocks = nn.ModuleList([
            TransformerBlock(c_hidden, n_heads, drop=drop)
            for _ in range(n_attn_layers)
        ])

        # One logit per timestep
        self.token_head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(c_hidden, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths):
        # x: (B, T, F), LEFT-padded
        B, T, _ = x.shape

        h = x.transpose(1, 2)        # (B, F, T)
        h = self.stem(h)             # (B, C, T)
        h = self.resblocks(h)        # (B, C, T)
        h = h.transpose(1, 2)        # (B, T, C)

        h = h + self.pos[:, :T, :]

        # left-padding mask: real positions are at the RIGHT end
        idx = torch.arange(T, device=x.device).unsqueeze(0)
        real = (T - 1 - idx) < lengths.to(x.device).unsqueeze(1)
        key_padding_mask = ~real

        for blk in self.attn_blocks:
            h = blk(h, key_padding_mask)

        logits = self.token_head(h).squeeze(-1)   # (B, T)
        return logits
# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    B, T, F = 8, 20, 244
    m = FraudCNNResAttnV2(n_features=F, window=T)
    xb = torch.randn(B, T, F)
    lb = torch.randint(1, T + 1, (B,))
    out = m(xb, lb)
    print("output shape :", out.shape)        # → torch.Size([8, 1])
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable params: {n_params:,}")

    # Ablation comparison — what each switch contributes (parameter-wise)
    for cfg in [
        dict(use_cls=False, use_max_pool=False, use_static_tower=False, n_attn_layers=1),
        dict(use_cls=False, use_max_pool=True,  use_static_tower=False, n_attn_layers=1),
        dict(use_cls=True,  use_max_pool=True,  use_static_tower=False, n_attn_layers=1),
        dict(use_cls=True,  use_max_pool=True,  use_static_tower=True,  n_attn_layers=1),
        dict(use_cls=True,  use_max_pool=True,  use_static_tower=True,  n_attn_layers=2),
    ]:
        mm = FraudCNNResAttnV2(n_features=F, window=T, **cfg)
        n  = sum(p.numel() for p in mm.parameters() if p.requires_grad)
        print(f"{cfg} -> {n:,} params")
