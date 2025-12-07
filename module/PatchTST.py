import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert (B, L, C) into patches → (B, embed_dim, num_patches)"""
    def __init__(self, c_in, embed_dim, patch_len, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=embed_dim,
            kernel_size=patch_len,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        # x: (B, L, C)
        x = x.permute(0, 2, 1)  # -> (B, C, L)
        return self.conv(x)     # -> (B, embed_dim, num_patches)


class TransformerEncoderLayer(nn.Module):
    """One transformer block"""
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class PatchTST(nn.Module):
    def __init__(
        self,
        c_in,        # 输入通道数
        seq_len,     # 输入序列长度
        patch_len,   # patch长度
        stride,      # patch步长
        embed_dim,   # embedding维度
        n_heads=4,
        ff_dim=128,
        num_layers=3,
        pred_len=6   # 输出维度
    ):
        super().__init__()

        self.c_in = c_in
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim
        self.pred_len = pred_len

        # ---- Patch Embedding ---- #
        self.patch_embed = PatchEmbedding(
            c_in=c_in,
            embed_dim=embed_dim,
            patch_len=patch_len,
            stride=stride
        )

        # ---- Transformer layers ---- #
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=n_heads,
                ff_dim=ff_dim
            )
            for _ in range(num_layers)
        ])

        # ---- Prediction head ---- #
        self.head = nn.Linear(embed_dim, pred_len)

    def forward(self, x):

        # ---- auto reshape ----
        if x.ndim == 2:            # (B, L)
            x = x.unsqueeze(-1)    # → (B, L, 1)

        if x.shape[-1] != self.c_in:
            # 支持 (B, C, L) → (B, L, C)
            if x.shape[1] == self.c_in:
                x = x.permute(0, 2, 1)
            else:
                raise RuntimeError(
                    f"Input has C={x.shape[-1]}, but model expects c_in={self.c_in}"
                )

        # --- Patch embed ---
        x = self.patch_embed(x)        # (B, embed_dim, P)
        x = x.permute(0, 2, 1)         # (B, P, E)

        # --- Transformer ---
        for layer in self.encoder_layers:
            x = layer(x)

        # --- Pool ---
        x = x.mean(dim=1)              # (B, E)

        # --- Head ---
        return self.head(x)


