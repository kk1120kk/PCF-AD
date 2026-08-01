
"""
Gated ConvNet with 1‑D per‑channel convolution for TIMIT frame‑level phoneme classification.
Architecture: 10 stacked GatedConvBlock, each containing:
    - depthwise 1D conv (kernel=7, groups=channels) → "1‑D convolution"
    - gated 1D conv (kernel=2) with length‑preserving right padding
    - residual connection
    - dropout only (no Batchnorm, as in the original paper setup)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConvBlock(nn.Module):
    def __init__(self, channels, kernel_size_1d=7, kernel_size_gated=2, dropout=0.0):
        super().__init__()
        # per‑channel 1‑D temporal convolution (context window -3..+3)
        self.depthwise_conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size_1d,
            padding=kernel_size_1d // 2,      # 'same' for odd kernel
            groups=channels,
            bias=True
        )
        # gated convolution (kernel_size_gated=2)
        self.conv_f = nn.Conv1d(channels, channels, kernel_size_gated, padding=0, bias=True)
        self.conv_g = nn.Conv1d(channels, channels, kernel_size_gated, padding=0, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.kernel_size_gated = kernel_size_gated

    def forward(self, x):
        residual = x
        out = self.depthwise_conv(x)          # (B, C, T)
        out = self.dropout(out)

        # Right‑pad by (kernel_size_gated - 1) to preserve length after conv with kernel=2
        pad_len = self.kernel_size_gated - 1  # = 1
        out_padded = F.pad(out, (0, pad_len)) # (B, C, T+1)
        f = self.conv_f(out_padded)
        g = torch.sigmoid(self.conv_g(out_padded))
        out = f * g                           # (B, C, T)
        out = self.dropout(out)
        out = out + residual                  # same length as input
        return out


class GatedConvNet1D(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=10,
                 kernel_size_1d=7, kernel_size_gated=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            GatedConvBlock(hidden_dim, kernel_size_1d, kernel_size_gated, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, D, T) — as produced by the data loader
        x = x.transpose(1, 2)                # (B, T, D)
        x = self.input_proj(x)               # (B, T, hidden_dim)
        x = x.transpose(1, 2)                # (B, hidden_dim, T)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.output_proj(x)         # (B, num_classes, T)
        logits = logits.transpose(1, 2)      # (B, T, num_classes)
        return logits


if __name__ == "__main__":
    B, D, T, C = 4, 240, 200, 39
    model = GatedConvNet1D(input_dim=D, num_classes=C, hidden_dim=128, num_layers=10)
    x = torch.randn(B, D, T)
    y = model(x)
    print(y.shape)   # expected (4, 200, 39)