# models/pdos_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    """带 bottleneck 的 1D 残差块"""
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        mid_channels = channels * 2  # 扩张通道提高表达能力
        self.conv1 = nn.Conv1d(channels, mid_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(mid_channels, channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, C, L)
        residual = x
        out = self.relu(self.conv1(x))
        out = self.dropout(self.conv2(out))
        out = out + residual
        out = out.permute(0, 2, 1)   # (B, L, C)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        return out


class PhDOSEncoder(nn.Module):
    """
    改进版 PDOS 编码器：多尺度卷积 + 残差块 + Transformer + MLP
    输入: (B, 64)
    输出: (B, latent_size)
    """
    def __init__(self, latent_size=256, hidden_channels=96, num_residual=2, num_transformer=2, dropout=0.1):
        super().__init__()

        # ---------- 多尺度卷积 ----------
        self.multi_conv = nn.ModuleList([
            nn.Conv1d(1, hidden_channels // 3, k, padding=k // 2)
            for k in [3, 5, 7]
        ])
        self.relu = nn.ReLU()

        # ---------- 残差块 ----------
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(hidden_channels, kernel_size=3, dropout=dropout)
            for _ in range(num_residual)
        ])

        # ---------- Transformer ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels, nhead=4, dim_feedforward=hidden_channels * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer)

        # ---------- 全局池化 + MLP ----------
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, latent_size),
            nn.ReLU(),
            nn.LayerNorm(latent_size),
            nn.Dropout(dropout),
            nn.Linear(latent_size, latent_size)
        )

    def forward(self, x):
        """
        x: (B, 64)
        """
        # -------- 物理先验归一化 --------
        x = F.relu(x)  # 保证非负
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化到积分为1

        # -------- 多尺度卷积特征提取 --------
        x = x.unsqueeze(1)  # (B, 1, 64)
        x = torch.cat([conv(x) for conv in self.multi_conv], dim=1)  # (B, hidden_channels, 64)
        x = self.relu(x)

        # -------- 残差块 --------
        for block in self.res_blocks:
            x = block(x)  # (B, hidden_channels, 64)

        # -------- Transformer 全局建模 --------
        x = x.permute(0, 2, 1)  # (B, L, C)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)

        # -------- 全局池化 + MLP --------
        x = self.global_pool(x).squeeze(-1)
        latent = self.fc(x)
        return latent
