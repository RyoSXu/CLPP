# models/phdos_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, expand_ratio=2, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(self.fc2(out))
        out = self.norm(out + x)
        return out


class PhDOSDecoder(nn.Module):
    """
    从结构 latent 解码 PDOS 的改进版模型
    """
    def __init__(self, encoder_struc, latent_size=256, output_size=64,
                 hidden_size=512, n_res_blocks=4, dropout=0.1, freeze_encoder=True):
        super().__init__()
        self.encoder_struc = encoder_struc
        if freeze_encoder:
            for p in self.encoder_struc.parameters():
                p.requires_grad = False
            self.encoder_struc.eval()

        # 初始投影
        self.fc_in = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size)
        )

        # 残差块堆叠
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, expand_ratio=2, dropout=dropout) for _ in range(n_res_blocks)]
        )

        # 输出层
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, structure_batch):
        # 编码结构
        z = self.encoder_struc(structure_batch)   # (B, latent_size)

        # 解码
        h = self.fc_in(z)
        h = self.res_blocks(h)
        phdos_pred = self.fc_out(h)

        # 保证物理合理性
        phdos_pred = F.relu(phdos_pred)
        phdos_pred = phdos_pred / (phdos_pred.sum(dim=-1, keepdim=True) + 1e-8)

        return phdos_pred
