# models/contrastive_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    """
    多模态对比学习模型
    输入：晶体结构 latent / PDOS latent
    输出：对比损失
    """
    def __init__(self, encoder_struc, encoder_phdos, temperature=0.1, lr=1e-3, latent_size=256):
        super().__init__()
        self.encoder_struc = encoder_struc  # M3GNet
        self.encoder_phdos = encoder_phdos
        self.temperature = temperature
        self.lr = lr
        
        self.projector_struc = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )
        self.projector_phdos = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size)
        )
        
    def forward(self, graph_batch, phdos_batch):
        z_struc = self.encoder_struc(graph_batch)
        z_phdos = self.encoder_phdos(phdos_batch)

        z_struc = self.projector_struc(z_struc)
        z_phdos = self.projector_phdos(z_phdos)

        z_struc_norm = F.normalize(z_struc, dim=1)
        z_phdos_norm = F.normalize(z_phdos, dim=1)

        sim_matrix = torch.matmul(z_struc_norm, z_phdos_norm.T) / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
        return loss

    @torch.no_grad()
    def get_similarity(self, graph_batch, phdos_batch):
        z_struc = self.encoder_struc(graph_batch)
        z_phdos = self.encoder_phdos(phdos_batch)
        z_struc = self.projector_struc(z_struc)
        z_phdos = self.projector_phdos(z_phdos)
        z_struc_norm = F.normalize(z_struc, dim=1)
        z_phdos_norm = F.normalize(z_phdos, dim=1)
        return torch.matmul(z_struc_norm, z_phdos_norm.T)
