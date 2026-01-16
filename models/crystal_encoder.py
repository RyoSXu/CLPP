# models/crystal_encoder.py
import torch
import torch.nn as nn
from matgl.models import M3GNet

class CrystalEncoder(nn.Module):
    """
    晶体结构编码器：使用 MatGL M3GNet
    输入: DGLGraph
    输出: latent_size 维向量
    """
    def __init__(self, hidden_size=128, latent_size=256, nblocks=3):
        super().__init__()
        # M3GNet Block 构建
        self.m3gnet = M3GNet(
            dim_node_embedding=hidden_size,
            dim_edge_embedding=hidden_size,
            nblocks=nblocks,
            units=hidden_size,
            cutoff=5.0,
            threebody_cutoff=4.0,
            readout_type="weighted_atom",
            task_type="regression",
            ntargets=latent_size,   
            activation_type="swish",
            dropout=0.1
        )

    def forward(self, graph):
        """
        graph: DGLGraph, 来自 Dataset
        """
        out = self.m3gnet(graph)
        return out
