# models/train_contrastive.py
import os
import torch
from dgl.data.utils import split_dataset
from matgl.graph.data import MGLDataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
import warnings
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.crystal_encoder import CrystalEncoder
from utils.collate import collate_fn_contrastive
from datasets.crystal_phdos_dataset import CrystalPhDOSDataset
from models.phdos_encoder import PhDOSEncoder
from models.contrastive_model import ContrastiveModel
warnings.simplefilter("ignore")

# -----------------------------
# Load config
# -----------------------------
config_path = "configs/contrastive.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# -----------------------------
# LightningModule 封装 ContrastiveModel
# -----------------------------
class ContrastiveLightning(pl.LightningModule):
    def __init__(self, encoder_struc, encoder_phdos, temperature=0.07, lr=1e-3, latent_size=256):
        super().__init__()
        self.model = ContrastiveModel(encoder_struc, encoder_phdos, temperature, lr, latent_size)
        self.lr = lr

    def forward(self, graph_batch, phdos_batch):
        z_struc = self.model.encoder_struc(g=graph_batch, return_all_layer_output=False)
        z_phdos = self.model.encoder_phdos(phdos_batch)
        # 归一化
        z_struc_norm = F.normalize(z_struc, dim=1)
        z_phdos_norm = F.normalize(z_phdos, dim=1)
        return torch.matmul(z_struc_norm, z_phdos_norm.T)

    def training_step(self, batch, batch_idx):
        graph_batch, phdos_batch, _ = batch
        loss = self.model(graph_batch, phdos_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        graph_batch, phdos_batch, _ = batch
        loss = self.model(graph_batch, phdos_batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# -----------------------------
# Dataset & DataLoader
# -----------------------------
dataset = CrystalPhDOSDataset(data_dir=cfg["dataset"]["data_dir"])
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
os.makedirs(cfg["train"]["data_checkpoint_dir"], exist_ok=True)
torch.save({
    "train_data": train_data,
    "val_data": val_data,
    "test_data": test_data
}, os.path.join(cfg["train"]["data_checkpoint_dir"], "dataset_splits.pt"))

train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_contrastive,
    batch_size=cfg["train"]["batch_size"],
    num_workers=cfg["train"].get("num_workers", 0)
)

# -----------------------------
# Model
# -----------------------------
crystal_encoder = CrystalEncoder(
    hidden_size=cfg["model"]["hidden_size"],
    nblocks=cfg["model"]["nblocks"],
    latent_size=cfg["model"]["latent_size"],
)
phdos_encoder = PhDOSEncoder(latent_size=cfg['model']['latent_size'])  
model = ContrastiveLightning(
    encoder_struc=crystal_encoder,
    encoder_phdos=phdos_encoder,
    temperature=cfg["model"].get("temperature", 0.07),
    lr=cfg["train"].get("lr", 1e-3),
    latent_size=cfg["model"]["latent_size"]
)

# -----------------------------
# Callbacks
# -----------------------------
os.makedirs(cfg["train"]["log_dir"], exist_ok=True)
os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
os.makedirs(cfg["train"]["encoder_checkpoint_dir"], exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=cfg["train"]["checkpoint_dir"],
    filename="contrastive-best-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)

lr_monitor = LearningRateMonitor(logging_interval="step")

# -----------------------------
# Trainer
# -----------------------------
trainer = pl.Trainer(
    max_epochs=cfg["train"]["num_epochs"],
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    default_root_dir=cfg["train"]["log_dir"],
    callbacks=[checkpoint_callback, lr_monitor],
    log_every_n_steps=10
)

# -----------------------------
# Start training
# -----------------------------
trainer.fit(model, train_loader, val_loader)

# -----------------------------
# 保存最优 encoder 以便回归使用
# -----------------------------
best_checkpoint_path = checkpoint_callback.best_model_path
best_model = ContrastiveLightning.load_from_checkpoint(
    best_checkpoint_path,
    encoder_struc=crystal_encoder,
    encoder_phdos=phdos_encoder
)

# 保存 encoder
torch.save({
    "encoder_state_dict": best_model.model.encoder_struc.state_dict()
}, os.path.join(cfg["train"]["encoder_checkpoint_dir"], "encoder_checkpoint.pt"))

# 测试
test_loss = trainer.validate(best_model, test_loader)
print("Test Loss:", test_loss)