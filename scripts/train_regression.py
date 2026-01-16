# models/train_regression.py
import os
import torch
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
import warnings
from dgl.data.utils import split_dataset
from matgl.graph.data import MGLDataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.collate import collate_fn_contrastive
from datasets.crystal_phdos_dataset import CrystalPhDOSDataset
from models.phdos_decoder import PhDOSDecoder
from models.crystal_encoder import CrystalEncoder
warnings.simplefilter("ignore")
class RegressionLightning(pl.LightningModule):
    def __init__(self, regression_model, dataset, save_path=None, lr=1e-3):
        super().__init__()
        self.model = regression_model
        self.dataset = dataset
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.save_path = save_path
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.test_results = []

    def forward(self, graph_batch):
        # graph_batch -> latent -> MLP
        return self.model(graph_batch)

    def training_step(self, batch, batch_idx):
        graph_batch, phdos, idxs = batch
        phdos_pred = self.model(graph_batch)
        loss = self.criterion(phdos_pred, phdos)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        graph_batch, phdos, idxs = batch
        phdos_pred = self.model(graph_batch)
        loss = self.criterion(phdos_pred, phdos)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        graph_batch, phdos, idxs = batch
        phdos_pred = self.model(graph_batch)
        loss = self.criterion(phdos_pred, phdos)
        phdos_pred_real = self.dataset.reverse_minmax(phdos_pred, idxs)
        phdos_real = self.dataset.reverse_minmax(phdos, idxs)
        #loss = self.criterion(phdos_pred_real, phdos_real)
        for i, idx in enumerate(idxs):
            self.test_results.append({
                "id": self.dataset.ids[idx],
                "pred": phdos_pred_real[i].cpu().numpy(),
                "true": phdos_real[i].cpu().numpy()
            })
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_epoch_end(self, outputs):
        if self.save_path and self.test_results:
            rows = []
            for item in self.test_results:
                rows.append([item["id"]] + item["pred"].tolist() + item["true"].tolist())

            seq_len = len(self.test_results[0]["pred"])
            columns = ["id"] + [f"pred_{i}" for i in range(seq_len)] + [f"true_{i}" for i in range(seq_len)]
            df = pd.DataFrame(rows, columns=columns)
            df.to_csv(self.save_path, index=False)
            print(f"Test results saved to {self.save_path}")
            
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

# -----------------------------
# Load config
# -----------------------------
config_path0 = "configs/contrastive.yaml"
with open(config_path0) as f:
    cfg0 = yaml.safe_load(f)
    
config_path = "configs/phdos_decoder.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# -----------------------------
# Dataset & DataLoader
# -----------------------------
dataset = CrystalPhDOSDataset(data_dir=cfg0["dataset"]["data_dir"])
splits = torch.load(os.path.join(cfg0["train"]["data_checkpoint_dir"], "dataset_splits.pt"))
train_data = splits["train_data"]
val_data = splits["val_data"]
test_data = splits["test_data"]

train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_contrastive,
    batch_size=cfg0["train"]["batch_size"],
    num_workers=cfg0["train"].get("num_workers", 0)
)

# -----------------------------
# Load Phase1 encoder
# -----------------------------
crystal_encoder = CrystalEncoder(
    hidden_size=cfg0["model"]["hidden_size"],
    nblocks=cfg0["model"]["nblocks"],
    latent_size=cfg0["model"]["latent_size"],
)

# 载入 Phase1 训练好的 checkpoint
encoder_path = os.path.join(cfg0["train"]["encoder_checkpoint_dir"], "encoder_checkpoint.pt")
checkpoint = torch.load(encoder_path)
crystal_encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)

# -----------------------------
# Regression model
# -----------------------------
reg_model = PhDOSDecoder(
    encoder_struc=crystal_encoder,
    latent_size=cfg0["model"]["latent_size"],
    output_size=cfg["model"]["output_size"],
    freeze_encoder=True
)
model = RegressionLightning(
    regression_model=reg_model,
    dataset=dataset,
    save_path=cfg["train"].get("test_result_path", "./results/test_results.csv"),
    lr=cfg["train"]["lr"]
)

# -----------------------------
# Callbacks
# -----------------------------
os.makedirs(cfg["train"]["log_dir"], exist_ok=True)
os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=cfg["train"]["checkpoint_dir"],
    filename="regression-best-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)

# -----------------------------
# Trainer
# -----------------------------
trainer = pl.Trainer(
    max_epochs=cfg["train"]["num_epochs"],
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    default_root_dir=cfg["train"]["log_dir"],
    log_every_n_steps=10,
    callbacks=[checkpoint_callback] 
)

# -----------------------------
# Start training
# -----------------------------
trainer.fit(model, train_loader, val_loader)

# -----------------------------
# 测试 & 保存 CSV
# -----------------------------
best_model_path = checkpoint_callback.best_model_path
best_model = RegressionLightning.load_from_checkpoint(
    best_model_path,
    regression_model=reg_model,
    dataset=dataset,
    save_path=cfg["train"].get("test_result_path", "./results/test_results.csv"),
    lr=cfg["train"]["lr"]
)

trainer.test(best_model, test_loader)
