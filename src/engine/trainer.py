import torch
import pytorch_lightning as pl
from src.utils.optimizers import build_optimizer, build_scheduler
import models.base_model as PICNet
from models.backbone.loss_net import PCINet_Loss

class PL_PICNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.matcher = PICNet(config)
        self.loss = PCINet_Loss(config)
        
    def configure_optimizers(self):
        # Construct the optimizer and learning rate scheduler based on the provided configuration.
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        self.matcher(batch)
        self.loss(batch)
        return {'loss': batch['loss']}

    def validation_step(self, batch, batch_idx):
        self.matcher(batch)
        self.loss(batch)
        return {'loss': batch['loss']}
    
    def test_step(self, batch, batch_idx):
        self.matcher(batch)
        self.loss(batch)
        return {'loss': batch['loss']}

