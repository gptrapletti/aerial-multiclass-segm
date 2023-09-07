from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchmetrics
from .utils import mask_to_labels

class AerialModule(pl.LightningModule):
    def __init__(self, backbone, lr):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.metric = torchmetrics.Dice(num_classes=6)
        # self.optimizer = optimizer
        # self.scheduler = scheduler
        self.lr = lr
    
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(input=preds, target=masks)
        metric = self.metric(preds=preds, target=mask_to_labels(masks))
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_metric', metric, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)
        metric = self.metric(preds=preds, target=mask_to_labels(masks))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        images, masks = batch
        preds = self(images)
        metric = self.metric(preds=preds, target=mask_to_labels(masks))
        self.log('test_metric', metric, prog_bar=True)
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode='min', 
            factor=0.1, 
            patience=10, 
            threshold=0.01, 
            threshold_mode='abs'
        )
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}
        
    
        
        