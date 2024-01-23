import pytorch_lightning as pl
import torch
from src.processing_utils import mask_to_labels

class AerialModule(pl.LightningModule):
    def __init__(self, backbone, loss_fn, metric, lr):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.metric = metric
        self.lr = lr
        self.save_hyperparameters()
        # self.save_hyperparameters(ignore=['backbone', 'loss_fn', 'metric'])
        # 'save_hyperparameters(): when loading from checkpoint, a Warning is raised saying that
        # backbone, loss_fn, and metric were basically saved two times. To avoid this 
        # 'self.save_hyperparameters(ignore=['backbone', 'loss_fn', 'metric'])' can be used, but
        # this means we have to manually pass a backbone, a loss_fn, and a metric where doing
        # AerialModule.load_from_checkpoint(...). See: 
        # https://pytorch-lightning.readthedocs.io/en/1.6.5/common/hyperparameters.html#lightningmodule-hyperparameters
        # This seems more cumbersome to me. In either case, the checkpoints had the same size.
    
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
        
    def on_train_epoch_start(self):
        return super().on_train_epoch_start()
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            threshold=1e-4, 
            threshold_mode='rel'
        )
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}
        
    
        
        