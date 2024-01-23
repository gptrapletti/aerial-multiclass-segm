import pytorch_lightning as pl
import torch
import numpy as np
import cv2
import os
from src.processing_utils import mask_to_labels, color_code_mask2

class AerialModule(pl.LightningModule):
    def __init__(self, backbone, loss_fn, metric, lr):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.metric = metric
        self.lr = lr
        self.activation_function = torch.nn.Softmax(dim=1)
        self.current_image_filename = None
        self.current_image = torch.zeros(size=(2000, 3000, 3))
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
        images, masks, _ = batch
        preds = self(images)
        loss = self.loss_fn(input=preds, target=masks)
        probs = self.activation_function(preds) # NOTE: need to do this because metric is torchmetrics.Dice and it does't apply an activation function internally
        metric = self.metric(preds=probs, target=mask_to_labels(masks))
        self.log('train_loss', loss, prog_bar=True, batch_size=images.shape[0])
        self.log('train_metric', metric, prog_bar=True, batch_size=images.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks, _ = batch
        preds = self(images)
        loss = self.loss_fn(preds, masks)
        probs = self.activation_function(preds)
        metric = self.metric(preds=probs, target=mask_to_labels(masks))
        self.log('val_loss', loss, prog_bar=True, batch_size=images.shape[0])
        self.log('val_metric', metric, prog_bar=True, batch_size=images.shape[0])
        
    def test_step(self, batch, batch_idx):
        images, masks, batch_patch_bboxs = batch
        logits = self(images) # [batch_size, 6, patch_size, patch_size]
        probs = self.activation_function(logits) # [batch_size, 6, patch_size, patch_size]
        metric = self.metric(preds=probs, target=mask_to_labels(masks))
        self.log('test_metric', metric, prog_bar=True, batch_size=images.shape[0])
        
        preds = torch.argmax(probs, dim=1) # [batch_size, patch_size, patch_size]
        # Set current image
        if self.current_image_filename is None:
            self.current_image_filename = batch_patch_bboxs[0][0]
                       
        self.stitch_patches(preds, batch_patch_bboxs, self.current_image_filename, self.current_image)
            
    def stitch_patches(self, preds, batch_patch_bboxs, current_image_filename, current_image): 
        for j, (pred_patch, patch_bbox) in enumerate(zip(preds, batch_patch_bboxs)):
            image_filename, bbox = patch_bbox
            # To 3 channels (needed for viz)
            pred_patch = torch.concat((pred_patch.unsqueeze(-1), pred_patch.unsqueeze(-1), pred_patch.unsqueeze(-1)), axis=-1) # [patch_size, patch_size, 3]
            # To category IDs
            color_coded_patch = color_code_mask2(pred_patch) # [patch_size, patch_size, 3]
            
            if image_filename == current_image_filename:
                # Stitch patch into current image
                current_image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :] = color_coded_patch
            else:
                # A patch from a new image arrived, so the previous image is complete and can be saved.
                current_image_for_viz = current_image.detach().cpu().numpy().astype(np.uint8)
                current_image_for_viz = cv2.cvtColor(current_image_for_viz, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename=os.path.join('temp', current_image_filename), img=current_image) # TODO: fix this dir         
                # Init new current image
                current_image_filename = image_filename
                current_image = torch.zeros(size=(2000, 3000, 3))
                # Add the new patch
                current_image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :] = color_coded_patch
        
    def on_train_epoch_start(self): # NOTE: why did I add this? Maybe for the AerialSampler?
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
        
    
        
        