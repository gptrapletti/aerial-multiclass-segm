import pytorch_lightning as pl
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from src.processing_utils import mask_to_labels, color_code_pred_mask, color_code_gt_mask

class AerialModule(pl.LightningModule):
    def __init__(self, backbone, loss_fn, metric, lr, output_path):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.metric = metric
        self.lr = lr
        self.output_path = output_path
        self.activation_function = torch.nn.Softmax(dim=1)
        self.current_image_filename = None
        self.current_image = np.zeros(shape=(2000, 3000, 6))
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
        logits = self(images) # [B, 6, S, S], S = patch side
        probs = self.activation_function(logits) # [B, 6, S, S]
        metric = self.metric(preds=probs, target=mask_to_labels(masks))
        self.log('test_metric', metric, prog_bar=True, batch_size=images.shape[0])
        
        probs = probs.detach().cpu()
        preds = torch.argmax(probs, dim=1) # [B, S, S], single channel images with category IDs
        preds = torch.nn.functional.one_hot(preds, num_classes=6) # [B, S, S, 6], 6 channels, with 1 for category with higher prediction
                      
        self.stitch_patches_and_save(preds, batch_patch_bboxs)
            
    def stitch_patches_and_save(self, preds, batch_patch_bboxs):
        # Set current image
        if self.current_image_filename is None:
            self.current_image_filename = batch_patch_bboxs[0][0]        
        
        for j, (pred_patch, patch_bbox) in enumerate(zip(preds, batch_patch_bboxs)):
            # pred_patch shape = [S, S, 6]
            image_filename, bbox = patch_bbox
                      
            if image_filename == self.current_image_filename:
                # Stitch patch into current image
                self.current_image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :] = pred_patch
            else:
                # A patch from a new image arrived, so the previous image is complete and can be saved.
                current_image_for_viz = color_code_pred_mask(self.current_image)
                self.save_comparison_plot(filename=self.current_image_filename, current_image=current_image_for_viz)
                current_image_for_viz = cv2.cvtColor(current_image_for_viz, cv2.COLOR_RGB2BGR)
                predicted_masks_output_dirpath = os.path.join(self.output_path, 'inference', 'predicted_masks')
                if not os.path.exists(predicted_masks_output_dirpath):
                    os.makedirs(predicted_masks_output_dirpath)
                output_filepath = os.path.join(predicted_masks_output_dirpath, self.current_image_filename + '.jpg')
                cv2.imwrite(filename=output_filepath, img=current_image_for_viz)                     
                # Init new current image
                self.current_image_filename = image_filename
                self.current_image = np.zeros(shape=(2000, 3000, 6))
                # Add the new patch
                self.current_image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :] = pred_patch
                
    def save_comparison_plot(self, filename, current_image):
        image_path = os.path.join(f"{os.environ['HOME']}/ds/aerial-multiclass-segm/data/images", filename + '.jpg')
        gt_mask_path = os.path.join(f"{os.environ['HOME']}/ds/aerial-multiclass-segm/data/masks", filename + '.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(gt_mask_path)[..., 0]
        gt_mask = color_code_gt_mask(gt_mask)
        
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 16))

        axes[0].imshow(image)
        axes[0].set_title(f"image {filename}", fontsize=22)
        axes[0].axis('off')

        axes[1].imshow(gt_mask)
        axes[1].set_title("GT mask", fontsize=22)
        axes[1].axis('off')

        axes[2].imshow(current_image)
        axes[2].set_title("PRED mask", fontsize=22)
        axes[2].axis('off')

        plt.tight_layout()
        output_dirpath = os.path.join(self.output_path, 'inference', 'comparison_plots')
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)
        plt.savefig(os.path.join(output_dirpath, filename + '.jpg'))
        plt.close()     
                
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
        
    
        
        