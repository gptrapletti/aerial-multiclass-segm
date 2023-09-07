import yaml
import torch
import pytorch_lightning as pl

from src.datamodule import AerialDataModule
from src.backbone_smp import unet_smp
from src.module import AerialModule
from src.callbacks import get_callbacks
from src.logger import get_logger

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
    
torch.set_float32_matmul_precision('medium')
    
datamodule = AerialDataModule(
    data_path = cfg['data_path'], 
    split_path = cfg['split_path'],
    n_random_patches_per_image = cfg['n_random_patches_per_image'],
    patch_size = cfg['patch_size'],
    train_batch_size = cfg['train_batch_size'],
    val_batch_size = cfg['val_batch_size'],
    num_workers = cfg['num_workers'],
    overlap = cfg['overlap'],
)

model = AerialModule(backbone=unet_smp, lr=cfg['lr'])

trainer = pl.Trainer(
    max_epochs = cfg['epochs'], 
    accelerator = 'gpu',
    callbacks = get_callbacks(cfg['run_name']),
    logger = get_logger(experiment_name=cfg['experiment_name'], run_name=cfg['run_name'])
)

trainer.fit(model=model, datamodule=datamodule)

