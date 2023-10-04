import yaml
import torch
import pytorch_lightning as pl
import datetime
import os

from src.datamodule import AerialDataModule
from src.backbone_smp import unet_smp
from src.module import AerialModule
from src.callbacks import get_callbacks

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

now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logger = pl.loggers.MLFlowLogger(
    experiment_name = cfg['experiment_name'], 
    run_name = f"{now}_{cfg['run_description']}"
)

print(f'\nExperiment name: {cfg["experiment_name"]}')
print(f'Experiment ID: {logger.experiment_id}')
print(f'Run name: {logger._run_name}')
print(f'Run ID: {logger.run_id}\n')

callbacks = get_callbacks(ckp_dst_path=os.path.join('mlruns', logger.experiment_id, logger.run_id, 'checkpoints'))

trainer = pl.Trainer(
    max_epochs = cfg['epochs'], 
    accelerator = 'gpu',
    callbacks = callbacks,
    logger = logger
)

trainer.fit(model=model, datamodule=datamodule)

logger.experiment.log_artifact(run_id = logger.run_id, local_path = 'config.yaml')