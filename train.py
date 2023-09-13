import yaml
import torch
import pytorch_lightning as pl
import mlflow
import datetime
import shutil
import os

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

try:
    experiment_id = mlflow.create_experiment(
        name = cfg['experiment_name'],
        # artifact_location = 'mlruns',
    )
except:
    experiment_id = mlflow.get_experiment_by_name(cfg['experiment_name']).experiment_id


now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

mlflow.start_run(
    experiment_id = experiment_id,
    run_name = f"{now}_{cfg['run_description']}"
  )

run_id = mlflow.active_run().info.run_id

print()
print(f'Experiment name: {cfg["experiment_name"]}')
print(f'Experiment ID: {experiment_id}')
print(f'Run name: {now}')
print(f'Run ID: {run_id}')
print()

# logger = get_logger(experiment_name=cfg['experiment_name'], run_name=cfg['run_name'])
logger = get_logger(experiment_name=cfg['experiment_name'], run_id=run_id)

trainer = pl.Trainer(
    max_epochs = cfg['epochs'], 
    accelerator = 'gpu',
    callbacks = get_callbacks(os.path.join('mlruns', experiment_id, run_id, 'checkpoints')),
    logger = logger
)

trainer.fit(model=model, datamodule=datamodule)

mlflow.log_artifact('config.yaml')

mlflow.end_run()

