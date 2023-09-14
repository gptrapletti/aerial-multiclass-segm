import os
import pytorch_lightning as pl
from typing import List

def get_callbacks(ckp_dst_path: str) -> List[pl.Callback]:
    '''Returns a list of callbacks.

    Args:
        ckp_dst_path: path to the directory where checkpoints will be saved.

    Returns:
        List of callbacks.
    '''
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath = ckp_dst_path,
            filename = '{epoch}',
            monitor = 'val_loss',
            mode = 'min',
            save_top_k = 1,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    ]
    
    return callbacks