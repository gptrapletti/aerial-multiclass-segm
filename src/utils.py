from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.exceptions import _TunerExitException


def tune_lr(trainer, model, datamodule, iter=100):
    '''Finds the best initial learning rate and sets it in the LightningModule object.'''
    tuner = Tuner(trainer)
    # After having found the best lr, it always throws this error (it may be a bug)
    try:
        tuner.lr_find(model=model, datamodule=datamodule, num_training=iter)
    except _TunerExitException:
        pass
    
    # # The best lr is set automatically into the module object
    # print(f'Tuned learning rate: {model.lr}')