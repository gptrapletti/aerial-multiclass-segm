import os
import hydra
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from src.callbacks import instantiate_callbacks
from src.utils import tune_lr

load_dotenv()

@hydra.main(version_base=None, config_path='configs', config_name='config.yaml')
def main(cfg):
              
    print(OmegaConf.to_yaml(cfg))
    
    torch.set_float32_matmul_precision('medium')
    
    print('\nInstantiate datamodule')
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    print('\nInstantiate model')   
    module = hydra.utils.instantiate(cfg.module)
    
    print('\nInstantiate trainer')    
    logger = hydra.utils.instantiate(cfg.logger)      
    callbacks = instantiate_callbacks(cfg.callbacks)    
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
       
    # print('\nFinding best learning rate...')
    # tune_lr(trainer=trainer, model=module, datamodule=datamodule)
    # # Re-init trainer, otherwise it still operates as a tuner
    # trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
       
    print('\nTraining...')         
    trainer.fit(model=module, datamodule=datamodule)

if __name__ == '__main__':
    main()
    
    