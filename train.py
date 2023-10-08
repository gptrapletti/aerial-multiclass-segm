import os
import hydra
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from src.callbacks import instantiate_callbacks

load_dotenv()

@hydra.main(version_base=None, config_path='configs', config_name='config.yaml')
def main(cfg):
              
    print(OmegaConf.to_yaml(cfg))
    
    torch.set_float32_matmul_precision('medium')
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    module = hydra.utils.instantiate(cfg.module)
    
    logger = hydra.utils.instantiate(cfg.logger)
      
    callbacks = instantiate_callbacks(cfg.callbacks)
    
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    trainer.fit(model=module, datamodule=datamodule)

if __name__ == '__main__':
    main()
    
    