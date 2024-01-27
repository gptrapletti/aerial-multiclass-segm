import os
import hydra
import torch
from omegaconf import OmegaConf
from dotenv import load_dotenv
from src.module import AerialModule
from src.callbacks import instantiate_callbacks

load_dotenv()

@hydra.main(version_base=None, config_path='configs', config_name='config.yaml')
def main(cfg):
              
    print(OmegaConf.to_yaml(cfg))
    
    torch.set_float32_matmul_precision('medium')
    
    print('\nInstantiate datamodule')
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    print('\nLoad model checkpoint')   
    module = AerialModule.load_from_checkpoint(cfg.best_ckp_path, output_path=cfg.output_path) # TODO: output_path should be removed when I'll have a new checkpoint with also the module's output_path parameter saved in it
    module = module.eval()
    
    print('\nInstantiate trainer')    
    logger = hydra.utils.instantiate(cfg.logger)      
    callbacks = instantiate_callbacks(cfg.callbacks)    
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
            
    print('\nInference...')         
    trainer.test(model=module, datamodule=datamodule)

if __name__ == '__main__':
    main()