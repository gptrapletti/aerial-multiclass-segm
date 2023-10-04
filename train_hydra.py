import os
import hydra
import torch
from dotenv import load_dotenv
from src.callbacks import get_callbacks

load_dotenv()

@hydra.main(version_base=None, config_path='configs', config_name='config.yaml')
def main(cfg):
    
    for k, v in cfg.items():
        print(f'{k}\n\t{v}\n\n')
        
    torch.set_float32_matmul_precision('medium')
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    module = hydra.utils.instantiate(cfg.module)
    
    logger = hydra.utils.instantiate(cfg.logger)
    
    callbacks = get_callbacks(ckp_dst_path=os.path.join('mlruns', logger.experiment_id, logger.run_id, 'checkpoints'))
    
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    trainer.fit(model=module, datamodule=datamodule)

if __name__ == '__main__':
    main()