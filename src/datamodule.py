import pytorch_lightning as pl
import os
import json

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class AerialDataModule(pl.LightningDataModule):
    '''Datamodule class for aerial multiclass segmentation.
    
    Args:
        data_path: data path, containing dirs "images" and "masks".
        split: dictionary with train, val and test idxs.
        num_workers: number of workers/processes.
    '''
    def __init__(
        self,
        data_path: str,
        split_path: str,
        num_workers: int,
    ):
        super().__init__()
        self.data_path = data_path
        with open(split_path, 'r') as file:
            self.split = json.load(file)
        self.num_workers = num_workers
        
    def prepare_data(self):
        image_filepaths = sorted([os.path.join(self.data_path, 'images', filename) for filename in os.listdir(os.path.join(self.data_path, 'images'))])
        mask_filepaths = sorted([os.path.join(self.data_path, 'masks', filename) for filename in os.listdir(os.path.join(self.data_path, 'masks'))])
        train_idxs = [int(x) for x in self.split['train']]
        val_idxs = [int(x) for x in self.split['val']]
        test_idxs = [int(x) for x in self.split['test']]
        self.train_image_filepaths = [image_filepaths[i] for i in range(len(image_filepaths)) if i+1 in train_idxs] # +1 because filenames start from 001, not from 000
        self.train_mask_filepaths = [mask_filepaths[i] for i in range(len(mask_filepaths)) if i+1 in train_idxs]
        self.val_image_filepaths = [image_filepaths[i] for i in range(len(image_filepaths)) if i+1 in val_idxs]
        self.val_mask_filepaths = [mask_filepaths[i] for i in range(len(mask_filepaths)) if i+1 in val_idxs]
        self.test_image_filepaths = [image_filepaths[i] for i in range(len(image_filepaths)) if i+1 in test_idxs]
        self.test_mask_filepaths = [mask_filepaths[i] for i in range(len(mask_filepaths)) if i+1 in test_idxs]
        
    def setup(self):
        pass
    
    def train_dataloader(self):
        pass
    
    def val_dataloader(self):
        pass
    
    def test_dataloader(self):
        pass
    
    
if __name__ == '__main__':
    import yaml
    
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    dm = AerialDataModule(data_path=cfg['data_path'], split_path=cfg['split_path'], num_workers=cfg['num_workers'])
    dm.prepare_data()
    print(len(dm.train_image_filepaths), len(dm.val_image_filepaths), len(dm.test_image_filepaths))