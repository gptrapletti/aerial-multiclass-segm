import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import json
from typing import Optional, List
import albumentations as A
from .dataset import TrainingDataset, ValidationDataset
from .sampler import AerialSampler

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
        n_random_patches_per_image: int,
        patch_size: int,
        overlap: float,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        train_tranforms: Optional[A.Compose] = None,
        val_transforms: Optional[A.Compose] = None
        
    ):
        super().__init__()
        self.data_path = data_path
        with open(split_path, 'r') as file:
            self.split = json.load(file)
        self.n_random_patches_per_image = n_random_patches_per_image
        self.patch_size = patch_size
        self.overlap = overlap
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_transforms = train_tranforms
        self.val_transforms = val_transforms
        
    def prepare_data(self):
        self.image_filepaths = sorted([os.path.join(self.data_path, 'images', filename) for filename in os.listdir(os.path.join(self.data_path, 'images'))])[:10] ### ! remove
        self.mask_filepaths = sorted([os.path.join(self.data_path, 'masks', filename) for filename in os.listdir(os.path.join(self.data_path, 'masks'))])[:10] ### ! remove
        self.train_idxs = [int(x) for x in self.split['train']]
        self.val_idxs = [int(x) for x in self.split['val']]
        self.test_idxs = [int(x) for x in self.split['test']]
        
    def setup(self, stage):
        if stage == 'fit':
            # Train
            train_image_filepaths = [self.image_filepaths[i] for i in range(len(self.image_filepaths)) if i+1 in self.train_idxs] # +1 because filenames start from 001, not from 000
            train_mask_filepaths = [self.mask_filepaths[i] for i in range(len(self.mask_filepaths)) if i+1 in self.train_idxs]
            self.train_dataset = TrainingDataset(
                image_filepaths = train_image_filepaths,
                mask_filepaths = train_mask_filepaths,
                n_random_patches_per_image = self.n_random_patches_per_image,
                patch_size = self.patch_size,
                transforms = self.train_transforms
            )   
            # Val
            val_image_filepaths = [self.image_filepaths[i] for i in range(len(self.image_filepaths)) if i+1 in self.val_idxs]
            val_mask_filepaths = [self.mask_filepaths[i] for i in range(len(self.mask_filepaths)) if i+1 in self.val_idxs]
            self.val_dataset = ValidationDataset(
                image_filepaths = val_image_filepaths,
                mask_filepaths = val_mask_filepaths,
                patch_size = self.patch_size,
                overlap = self.overlap,
                transforms = self.val_transforms
            )
            
        if stage == 'test':
            test_image_filepaths = [self.image_filepaths[i] for i in range(len(self.image_filepaths)) if i+1 in self.test_idxs]
            test_mask_filepaths = [self.mask_filepaths[i] for i in range(len(self.mask_filepaths)) if i+1 in self.test_idxs]
            pass           
    
    def train_dataloader(self):
        sampler = AerialSampler(self.train_dataset)
        return DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size, shuffle=False, sampler=sampler, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        pass
    
    
if __name__ == '__main__':
    import yaml
    
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    dm = AerialDataModule(
        data_path=cfg['data_path'], 
        split_path=cfg['split_path'],
        n_random_patches_per_image=cfg['n_random_patches_per_image'],
        patch_size=cfg['patch_size'],
        train_batch_size=cfg['train_batch_size'],
        val_batch_size=cfg['val_batch_size'],
        num_workers=cfg['num_workers'],
        
    )
    dm.prepare_data()
    print(len(dm.train_image_filepaths), len(dm.val_image_filepaths), len(dm.test_image_filepaths))