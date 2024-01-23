import os
import torch
from typing import List, Optional
import numpy as np
import yaml
from PIL import Image
import albumentations as A
from src.processing_utils import generate_random_non_overlapping_bboxs, get_grid_bboxs, mask_to_one_hot

class AerialDataset(torch.utils.data.Dataset):
    '''Generic dataset class'''
    def __init__(
        self,
        image_filepaths: List[str],
        mask_filepaths: List[str],
        transforms: Optional[A.Compose] = None
    ):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.mask_filepaths = mask_filepaths
        self.transforms = transforms
        
    def __getitem__(self, idx):
        '''Use generated bboxs to load the corresponding image and mask and 
        return a single patch. Leverages PIL's lazy loading to fast loading.
        '''
        filename = os.path.splitext(self.patch_bboxs[idx][0])[0] # without extension
        bbox = self.patch_bboxs[idx][1]
        # Get image patch
        with Image.open(os.path.join(self.images_dir, filename + '.jpg')) as image:
            image_patch = image.crop((bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])) # PIL works in (width, height)
            image_patch = np.array(image_patch)
        # Get mask patch
        with Image.open(os.path.join(self.masks_dir, filename + '.png')) as mask:
            mask_patch = mask.crop((bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]))
            mask_patch = np.array(mask_patch)
        
        needs_standardization = True
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image_patch, mask=mask_patch)
            image_patch = transformed['image']
            mask_patch = transformed['mask']
            
            if any(isinstance(t, A.Normalize) for t in self.transforms.transforms):
                needs_standardization = False
        
        # Standardization
        if needs_standardization:
            image_patch = image_patch / 255.0               
        
        mask_patch = mask_to_one_hot(mask=mask_patch, n_classes=6)
                
        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1) # to shape [C, H, W]
        mask_patch = torch.from_numpy(mask_patch).permute(2, 0, 1)
        
        image_patch = image_patch.type(torch.float32)
        mask_patch = mask_patch.type(torch.float32) # or torch.uint8?
        
        return image_patch, mask_patch, (filename, bbox)
    
    
class TrainingDataset(AerialDataset):
    '''Dataset class for training, with random non-overlapping patches, lazy
    loading via PIL.
    '''
    def __init__(
        self, 
        image_filepaths: List[str],
        mask_filepaths: List[str],
        n_random_patches_per_image: int,
        patch_size: int,
        transforms: Optional[A.Compose] = None
    ):
        super().__init__(
            image_filepaths = image_filepaths,
            mask_filepaths = mask_filepaths,
            transforms = transforms,
        )
        self.n_random_patches_per_image = n_random_patches_per_image
        self.patch_size = patch_size        
        self.patch_bboxs = self.generate_patch_bboxs()
        self.images_dir = os.path.dirname(image_filepaths[0])
        self.masks_dir = os.path.dirname(mask_filepaths[0])
   
    def generate_patch_bboxs(self):
        '''For each image and its mask, generates random patches bounding box coordinates.
        '''
        patch_bboxs = []
        for i in range(len(self.image_filepaths)):
            filename = os.path.basename(self.image_filepaths[i])            
            # Generates bboxs for image i
            bboxs = generate_random_non_overlapping_bboxs(
                n_bboxs = self.n_random_patches_per_image,
                side = self.patch_size,
                max_height= 2000,
                max_width = 3000,
                max_iter=5000
            )

            for bbox in bboxs:
                patch_bboxs.append((filename, bbox))        
        
        return patch_bboxs
    
    def reset_patch_bboxs(self):
        '''To generate again the list of bboxs.
        '''
        self.patch_bboxs = self.generate_patch_bboxs()
    
    def __len__(self):
        return len(self.patch_bboxs)


class ValidationDataset(AerialDataset):
    '''Dataset class for validation, where patches are extracted
    in a grid with no overlap.'''
    def __init__(
        self,
        image_filepaths: List[str],
        mask_filepaths: List[str],
        patch_size: int,
        overlap: int,
        transforms: Optional[A.Compose] = None        
    ):
        super().__init__(
            image_filepaths = image_filepaths,
            mask_filepaths = mask_filepaths,
            transforms = transforms,
        )
        self.patch_size = patch_size
        self.overlap = overlap     
        self.grid_bboxs = get_grid_bboxs( # create grid bboxs, the same for all images
            side = self.patch_size, 
            overlap = self.overlap, 
            max_height = 2000,
            max_width = 3000
        )
        self.patch_bboxs = self.generate_grid_patch_bboxs()
        self.images_dir = os.path.dirname(image_filepaths[0])
        self.masks_dir = os.path.dirname(mask_filepaths[0])
    
    def generate_grid_patch_bboxs(self):
        '''As for training, creates a list with bboxs for each image and mask'''
        patch_bboxs = []
        for i in range(len(self.image_filepaths)):
            filename = os.path.basename(self.image_filepaths[i])
            for bbox in self.grid_bboxs:
                patch_bboxs.append((filename, bbox))

        return patch_bboxs
    
    def __len__(self):
        return len(self.patch_bboxs)           
    

if __name__ == '__main__':
    # To test dataloading speed
    
    from tqdm import tqdm
    from sampler import AerialSampler
    from transforms import train_transforms
    
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    image_filepaths = sorted([os.path.join(cfg['images_dir'], filename) for filename in os.listdir(os.path.join(cfg['images_dir']))])
    mask_filepaths = sorted([os.path.join(cfg['masks_dir'], filename) for filename in os.listdir(os.path.join(cfg['masks_dir']))])
        
    train_dataset = TrainingDataset(
        image_filepaths = image_filepaths[:250],
        mask_filepaths = mask_filepaths[:250],
        transforms = train_transforms,
        n_random_patches_per_image = cfg['n_random_patches_per_image'],
        patch_size = cfg['patch_size']
    )
    
    # Just a check
    print(len(train_dataset))    
    print(train_dataset[54][0].shape, train_dataset[54][1].shape)
    print(train_dataset[54][0].dtype, train_dataset[54][1].dtype)
       
    # # Test dataloader speed
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=14, shuffle=True)
    # for batch in tqdm(dataloader):
    #     continue
    
    # # Proof that without the sampler, the same bboxs are used for all the epochs
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=14, shuffle=False)
    # # Epoch 1
    # total1 = 0
    # for batch in tqdm(dataloader):
    #     total1 += batch[0].sum().item()
    # print(total1)
    # # Epoch 2
    # total2 = 0
    # for batch in tqdm(dataloader):
    #     total2 += batch[0].sum().item()  
    # print(total2)
    
    # # # Test speed with sampler
    # sampler = AerialSampler(dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32, num_workers=12)
    # for batch in tqdm(dataloader):
    #     continue
    
    # # Proof that with the sampler, different bboxs are used for all the epochs
    # sampler = AerialSampler(dataset)
    # dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=32, num_workers=12)
    # # Epoch 1
    # total1 = 0
    # for batch in tqdm(dataloader):
    #     total1 += batch[0].sum().item()
    # print(total1)
    # # Epoch 2
    # total2 = 0
    # for batch in tqdm(dataloader):
    #     total2 += batch[0].sum().item()  
    # print(total2)
    
    val_dataset = ValidationDataset(
        image_filepaths = image_filepaths[250:350],
        mask_filepaths = mask_filepaths[250:350],
        patch_size = cfg['patch_size'],
        overlap = 0
    )
    
    # Just a check
    print(len(val_dataset))    
    print(val_dataset[54][0].shape, val_dataset[54][1].shape)
    print(val_dataset[54][0].dtype, val_dataset[54][1].dtype)
    
    # # Test dataloader speed
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=96, num_workers=14, shuffle=False)
    # for batch in tqdm(val_dataloader):
    #     continue