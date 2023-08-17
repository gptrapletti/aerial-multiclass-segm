import os
import torch
from collections import OrderedDict
import random
import cv2
from typing import List
import numpy as np
import yaml
from PIL import Image
from utils import generate_random_non_overlapping_bboxs

class TrainingDataset1(torch.utils.data.Dataset):
    '''Dataset for training, with random overlapping patches, cache,
    for masks 2000x3000.'''
    def __init__(
        self, 
        image_filepaths: List[str],
        mask_filepaths: List[str],
        n_random_patches_per_image: int,
    ):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.mask_filepaths = mask_filepaths
        self.n_random_patches_per_image = n_random_patches_per_image
        self.patch_bboxs = self.generate_patch_bboxs()
        self.cache = OrderedDict()
        self.images_dir = os.path.dirname(image_filepaths[0])
        self.masks_dir = os.path.dirname(mask_filepaths[0])

    def get_random_bbox(self, side, max_height, max_width):
        top_left = random.randint(0, max_height - side), random.randint(0, max_width - side)
        bottom_right = top_left[0] + side, top_left[1] + side
        return (top_left, bottom_right)
    
    def generate_patch_bboxs(self):
        '''For each image and its mask, generates patches bounding box coordinates.
        '''
        patch_bboxs = []
        for i in range(len(self.image_filepaths)):
            for j in range(self.n_random_patches_per_image):
                filename = os.path.basename(self.image_filepaths[i])
                bbox = self.get_random_bbox(side=256, max_height=2000, max_width=3000)
                patch_bboxs.append((filename, bbox))
        
        return patch_bboxs

    def load_and_process(self, filename):
        '''Loads and returns full image and mask, does some processing.
        If image and mask are not cached, it caches them.
        '''
        if filename in self.cache:
            # Return cached image and mask if they are in the cache
            image = self.cache[filename]['image']
            mask = self.cache[filename]['mask']
            # # Move the accessed entry to the end (to show it's recently used) ### ! probably useless if cache len max set to 1
            # self.cache.move_to_end(filename)
        else:
            # Load image
            image = cv2.imread(os.path.join(self.images_dir, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (3000, 2000))
            # Load mask
            mask = np.load(os.path.join(self.masks_dir, os.path.splitext(filename)[0] + '.npy'), allow_pickle=True)          
            # Add loaded image and mask to the cache
            self.cache[filename] = {'image': image, 'mask': mask}            
            # If cache exceeds size limit, remove the least recently used item (first item)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        
        return image, mask

    def __getitem__(self, idx):
        '''Extract and returns a single image patch and a single mask patch from the cached image
        '''
        filename = self.patch_bboxs[idx][0]
        bbox = self.patch_bboxs[idx][1]
        image, mask = self.load_and_process(filename)
        image_patch = image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :]
        mask_patch = mask[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]
        
        
        
        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).type(torch.float16)
        mask_patch = torch.from_numpy(mask_patch).type(torch.int8)
        
        return image_patch, mask_patch
    
    def __len__(self):
        return len(self.patch_bboxs)
    
    
class TrainingDataset2(torch.utils.data.Dataset):
    '''Dataset class for training, with random non-overlapping patches, lazy
    loading via PIL.
    '''
    def __init__(
        self, 
        image_filepaths: List[str],
        mask_filepaths: List[str],
        n_random_patches_per_image: int,
    ):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.mask_filepaths = mask_filepaths
        self.n_random_patches_per_image = n_random_patches_per_image
        self.patch_bboxs = self.generate_patch_bboxs()
        self.cache = OrderedDict()
        self.images_dir = os.path.dirname(image_filepaths[0])
        self.masks_dir = os.path.dirname(mask_filepaths[0])

    def get_random_bbox(self, side, max_height, max_width):
        top_left = random.randint(0, max_height - side), random.randint(0, max_width - side)
        bottom_right = top_left[0] + side, top_left[1] + side
        return (top_left, bottom_right)
    
    def generate_patch_bboxs(self):
        '''For each image and its mask, generates patches bounding box coordinates.
        '''
        patch_bboxs = []
        for i in range(len(self.image_filepaths)):
            for j in range(self.n_random_patches_per_image):
                filename = os.path.basename(self.image_filepaths[i])
                bbox = self.get_random_bbox(side=512, max_height=4000, max_width=6000)
                patch_bboxs.append((filename, bbox))
        
        return patch_bboxs

    def __getitem__(self, idx):
        '''Extract and returns a single image patch and a single mask patch from the cached image
        '''
        filename = self.patch_bboxs[idx][0]
        bbox = self.patch_bboxs[idx][1]
        # Get image patch
        with Image.open(os.path.join(self.images_dir, filename)) as image:
            image_patch = image.crop((bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])) # PIL works in (width, height)
            image_patch = image_patch.resize(size=(256, 256))
            image_patch = np.array(image_patch)
        # Get mask patch
        with Image.open(os.path.join(self.masks_dir, filename)) as mask:
            mask_patch = mask.crop((bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]))
            mask_patch = mask_patch.resize(size=(256, 256))
            mask_patch = np.array(mask_patch)
                
        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).type(torch.int8)
        mask_patch = torch.from_numpy(mask_patch).type(torch.int8)
        
        return image_patch, mask_patch
    
    def __len__(self):
        return len(self.patch_bboxs)
    

if __name__ == '__main__':
    from tqdm import tqdm
    
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
        
    image_filepaths = sorted([os.path.join(cfg['images_dir'], filename) for filename in os.listdir(os.path.join(cfg['images_dir']))])
    mask_filepaths = sorted([os.path.join(cfg['masks_dir'], filename) for filename in os.listdir(os.path.join(cfg['masks_dir']))])
        
    dataset = TrainingDataset(
        image_filepaths = image_filepaths,
        mask_filepaths =mask_filepaths,
        n_random_patches_per_image = cfg['n_random_patches_per_image']
    )
    
    print(len(dataset))    
    print(dataset[54][0].shape, dataset[54][1].shape)
    print(dataset[54][0].dtype, dataset[54][1].dtype)
    
    # for item in tqdm(dataset):
    #     continue

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=14, shuffle=True)
    
    for batch in tqdm(dataloader):
        continue