import torch
from collections import OrderedDict
import random
import cv2

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, image_filepaths, cache_size=10):
        super().__init__()
        self.image_filepaths = image_filepaths
        self.n_patch_per_image = 100
        self.patch_bboxs = self.generate_patch_bboxs()
        self.cache_size = cache_size
        self.image_cache = OrderedDict()

    def get_random_bbox_coords(self, side, max_height, max_width):
        top_left = random.randint(0, max_height - side), random.randint(0, max_width - side)
        bottom_right = top_left[0] + side, top_left[1] + side
        return (top_left, bottom_right)
    
    def generate_patch_bboxs(self):
        patch_bboxs = []
        for i in range(len(self.image_filepaths)):
            for j in range(self.n_patch_per_image):
                filepath = image_filepaths[i]
                bbox = self.get_random_bbox_coords(side=256, max_height=2000, max_width=3000)
                patch_bboxs.append((filepath, bbox))
        
        return patch_bboxs

    def load_and_process_image(self, filepath):
        if filepath in self.image_cache:
            # Return cached image if it's in the cache
            image = self.image_cache[filepath]
            # Move the accessed entry to the end (to show it's recently used)
            self.image_cache.move_to_end(filepath)
        else:
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (3000, 2000))
            
            # Add the loaded image to the cache
            self.image_cache[filepath] = image
            
            # If cache exceeds size limit, remove the least recently used item (first item)
            if len(self.image_cache) > self.cache_size:
                self.image_cache.popitem(last=False)
        
        return image

    def __getitem__(self, idx):
        filepath = self.patch_bboxs[idx][0]
        bbox = self.patch_bboxs[idx][1]
        image = self.load_and_process_image(filepath)
        patch = image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :]
        return patch
    
    def __len__(self):
        return len(self.patch_bboxs)