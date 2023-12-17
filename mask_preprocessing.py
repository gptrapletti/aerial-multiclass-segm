# Load the masks as PNGs, turn them to semantic masks with a single channel with category IDs as values, save them to disk.

import os
import numpy as np
from src.processing_utils import from_png_to_semantic_mask, create_filename
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import yaml
from PIL import Image
import skimage

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

original_mask_filepaths = sorted([os.path.join('data/masks_original', filename) for filename in os.listdir(os.path.join('data/masks_original'))])

if not os.path.exists(cfg['masks_dir']):
    os.makedirs(cfg['masks_dir'])

def process_file(idx_filepath_tuple):
    idx, filepath = idx_filepath_tuple
    mask = cv2.imread(filepath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # required for conversion to color to category label
    mask = cv2.resize(mask, (3000, 2000), interpolation=cv2.INTER_NEAREST)
    semantic_mask = from_png_to_semantic_mask(mask) 
    semantic_mask = semantic_mask.astype(np.uint8)
    filename = create_filename(idx, extension='.png')
    cv2.imwrite(os.path.join(cfg['masks_dir'], filename), semantic_mask)
    
original_mask_filepaths_with_idxs = [(i+1, filepath) for i, filepath in enumerate(original_mask_filepaths)]

with Pool(processes=cfg['num_workers']) as pool: 
    semantic_masks = list(tqdm(pool.imap(process_file, original_mask_filepaths_with_idxs), total=len(original_mask_filepaths_with_idxs)))