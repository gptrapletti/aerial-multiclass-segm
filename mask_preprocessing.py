# Load the masks as PNGs, turn them to semantic masks with a single channel with category IDs as values, save them to disk.

import os
import numpy as np
from src.utils import from_png_to_semantic_mask
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
    
def process_file(filepath):
    filename = os.path.basename(filepath).split('.')[0]
    mask = cv2.imread(filepath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # required for conversion to color to category label
    mask = cv2.resize(mask, (3000, 2000), interpolation=cv2.INTER_NEAREST)
    semantic_mask = from_png_to_semantic_mask(mask) 
    semantic_mask = semantic_mask.astype(np.uint8)
    cv2.imwrite(os.path.join(cfg['masks_dir'], filename) + '.png', semantic_mask)

with Pool(processes=cfg['n_workers']) as pool: 
    semantic_masks = list(tqdm(pool.imap(process_file, original_mask_filepaths), total=len(original_mask_filepaths)))