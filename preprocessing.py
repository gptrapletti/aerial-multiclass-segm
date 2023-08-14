# Load the masks as PNGs, turn them to semantic masks with a single channel with category IDs as values, save them to disk.

import os
import numpy as np
from utils import from_png_to_semantic_mask
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

mask_filepaths = sorted([os.path.join('data/masks_original', filename) for filename in os.listdir(os.path.join('data/masks_original'))])

if not os.path.exists('data/masks'):
    os.makedirs('data/masks')
    
def process_file(filepath):
    filename = os.path.basename(filepath).split('.')[0]
    mask = cv2.imread(filepath)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (3000, 2000), interpolation=cv2.INTER_NEAREST)
    semantic_mask = from_png_to_semantic_mask(mask)
    semantic_mask = semantic_mask.astype(np.uint8)
    np.save(file=f'data/masks/{filename}.npy', arr=semantic_mask)

with Pool(processes=cpu_count()) as pool: 
    semantic_masks = list(tqdm(pool.imap(process_file, mask_filepaths), total=len(mask_filepaths)))


# # To save a unique array
# def process_file(filepath):
#     filename = os.path.basename(filepath).split('.')[0]
#     mask = cv2.imread(filepath)
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#     mask = cv2.resize(mask, (3000, 2000), interpolation=cv2.INTER_NEAREST)
#     semantic_mask = from_png_to_semantic_mask(mask)
#     semantic_mask = semantic_mask.astype(np.uint8)
#     return semantic_mask


# with Pool(processes=cpu_count()) as pool: 
#     semantic_masks_ls = list(tqdm(pool.imap(process_file, mask_filepaths), total=len(mask_filepaths)))
#     semantic_masks_arr = np.stack(semantic_masks_ls, axis=0)
#     np.save(file=f'data/masks.npy', arr=semantic_masks_arr)
    