# Use number of pixels per category per image to find train-val-test splits where the frequencies of
# each category are roughly the same of the entire dataset. Saves good split alternatives as JSON file.
# See notebook `01_data_exploration.ipynb` for the choice of the best split.

import yaml
import random
import json
from multiprocessing import Pool

N_ITERATIONS = 10000000
THRESHOLD = 0.0001
N_PROCESSES = 14
    
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

categories = [str(i) for i in range(0, 6)]

def compute_category_perc_in_set(pixel_counts):
    '''Gets dictionary with pixel count for every image
    key = filename (str), value = pixel counts (dict)    
    '''
    # Sum all pixel counts separately by category for this set
    pixel_counts_set = {category: 0 for category in categories}
    for filename in pixel_counts:
        pixel_counts_i = pixel_counts[filename]
        for category in pixel_counts_i:
            pixel_counts_set[category] += pixel_counts_i[category]
            
    # Turn to percs
    pixel_percs_set = {category: round(pixel_counts_set[category]/(1000*1500*len(pixel_counts)), 4) for category in pixel_counts_set}
    
    return pixel_percs_set

# Load pixel counts
with open('data/category_pixel_counts.json', 'r') as file:
    all_pixel_counts = json.load(file)

original_percs = compute_category_perc_in_set(all_pixel_counts)

def try_split(_):
    ## Get splits for this iteration
    all_filenames_i = random.sample(list(all_pixel_counts.keys()), len(all_pixel_counts))
    train_filenames_i = all_filenames_i[:cfg['train_split']]
    val_filenames_i = all_filenames_i[cfg['train_split']:cfg['train_split']+cfg['val_split']]
    test_filenames_i = all_filenames_i[cfg['train_split']+cfg['val_split']:]
    train_pixel_counts_i = {filename: all_pixel_counts[filename] for filename in train_filenames_i}
    val_pixel_counts_i = {filename: all_pixel_counts[filename] for filename in val_filenames_i}
    test_pixel_counts_i = {filename: all_pixel_counts[filename] for filename in test_filenames_i}
    ## Compute category distribution
    train_percs_i = compute_category_perc_in_set(train_pixel_counts_i)
    val_percs_i = compute_category_perc_in_set(val_pixel_counts_i)
    test_percs_i = compute_category_perc_in_set(test_pixel_counts_i)
    ## Compute differences vs original distribution
    # Train
    train_diffs_i = {}
    for key in train_percs_i:
        delta = round(abs(original_percs[key] - train_percs_i[key]), 4)
        train_diffs_i[key] = delta
    # Val
    val_diffs_i = {}
    for key in val_percs_i:
        delta = round(abs(original_percs[key] - val_percs_i[key]), 4)
        val_diffs_i[key] = delta
    # Test
    test_diffs_i = {}
    for key in test_percs_i:
        delta = round(abs(original_percs[key] - test_percs_i[key]), 4)
        test_diffs_i[key] = delta
        
    # ## Check v1: check if iteration sets are similar to the overall dataset
    # # Train
    # train_check = [perc <= THRESHOLD for perc in train_diffs_i.values()]
    # if sum(train_check) == len(train_check):
    #     train_passes = True
    # else:
    #     train_passes = False
    # # Val
    # val_check = [perc <= THRESHOLD for perc in val_diffs_i.values()]
    # if sum(val_check) == len(val_check):
    #     val_passes = True
    # else:
    #     val_passes = False

    ## Check v2: check if rare classes (4 and ) have frequency similar to the overall dataset
    # Train
    train_check = [perc <= THRESHOLD for perc in train_diffs_i.values()]
    if sum(train_check[-2:]) == 2:
        train_passes = True
    else:
        train_passes = False
    # Val
    val_check = [perc <= THRESHOLD for perc in val_diffs_i.values()]
    if sum(val_check[-2:]) == 2:
        val_passes = True
    else:
        val_passes = False
    # Test
    test_check = [perc <= THRESHOLD for perc in test_diffs_i.values()]
    if sum(test_check[-2:]) == 2:
        test_passes = True
    else:
        test_passes = False

    # Return infos if it's a good split
    if train_passes and val_passes and test_passes:
        good_split = {
            'train': train_filenames_i, 
            'val': val_filenames_i, 
            'test': test_filenames_i, 
            'train_diffs': train_diffs_i, 
            'val_diffs': val_diffs_i,
            'test_diffs': test_diffs_i,
        }
           
        return good_split
     
with Pool(processes=N_PROCESSES) as pool:
    good_splits = list(pool.map(try_split, list(range(N_ITERATIONS))))
        
# Filter out None values
good_splits = [split for split in good_splits if split is not None]

with open('data/good_splits.json', 'w') as f:
    json.dump(good_splits, f)
