import numpy as np

def from_png_to_semantic_mask (mask: np.ndarray):
    '''Gets a 3 channels RGB mask array and turns it into a single
    channel semantic segmentation mask with category IDs as
    pixel values   
    '''    

    category_colors = {
        'background': [(112, 150, 146),
        (2, 135, 115),
        (9, 143, 150),
        (0, 0, 0),
        (119, 11, 32),
        (102, 51, 0),
        (255, 0, 0),
        (190, 153, 153),
        (0, 50, 89),
        (153, 153, 153)],
        'ground': [(128, 64, 128), (112, 103, 87), (130, 76, 0), (48, 41, 30)],
        'vegetation': [(0, 102, 0), (107, 142, 35), (51, 51, 0), (190, 250, 190)],
        'buildings': [(70, 70, 70), (102, 102, 156), (254, 228, 12), (254, 148, 12)],
        'water': [(28, 42, 168)],
        'person': [(255, 22, 96)]
    }


    category_ids = {
        'background': 0,
        'ground': 1,
        'vegetation': 2,
        'buildings': 3,
        'water': 4,
        'person': 5
    }

    for category in category_colors:
        for color in category_colors[category]:
            color_is_present = np.all(mask == color, axis=2)
            mask[color_is_present] = category_ids[category]
            
    return mask[0] # the 3 color channels are now the same, so one is enough






if __name__ == '__main__':
    
    import os
    import cv2
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    
    mask_filepaths = sorted([os.path.join('data/masks', filename) for filename in os.listdir(os.path.join('data/masks'))])
    
    def process_file(filepath):
        mask = cv2.imread(filepath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        semantic_mask = from_png_to_semantic_mask(mask)
        return semantic_mask

    with Pool(processes=cpu_count()) as pool:  # Use all available CPU cores
        semantic_masks = list(tqdm(pool.imap(process_file, mask_filepaths), total=len(mask_filepaths)))
   

    