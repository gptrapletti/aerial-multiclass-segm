import numpy as np
from typing import List, Tuple
import random

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
            
    return mask[..., 0] # the 3 color channels are now the same, so one is enough


def get_random_bbox_coords(side, max_height, max_width):
    top_left = random.randint(0, max_height - side), random.randint(0, max_width - side)
    bottom_right = top_left[0] + side, top_left[1] + side
    return (top_left, bottom_right)


def get_grid_bboxs(side: int, overlap: float, max_height: int, max_width: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    '''
    Gets bounding box coordinates for patches, following a grid fashion.
    
    Args:
        side: patch dimension.
        overlap: percentage of overlap between patches.
        max_height: maximum height of the image.
        max_width: maximum width of the image.
        
    
    '''
    stride = side - int(side*overlap)
    n_bboxs_along_height = 1 + ((max_height - side) // stride)
    n_bboxs_along_width = 1 + ((max_width - side) // stride)

    bboxs = []
    bbox_reference = ((0, 0), (256, 256))
    bboxs.append(bbox_reference)

    for i in range(n_bboxs_along_height):
        
        for j in range(n_bboxs_along_width -1):
            top_left = (bboxs[-1][0][0], bboxs[-1][0][1] + stride)
            bottom_right = (bboxs[-1][1][0], bboxs[-1][1][1] + stride)
            new_bbox = (top_left, bottom_right)
            bboxs.append(new_bbox)
            
        last_bbox = ( (bboxs[-1][0][0], max_width - side), (bboxs[-1][1][0], max_width) )
        bboxs.append(last_bbox)
        
        bbox_reference = ( (bbox_reference[0][0]+stride, bbox_reference[0][1]), (bbox_reference[1][0]+stride, bbox_reference[1][1]) )
        bboxs.append(bbox_reference)

    # Remove last boox for it is out of bounds.
    bboxs = bboxs[:-1] 

    # Do last row (from the end of the height)
    bbox_start = ( (max_height - side, 0), (max_height, side) )
    bboxs.append(bbox_start)

    for j in range(n_bboxs_along_width -1):
        top_left = (bboxs[-1][0][0], bboxs[-1][0][1] + stride)
        bottom_right = (bboxs[-1][1][0], bboxs[-1][1][1] + stride)
        new_bbox = (top_left, bottom_right)
        bboxs.append(new_bbox)

    last_bbox = ( (bboxs[-1][0][0], max_width - side), (bboxs[-1][1][0], max_width) )
    bboxs.append(last_bbox)
    
    return bboxs




if __name__ == '__main__':
    
    import os
    import cv2
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    
    mask_filepaths = sorted([os.path.join('data/masks', filename) for filename in os.listdir(os.path.join('data/masks'))]) ### ! remove the [:10]
    
    def process_file(filepath):
        mask = cv2.imread(filepath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        semantic_mask = from_png_to_semantic_mask(mask)
        return semantic_mask

    with Pool(processes=cpu_count()) as pool:  # use all available CPU cores
        semantic_masks = list(tqdm(pool.imap(process_file, mask_filepaths), total=len(mask_filepaths)))
        
        
        


    

    