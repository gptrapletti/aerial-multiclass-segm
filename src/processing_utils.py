import numpy as np
from typing import List, Tuple
import random
from shapely import Polygon
import torch

def create_filename(idx, extension='.jpg'):
    n_zeroes = 3 - len(str(idx))
    filename = '0' * n_zeroes + str(idx) + extension
    return filename

def from_png_to_semantic_mask (mask: np.ndarray):
    '''Gets a 3 channels RGB mask array and turns it into a single
    channel semantic segmentation mask with category IDs as
    pixel values   
    '''    
    ## First version
    # category_colors = {
    #     'other': [(112, 150, 146),
    #     (2, 135, 115),
    #     (9, 143, 150),
    #     (0, 0, 0),
    #     (119, 11, 32),
    #     (102, 51, 0),
    #     (255, 0, 0),
    #     (190, 153, 153),
    #     (0, 50, 89),
    #     (153, 153, 153)],
    #     'ground': [(128, 64, 128), (112, 103, 87), (130, 76, 0), (48, 41, 30)],
    #     'vegetation': [(0, 102, 0), (107, 142, 35), (51, 51, 0), (190, 250, 190)],
    #     'buildings': [(70, 70, 70), (102, 102, 156), (254, 228, 12), (254, 148, 12)],
    #     'water': [(28, 42, 168)],
    #     'person': [(255, 22, 96)]
    # }
    
    # Second version
    category_colors = {
        'other': [(112, 150, 146), (2, 135, 115), (9, 143, 150), (0, 0, 0), (119, 11, 32), (102, 51, 0), (255, 0, 0)],
        'ground': [(128, 64, 128), (112, 103, 87), (130, 76, 0), (48, 41, 30)],
        'vegetation': [(0, 102, 0), (107, 142, 35), (51, 51, 0), (190, 250, 190)],
        'buildings': [(70, 70, 70), (102, 102, 156), (254, 228, 12), (254, 148, 12), (0, 50, 89), (190, 153, 153), (153, 153, 153)],
        'water': [(28, 42, 168)],
        'person': [(255, 22, 96)]
    }

    category_ids = {
        'other': 0,
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


def get_random_bbox(side, max_height, max_width):
    '''Gets size of the bounding box and source image properties and returns
    the coordinates of the bounding box of a patch.
    '''
    top_left = random.randint(0, max_height - side), random.randint(0, max_width - side)
    bottom_right = top_left[0] + side, top_left[1] + side
    return (top_left, bottom_right)


def shapely_friendly_bbox(bbox):
    '''Convertes a bounding box from format ((A, B), (C, D)) to format
    required by Shapely to create a polygon out of it. 
    '''
    return (bbox[0], (bbox[0][0], bbox[1][1]), bbox[1], (bbox[1][0], bbox[0][1]))


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
    bbox_reference = ((0, 0), (side, side))
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


def generate_random_non_overlapping_bboxs(n_bboxs, side, max_height, max_width, max_iter=100):
    '''Generates random bounding boxes with no overlap.
    
    Args:
        n_bboxs: number of bounding boxes to generate.
        side: bounding box dimension.
        max_height: maximum height of the image.
        max_width: maximum width of the image.
        
    Returns:
        List of bounding boxes.
    '''
    bboxs = []
    iter_counter = 0 # to avoid it getting stuck searching for a bbox when there is no more space
    while len(bboxs) != n_bboxs and iter_counter <= max_iter:
        iter_counter += 1
        bbox_i = get_random_bbox(side=side, max_height=max_height, max_width=max_width)
        if len(bboxs) == 0:
            bboxs.append(bbox_i)
        else:
            is_overlapping = False
            bbox_i_polygon = Polygon(shapely_friendly_bbox(bbox_i))
            for bbox in bboxs:
                bbox_polygon = Polygon(shapely_friendly_bbox(bbox))
                intersection = bbox_i_polygon.intersects(bbox_polygon)
                if intersection:
                    is_overlapping = True
                    break
            if not is_overlapping:
                bboxs.append(bbox_i)
    
    return bboxs


def mask_to_one_hot(mask: np.ndarray, n_classes: int) -> np.ndarray:
    '''Function to turn a patch mask with indexes (shape=[H, W]) to a
    one-hot encoded patch mask (shape=[H, W, C], where C is the number of classes).
    
    Args:
        mask: patch mask array.
        n_classes: number of classes.
        
    Returns:
        one-hot encoded patch mask.
    '''
    mask_hot = np.zeros(shape=(mask.shape[0], mask.shape[1], n_classes))
    for class_i in range(n_classes):
        mask_hot[..., class_i] = np.where(mask == class_i, 1, 0)
        
    return mask_hot


def mask_to_labels (masks: torch.Tensor) -> torch.Tensor:
    '''Turns mask batch tensor from one-hot encoding and shape [B, C, H, W]
    to mask with category labels with shape [B, 1, H, W].
    
    Args:
        masks: one-hot encoded mask batch tensor.
    
    Returns:
        mask batch tensor with labels
    '''
    index_masks = torch.zeros(masks.shape[0], 1, masks.shape[2], masks.shape[3]).to(masks.device)
    for idx, mask in enumerate(masks):
        for label, channel in enumerate(mask):
            index_masks[idx] += channel * label
            
    return index_masks.type(torch.uint8)


def color_code_mask1(mask):
    '''To turn a mask as a np.array with shape [H, W, 3] with category IDs (1, 2, 3, etc)
    to a color coded mask for visualization.
    
    Args:
        mask (np.array): mask with category IDs.
        
    Return:
        np.array: color coded mask [H, W, 3].
    '''
    color_mapping = {
        0: (0, 0, 0), # other
        1: (125, 125, 125), # ground
        2: (0, 255, 0), # vegetation
        3: (90, 60, 0), # buildings
        4: (0, 0, 255), # water
        5: (255, 0, 0) # people
    }
    
    for color_id in color_mapping:
        filter = np.all(mask == (color_id, color_id, color_id), axis=2)
        mask[filter] = color_mapping[color_id]
        
    return mask


def color_code_mask2(mask):
    '''To turn a mask as a torch tensor with shape [H, W, 3] with category IDs (1, 2, 3, etc)
    to a color coded mask for visualization using PyTorch.

    Args:
        mask (torch.Tensor): mask with category IDs.
        
    Return:
        torch.Tensor: color coded mask [H, W, 3].
    '''
    color_mapping = {
        0: [0, 0, 0], # other
        1: [125, 125, 125], # ground
        2: [0, 255, 0], # vegetation
        3: [90, 60, 0], # buildings
        4: [0, 0, 255], # water
        5: [255, 0, 0] # people
    }
    mask_arr = mask.detach().cpu().numpy()
    colored_mask = np.zeros_like(mask_arr)
    
    for color_id, color in color_mapping.items():
        filter = np.all(mask_arr == (color_id, color_id, color_id), axis=2)
        colored_mask[filter] = color

    return torch.tensor(colored_mask)


def aerial_collate_fn(batch):
    images, masks, metadata = zip(*batch) # batch = [(image1, mask1, metadata1), (image2, mask2, metadata2), ...]
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    
    return images, masks, metadata


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

    with Pool(processes=cpu_count()) as pool:  # use all available CPU cores
        semantic_masks = list(tqdm(pool.imap(process_file, mask_filepaths), total=len(mask_filepaths)))
        
        
        


    

    