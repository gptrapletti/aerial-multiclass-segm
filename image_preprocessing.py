# To resize images to 2000x3000 and save to disk

import os
import cv2
import yaml
from tqdm import tqdm

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

image_filepaths = sorted([os.path.join('data/images_4000_6000', filename) for filename in os.listdir(os.path.join('data/images_4000_6000'))])

for i in tqdm(range(len(image_filepaths))):
    filepath = image_filepaths[i]
    filename = os.path.splitext(os.path.basename(filepath))[0] # without extension
    image = cv2.imread(filepath)
    image = cv2.resize(image, (3000, 2000))
    dst_filepath = os.path.join(cfg['images_dir'], filename + '.jpg')
    cv2.imwrite(dst_filepath, image)
  

# # # # Extract grid patches and save to disk
# bboxs = get_grid_bboxs(
#     side = cfg['patch_size'],
#     overlap = cfg['patch_overlap'],
#     max_height = 2000, # because image are going to be resized
#     max_width = 3000
# )
#
# for i in tqdm(range(len(image_filepaths))):
#     filepath = image_filepaths[i]
#     filename = os.path.splitext(os.path.basename(filepath))[0] # without extension
#     image = cv2.imread(filepath)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (3000, 2000))
#     for bbox in bboxs:
#         patch = image[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :]
#         patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
#         coords = str(bbox).replace('(', '').replace(')', '').replace(',', '').replace(' ', '_') # ((1, 2), (3, 4)) -> 1_2_3_4
#         patch_name = filename + '_' + coords + '.png'
#         patch_filepath = os.path.join(cfg['patches_dir'], patch_name)
#         cv2.imwrite(patch_filepath, patch)

