# Aerial Semantic Segmentation

![drone_cover_image](data/aerial_drone.jpg)

## Introduction
This deep learning project aims to segment aerial images captured by drones. The focus is on multiclass segmentation to categorize different features in these high-resolution images.

## Repository structure
Below are outlined the contents of the repository.
- `configs`: Hydra configuration files, used for setting up and managing configurations in a structured manner, facilitating the customization and adjustment of various parameters during the model training​.
- `data`: aerial drone photography images and their segmentation masks.
- `notebooks`: notebooks for data exploration and some augmentations tests.
- `src`: source code for training.
- `train.py`: training script.
- some preprocessing and utility scripts.

## Data
The dataset consists of 400 aerial photographs of urban areas from drone, featuring people, buildings, cars, streets, trees, vegetation, lakes, etc. The images come with corresponding semantic segmentation masks.

The dataset can be found here: https://www.tugraz.at/index.php?id=22387


