# Aerial Semantic Segmentation

![drone_cover_image](data/aerial_drone.jpg)

## Introduction
This deep learning project aims to segment aerial images captured by drones. The focus is on multiclass segmentation to categorize different features in these high-resolution images. The projects is based on PyTorch Lightning to streamline the workflow, and Hydra to efficiently manage configurations and experiments.

## Repository structure
Below are outlined the contents of the repository.
- `configs`: Hydra configuration files, used for setting up and managing configurations in a structured manner, facilitating the customization and adjustment of various parameters during the model training​.
- `data`: aerial drone photography images and their segmentation masks.
- `notebooks`: notebooks for data exploration and some augmentations tests.
- `src`: source code for training.
- `train.py`: training script.
- some preprocessing and utility scripts.

## Data
The dataset consists of 400 aerial drone photographs of urban areas, featuring people, buildings, cars, streets, trees, vegetation, ponds, etc. The images come with corresponding semantic segmentation masks. The dataset can be found here: https://www.tugraz.at/index.php?id=22387

## Pre-processing
The images undergo preprocessing to reduce their size, optimizing disk space usage while ensuring they remain sufficiently detailed to pose a challenge. The original 24 categories are mapped into 6 broader groups (for instance, 'roof', 'wall', 'window', and 'door' are combined under 'buildings'). This restructuring not only preserves but accentuates class imbalances, aiming to train a model that is particularly robust in handling such disparities.

Masks are pre-processed for efficient data loading by converting them from 3-channel color-encoded images to single-channel images with label IDs. The `multiprocessing` library is utilized to expedite this processing. Alternatively, using a dataset that returns the processed semantic mask in conjunction with a dataloader configured with the `num_workers` argument can achieve comparable computation times.

## Patch extraction
Images are divided into patches for model input, a process that occurs during runtime. Specifically, in the data loading phase, each image is read from disk, and a set of bounding box coordinates for the patches is determined. Subsequently, the dataset returns these individual patches, taking advantage of PIL's lazy loading feature to expedite the cropping process. This method enhances the efficiency of data handling and preprocessing.

In detail, the dataset object, during training, compiles a list containing the file paths of images and the bounding box coordinates for each patch. Utilizing this information, the dataset loads and crops the image to provide the desired patch. The training dataloader then delivers batches of patches in the shape `[n_patches, 3, patch_size, patch_size]`, where `n_patches` is determined based on the GPU memory capacity.
For validation and testing, patches are systematically extracted in a grid pattern, resulting in a consistent number of patches per image. In the validation phase, patches are set with zero overlap, whereas in testing, patches may overlap (by 25% or 50%). An averaging method can be applied to these overlapping patches to obtain the final prediction, allowing for a complete and integrated view of each image.

## Model architecture
The model utilizes a U-Net architecture from the `segmentation_models_pytorch` library, featuring a ResNet34 encoder and a decoder, for precise pixel-level classification required in a multiclass semantic segmentation task.

## Training
Various training experiments were conducted, initially focusing on evaluating different loss functions—specifically CrossEntropy and Dice—over a fixed number of epochs. The training process and its outcomes are meticulously logged using MLFlow for comprehensive tracking and analysis. These experiments were performed using an NVIDIA GeForce RTX 3050 GPU.

## To-Do List

- [ ] Beautify test_step (remove TODOs, etc.), implement saving plots for comparison prediction vs GT (maybe a PDF?).
- [ ] Try inference on patches with best model.
- [ ] Implement train and val metric as unique Dice for all classes; for test: return Dice per class + Dice for all classes.
- [ ] Write test dataset class.
- [ ] Improve out losses weights are sent to GPU (internally, see loss classes).
- [ ] If segmentation results are not good along the edges, implement a loss that penalizes when the model is approximative there.
- [ ] Write custom architecture. Can add attention mechanisms too.
- [ ] Consider whether to weight more the "other" class in the loss computation, since it encompasses a large variety of different kinds of objects.


