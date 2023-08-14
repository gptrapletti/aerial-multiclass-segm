## Sources
- Dataset website: https://www.tugraz.at/index.php?id=22387
- Repo for similar dataset: https://github.com/aqbewtra/Multi-Class-Aerial-Segmentation/tree/main

## General
- Masks when loaded should be changed to match the new categories, that is, turn the 3 channels mask into a single channel mask with progressive category IDs or into a one-hot-encoded tensor.
    - Another option would be to do this as pre-processing and save those arrays to disk.
- Crop 512x512 patches, then resize to 256x256 before training.
    - Ideally all at runtime (https://towardsdatascience.com/slicing-images-into-overlapping-patches-at-runtime-911fa38618d7)
- Extract patches in a random fashion for training, while in a grid fashion for validation and testing. Extract patches with overlap (25% or 50%) for val and test, but without addressing the overlap via averaging.
- Later, after trainin and testing, on the test set, we want to see a complete prediction on each image. So overlapping predictions are averaged to create a full mask (3000x4000 px) and a whole-image metric can also be computed.

## Patch extraction
- Method 1:
    - For trainin, in the dataset create a list with image filepath and bounding box coordinates for each patch. Them the `__getitem__` will use those infos to load the image and crop the patch, returning it as the item.
    - The train Dataloader will return batches of patches with shape `[N_PATCHES, 3, 256, 256]`, where `N_PATCHES` is an adequate number of patches in terms of GPU memory.
    - During validation and testing, the patches are extracted in a grid fashion thus yield a constant number of patches per image, which can be returned as a tensor with shape `[N_PATCHES_VAL, 3, 256, 256]` or can be done something similar to the train dataset.