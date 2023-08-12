## Sources
- Dataset website: https://www.tugraz.at/index.php?id=22387
- Repo for similar dataset: https://github.com/aqbewtra/Multi-Class-Aerial-Segmentation/tree/main

## General
- Masks when loaded should be changed to match the new categories, that is, turn the 3 channels mask into a single channel mask with progressive category IDs or into a one-hot-encoded tensor.
    - Another option would be to do this as pre-processing and save those arrays to disk.
- Crop 512x512 patches, then resize to 256x256 before training.
    - Ideally all at runtime (https://towardsdatascience.com/slicing-images-into-overlapping-patches-at-runtime-911fa38618d7)
- Extract patches with overlap (25% or 50%), for training, val and test. For val and test evaluate without addressing the overlap via averaging.
- Extract patches in a random fashion for training, while in a grid fashion for validation and testing.
- On the test set, we want to see a complete prediction on each image. So overlapping predictions are averaged to create a full mask (3000x4000 px) and
a whole-image metric can also be computed.