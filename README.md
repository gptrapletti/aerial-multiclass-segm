## Sources
- Dataset website: https://www.tugraz.at/index.php?id=22387
- Repo for similar dataset: https://github.com/aqbewtra/Multi-Class-Aerial-Segmentation/tree/main

## General
- Drone images, not satellite eheh!
- Images and masks are resized from  4000x6000 pixels to 2000x3000 pixels and saved to disk. This mitigates computation while still keeping the images large enough to be challenging.
- Masks are also remapped to single channel masks with progressive category labels as pixel values and saved to disk.
- Both images and masks are renamed 1-400.
- Extract 256x256 px patches in a random fashion for training, while in a grid fashion for validation and testing. For validation, patch overlap in set to zero, while for testing patches have an overlap (25% or 50%) and averaging can be done to get the final prediction and see a complete prediction on each image. Compute whole-image Dice for the test phase?

## Patch extraction
- -Ideally patches are extracted at runtime (https://towardsdatascience.com/slicing-images-into-overlapping-patches-at-runtime-911fa38618d7).
- Method 1:
    - For trainin, in the dataset create a list with image filepath and bounding box coordinates for each patch. Them the `__getitem__` will use those infos to load the image and crop the patch, returning it as the item.
    - The train Dataloader will return batches of patches with shape `[N_PATCHES, 3, 256, 256]`, where `N_PATCHES` is an adequate number of patches in terms of GPU memory.
    - During validation and testing, the patches are extracted in a grid fashion thus yield a constant number of patches per image, which can be returned as a tensor with shape `[N_PATCHES_VAL, 3, 256, 256]` or can be done something similar to the train dataset.


## Multiprocessing
Process the masks from RGB to semantic masks with category ID values:
- via `multiprocessing` as in `utils.py`, n=400: 13min 31s
- via a dataset that returns the processed semantic mask and a dataloader with `num_workers=16`, n=400: 14m 38s
Conclusion: the two methods have the same speed.


## How to deal with the "other" category.
- Custom Loss Function: Design a custom loss function that effectively ignores the pixels of objects you are not interested in. This can be complex but might allow for more nuanced ontrol of how your model learns from the 'unwanted' objects. For example, the loss for the 'unwanted' objects could be set to zero, so the model is not penalized for incorrectly classifying these objects.

- Use of Weight Maps in Loss Function: To down-weight the contribution of uninterested objects in the loss function, use a weight map that assigns lower weights to pixels of uninterested objects and higher weights to pixels of interested objects. This tells the model that misclassifying the uninterested objects is less penalizing than misclassifying the interested objects.


## Hydra for configs
`config.yaml` contains also classes to use to instantiate objects. In the training script, Hydra is used to read the yaml file and return the cfg dict, that contains the classes to use. Object are instantiated from those classes using `hydra.utils.instantiate()` and then passed to the LightningModule, DataModule, etc (see nb_01.ipynb).

# NOTES TO ADD TO FINAL README
- Use of Hydra to easily plug in and out diffent options (losses, etc.)

# TODO-LONG-TERM
- Write test dataset class as described above.
- Train and val metric: unique Dice for all classes; metric for test: Dice divided by class.

# TODO

- focal loss in practice: why focal loss with gamma=0 doesn't fall back to CE loss???
- Understand how combine losses.
- Find out best LR.
- Test losses
    - test each loss for N epochs.
    - CE, focal, weighted CE, weighted focal, the best of those combined with Dice loss.




