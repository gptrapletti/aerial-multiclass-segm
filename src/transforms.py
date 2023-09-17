import albumentations as A

train_transforms = A.Compose([
    # Dual transforms
    A.Resize(height=256, width=256, interpolation=3, always_apply=True),
    A.Affine(
        scale = (0.8, 1.2),
        rotate = (-360, 360),
        shear = (-20, 20),
        p = 0.5
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # Image only transforms
    A.ColorJitter(
        brightness = 0.5,
        contrast = 0.5,
        saturation = 0.5,
        hue = 0,
        p = 0.5
    ),
    A.CLAHE(p=0.5),
    # A.Normalize(mean=(0.4456, 0.4436, 0.4018), std=(0.2220, 0.2154, 0.2298), p=1) # mean and std computed on this dataset.    
])

val_transforms = A.Compose([
    A.Resize(height=256, width=256, interpolation=3, always_apply=True)
    # A.Normalize(mean=(0.4456, 0.4436, 0.4018), std=(0.2220, 0.2154, 0.2298), p=1) # mean and std computed on this dataset
])

