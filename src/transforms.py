"""
MONAI Transform Pipeline for Pancreas CT Segmentation.
Defines preprocessing transforms for training and inference.
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
    EnsureTyped,
    Resized,
)


def get_train_transforms(spatial_size=(224, 224), keys=["image", "label"]):
    """
    Get training transforms with data augmentation.
    
    Args:
        spatial_size: Target 2D spatial size for slices
        keys: Dictionary keys for image and label
        
    Returns:
        Composed transform pipeline
    """
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=keys, source_key="image"),
        EnsureTyped(keys=keys),
    ])


def get_val_transforms(keys=["image", "label"]):
    """
    Get validation/test transforms without augmentation.
    
    Args:
        keys: Dictionary keys for image and label
        
    Returns:
        Composed transform pipeline
    """
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=keys, source_key="image"),
        EnsureTyped(keys=keys),
    ])


def get_inference_transforms():
    """
    Get inference transforms for single image prediction.
    
    Returns:
        Composed transform pipeline
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ])
