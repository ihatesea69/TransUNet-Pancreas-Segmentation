"""
Dataset classes for pancreas CT segmentation.
Implements SlicingDataset for extracting 2D slices from 3D volumes.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd


class SlicingDataset(Dataset):
    """
    Dataset that extracts 2D axial slices from 3D CT volumes.
    Only includes slices that contain pancreas labels (non-empty masks).
    
    Args:
        data_list: List of dictionaries with 'image' and 'label' paths
        transform: MONAI transform pipeline for preprocessing
        slice_axis: Axis along which to extract slices (0=axial, 1=coronal, 2=sagittal)
        include_empty: Whether to include slices without pancreas labels
        cache_data: Whether to cache loaded volumes in memory
    """
    
    def __init__(
        self,
        data_list,
        transform=None,
        slice_axis=0,
        include_empty=False,
        cache_data=False,
        target_size=(224, 224),
    ):
        self.data_list = data_list
        self.transform = transform
        self.slice_axis = slice_axis
        self.include_empty = include_empty
        self.cache_data = cache_data
        self.target_size = target_size
        
        # Build slice index mapping
        self.slice_indices = []
        self.cached_volumes = {}
        
        self._build_slice_index()
    
    def _build_slice_index(self):
        """Build mapping from global index to (volume_idx, slice_idx)."""
        print("Building slice index...")
        
        for vol_idx, data_dict in enumerate(self.data_list):
            # Load volume to get slice count
            if self.transform is not None:
                data = self.transform(data_dict)
                image = data["image"]
                label = data["label"]
            else:
                # Basic loading without transform
                import nibabel as nib
                image = nib.load(data_dict["image"]).get_fdata()
                label = nib.load(data_dict["label"]).get_fdata()
            
            # Convert to numpy if tensor
            if torch.is_tensor(image):
                image = image.numpy()
            if torch.is_tensor(label):
                label = label.numpy()
            
            # Handle channel dimension
            if image.ndim == 4:
                image = image[0]
            if label.ndim == 4:
                label = label[0]
            
            # Cache if requested
            if self.cache_data:
                self.cached_volumes[vol_idx] = {"image": image, "label": label}
            
            # Get number of slices along axis
            num_slices = image.shape[self.slice_axis]
            
            for slice_idx in range(num_slices):
                # Get slice label to check if non-empty
                if self.slice_axis == 0:
                    slice_label = label[slice_idx, :, :]
                elif self.slice_axis == 1:
                    slice_label = label[:, slice_idx, :]
                else:
                    slice_label = label[:, :, slice_idx]
                
                has_label = np.any(slice_label > 0)
                
                if has_label or self.include_empty:
                    self.slice_indices.append((vol_idx, slice_idx))
        
        print(f"Total slices: {len(self.slice_indices)}")
    
    def __len__(self):
        return len(self.slice_indices)
    
    def __getitem__(self, idx):
        vol_idx, slice_idx = self.slice_indices[idx]
        
        # Get volume data
        if self.cache_data and vol_idx in self.cached_volumes:
            image = self.cached_volumes[vol_idx]["image"]
            label = self.cached_volumes[vol_idx]["label"]
        else:
            data_dict = self.data_list[vol_idx]
            if self.transform is not None:
                data = self.transform(data_dict)
                image = data["image"]
                label = data["label"]
            else:
                import nibabel as nib
                image = nib.load(data_dict["image"]).get_fdata()
                label = nib.load(data_dict["label"]).get_fdata()
            
            # Convert to numpy if tensor
            if torch.is_tensor(image):
                image = image.numpy()
            if torch.is_tensor(label):
                label = label.numpy()
            
            # Handle channel dimension
            if image.ndim == 4:
                image = image[0]
            if label.ndim == 4:
                label = label[0]
        
        # Extract slice
        if self.slice_axis == 0:
            image_slice = image[slice_idx, :, :]
            label_slice = label[slice_idx, :, :]
        elif self.slice_axis == 1:
            image_slice = image[:, slice_idx, :]
            label_slice = label[:, slice_idx, :]
        else:
            image_slice = image[:, :, slice_idx]
            label_slice = label[:, :, slice_idx]
        
        # Resize to target size
        image_slice = self._resize_slice(image_slice, self.target_size)
        label_slice = self._resize_slice(label_slice, self.target_size, is_label=True)
        
        # Merge tumor (class 2) with pancreas (class 1) for binary segmentation
        # 0 = background, 1 = pancreas + tumor
        label_slice = np.where(label_slice > 0, 1, 0)
        
        # Convert to tensor and add channel dimension
        image_tensor = torch.from_numpy(image_slice).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label_slice).long()
        
        return image_tensor, label_tensor
    
    def _resize_slice(self, slice_data, target_size, is_label=False):
        """Resize slice to target size using interpolation."""
        from scipy.ndimage import zoom
        
        current_size = slice_data.shape
        zoom_factors = (target_size[0] / current_size[0], target_size[1] / current_size[1])
        
        if is_label:
            # Use nearest neighbor for labels
            resized = zoom(slice_data, zoom_factors, order=0)
        else:
            # Use bilinear for images
            resized = zoom(slice_data, zoom_factors, order=1)
        
        return resized


def load_data_splits(json_path):
    """
    Load train/val/test splits from JSON file.
    
    Args:
        json_path: Path to JSON file with splits
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    with open(json_path, "r") as f:
        splits = json.load(f)
    return splits


def create_data_list(image_dir, label_dir, file_ids):
    """
    Create list of data dictionaries for MONAI transforms.
    
    Args:
        image_dir: Directory containing image files
        label_dir: Directory containing label files
        file_ids: List of file identifiers
        
    Returns:
        List of dictionaries with 'image' and 'label' keys
    """
    data_list = []
    for file_id in file_ids:
        data_list.append({
            "image": os.path.join(image_dir, file_id),
            "label": os.path.join(label_dir, file_id.replace("_0000", "")),
        })
    return data_list
