"""
Visualization utilities for pancreas CT segmentation.
Provides functions to display CT slices with mask overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def show_slices(image, label=None, title="CT Slices", figsize=(15, 5)):
    """
    Display axial, coronal, and sagittal slices of a 3D volume.
    
    Args:
        image: 3D numpy array (D, H, W) or (C, D, H, W)
        label: Optional 3D numpy array with same shape as image
        title: Figure title
        figsize: Figure size tuple
    """
    # Handle channel dimension
    if image.ndim == 4:
        image = image[0]
    if label is not None and label.ndim == 4:
        label = label[0]
    
    # Get center slices
    d, h, w = image.shape
    axial_idx = d // 2
    coronal_idx = h // 2
    sagittal_idx = w // 2
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Axial slice (top-down view)
    axes[0].imshow(image[axial_idx, :, :], cmap="gray")
    if label is not None:
        mask = np.ma.masked_where(label[axial_idx, :, :] == 0, label[axial_idx, :, :])
        axes[0].imshow(mask, cmap="jet", alpha=0.5, vmin=0, vmax=2)
    axes[0].set_title(f"Axial (slice {axial_idx})")
    axes[0].axis("off")
    
    # Coronal slice (front view)
    axes[1].imshow(image[:, coronal_idx, :], cmap="gray")
    if label is not None:
        mask = np.ma.masked_where(label[:, coronal_idx, :] == 0, label[:, coronal_idx, :])
        axes[1].imshow(mask, cmap="jet", alpha=0.5, vmin=0, vmax=2)
    axes[1].set_title(f"Coronal (slice {coronal_idx})")
    axes[1].axis("off")
    
    # Sagittal slice (side view)
    axes[2].imshow(image[:, :, sagittal_idx], cmap="gray")
    if label is not None:
        mask = np.ma.masked_where(label[:, :, sagittal_idx] == 0, label[:, :, sagittal_idx])
        axes[2].imshow(mask, cmap="jet", alpha=0.5, vmin=0, vmax=2)
    axes[2].set_title(f"Sagittal (slice {sagittal_idx})")
    axes[2].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_prediction_comparison(image, ground_truth, prediction, slice_idx=None, figsize=(15, 5)):
    """
    Display original CT, ground truth mask, and model prediction side by side.
    
    Args:
        image: 2D or 3D numpy array
        ground_truth: 2D or 3D numpy array with same shape
        prediction: 2D or 3D numpy array with same shape
        slice_idx: Optional slice index for 3D volumes
        figsize: Figure size tuple
    """
    # Handle 3D volumes
    if image.ndim == 3:
        if slice_idx is None:
            slice_idx = image.shape[0] // 2
        image = image[slice_idx]
        ground_truth = ground_truth[slice_idx]
        prediction = prediction[slice_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original CT
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original CT")
    axes[0].axis("off")
    
    # Ground Truth (green overlay)
    axes[1].imshow(image, cmap="gray")
    gt_mask = np.ma.masked_where(ground_truth == 0, ground_truth)
    axes[1].imshow(gt_mask, cmap="Greens", alpha=0.6, vmin=0, vmax=2)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    # Prediction (red overlay)
    axes[2].imshow(image, cmap="gray")
    pred_mask = np.ma.masked_where(prediction == 0, prediction)
    axes[2].imshow(pred_mask, cmap="Reds", alpha=0.6, vmin=0, vmax=2)
    axes[2].set_title("Model Prediction")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history, figsize=(12, 4)):
    """
    Plot training and validation loss/metric curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_dice' keys
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice score curve
    axes[1].plot(epochs, history["val_dice"], "g-", label="Val Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Validation Dice Score")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def show_attention_map(image, attention_weights, figsize=(10, 5)):
    """
    Overlay attention heatmap on the original image.
    
    Args:
        image: 2D numpy array (H, W)
        attention_weights: 2D numpy array (H, W) with attention values
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Attention overlay
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(attention_weights, cmap="hot", alpha=0.6)
    axes[1].set_title("Attention Map Overlay")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()
