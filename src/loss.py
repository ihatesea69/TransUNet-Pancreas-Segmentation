"""
Loss functions for pancreas segmentation.
Implements HybridLoss combining Dice Loss and Cross-Entropy Loss.
"""

import torch
import torch.nn as nn
from monai.losses import DiceLoss


class HybridLoss(nn.Module):
    """
    Hybrid loss combining Dice Loss and Cross-Entropy Loss.
    
    Total_Loss = dice_weight * DiceLoss + ce_weight * CrossEntropyLoss
    
    Args:
        dice_weight: Weight for Dice Loss component (default: 0.5)
        ce_weight: Weight for Cross-Entropy Loss component (default: 0.5)
        num_classes: Number of segmentation classes (default: 2)
        softmax: Whether to apply softmax in Dice Loss (default: True)
        include_background: Whether to include background in Dice calculation (default: True)
    """
    
    def __init__(
        self,
        dice_weight=0.5,
        ce_weight=0.5,
        num_classes=2,
        softmax=True,
        include_background=True,
    ):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        
        # Dice Loss from MONAI
        self.dice_loss = DiceLoss(
            softmax=softmax,
            include_background=include_background,
            to_onehot_y=True,
        )
        
        # Cross-Entropy Loss from PyTorch
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        Compute hybrid loss.
        
        Args:
            predictions: Model output logits (B, C, H, W)
            targets: Ground truth labels (B, H, W) as long tensor
            
        Returns:
            Combined loss value
        """
        # Dice loss expects (B, C, H, W) for both pred and target
        # Target needs to be (B, 1, H, W) for to_onehot_y
        targets_dice = targets.unsqueeze(1).float()
        dice = self.dice_loss(predictions, targets_dice)
        
        # Cross-entropy expects (B, C, H, W) predictions and (B, H, W) targets
        ce = self.ce_loss(predictions, targets)
        
        # Combine losses
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        return total_loss


class DiceCELoss(nn.Module):
    """
    Alternative implementation using MONAI's DiceCELoss directly.
    """
    
    def __init__(self, softmax=True, include_background=True, lambda_dice=0.5, lambda_ce=0.5):
        super().__init__()
        from monai.losses import DiceCELoss as MonaiDiceCELoss
        
        self.loss = MonaiDiceCELoss(
            softmax=softmax,
            include_background=include_background,
            to_onehot_y=True,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
        )
    
    def forward(self, predictions, targets):
        targets = targets.unsqueeze(1)
        return self.loss(predictions, targets)
