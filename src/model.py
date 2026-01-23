"""
TransUNet Architecture for Medical Image Segmentation.

Architecture Overview:
1. CNN Encoder (ResNet50): Extracts multi-scale features
2. Transformer Bottleneck (ViT): Captures global context via self-attention
3. CNN Decoder (U-Net style): Reconstructs spatial resolution with skip connections
4. Segmentation Head: 1x1 conv to output class predictions

Reference: Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
arXiv:2102.04306
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# Transformer Components
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Store for visualization
        self.attention_weights = attn.detach()
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for processing CNN feature maps.
    
    Takes flattened feature patches as input and applies self-attention.
    """
    
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0,
        dropout=0.0,
        num_patches=196,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        # x: (B, N, embed_dim) where N is number of patches
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x
    
    def get_attention_maps(self):
        """Return attention weights from all layers."""
        return [block.attn.attention_weights for block in self.blocks]


# ============================================================================
# CNN Encoder (ResNet-based)
# ============================================================================

class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class CNNEncoder(nn.Module):
    """
    CNN Encoder extracting multi-scale features.
    
    Returns features at 4 scales: 1/4, 1/8, 1/16, 1/32 of input resolution.
    """
    
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Encoder stages
        self.layer1 = self._make_layer(base_channels, base_channels, 3)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 4, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 6, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 3, stride=2)
        
        # Channel dimensions for skip connections
        self.skip_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Returns skip connections at each scale
        x = self.stem(x)  # 1/4
        
        skip1 = self.layer1(x)   # 1/4, 64 channels
        skip2 = self.layer2(skip1)  # 1/8, 128 channels
        skip3 = self.layer3(skip2)  # 1/16, 256 channels
        skip4 = self.layer4(skip3)  # 1/32, 512 channels
        
        return [skip1, skip2, skip3, skip4]


# ============================================================================
# CNN Decoder (U-Net style)
# ============================================================================

class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class CNNDecoder(nn.Module):
    """U-Net style decoder with skip connections."""
    
    def __init__(self, encoder_channels, embed_dim=768, base_channels=64):
        super().__init__()
        
        # Project transformer output back to spatial features
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, encoder_channels[-1], 1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True),
        )
        
        # Decoder blocks (from deepest to shallowest)
        self.decoder4 = DecoderBlock(encoder_channels[-1], encoder_channels[-2], base_channels * 4)
        self.decoder3 = DecoderBlock(base_channels * 4, encoder_channels[-3], base_channels * 2)
        self.decoder2 = DecoderBlock(base_channels * 2, encoder_channels[-4], base_channels)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Final upsampling to match input resolution
        self.final_up = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
    
    def forward(self, x, skips, feature_size):
        """
        Args:
            x: Transformer output (B, N, embed_dim)
            skips: List of skip connections from encoder [skip1, skip2, skip3, skip4]
            feature_size: Spatial size of deepest feature map (H, W)
        """
        B, N, C = x.shape
        H, W = feature_size
        
        # Reshape transformer output to spatial
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        
        # Decoder with skip connections
        x = self.decoder4(x, skips[2])  # 1/16 -> 1/8
        x = self.decoder3(x, skips[1])  # 1/8 -> 1/4
        x = self.decoder2(x, skips[0])  # 1/4 -> 1/2
        x = self.decoder1(x)            # 1/2 -> 1/1
        x = self.final_up(x)            # Final upsampling
        
        return x


# ============================================================================
# TransUNet Model
# ============================================================================

class TransUNet(nn.Module):
    """
    TransUNet: Transformer + U-Net for Medical Image Segmentation.
    
    Architecture:
        Input -> CNN Encoder -> Transformer Bottleneck -> CNN Decoder -> Output
    
    Args:
        img_size: Input image size (default: 224)
        in_channels: Number of input channels (default: 1 for CT)
        out_channels: Number of output classes (default: 2)
        embed_dim: Transformer embedding dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        num_layers: Number of transformer layers (default: 12)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        dropout: Dropout rate (default: 0.1)
        base_channels: Base channel count for CNN (default: 64)
    """
    
    def __init__(
        self,
        img_size=224,
        in_channels=1,
        out_channels=2,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_ratio=4.0,
        dropout=0.1,
        base_channels=64,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        
        # CNN Encoder
        self.encoder = CNNEncoder(in_channels, base_channels)
        encoder_out_channels = self.encoder.skip_channels[-1]  # 512
        
        # Calculate feature map size after encoder (1/32 of input)
        self.feature_size = img_size // 32
        self.num_patches = self.feature_size ** 2
        
        # Patch embedding: project CNN features to transformer dimension
        self.patch_embed = nn.Sequential(
            nn.Conv2d(encoder_out_channels, embed_dim, kernel_size=1),
            nn.Flatten(2),  # (B, embed_dim, H*W)
        )
        
        # Vision Transformer
        self.transformer = VisionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_patches=self.num_patches,
        )
        
        # CNN Decoder
        self.decoder = CNNDecoder(
            encoder_channels=self.encoder.skip_channels,
            embed_dim=embed_dim,
            base_channels=base_channels,
        )
        
        # Segmentation head
        self.seg_head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image (B, C, H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        B = x.shape[0]
        
        # CNN Encoder: extract multi-scale features
        skips = self.encoder(x)  # [skip1, skip2, skip3, skip4]
        
        # Patch embedding: project deepest features to transformer dimension
        x = self.patch_embed(skips[-1])  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        
        # Transformer: global context modeling
        x = self.transformer(x)
        
        # CNN Decoder: upsample with skip connections
        feature_size = (self.feature_size, self.feature_size)
        x = self.decoder(x, skips, feature_size)
        
        # Segmentation head
        x = self.seg_head(x)
        
        return x
    
    def get_attention_maps(self):
        """Get attention maps from transformer for visualization."""
        return self.transformer.get_attention_maps()


# ============================================================================
# Model Factory
# ============================================================================

def create_transunet(
    img_size=224,
    in_channels=1,
    out_channels=2,
    variant="base",
):
    """
    Create TransUNet model with predefined configurations.
    
    Args:
        img_size: Input image size
        in_channels: Number of input channels
        out_channels: Number of output classes
        variant: Model variant ('small', 'base', 'large')
        
    Returns:
        TransUNet model instance
    """
    configs = {
        "small": {
            "embed_dim": 384,
            "num_heads": 6,
            "num_layers": 6,
            "base_channels": 32,
        },
        "base": {
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "base_channels": 64,
        },
        "large": {
            "embed_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "base_channels": 64,
        },
    }
    
    config = configs.get(variant, configs["base"])
    
    return TransUNet(
        img_size=img_size,
        in_channels=in_channels,
        out_channels=out_channels,
        **config,
    )


if __name__ == "__main__":
    # Test model
    model = create_transunet(img_size=224, in_channels=1, out_channels=2, variant="base")
    x = torch.randn(1, 1, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
