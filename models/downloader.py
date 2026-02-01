"""
Model downloader and management for Video Quality Metrics.

Handles automatic downloading of pretrained models:
- I3D for FVD (via torchvision R3D-18)
- InceptionV3 for FID
- RAFT for optical flow (optional)
"""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path


# Default cache directory (follows torch hub convention)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "video_quality_metrics"


def get_cache_dir() -> Path:
    """Get the model cache directory, creating it if necessary."""
    cache_dir = Path(os.environ.get("VQ_METRICS_CACHE", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Video Encoder (for FVD)
# ============================================================================

_video_encoder: Optional[torch.nn.Module] = None
_video_encoder_device: Optional[torch.device] = None


def get_video_encoder(device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Get pretrained video encoder for FVD calculation.
    
    Uses torchvision's R3D-18 (3D ResNet) pretrained on Kinetics-400.
    This is a well-established backbone for video feature extraction.
    
    Args:
        device: Target device (auto-detected if None)
        
    Returns:
        Pretrained video encoder in eval mode
    """
    global _video_encoder, _video_encoder_device
    
    if device is None:
        device = get_device()
    
    # Return cached encoder if available on same device
    if _video_encoder is not None and _video_encoder_device == device:
        return _video_encoder
    
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        
        print("[VQ Metrics] Loading R3D-18 video encoder (pretrained on Kinetics-400)...")
        
        # Load with pretrained weights
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        
        # Remove final classification layer to get features
        # R3D-18 outputs 512-dim features before the final FC
        model.fc = torch.nn.Identity()
        
        model = model.to(device)
        model.eval()
        
        _video_encoder = model
        _video_encoder_device = device
        
        print("[VQ Metrics] R3D-18 loaded successfully (512-dim features)")
        return model
        
    except ImportError as e:
        raise ImportError(
            "torchvision is required for pretrained video encoder. "
            "Install with: pip install torchvision>=0.15.0"
        ) from e


# ============================================================================
# Image Encoder (for FID)
# ============================================================================

_image_encoder: Optional[torch.nn.Module] = None
_image_encoder_device: Optional[torch.device] = None


def get_image_encoder(device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Get pretrained image encoder for FID calculation.
    
    Uses torchvision's Inception V3 pretrained on ImageNet.
    Features are extracted from the final average pooling layer (2048-dim).
    
    Args:
        device: Target device (auto-detected if None)
        
    Returns:
        Pretrained image encoder in eval mode
    """
    global _image_encoder, _image_encoder_device
    
    if device is None:
        device = get_device()
    
    if _image_encoder is not None and _image_encoder_device == device:
        return _image_encoder
    
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        print("[VQ Metrics] Loading Inception V3 encoder (pretrained on ImageNet)...")
        
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, transform_input=False)
        
        # Remove classification head
        model.fc = torch.nn.Identity()
        
        # Disable auxiliary outputs
        model.aux_logits = False
        model.AuxLogits = None
        
        model = model.to(device)
        model.eval()
        
        _image_encoder = model
        _image_encoder_device = device
        
        print("[VQ Metrics] Inception V3 loaded successfully (2048-dim features)")
        return model
        
    except ImportError as e:
        raise ImportError(
            "torchvision is required for pretrained image encoder. "
            "Install with: pip install torchvision>=0.15.0"
        ) from e


# ============================================================================
# Optical Flow (RAFT)
# ============================================================================

_raft_model: Optional[torch.nn.Module] = None
_raft_device: Optional[torch.device] = None


def get_raft_model(device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Get pretrained RAFT optical flow model.
    
    Uses torchvision's RAFT implementation pretrained on FlyingChairs/Sintel.
    
    Args:
        device: Target device (auto-detected if None)
        
    Returns:
        Pretrained RAFT model in eval mode
    """
    global _raft_model, _raft_device
    
    if device is None:
        device = get_device()
    
    if _raft_model is not None and _raft_device == device:
        return _raft_model
    
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        
        print("[VQ Metrics] Loading RAFT-Small optical flow model...")
        
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights)
        
        model = model.to(device)
        model.eval()
        
        _raft_model = model
        _raft_device = device
        
        print("[VQ Metrics] RAFT-Small loaded successfully")
        return model
        
    except ImportError as e:
        raise ImportError(
            "torchvision>=0.14.0 is required for RAFT. "
            "Install with: pip install torchvision>=0.15.0"
        ) from e


# ============================================================================
# Preprocessing utilities
# ============================================================================

def get_video_transform():
    """Get preprocessing transform for video encoder (R3D-18)."""
    from torchvision.transforms import Compose, Normalize
    
    # R3D-18 expects: [B, C, T, H, W] normalized with ImageNet stats
    # Input should be resized to 112x112 or 224x224
    return Compose([
        Normalize(mean=[0.43216, 0.394666, 0.37645], 
                  std=[0.22803, 0.22145, 0.216989])
    ])


def get_image_transform():
    """Get preprocessing transform for image encoder (Inception V3)."""
    from torchvision.transforms import Compose, Normalize, Resize
    
    # Inception V3 expects 299x299, normalized with ImageNet stats
    return Compose([
        Resize((299, 299), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], 
                  std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# Model info
# ============================================================================

MODEL_INFO = {
    "r3d_18": {
        "name": "R3D-18 (3D ResNet)",
        "pretrained_on": "Kinetics-400",
        "feature_dim": 512,
        "input_size": "T×112×112 or T×224×224",
        "approx_size_mb": 60,
    },
    "inception_v3": {
        "name": "Inception V3",
        "pretrained_on": "ImageNet-1K",
        "feature_dim": 2048,
        "input_size": "299×299",
        "approx_size_mb": 100,
    },
    "raft_small": {
        "name": "RAFT-Small",
        "pretrained_on": "FlyingChairs + FlyingThings3D + Sintel + KITTI",
        "approx_size_mb": 20,
    },
}


def print_model_info():
    """Print information about available models."""
    print("\n=== Video Quality Metrics - Pretrained Models ===\n")
    for key, info in MODEL_INFO.items():
        print(f"  {key}:")
        for k, v in info.items():
            print(f"    {k}: {v}")
        print()
