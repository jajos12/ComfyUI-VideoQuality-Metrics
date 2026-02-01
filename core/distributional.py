"""
Distributional metrics for video quality assessment.

Uses pretrained models:
- R3D-18 (Kinetics-400) for FVD
- Inception V3 (ImageNet) for FID
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_statistics(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and covariance of feature vectors.
    
    Args:
        features: Tensor [N, D] - N samples of D-dimensional features
        
    Returns:
        mean: Tensor [D]
        cov: Tensor [D, D]
    """
    n = features.shape[0]
    mean = features.mean(dim=0)
    
    if n < 2:
        # Return identity covariance for single sample
        cov = torch.eye(features.shape[1], device=features.device)
        return mean, cov
    
    centered = features - mean
    cov = (centered.T @ centered) / (n - 1)
    
    # Add small regularization for numerical stability
    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
    
    return mean, cov


def frechet_distance(mu1: torch.Tensor, sigma1: torch.Tensor,
                     mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    Compute Fréchet distance between two Gaussian distributions.
    
    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1, sigma1: Mean and covariance of first distribution
        mu2, sigma2: Mean and covariance of second distribution
        
    Returns:
        Fréchet distance (scalar)
    """
    diff = mu1 - mu2
    
    # Compute sqrt of product of covariances using eigendecomposition
    product = sigma1 @ sigma2
    
    # Eigenvalue decomposition for numerical stability
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(product)
        eigenvalues = torch.clamp(eigenvalues.real, min=0)
        sqrt_product = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
        sqrt_product = sqrt_product.real
    except RuntimeError:
        # Fallback: use identity if eigendecomposition fails
        sqrt_product = torch.zeros_like(sigma1)
    
    trace_term = torch.trace(sigma1 + sigma2 - 2 * sqrt_product)
    fd = torch.dot(diff, diff) + trace_term
    
    return torch.clamp(fd, min=0)


def extract_video_features(videos: torch.Tensor) -> torch.Tensor:
    """
    Extract features from videos using pretrained R3D-18.
    
    Args:
        videos: Tensor [N, T, H, W, C] - batch of videos
        
    Returns:
        features: Tensor [N, 512]
    """
    try:
        from ..models.downloader import get_video_encoder
    except (ImportError, ValueError):
        from models.downloader import get_video_encoder
    
    device = videos.device
    encoder = get_video_encoder(device)
    
    # Convert from [N, T, H, W, C] to [N, C, T, H, W]
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)
    
    videos = videos.permute(0, 4, 1, 2, 3).contiguous()  # [N, C, T, H, W]
    
    # Resize to expected input size (112x112 for R3D-18)
    N, C, T, H, W = videos.shape
    if H != 112 or W != 112:
        # Reshape to [N*T, C, H, W] for resize, then back
        videos = videos.permute(0, 2, 1, 3, 4).reshape(N * T, C, H, W)
        videos = F.interpolate(videos, size=(112, 112), mode='bilinear', align_corners=False)
        videos = videos.reshape(N, T, C, 112, 112).permute(0, 2, 1, 3, 4)  # Back to [N, C, T, H, W]
    
    # Normalize with R3D-18 stats (Kinetics mean/std)
    # Shape: [1, 3, 1, 1, 1] for broadcasting over [N, C, T, H, W]
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
    videos = (videos - mean) / std
    
    with torch.no_grad():
        features = encoder(videos)
    
    return features


def extract_image_features(images: torch.Tensor) -> torch.Tensor:
    """
    Extract features from images using pretrained Inception V3.
    
    Args:
        images: Tensor [N, H, W, C] - batch of images
        
    Returns:
        features: Tensor [N, 2048]
    """
    try:
        from ..models.downloader import get_image_encoder
    except (ImportError, ValueError):
        from models.downloader import get_image_encoder
    
    device = images.device
    encoder = get_image_encoder(device)
    
    # Convert from [N, H, W, C] to [N, C, H, W]
    images = images.permute(0, 3, 1, 2).contiguous()
    
    # Resize to 299x299 (Inception V3 input size)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    
    with torch.no_grad():
        features = encoder(images)
    
    return features


def calculate_fvd(videos1: torch.Tensor, videos2: torch.Tensor) -> dict:
    """
    Calculate Fréchet Video Distance between two sets of videos.
    
    Uses pretrained R3D-18 (Kinetics-400) for feature extraction.
    
    Args:
        videos1: Tensor [N1, T, H, W, C] - first set of videos
        videos2: Tensor [N2, T, H, W, C] - second set of videos
        
    Returns:
        dict with:
            - fvd: Fréchet Video Distance (lower = more similar)
            - n_samples: Number of samples used
            - feature_dim: Dimensionality of features
    """
    # Handle single video case
    if videos1.dim() == 4:
        videos1 = videos1.unsqueeze(0)
    if videos2.dim() == 4:
        videos2 = videos2.unsqueeze(0)
    
    # Extract features
    features1 = extract_video_features(videos1)
    features2 = extract_video_features(videos2)
    
    # Compute statistics
    mu1, sigma1 = compute_statistics(features1)
    mu2, sigma2 = compute_statistics(features2)
    
    # Compute FVD
    fvd = frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return {
        'fvd': fvd,
        'n_samples': (videos1.shape[0], videos2.shape[0]),
        'feature_dim': features1.shape[1]
    }


def calculate_fid(images1: torch.Tensor, images2: torch.Tensor) -> dict:
    """
    Calculate Fréchet Inception Distance between two sets of images.
    
    Uses pretrained Inception V3 (ImageNet) for feature extraction.
    
    Args:
        images1: Tensor [N1, H, W, C] - first set of images
        images2: Tensor [N2, H, W, C] - second set of images
        
    Returns:
        dict with:
            - fid: Fréchet Inception Distance (lower = more similar)
            - n_samples: Number of samples used
            - feature_dim: Dimensionality of features
    """
    # Extract features
    features1 = extract_image_features(images1)
    features2 = extract_image_features(images2)
    
    # Compute statistics
    mu1, sigma1 = compute_statistics(features1)
    mu2, sigma2 = compute_statistics(features2)
    
    # Compute FID
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return {
        'fid': fid,
        'n_samples': (images1.shape[0], images2.shape[0]),
        'feature_dim': features1.shape[1]
    }


def calculate_video_fid(video1: torch.Tensor, video2: torch.Tensor) -> dict:
    """
    Calculate frame-averaged FID between two videos.
    
    Args:
        video1: Tensor [T1, H, W, C] - first video
        video2: Tensor [T2, H, W, C] - second video
        
    Returns:
        dict with:
            - video_fid: Mean FID across frames
            - per_frame_distances: Feature distances per frame pair
    """
    T1, T2 = video1.shape[0], video2.shape[0]
    
    # Extract features for all frames
    features1 = extract_image_features(video1)
    features2 = extract_image_features(video2)
    
    # If different lengths, compute overall FID
    if T1 != T2:
        mu1, sigma1 = compute_statistics(features1)
        mu2, sigma2 = compute_statistics(features2)
        fid = frechet_distance(mu1, sigma1, mu2, sigma2)
        return {
            'video_fid': fid.item(),
            'note': 'Different video lengths - pooled comparison'
        }
    
    # Compute per-frame feature distances
    per_frame_distances = []
    for t in range(T1):
        dist = torch.sum((features1[t] - features2[t]) ** 2).item()
        per_frame_distances.append(dist)
    
    return {
        'video_fid': sum(per_frame_distances) / len(per_frame_distances),
        'per_frame_distances': per_frame_distances
    }
