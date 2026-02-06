"""
Synthetic video generators for testing.

Provides factory functions to create test videos with known properties.
"""
import torch
import random


def create_static_video(frames: int = 16, h: int = 128, w: int = 128) -> torch.Tensor:
    """
    Create a static video with no motion.
    
    A white square remains fixed in the center for all frames.
    Expected: Perfect smoothness (1.0), zero warping error.
    
    Returns:
        Tensor [1, T, H, W, 3]
    """
    video = torch.zeros(1, frames, h, w, 3)
    center_y, center_x = h // 2, w // 2
    square_size = 10
    video[:, :, center_y:center_y+square_size, center_x:center_x+square_size, :] = 1.0
    return video


def create_smooth_video(frames: int = 16, h: int = 128, w: int = 128, 
                        speed: int = 3) -> torch.Tensor:
    """
    Create a video with smooth linear motion.
    
    A white square moves horizontally at constant velocity.
    Expected: High smoothness (>0.3), low warping error.
    
    Args:
        speed: Pixels per frame movement
        
    Returns:
        Tensor [1, T, H, W, 3]
    """
    video = torch.zeros(1, frames, h, w, 3)
    square_size = 10
    y_pos = h // 2
    
    for t in range(frames):
        x_pos = int(5 + t * speed)
        if x_pos + square_size < w:
            video[0, t, y_pos:y_pos+square_size, x_pos:x_pos+square_size, :] = 1.0
    
    return video


def create_jittery_video(frames: int = 16, h: int = 128, w: int = 128,
                         seed: int = 42) -> torch.Tensor:
    """
    Create a video with jittery random motion.
    
    A white square jumps to random positions each frame.
    Expected: Low smoothness (<0.2), high warping error.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Tensor [1, T, H, W, 3]
    """
    random.seed(seed)
    video = torch.zeros(1, frames, h, w, 3)
    square_size = 10
    
    for t in range(frames):
        x_pos = random.randint(5, w - square_size - 5)
        y_pos = random.randint(5, h - square_size - 5)
        video[0, t, y_pos:y_pos+square_size, x_pos:x_pos+square_size, :] = 1.0
    
    return video


def create_flickering_video(frames: int = 16, h: int = 128, w: int = 128) -> torch.Tensor:
    """
    Create a video with flickering brightness.
    
    Static content but alternating bright/dark frames.
    Expected: High flickering score.
    
    Returns:
        Tensor [1, T, H, W, 3]
    """
    video = torch.zeros(1, frames, h, w, 3)
    
    for t in range(frames):
        brightness = 1.0 if t % 2 == 0 else 0.3
        video[0, t, 50:78, 50:78, :] = brightness
    
    return video


def create_accelerating_video(frames: int = 16, h: int = 128, w: int = 128) -> torch.Tensor:
    """
    Create a video with accelerating motion.
    
    A white square accelerates (increasing speed over time).
    Expected: Lower smoothness than constant velocity.
    
    Returns:
        Tensor [1, T, H, W, 3]
    """
    video = torch.zeros(1, frames, h, w, 3)
    square_size = 10
    y_pos = h // 2
    
    for t in range(frames):
        # Quadratic position = acceleration
        x_pos = int(5 + 0.5 * t * t)
        if x_pos + square_size < w:
            video[0, t, y_pos:y_pos+square_size, x_pos:x_pos+square_size, :] = 1.0
    
    return video


def create_noisy_image_pair(h: int = 256, w: int = 256, 
                            noise_level: float = 0.05) -> tuple:
    """
    Create a reference and distorted image pair.
    
    Args:
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        (reference, distorted) tensors [1, H, W, 3]
    """
    reference = torch.rand(1, h, w, 3)
    noise = torch.randn_like(reference) * noise_level
    distorted = (reference + noise).clamp(0, 1)
    return reference, distorted
