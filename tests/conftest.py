"""
Pytest fixtures for VideoQuality tests.

Provides shared test data generators and utilities.
"""
import pytest
import torch
from pathlib import Path


@pytest.fixture
def static_video():
    """Create a static video (no motion) for testing."""
    video = torch.zeros(1, 16, 128, 128, 3)
    video[:, :, 64:74, 64:74, :] = 1.0
    return video


@pytest.fixture
def smooth_video():
    """Create a video with smooth linear motion."""
    video = torch.zeros(1, 16, 128, 128, 3)
    for t in range(16):
        x_pos = int(5 + t * 3)
        video[0, t, 64:74, x_pos:x_pos+10, :] = 1.0
    return video


@pytest.fixture
def jittery_video():
    """Create a video with jittery random motion."""
    import random
    random.seed(42)
    video = torch.zeros(1, 16, 128, 128, 3)
    for t in range(16):
        x_pos = random.randint(5, 113)
        y_pos = random.randint(5, 113)
        video[0, t, y_pos:y_pos+10, x_pos:x_pos+10, :] = 1.0
    return video


@pytest.fixture
def sample_image_pair():
    """Create a pair of similar images for fidelity testing."""
    reference = torch.rand(1, 256, 256, 3)
    # Add slight noise for distorted version
    distorted = reference + torch.randn_like(reference) * 0.05
    distorted = distorted.clamp(0, 1)
    return reference, distorted
