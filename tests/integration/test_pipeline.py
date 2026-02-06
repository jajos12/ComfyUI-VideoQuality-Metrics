"""
Comprehensive test suite for Video Quality Metrics.

Tests all core modules with pretrained model support.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fidelity import calculate_psnr, calculate_ssim
from core.color import calculate_ciede2000


def test_fidelity_metrics():
    """Test PSNR and SSIM on identical and noisy images."""
    print("=" * 50)
    print("FIDELITY METRICS TEST")
    print("=" * 50)
    
    img1 = torch.rand((1, 256, 256, 3))
    img2 = img1.clone()
    
    # Identical images
    psnr = calculate_psnr(img1, img2).item()
    ssim = calculate_ssim(img1, img2).item()
    
    print(f"Identical images:")
    print(f"  PSNR: {psnr:.2f} (expected: 100.00)")
    print(f"  SSIM: {ssim:.4f} (expected: 1.0000)")
    
    assert psnr == 100.0, "PSNR should be 100 for identical images"
    assert abs(ssim - 1.0) < 0.001, "SSIM should be 1.0 for identical images"
    
    # Noisy image
    img3 = img1 + 0.05 * torch.randn_like(img1)
    img3 = torch.clamp(img3, 0, 1)
    
    psnr_noisy = calculate_psnr(img1, img3).item()
    ssim_noisy = calculate_ssim(img1, img3).item()
    
    print(f"Noisy image:")
    print(f"  PSNR: {psnr_noisy:.2f}")
    print(f"  SSIM: {ssim_noisy:.4f}")
    
    assert psnr_noisy < psnr, "PSNR should decrease with noise"
    assert ssim_noisy < ssim, "SSIM should decrease with noise"
    
    print("✓ Fidelity metrics OK\n")


def test_color_metrics():
    """Test CIEDE2000 color difference."""
    print("=" * 50)
    print("COLOR METRICS TEST")
    print("=" * 50)
    
    img1 = torch.rand((1, 128, 128, 3))
    img2 = img1.clone()
    
    # Identical images
    ciede = calculate_ciede2000(img1, img2).item()
    print(f"Identical images - CIEDE2000: {ciede:.4f} (expected: 0.00)")
    assert abs(ciede) < 0.001, "CIEDE2000 should be 0 for identical images"
    
    # Color shifted
    img3 = img1.clone()
    img3[..., 0] += 0.1
    img3 = torch.clamp(img3, 0, 1)
    
    ciede_shift = calculate_ciede2000(img1, img3).item()
    print(f"Color shifted - CIEDE2000: {ciede_shift:.4f}")
    assert ciede_shift > 0, "CIEDE2000 should detect color shift"
    
    print("✓ Color metrics OK\n")


def test_temporal_metrics_lk():
    """Test temporal metrics with Lucas-Kanade (no pretrained model needed)."""
    print("=" * 50)
    print("TEMPORAL METRICS TEST (Lucas-Kanade)")
    print("=" * 50)
    
    from core.temporal import (
        calculate_warping_error, 
        calculate_temporal_flickering, 
        calculate_motion_smoothness
    )
    
    # Create synthetic video (smooth motion)
    T, H, W, C = 8, 64, 64, 3
    video = torch.zeros(1, T, H, W, C)
    
    # Moving gradient
    for t in range(T):
        offset = t * 5
        for x in range(W):
            video[0, t, :, x, :] = (x + offset) / (W + T * 5)
    
    # Warping error (using Lucas-Kanade)
    warp_result = calculate_warping_error(video, bidirectional=False, use_raft=False)
    print(f"Warping Error: {warp_result['warping_error'].item():.4f}")
    
    # Flickering
    flicker_result = calculate_temporal_flickering(video, window_size=3)
    print(f"Flickering Score: {flicker_result['flickering_score'].item():.4f}")
    
    # Motion smoothness
    smooth_result = calculate_motion_smoothness(video, use_raft=False)
    print(f"Smoothness Score: {smooth_result['smoothness_score'].item():.4f}")
    
    print("✓ Temporal metrics OK\n")


def test_distributional_metrics():
    """Test FVD and FID with pretrained models."""
    print("=" * 50)
    print("DISTRIBUTIONAL METRICS TEST (Pretrained)")
    print("=" * 50)
    
    from core.distributional import calculate_fvd, calculate_fid
    
    # Note: This will download models on first run (~160MB total)
    
    # Create synthetic videos
    print("Creating test data...")
    video1 = torch.rand(2, 8, 64, 64, 3)
    video2 = video1.clone() + 0.1 * torch.randn_like(video1)
    video2 = torch.clamp(video2, 0, 1)
    
    # FVD
    print("Computing FVD (will download R3D-18 on first run)...")
    fvd_result = calculate_fvd(video1, video2)
    print(f"FVD: {fvd_result['fvd'].item():.2f}")
    print(f"Feature dim: {fvd_result['feature_dim']}")
    
    # FID for frames
    print("Computing FID (will download Inception V3 on first run)...")
    frames1 = video1[0]  # [T, H, W, C]
    frames2 = video2[0]
    
    fid_result = calculate_fid(frames1, frames2)
    print(f"FID: {fid_result['fid'].item():.2f}")
    print(f"Feature dim: {fid_result['feature_dim']}")
    
    print("✓ Distributional metrics OK\n")


def test_visualization():
    """Test radar chart and metric normalization."""
    print("=" * 50)
    print("VISUALIZATION TEST")
    print("=" * 50)
    
    from utils.plotting import normalize_metrics, generate_radar_chart_tensor
    
    raw_metrics = {
        'psnr': 35.5,
        'ssim': 0.92,
        'ciede2000': 2.5,
        'warping_error': 0.08,
    }
    
    normalized = normalize_metrics(raw_metrics)
    print(f"Normalized metrics: {normalized}")
    
    for k, v in normalized.items():
        assert 0 <= v <= 1, f"Normalized {k} should be in [0, 1]"
    
    chart = generate_radar_chart_tensor(normalized, size=128)
    print(f"Radar chart shape: {chart.shape}")
    assert chart.shape == (1, 128, 128, 3), "Chart should be [1, H, W, 3]"
    
    print("✓ Visualization OK\n")


def main():
    print("\n" + "=" * 60)
    print("VIDEO QUALITY METRICS - TEST SUITE")
    print("=" * 60 + "\n")
    
    test_fidelity_metrics()
    test_color_metrics()
    test_temporal_metrics_lk()
    test_visualization()
    
    # Optional: Test pretrained models (requires download)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', 
                        help='Run full tests including pretrained model downloads')
    args, _ = parser.parse_known_args()
    
    if args.full:
        test_distributional_metrics()
    else:
        print("=" * 50)
        print("SKIPPING DISTRIBUTIONAL TESTS")
        print("(Run with --full to download pretrained models)")
        print("=" * 50 + "\n")
    
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
