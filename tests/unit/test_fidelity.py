"""
Unit tests for perceptual fidelity metrics.

Tests: calculate_psnr, calculate_ssim, calculate_ciede2000
"""
import pytest
import torch

from core.fidelity import calculate_psnr, calculate_ssim
from core.color import calculate_ciede2000
from tests.fixtures.synthetic_videos import create_noisy_image_pair


class TestPSNR:
    """Tests for Peak Signal-to-Noise Ratio."""
    
    def test_identical_images_high_psnr(self):
        """Identical images should have very high (or infinite) PSNR."""
        image = torch.rand(1, 256, 256, 3)
        psnr = calculate_psnr(image, image)
        
        # Identical = infinite PSNR (clamped to high value)
        assert psnr.item() > 50
    
    def test_noisy_image_lower_psnr(self):
        """Noisy image should have lower PSNR than identical."""
        reference, distorted = create_noisy_image_pair(noise_level=0.1)
        psnr = calculate_psnr(distorted, reference)
        
        # Should be reasonable but not infinite
        assert 20 < psnr.item() < 50
    
    def test_psnr_symmetry(self):
        """PSNR should be symmetric: PSNR(a,b) == PSNR(b,a)."""
        reference, distorted = create_noisy_image_pair()
        psnr_ab = calculate_psnr(reference, distorted)
        psnr_ba = calculate_psnr(distorted, reference)
        
        assert psnr_ab.item() == pytest.approx(psnr_ba.item(), abs=0.01)
    
    def test_more_noise_lower_psnr(self):
        """Higher noise level should result in lower PSNR."""
        ref1, dist1 = create_noisy_image_pair(noise_level=0.05)
        ref2, dist2 = create_noisy_image_pair(noise_level=0.20)
        
        # Use same reference for fair comparison
        reference = torch.rand(1, 256, 256, 3)
        low_noise = reference + torch.randn_like(reference) * 0.05
        high_noise = reference + torch.randn_like(reference) * 0.20
        
        psnr_low = calculate_psnr(low_noise.clamp(0,1), reference)
        psnr_high = calculate_psnr(high_noise.clamp(0,1), reference)
        
        assert psnr_low > psnr_high


class TestSSIM:
    """Tests for Structural Similarity Index."""
    
    def test_identical_images_ssim_one(self):
        """Identical images should have SSIM = 1.0."""
        image = torch.rand(1, 256, 256, 3)
        ssim = calculate_ssim(image, image)
        
        assert ssim.item() == pytest.approx(1.0, abs=0.01)
    
    def test_ssim_range(self):
        """SSIM should be in [0, 1] range."""
        reference, distorted = create_noisy_image_pair()
        ssim = calculate_ssim(distorted, reference)
        
        assert 0.0 <= ssim.item() <= 1.0
    
    def test_noisy_image_lower_ssim(self):
        """Noisy image should have lower SSIM than identical."""
        reference, distorted = create_noisy_image_pair(noise_level=0.1)
        ssim = calculate_ssim(distorted, reference)
        
        assert ssim.item() < 1.0
        assert ssim.item() > 0.5  # Should still be reasonably similar


class TestCIEDE2000:
    """Tests for CIEDE2000 color difference."""
    
    def test_identical_images_zero_delta(self):
        """Identical images should have ΔE00 = 0."""
        image = torch.rand(1, 256, 256, 3)
        delta_e = calculate_ciede2000(image, image)
        
        assert delta_e.item() == pytest.approx(0.0, abs=0.01)
    
    def test_color_shift_positive_delta(self):
        """Color-shifted images should have positive ΔE00."""
        reference = torch.ones(1, 256, 256, 3) * 0.5  # Gray
        shifted = reference.clone()
        shifted[:, :, :, 0] += 0.1  # Add red
        
        delta_e = calculate_ciede2000(shifted.clamp(0,1), reference)
        
        assert delta_e.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
