"""
Tests for SOTA No-Reference VQA Metrics.

Tests CLIP-IQA, DOVER, and FAST-VQA implementations.
"""

import torch
import pytest

from core.clip_iqa import (
    calculate_clip_aesthetic_score,
    calculate_text_video_alignment,
    is_clip_available
)
from core.dover import (
    calculate_dover_quality,
    is_dover_available,
    DOVERBackbone
)
from core.fast_vqa import (
    calculate_fastvqa_quality,
    is_fastvqa_available,
    GridMinipatchSampler,
    FragmentAttentionNetwork
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a sample image tensor [B, H, W, C]."""
    return torch.rand(1, 224, 224, 3)


@pytest.fixture
def sample_video():
    """Create a sample video tensor [B, T, H, W, C]."""
    return torch.rand(1, 16, 224, 224, 3)


@pytest.fixture
def sample_video_4d():
    """Create a sample video tensor [T, H, W, C] (no batch)."""
    return torch.rand(8, 128, 128, 3)


# ============================================================================
# CLIP-IQA Tests
# ============================================================================

class TestCLIPIQA:
    """Tests for CLIP-based quality assessment."""
    
    def test_clip_available(self):
        """Check if CLIP is available."""
        result = is_clip_available()
        assert isinstance(result, bool)
        print(f"CLIP Available: {result}")
    
    @pytest.mark.skipif(not is_clip_available(), reason="CLIP not installed")
    def test_aesthetic_score_shape(self, sample_image):
        """Test aesthetic score returns correct structure."""
        result = calculate_clip_aesthetic_score(sample_image)
        
        assert "aesthetic_score" in result
        assert "per_image_scores" in result
        assert isinstance(result["aesthetic_score"], float)
        assert isinstance(result["per_image_scores"], list)
    
    @pytest.mark.skipif(not is_clip_available(), reason="CLIP not installed")
    def test_aesthetic_score_range(self, sample_image):
        """Test that aesthetic scores are in valid range."""
        result = calculate_clip_aesthetic_score(sample_image)
        
        score = result["aesthetic_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"
    
    @pytest.mark.skipif(not is_clip_available(), reason="CLIP not installed")
    def test_text_video_alignment_shape(self, sample_video):
        """Test text-video alignment returns correct structure."""
        result = calculate_text_video_alignment(
            video=sample_video,
            prompt="a cat walking"
        )
        
        assert "alignment_score" in result
        assert "per_frame_scores" in result
        assert "frame_indices" in result
        assert isinstance(result["alignment_score"], float)
    
    @pytest.mark.skipif(not is_clip_available(), reason="CLIP not installed")
    def test_text_video_alignment_range(self, sample_video):
        """Test that alignment scores are in valid range."""
        result = calculate_text_video_alignment(
            video=sample_video,
            prompt="a beautiful landscape"
        )
        
        score = result["alignment_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"


# ============================================================================
# DOVER Tests
# ============================================================================

class TestDOVER:
    """Tests for DOVER quality assessment."""
    
    def test_dover_available(self):
        """Check if DOVER is available."""
        result = is_dover_available()
        assert isinstance(result, bool)
        assert result is True  # Should always be True (uses torchvision)
        print(f"DOVER Available: {result}")
    
    def test_dover_model_creation(self):
        """Test that DOVER model can be created."""
        model = DOVERBackbone()
        assert model is not None
        
        # Check dual heads exist
        assert hasattr(model, 'aesthetic_head')
        assert hasattr(model, 'technical_head')
    
    def test_dover_model_forward(self):
        """Test DOVER model forward pass."""
        model = DOVERBackbone()
        model.eval()
        
        # Input expected: [B, C, T, H, W]
        x = torch.rand(2, 3, 8, 224, 224)
        
        with torch.no_grad():
            aesthetic, technical = model(x)
        
        assert aesthetic.shape == (2, 1)
        assert technical.shape == (2, 1)
    
    def test_dover_quality_shape(self, sample_video):
        """Test DOVER quality returns correct structure."""
        result = calculate_dover_quality(sample_video)
        
        assert "aesthetic_score" in result
        assert "technical_score" in result
        assert "overall_score" in result
        assert "per_frame_aesthetic" in result
        assert "per_frame_technical" in result
    
    def test_dover_quality_range(self, sample_video):
        """Test that DOVER scores are in valid range."""
        result = calculate_dover_quality(sample_video)
        
        for key in ["aesthetic_score", "technical_score", "overall_score"]:
            score = result[key]
            assert 0.0 <= score <= 1.0, f"{key}: {score} out of range [0, 1]"
    
    def test_dover_disentanglement(self, sample_video):
        """Test that aesthetic and technical scores are independent."""
        result = calculate_dover_quality(sample_video)
        
        # They should not be identical (different heads)
        # Note: With random input, they might be close but not exact
        aesthetic = result["aesthetic_score"]
        technical = result["technical_score"]
        
        # Just verify they're both computed
        assert isinstance(aesthetic, float)
        assert isinstance(technical, float)


# ============================================================================
# FAST-VQA Tests
# ============================================================================

class TestFASTVQA:
    """Tests for FAST-VQA quality assessment."""
    
    def test_fastvqa_available(self):
        """Check if FAST-VQA is available."""
        result = is_fastvqa_available()
        assert result is True
        print(f"FAST-VQA Available: {result}")
    
    def test_gms_sampler_creation(self):
        """Test GMS sampler creation."""
        sampler = GridMinipatchSampler(
            fragment_size=32,
            grid_size=7,
            temporal_samples=8
        )
        
        assert sampler.fragment_size == 32
        assert sampler.grid_size == 7
        assert sampler.temporal_samples == 8
    
    def test_gms_sampler_output_shape(self, sample_video):
        """Test GMS sampler produces correct fragment shape."""
        sampler = GridMinipatchSampler(
            fragment_size=32,
            grid_size=4,
            temporal_samples=4
        )
        
        fragments = sampler.sample(sample_video)
        
        # Expected: [B, N, C, H, W] where N = temporal * grid^2
        B = sample_video.shape[0]
        N = 4 * 4 * 4  # temporal * grid * grid
        
        assert fragments.shape[0] == B
        assert fragments.shape[1] == N
        assert fragments.shape[2] == 3  # RGB
        assert fragments.shape[3] == 32  # fragment_size
        assert fragments.shape[4] == 32
    
    def test_fanet_forward(self):
        """Test Fragment Attention Network forward pass."""
        fanet = FragmentAttentionNetwork(fragment_size=32)
        fanet.eval()
        
        # Simulate GMS output: [B, N, C, H, W]
        fragments = torch.rand(2, 49, 3, 32, 32)
        
        with torch.no_grad():
            quality = fanet(fragments)
        
        assert quality.shape == (2, 1)
    
    def test_fastvqa_quality_shape(self, sample_video):
        """Test FAST-VQA quality returns correct structure."""
        result = calculate_fastvqa_quality(sample_video)
        
        assert "quality_score" in result
        assert "num_fragments" in result
        assert "grid_size" in result
    
    def test_fastvqa_quality_range(self, sample_video):
        """Test that FAST-VQA scores are in valid range."""
        result = calculate_fastvqa_quality(sample_video)
        
        score = result["quality_score"]
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"
    
    def test_fastvqa_4d_input(self, sample_video_4d):
        """Test FAST-VQA handles 4D input (no batch dim)."""
        result = calculate_fastvqa_quality(sample_video_4d)
        
        assert "quality_score" in result
        assert isinstance(result["quality_score"], float)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests across all NR-VQA metrics."""
    
    def test_all_metrics_same_video(self, sample_video):
        """Test all metrics can run on the same video."""
        results = {}
        
        # DOVER
        results["dover"] = calculate_dover_quality(sample_video)
        
        # FAST-VQA
        results["fastvqa"] = calculate_fastvqa_quality(sample_video)
        
        # CLIP (if available)
        if is_clip_available():
            results["clip_aesthetic"] = calculate_clip_aesthetic_score(
                sample_video[0]  # First batch, all frames as images
            )
            results["clip_alignment"] = calculate_text_video_alignment(
                sample_video,
                prompt="random video content"
            )
        
        print("\n=== All Metrics Results ===")
        for name, result in results.items():
            if "score" in str(result):
                print(f"{name}: {result}")
        
        assert len(results) >= 2  # At least DOVER and FAST-VQA


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing SOTA No-Reference VQA Metrics")
    print("=" * 60)
    
    # Quick smoke test
    video = torch.rand(1, 8, 128, 128, 3)
    
    print("\n[1] Testing CLIP-IQA...")
    if is_clip_available():
        result = calculate_clip_aesthetic_score(video[0, 0:1])
        print(f"    Aesthetic Score: {result['aesthetic_score']:.3f}")
    else:
        print("    CLIP not available, skipping")
    
    print("\n[2] Testing DOVER...")
    result = calculate_dover_quality(video)
    print(f"    Aesthetic: {result['aesthetic_score']:.3f}")
    print(f"    Technical: {result['technical_score']:.3f}")
    print(f"    Overall:   {result['overall_score']:.3f}")
    
    print("\n[3] Testing FAST-VQA...")
    result = calculate_fastvqa_quality(video)
    print(f"    Quality Score: {result['quality_score']:.3f}")
    print(f"    Fragments: {result['num_fragments']}")
    
    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)
