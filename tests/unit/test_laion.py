"""
Unit tests for LAION Aesthetic Predictor.

Uses mocking to simulate Transformers/CLIP availability and verify:
1. Input shape handling.
2. MLP forward pass logic.
3. Fallback handling when libs missing.
"""
import torch
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Fix relative imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Core imports
from core.laion_aesthetic import calculate_laion_aesthetic_score, AestheticPredictor, is_laion_available

class MockBatchEncoding(dict):
    def to(self, device):
        return self

class TestLAIONAesthetic:
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock transformers and weight download."""
        with patch("core.laion_aesthetic._get_clip_model") as mock_clip, \
             patch("core.laion_aesthetic._load_aesthetic_model") as mock_mlp, \
             patch("core.laion_aesthetic.is_laion_available", return_value=True):
            
            # Setup Mock CLIP
            clip_model = MagicMock()
            # features [1, 768]
            clip_model.get_image_features.return_value = torch.rand(1, 768)
            
            processor = MagicMock()
            # Return custom dict that supports .to()
            processor.return_value = MockBatchEncoding({"pixel_values": torch.rand(1, 3, 224, 224)})
            
            mock_clip.return_value = (clip_model, processor)
            
            # Setup Mock MLP
            mlp_model = MagicMock()
            # Score [1, 1]
            mlp_model.return_value = torch.tensor([[7.5]])
            mock_mlp.return_value = mlp_model
            
            yield mock_clip, mock_mlp

    def test_calculate_score_shape(self, mock_dependencies):
        """Test calculation with correct shape."""
        # [B, T, H, W, C]
        video = torch.rand(1, 4, 224, 224, 3)
        
        res = calculate_laion_aesthetic_score(video)
        
        assert "aesthetic_score" in res
        assert res["aesthetic_score"] == 7.5
        assert len(res["per_frame_scores"]) == 4

    def test_calculate_score_fallback(self):
        """Test fallback when import fails."""
        with patch("core.laion_aesthetic._get_clip_model", side_effect=ImportError):
            res = calculate_laion_aesthetic_score(torch.rand(1, 1, 224, 224, 3))
            assert res['aesthetic_score'] == 0.0
            assert "error" in res

    def test_mlp_structure(self):
        """Test MLP architecture definition."""
        model = AestheticPredictor(input_dim=768)
        # Check first layer
        assert isinstance(model.layers[0], torch.nn.Linear)
        assert model.layers[0].in_features == 768
        assert model.layers[0].out_features == 1024
        
        # Check last layer
        assert model.layers[-1].out_features == 1

if __name__ == "__main__":
    pytest.main([__file__])
