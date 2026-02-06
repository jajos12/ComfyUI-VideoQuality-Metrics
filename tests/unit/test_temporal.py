"""
Unit tests for temporal consistency metrics.

Tests: calculate_motion_smoothness, calculate_warping_error, calculate_temporal_flickering
"""
import pytest
import torch

from core.temporal import (
    calculate_motion_smoothness,
    calculate_warping_error,
    calculate_temporal_flickering
)
from tests.fixtures.synthetic_videos import (
    create_static_video,
    create_smooth_video,
    create_jittery_video,
    create_flickering_video
)


class TestMotionSmoothness:
    """Tests for calculate_motion_smoothness function."""
    
    def test_static_video_perfect_smoothness(self):
        """Static video should have smoothness = 1.0 and jerk = 0."""
        video = create_static_video()
        result = calculate_motion_smoothness(video)
        
        assert result['smoothness_score'].item() == pytest.approx(1.0, abs=0.01)
        assert result['mean_jerk'].item() == pytest.approx(0.0, abs=0.001)
    
    def test_smooth_video_high_smoothness(self):
        """Smooth linear motion should have reasonable smoothness."""
        video = create_smooth_video()
        result = calculate_motion_smoothness(video)
        
        # Smooth motion should score > 0.2
        assert result['smoothness_score'].item() > 0.2
    
    def test_jittery_video_low_smoothness(self):
        """Jittery motion should have lower smoothness than smooth motion."""
        smooth_video = create_smooth_video()
        jittery_video = create_jittery_video()
        
        smooth_result = calculate_motion_smoothness(smooth_video)
        jittery_result = calculate_motion_smoothness(jittery_video)
        
        assert smooth_result['smoothness_score'] > jittery_result['smoothness_score']
    
    def test_ordering_static_smooth_jittery(self):
        """Smoothness should follow: static >= smooth >= jittery."""
        static = calculate_motion_smoothness(create_static_video())
        smooth = calculate_motion_smoothness(create_smooth_video())
        jittery = calculate_motion_smoothness(create_jittery_video())
        
        assert static['smoothness_score'] >= smooth['smoothness_score']
        assert smooth['smoothness_score'] >= jittery['smoothness_score']
    
    def test_edge_case_two_frames(self):
        """Videos with < 3 frames should return default values."""
        video = torch.rand(1, 2, 128, 128, 3)
        result = calculate_motion_smoothness(video)
        
        assert result['smoothness_score'].item() == 1.0
        assert result['mean_jerk'].item() == 0.0
    
    def test_score_range(self):
        """Smoothness score should always be in [0, 1]."""
        for video_fn in [create_static_video, create_smooth_video, create_jittery_video]:
            result = calculate_motion_smoothness(video_fn())
            score = result['smoothness_score'].item()
            assert 0.0 <= score <= 1.0


class TestTemporalFlickering:
    """Tests for calculate_temporal_flickering function."""
    
    def test_static_video_no_flickering(self):
        """Static brightness should have low flickering score."""
        video = create_static_video()
        result = calculate_temporal_flickering(video)
        
        assert result['flickering_score'].item() < 0.1
    
    def test_flickering_video_high_score(self):
        """Alternating brightness should have high flickering score."""
        video = create_flickering_video()
        result = calculate_temporal_flickering(video)
        
        # Flickering video should score higher than static
        static_result = calculate_temporal_flickering(create_static_video())
        assert result['flickering_score'] > static_result['flickering_score']


class TestWarpingError:
    """Tests for calculate_warping_error function."""
    
    def test_static_video_low_error(self):
        """Static video should have minimal warping error."""
        video = create_static_video()
        result = calculate_warping_error(video)
        
        # Static = no motion = low error
        assert result['warping_error'].item() < 0.1
    
    def test_flickering_higher_error_than_smooth(self):
        """Flickering video should have higher warping error than smooth motion."""
        smooth = calculate_warping_error(create_smooth_video())
        flickering = calculate_warping_error(create_flickering_video())
        
        # Flickering causes large photometric errors that warping cannot fix
        assert flickering['warping_error'] > smooth['warping_error']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
