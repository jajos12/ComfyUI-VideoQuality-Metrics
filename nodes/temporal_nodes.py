"""
Temporal consistency nodes for ComfyUI.
"""

from ..core.temporal import (
    calculate_warping_error,
    calculate_temporal_flickering,
    calculate_motion_smoothness
)
try:
    from ..utils.logger import get_logger
except (ImportError, ValueError):
    from utils.logger import get_logger

logger = get_logger()


class VQ_TemporalConsistency:
    """
    Analyze temporal consistency of video frames.
    
    Outputs warping error and flickering scores.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",),
            },
            "optional": {
                "bidirectional_flow": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("warping_error", "flickering_score", "summary")
    FUNCTION = "analyze"
    CATEGORY = "VideoQuality/Temporal"

    def analyze(self, video_frames, bidirectional_flow=True):
        # video_frames: [T, H, W, C] from ComfyUI
        video = video_frames.unsqueeze(0)  # Add batch dim: [1, T, H, W, C]
        
        warp_result = calculate_warping_error(video, bidirectional=bidirectional_flow)
        flicker_result = calculate_temporal_flickering(video)
        
        warping_error = warp_result['warping_error'].item()
        flickering_score = flicker_result['flickering_score'].item()
        
        logger.info(f"Temporal Consistency - Warping: {warping_error:.4f}, Flickering: {flickering_score:.4f} (Input shape: {video.shape})")
        
        summary = (
            f"Temporal Consistency Analysis\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Warping Error: {warping_error:.4f}\n"
            f"Flickering Score: {flickering_score:.4f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Lower warping error = better temporal consistency\n"
            f"Lower flickering = more stable brightness"
        )
        
        return (warping_error, flickering_score, summary)


class VQ_MotionSmoothness:
    """
    Analyze motion smoothness of video frames.
    
    Measures jerk (motion acceleration) to detect unnatural movements.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("smoothness_score", "mean_jerk", "summary")
    FUNCTION = "analyze"
    CATEGORY = "VideoQuality/Temporal"

    def analyze(self, video_frames):
        video = video_frames.unsqueeze(0)
        
        result = calculate_motion_smoothness(video)
        
        smoothness = result['smoothness_score'].item()
        jerk = result['mean_jerk'].item()
        
        logger.info(f"Motion Smoothness - Score: {smoothness:.4f}, Jerk: {jerk:.4f} (Input shape: {video.shape})")
        
        # Interpret the score
        if smoothness > 0.8:
            quality = "Excellent - Very smooth motion"
        elif smoothness > 0.6:
            quality = "Good - Minor jitter detected"
        elif smoothness > 0.4:
            quality = "Fair - Noticeable motion artifacts"
        else:
            quality = "Poor - Jerky, unnatural motion"
        
        summary = (
            f"Motion Smoothness Analysis\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Smoothness Score: {smoothness:.4f}\n"
            f"Mean Jerk: {jerk:.4f}\n"
            f"Assessment: {quality}"
        )
        
        return (smoothness, jerk, summary)


NODE_CLASS_MAPPINGS = {
    "VQ_TemporalConsistency": VQ_TemporalConsistency,
    "VQ_MotionSmoothness": VQ_MotionSmoothness,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_TemporalConsistency": "VQ Temporal Consistency (Warping/Flickering)",
    "VQ_MotionSmoothness": "VQ Motion Smoothness",
}
