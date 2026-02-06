"""
ComfyUI Nodes for CLIP-based Quality Assessment.

Provides:
- VQ_CLIPAestheticScore: Frame-level aesthetic quality scoring
- VQ_TextVideoAlignment: Prompt adherence scoring for videos
"""

import torch
from typing import Dict, Any, Tuple
try:
    from ..utils.logger import get_logger
except (ImportError, ValueError):
    from utils.logger import get_logger

logger = get_logger()

# Import core CLIP functions
try:
    from ..core.clip_iqa import (
        calculate_clip_aesthetic_score,
        calculate_text_video_alignment,
        is_clip_available
    )
except (ImportError, ValueError):
    from core.clip_iqa import (
        calculate_clip_aesthetic_score,
        calculate_text_video_alignment,
        is_clip_available
    )


class VQ_CLIPAestheticScore:
    """
    Calculates aesthetic quality score for images/video frames using CLIP.
    
    Uses CLIP's understanding of visual quality to score frames on a 0-1 scale.
    Higher scores indicate more aesthetically pleasing images.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("aesthetic_score", "summary",)
    FUNCTION = "calculate"
    CATEGORY = "Video Quality/CLIP"
    
    def calculate(self, images: torch.Tensor) -> Tuple[float, str]:
        if not is_clip_available():
            return (
                0.0,
                "⚠️ CLIP not available. Install: pip install transformers"
            )
        
        result = calculate_clip_aesthetic_score(images)
        score = result["aesthetic_score"]
        per_image = result["per_image_scores"]
        
        logger.info(f"CLIP Aesthetic Score: {score:.4f} (Input shape: {images.shape})")
        
        # Interpretation
        if score >= 0.7:
            quality = "High aesthetic quality"
        elif score >= 0.5:
            quality = "Moderate aesthetic quality"
        else:
            quality = "Low aesthetic quality"
        
        summary = (
            f"🎨 CLIP Aesthetic Score\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Overall: {score:.3f} ({quality})\n"
            f"Frames analyzed: {len(per_image)}\n"
            f"Score range: [{min(per_image):.3f}, {max(per_image):.3f}]"
        )
        
        return (score, summary)


class VQ_TextVideoAlignment:
    """
    Measures how well a video matches a text prompt using CLIP.
    
    Essential for evaluating text-to-video generation quality.
    Detects "prompt drift" where the video diverges from the original description.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "sample_frames": ("INT", {"default": 8, "min": 1, "max": 32}),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("alignment_score", "summary",)
    FUNCTION = "calculate"
    CATEGORY = "Video Quality/CLIP"
    
    def calculate(
        self,
        video: torch.Tensor,
        prompt: str,
        sample_frames: int = 8
    ) -> Tuple[float, str]:
        if not is_clip_available():
            return (
                0.0,
                "⚠️ CLIP not available. Install: pip install transformers"
            )
        
        if not prompt.strip():
            return (0.0, "⚠️ No prompt provided for alignment check.")
        
        result = calculate_text_video_alignment(
            video=video,
            prompt=prompt,
            sample_frames=sample_frames
        )
        
        score = result["alignment_score"]
        per_frame = result["per_frame_scores"]
        indices = result["frame_indices"]
        
        logger.info(f"CLIP Text-Video Alignment: {score:.4f} (Prompt: '{prompt[:30]}...', Frames: {len(per_frame)})")
        
        # Interpretation
        if score >= 0.7:
            alignment = "Strong alignment"
        elif score >= 0.5:
            alignment = "Moderate alignment"
        else:
            alignment = "Weak alignment (possible prompt drift)"
        
        # Detect temporal drift
        if len(per_frame) >= 3:
            drift = per_frame[-1] - per_frame[0]
            if drift < -0.1:
                drift_msg = f"⚠️ Temporal drift detected: {drift:+.3f}"
            else:
                drift_msg = f"Temporal stability: {drift:+.3f}"
        else:
            drift_msg = ""
        
        summary = (
            f"📝 Text-Video Alignment\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Prompt: \"{prompt[:50]}{'...' if len(prompt) > 50 else ''}\"\n"
            f"Alignment: {score:.3f} ({alignment})\n"
            f"Frames sampled: {len(per_frame)}\n"
            f"{drift_msg}"
        )
        
        return (score, summary)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "VQ_CLIPAestheticScore": VQ_CLIPAestheticScore,
    "VQ_TextVideoAlignment": VQ_TextVideoAlignment,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_CLIPAestheticScore": "VQ CLIP Aesthetic Score",
    "VQ_TextVideoAlignment": "VQ Text-Video Alignment",
}
