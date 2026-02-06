"""
ComfyUI Nodes for FAST-VQA Quality Assessment.

Provides:
- VQ_FASTVQAScore: Efficient no-reference video quality scoring
"""

import torch
from typing import Dict, Any, Tuple
try:
    from ..utils.logger import get_logger
except (ImportError, ValueError):
    from utils.logger import get_logger

logger = get_logger()

# Import core FAST-VQA functions
try:
    from ..core.fast_vqa import (
        calculate_fastvqa_quality,
        is_fastvqa_available
    )
except (ImportError, ValueError):
    from core.fast_vqa import (
        calculate_fastvqa_quality,
        is_fastvqa_available
    )


class VQ_FASTVQAScore:
    """
    FAST-VQA based no-reference video quality assessment.
    
    Uses Grid Mini-patch Sampling (GMS) for efficient evaluation
    of high-resolution videos. No reference video required.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("quality_score", "summary",)
    FUNCTION = "calculate"
    CATEGORY = "Video Quality/FAST-VQA"
    
    def calculate(self, video: torch.Tensor) -> Tuple[float, str]:
        if not is_fastvqa_available():
            return (0.0, "⚠️ FAST-VQA not available")
        
        result = calculate_fastvqa_quality(video=video)
        
        score = result["quality_score"]
        num_fragments = result["num_fragments"]
        grid_size = result["grid_size"]
        
        logger.info(f"FAST-VQA Score: {score:.4f}, Fragments: {num_fragments} (Input shape: {video.shape})")
        
        # Interpretation
        if score >= 0.7:
            quality = "High quality"
        elif score >= 0.5:
            quality = "Moderate quality"
        else:
            quality = "Low quality"
        
        summary = (
            f"⚡ FAST-VQA Quality Score\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Score: {score:.3f} ({quality})\n"
            f"Fragments: {num_fragments} ({grid_size}×{grid_size} grid)\n"
            f"Mode: No-Reference (internal)"
        )
        
        return (score, summary)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "VQ_FASTVQAScore": VQ_FASTVQAScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_FASTVQAScore": "VQ FAST-VQA Score (No-Reference)",
}
