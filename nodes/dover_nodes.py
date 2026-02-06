"""
ComfyUI Nodes for DOVER Quality Assessment.

Provides:
- VQ_DOVERQuality: Disentangled aesthetic and technical video quality scoring
"""

import torch
from typing import Dict, Any, Tuple

# Import core DOVER functions
try:
    from ..core.dover import (
        calculate_dover_quality,
        is_dover_available
    )
except (ImportError, ValueError):
    from core.dover import (
        calculate_dover_quality,
        is_dover_available
    )


class VQ_DOVERQuality:
    """
    DOVER-based video quality assessment with disentangled scores.
    
    Provides separate aesthetic (composition, color) and technical 
    (sharpness, noise) quality scores based on SOTA VQA research.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 8, "min": 1, "max": 32}),
            },
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING",)
    RETURN_NAMES = ("aesthetic_score", "technical_score", "overall_score", "summary",)
    FUNCTION = "calculate"
    CATEGORY = "Video Quality/DOVER"
    
    def calculate(
        self,
        video: torch.Tensor,
        num_frames: int = 8
    ) -> Tuple[float, float, float, str]:
        if not is_dover_available():
            return (
                0.0, 0.0, 0.0,
                "⚠️ DOVER dependencies not available. Install torchvision>=0.15"
            )
        
        result = calculate_dover_quality(
            video=video,
            num_frames=num_frames
        )
        
        aesthetic = result["aesthetic_score"]
        technical = result["technical_score"]
        overall = result["overall_score"]
        
        # Interpretation
        def interpret_score(score: float, category: str) -> str:
            if score >= 0.7:
                return f"High {category}"
            elif score >= 0.5:
                return f"Moderate {category}"
            else:
                return f"Low {category}"
        
        aesthetic_interp = interpret_score(aesthetic, "aesthetic quality")
        technical_interp = interpret_score(technical, "technical quality")
        
        # Detect imbalance (common in AI videos)
        imbalance = abs(aesthetic - technical)
        if imbalance > 0.2:
            if aesthetic > technical:
                imbalance_msg = "⚠️ Aesthetic > Technical: May have artifacts despite good composition"
            else:
                imbalance_msg = "⚠️ Technical > Aesthetic: Sharp but uninteresting composition"
        else:
            imbalance_msg = "✓ Balanced aesthetic and technical quality"
        
        summary = (
            f"🎬 DOVER Quality Assessment\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Aesthetic: {aesthetic:.3f} ({aesthetic_interp})\n"
            f"Technical: {technical:.3f} ({technical_interp})\n"
            f"Overall:   {overall:.3f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{imbalance_msg}\n"
            f"Frames analyzed: {len(result['frame_indices'])}"
        )
        
        return (aesthetic, technical, overall, summary)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "VQ_DOVERQuality": VQ_DOVERQuality,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_DOVERQuality": "VQ DOVER Quality (Aesthetic + Technical)",
}
