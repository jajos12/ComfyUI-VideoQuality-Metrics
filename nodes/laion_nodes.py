import torch
import logging
from core.laion_aesthetic import calculate_laion_aesthetic_score, is_laion_available

logger = logging.getLogger("ComfyUI-VideoQuality-Metrics")

class VQ_LAIONAestheticScore:
    """
    Node for calculating LAION Aesthetic Score (1-10) using CLIP+MLP.
    This is a "real" aesthetic predictor trained on human ratings.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # Can be single image or video batch
            },
            "optional": {
                "sample_frames": ("INT", {"default": 8, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("aesthetic_score", "score_summary")
    FUNCTION = "calculate_score"
    CATEGORY = "VideoQualityMetrics/Aesthetic"

    def calculate_score(self, image, sample_frames=8):
        if not is_laion_available():
            logger.error("Transformers library not found. Please run install_dependencies.py")
            return (0.0, "Error: Transformers/CLIP missing")

        logger.info(f"Calculating LAION Aesthetic Score for batch: {image.shape}")
        
        try:
            result = calculate_laion_aesthetic_score(
                video=image,
                sample_frames=sample_frames
            )
            
            score = result['aesthetic_score']
            
            # Interpret score
            if score >= 6.5: interpretation = "Excellent"
            elif score >= 5.5: interpretation = "Good"
            elif score >= 4.5: interpretation = "Average"
            else: interpretation = "Poor"
            
            summary = (
                f"LAION Aesthetic Score: {score:.2f} / 10\n"
                f"Rating: {interpretation}"
            )
            
            logger.info(f"Score: {score:.2f} ({interpretation})")
            
            return (score, summary)
            
        except Exception as e:
            logger.error(f"LAION calculation failed: {e}")
            return (0.0, f"Error: {str(e)}")

# Node mappings
NODE_CLASS_MAPPINGS = {
    "VQ_LAIONAestheticScore": VQ_LAIONAestheticScore
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_LAIONAestheticScore": "LAION Aesthetic Score (V2)"
}
