"""
Distributional quality nodes for ComfyUI.
"""

from ..core.distributional import calculate_fvd, calculate_fid, calculate_video_fid
try:
    from ..utils.logger import get_logger
except (ImportError, ValueError):
    from utils.logger import get_logger

logger = get_logger()


class VQ_FVD:
    """
    Calculate Fréchet Video Distance between two video batches.
    
    FVD measures distributional similarity in video feature space.
    Lower FVD = more similar video distributions.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_generated": ("IMAGE",),
                "video_reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("fvd", "summary")
    FUNCTION = "calculate"
    CATEGORY = "VideoQuality/Distributional"

    def calculate(self, video_generated, video_reference):
        # Add batch dimension if needed
        gen = video_generated.unsqueeze(0) if video_generated.dim() == 4 else video_generated
        ref = video_reference.unsqueeze(0) if video_reference.dim() == 4 else video_reference
        
        result = calculate_fvd(gen, ref)
        fvd_val = result['fvd'].item()
        
        logger.info(f"FVD Score: {fvd_val:.4f} (Input shapes: {gen.shape} vs {ref.shape})")
        
        # Interpret FVD score
        if fvd_val < 50:
            quality = "Excellent - Very similar to reference"
        elif fvd_val < 150:
            quality = "Good - Minor distributional differences"
        elif fvd_val < 300:
            quality = "Fair - Noticeable quality gap"
        else:
            quality = "Poor - Significant distributional divergence"
        
        summary = (
            f"Fréchet Video Distance (FVD)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"FVD Score: {fvd_val:.2f}\n"
            f"Assessment: {quality}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Lower FVD = closer to reference distribution"
        )
        
        return (fvd_val, summary)


class VQ_FID:
    """
    Calculate Fréchet Inception Distance for image batches.
    
    FID measures distributional similarity in image feature space.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_generated": ("IMAGE",),
                "images_reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("fid", "summary")
    FUNCTION = "calculate"
    CATEGORY = "VideoQuality/Distributional"

    def calculate(self, images_generated, images_reference):
        result = calculate_fid(images_generated, images_reference)
        fid_val = result['fid'].item()
        
        logger.info(f"FID Score: {fid_val:.4f} (Samples: {result['n_samples'][0]} vs {result['n_samples'][1]})")
        
        summary = (
            f"Fréchet Inception Distance (FID)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"FID Score: {fid_val:.2f}\n"
            f"Samples: {result['n_samples'][0]} vs {result['n_samples'][1]}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Lower FID = better image quality"
        )
        
        return (fid_val, summary)


class VQ_VideoFID:
    """
    Calculate frame-by-frame FID between two videos.
    
    Useful for comparing video-to-video transformations.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video1": ("IMAGE",),
                "video2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("mean_fid", "summary")
    FUNCTION = "calculate"
    CATEGORY = "VideoQuality/Distributional"

    def calculate(self, video1, video2):
        result = calculate_video_fid(video1, video2)
        mean_fid = result['video_fid']
        
        logger.info(f"Video Frame-by-Frame FID: {mean_fid:.4f} (Input shapes: {video1.shape} vs {video2.shape})")
        
        if 'per_frame_fid' in result:
            per_frame = result['per_frame_fid']
            frame_info = f"Per-frame range: {min(per_frame):.2f} - {max(per_frame):.2f}"
        else:
            frame_info = "Different video lengths - using pooled comparison"
        
        summary = (
            f"Video Frame-by-Frame FID\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Mean FID: {mean_fid:.2f}\n"
            f"{frame_info}"
        )
        
        return (mean_fid, summary)


NODE_CLASS_MAPPINGS = {
    "VQ_FVD": VQ_FVD,
    "VQ_FID": VQ_FID,
    "VQ_VideoFID": VQ_VideoFID,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_FVD": "VQ Fréchet Video Distance (FVD)",
    "VQ_FID": "VQ Fréchet Inception Distance (FID)",
    "VQ_VideoFID": "VQ Video Frame-by-Frame FID",
}
