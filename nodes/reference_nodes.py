from ..core.fidelity import calculate_psnr, calculate_ssim
from ..core.color import calculate_ciede2000
try:
    from ..utils.logger import get_logger
except (ImportError, ValueError):
    from utils.logger import get_logger

logger = get_logger()

class VQ_FullReferenceMetrics:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("psnr", "ssim", "ciede2000", "summary")
    FUNCTION = "measure"
    CATEGORY = "VideoQuality"

    def measure(self, images, reference):
        if images.shape != reference.shape:
            return (0.0, 0.0, 0.0, "Error: Image dimensions do not match.")

        psnr_val = calculate_psnr(images, reference).item()
        ssim_val = calculate_ssim(images, reference).item()
        ciede_val = calculate_ciede2000(images, reference).item()
        
        logger.info(f"Full-Reference Metrics - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, CIEDE2000: {ciede_val:.2f} (Input shape: {images.shape})")
        
        summary = f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}\nCIEDE2000: {ciede_val:.2f}"
        
        return (psnr_val, ssim_val, ciede_val, summary)

NODE_CLASS_MAPPINGS = {
    "VQ_FullReferenceMetrics": VQ_FullReferenceMetrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_FullReferenceMetrics": "VQ Full-Reference Metrics (PSNR/SSIM)",
}
