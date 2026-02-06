"""
Logging utility for ComfyUI-VideoQuality-Metrics.
"""
import logging
import sys

def get_logger(name: str = "VideoQualityMetrics"):
    """Get a configured logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(name)s] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # Default level
    
    return logger
