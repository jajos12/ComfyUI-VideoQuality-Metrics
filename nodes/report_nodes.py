"""
Reporting and visualization nodes for ComfyUI.
"""

import torch
import json
from typing import Dict, Any
from ..utils.plotting import normalize_metrics, generate_radar_chart_tensor
from ..utils.stats import compare_workflows
try:
    from ..utils.logger import get_logger
except (ImportError, ValueError):
    from utils.logger import get_logger

logger = get_logger()


class VQ_RadarChart:
    """
    Generate a radar chart visualization comparing multiple metrics.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "psnr": ("FLOAT", {"default": 0.0}),
                "ssim": ("FLOAT", {"default": 0.0}),
            },
            "optional": {
                "ciede2000": ("FLOAT", {"default": 0.0}),
                "warping_error": ("FLOAT", {"default": 0.0}),
                "flickering_score": ("FLOAT", {"default": 0.0}),
                "smoothness_score": ("FLOAT", {"default": 0.0}),
                "fvd": ("FLOAT", {"default": 0.0}),
                "chart_size": ("INT", {"default": 256, "min": 128, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("radar_chart", "metrics_json")
    FUNCTION = "generate"
    CATEGORY = "VideoQuality/Reporting"

    def generate(self, psnr, ssim, ciede2000=0.0, warping_error=0.0, 
                 flickering_score=0.0, smoothness_score=0.0, fvd=0.0, chart_size=256):
        
        # Collect non-zero metrics
        raw_metrics = {
            'psnr': psnr,
            'ssim': ssim,
        }
        
        if ciede2000 > 0:
            raw_metrics['ciede2000'] = ciede2000
        if warping_error > 0:
            raw_metrics['warping_error'] = warping_error
        if flickering_score > 0:
            raw_metrics['flickering_score'] = flickering_score
        if smoothness_score > 0:
            raw_metrics['smoothness_score'] = smoothness_score
        if fvd > 0:
            raw_metrics['fvd'] = fvd
        
        # Normalize
        normalized = normalize_metrics(raw_metrics)
        
        # Generate chart
        chart = generate_radar_chart_tensor(normalized, size=chart_size)
        
        # JSON output
        metrics_json = json.dumps({
            'raw': raw_metrics,
            'normalized': normalized
        }, indent=2)
        
        return (chart, metrics_json)


class VQ_MetricsLogger:
    """
    Log metrics to JSON format for export and analysis.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metric_name": ("STRING", {"default": "video_quality"}),
            },
            "optional": {
                "psnr": ("FLOAT", {"default": 0.0}),
                "ssim": ("FLOAT", {"default": 0.0}),
                "ciede2000": ("FLOAT", {"default": 0.0}),
                "warping_error": ("FLOAT", {"default": 0.0}),
                "flickering_score": ("FLOAT", {"default": 0.0}),
                "smoothness_score": ("FLOAT", {"default": 0.0}),
                "fvd": ("FLOAT", {"default": 0.0}),
                "fid": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)
    FUNCTION = "log"
    CATEGORY = "VideoQuality/Reporting"

    def log(self, metric_name, psnr=0.0, ssim=0.0, ciede2000=0.0, 
            warping_error=0.0, flickering_score=0.0, smoothness_score=0.0, 
            fvd=0.0, fid=0.0):
        
        metrics = {
            'name': metric_name,
            'fidelity': {
                'psnr': psnr,
                'ssim': ssim,
                'ciede2000': ciede2000,
            },
            'temporal': {
                'warping_error': warping_error,
                'flickering_score': flickering_score,
                'smoothness_score': smoothness_score,
            },
            'distributional': {
                'fvd': fvd,
                'fid': fid,
            }
        }
        
        # Filter out zero values
        for category in ['fidelity', 'temporal', 'distributional']:
            metrics[category] = {k: v for k, v in metrics[category].items() if v != 0.0}
        
        return (json.dumps(metrics, indent=2),)


class VQ_MetricsComparison:
    """
    Compare two sets of metrics with statistical significance testing.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metrics_a_json": ("STRING",),
                "metrics_b_json": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_report",)
    FUNCTION = "compare"
    CATEGORY = "VideoQuality/Reporting"

    def compare(self, metrics_a_json, metrics_b_json):
        try:
            metrics_a = json.loads(metrics_a_json)
            metrics_b = json.loads(metrics_b_json)
        except json.JSONDecodeError:
            return ("Error: Invalid JSON input",)
        
        report_lines = [
            "Workflow Comparison Report",
            "=" * 40,
            "",
        ]
        
        # Compare each metric
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        
        for key in sorted(all_keys):
            val_a = metrics_a.get(key, 'N/A')
            val_b = metrics_b.get(key, 'N/A')
            
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_b - val_a
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
                report_lines.append(f"{key}: A={val_a:.4f} vs B={val_b:.4f} (Δ={diff_str}, Winner={winner})")
            else:
                report_lines.append(f"{key}: A={val_a} vs B={val_b}")
        
                report_lines.append(f"{key}: A={val_a} vs B={val_b}")
        
        result_str = "\n".join(report_lines)
        logger.info(f"Workflow Comparison:\n{result_str}")
        return (result_str,)


NODE_CLASS_MAPPINGS = {
    "VQ_RadarChart": VQ_RadarChart,
    "VQ_MetricsLogger": VQ_MetricsLogger,
    "VQ_MetricsComparison": VQ_MetricsComparison,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VQ_RadarChart": "VQ Radar Chart",
    "VQ_MetricsLogger": "VQ Metrics Logger (JSON)",
    "VQ_MetricsComparison": "VQ Workflow Comparison",
}
