"""
ComfyUI Video Quality Metrics

A comprehensive suite of video quality assessment metrics.
"""

# Import all node mappings from submodules
from .nodes.reference_nodes import (
    NODE_CLASS_MAPPINGS as REFERENCE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as REFERENCE_NAMES
)
from .nodes.temporal_nodes import (
    NODE_CLASS_MAPPINGS as TEMPORAL_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as TEMPORAL_NAMES
)
from .nodes.distribution_nodes import (
    NODE_CLASS_MAPPINGS as DISTRIBUTION_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as DISTRIBUTION_NAMES
)
from .nodes.report_nodes import (
    NODE_CLASS_MAPPINGS as REPORT_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as REPORT_NAMES
)
from .nodes.clip_nodes import (
    NODE_CLASS_MAPPINGS as CLIP_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as CLIP_NAMES
)
from .nodes.dover_nodes import (
    NODE_CLASS_MAPPINGS as DOVER_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as DOVER_NAMES
)
from .nodes.fast_vqa_nodes import (
    NODE_CLASS_MAPPINGS as FASTVQA_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as FASTVQA_NAMES
)

# Merge all node mappings
NODE_CLASS_MAPPINGS = {
    **REFERENCE_NODES,
    **TEMPORAL_NODES,
    **DISTRIBUTION_NODES,
    **REPORT_NODES,
    **CLIP_NODES,
    **DOVER_NODES,
    **FASTVQA_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **REFERENCE_NAMES,
    **TEMPORAL_NAMES,
    **DISTRIBUTION_NAMES,
    **REPORT_NAMES,
    **CLIP_NAMES,
    **DOVER_NAMES,
    **FASTVQA_NAMES,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']



