"""
This sub-package contains *tool* models — utility / auxiliary models that
support the main generation pipelines but are not themselves the primary
content generator. Examples:

- depth estimation (DepthAnything)
- foreground segmentation / background removal (RMBG-1.4, SAM, ...)
- matting / alpha refinement
- keypoint / pose detection helpers

All tool models inherit from `BaseToolModel` (see `base.py`) so they share a
consistent constructor, device handling, and `__call__` / `predict` API.
"""

from models.tools.base import BaseToolModel
from models.tools.depth_anything import DepthAnythingModel
from models.tools.rmbg import RMBGModel

__all__ = ["BaseToolModel", "DepthAnythingModel", "RMBGModel"]
