"""
DepthAnythingModel — depth estimation wrapper (Depth-Anything).

Conforms to `BaseToolModel`. Used by gen_tpose for foreground / background
separation (combined with white-bg suppression).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from models.tools.base import BaseToolModel


class DepthAnythingModel(BaseToolModel):
    """Thin wrapper around HuggingFace `AutoModelForDepthEstimation`."""

    def _load(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(self.model_path)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def predict(self, image: Image.Image, **kwargs) -> np.ndarray:
        """
        Run depth prediction.

        Args:
            image: RGB PIL image.

        Returns:
            HxW float32 numpy array (depth, resized to the original image size).
        """
        self._ensure_loaded()
        img = np.array(image.convert("RGB"))
        h, w = img.shape[:2]
        inputs = {
            k: v.to(self.device)
            for k, v in self.processor(images=img, return_tensors="pt").items()
        }
        depth = self.model(**inputs).predicted_depth
        depth = F.interpolate(
            depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
        return depth
