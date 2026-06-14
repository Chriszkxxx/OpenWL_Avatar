"""
BaseToolModel — unified base class for all tool / utility models
(depth estimation, segmentation, background removal, matting, etc.).

Subclasses should override:
  - `_load()`        : load underlying model weights / processors
  - `predict(image)` : run inference and return the model's native output

The base class provides:
  - consistent `__init__(model_path, device)`
  - lazy-load via `_loaded` flag
  - convenient `__call__` alias for `predict`
  - `unload()` to release GPU memory
"""

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class BaseToolModel(ABC):
    """Abstract base class for tool / utility models."""

    def __init__(self, model_path: str, device: str = "cuda", lazy: bool = False):
        """
        Args:
            model_path: Local path or HuggingFace hub id of the model.
            device:     Inference device, e.g. "cuda" / "cpu".
            lazy:       If True, defer weight loading until first `predict` call.
        """
        self.model_path = model_path
        self.device = device
        self._loaded = False
        if not lazy:
            self._load()
            self._loaded = True

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _load(self) -> None:
        """Load weights / processors. Called once on init (or first call if lazy)."""

    @abstractmethod
    def predict(self, image: Image.Image, **kwargs) -> Any:
        """Run inference on a single PIL image."""

    # ------------------------------------------------------------------
    # Common utilities
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def __call__(self, image: Image.Image, **kwargs) -> Any:
        self._ensure_loaded()
        return self.predict(image, **kwargs)

    def unload(self) -> None:
        """Release GPU memory. Subclasses may override for custom cleanup."""
        for attr in ("model", "processor", "pipeline"):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except AttributeError:
                    pass
        self._loaded = False
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
