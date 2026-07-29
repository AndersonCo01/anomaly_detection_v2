"""
Model inference module.

Loads a trained PatchCore model and runs anomaly detection on
individual images. Returns anomaly scores, binary predictions,
and heatmaps showing defect localization.

In Anomalib v2, the Lightning model includes a pre_processor
(handles resize + normalize) and post_processor (handles score
normalization + thresholding). We pass raw images and let the
full pipeline run.

References: https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html
"""
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from anomalib.models import Patchcore

from anomaly_detection.utils import (
    setup_logging,
    PATCHCORE_MODEL_DIR,
    IMAGE_SIZE,
)

logger = setup_logging(__name__)


class DefectDetector:
    """
    Anomaly-based defect detector using PatchCore.

    Loads a trained PatchCore checkpoint and provides a simple
    predict() interface for single-image inference.

    Usage:
        detector = DefectDetector()
        result = detector.predict(image)
        print(result["label"])        # "Defective" or "Good"
        print(result["score"])        # anomaly score (float)
        print(result["heatmap"])      # (H, W) anomaly heatmap
    """

class DefectDetector:

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threshold: float = 0.5,
    ):
        self.device = torch.device("cpu")

        self.model_dir = Path(model_dir) if model_dir else PATCHCORE_MODEL_DIR
        self.threshold = threshold
        self.model: Optional[Patchcore] = None
        self._inference_count = 0

        logger.info(
            "DefectDetector initialized | model_dir=%s | threshold=%.2f | device=%s",
            self.model_dir,
            self.threshold,
            self.device,
        )

    def load_model(self) -> None:
        """
        Load the PatchCore model from checkpoint."""
        start = time.perf_counter()
        ckpt_files = list(self.model_dir.rglob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"No checkpoint found in {self.model_dir}. "
                f"Run 'python -m anomaly_detection.train' first."
            )

        ckpt_path = ckpt_files[0]
        logger.info("Loading model from %s", ckpt_path)

        self.model = Patchcore.load_from_checkpoint(
            str(ckpt_path),
            weights_only=False,
        )

        self.model.to(self.device)
        self.model.eval()

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Model loaded | time=%.0fms", elapsed_ms)

    def predict(self, image: np.ndarray) -> dict:
        """
        Run anomaly detection on a single image.

        Args:
            image: RGB numpy array (H, W, 3), uint8.

        Returns:
            Dictionary containing:
            - label
            - score
            - heatmap
            - latency_ms
        """
        if self.model is None:
            self.load_model()

        start = time.perf_counter()

        # Convert raw image to tensor: (H, W, 3) uint8 -> (1, 3, H, W) float [0, 1]
        # Do NOT apply ImageNet normalization here — the model's
        # built-in pre_processor handles resize + normalize.
        tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        # Run full pipeline: pre_processor -> model -> post_processor
        if hasattr(self.model, "pre_processor"):
            tensor = self.model.pre_processor(tensor)

        print("Input device:", tensor.device)
        print("Model device:", next(self.model.parameters()).device)

        with torch.no_grad():
            output = self.model(tensor)
              

        # Extract results from Anomalib v2 InferenceBatch
        score = self._extract_score(output)
        heatmap = self._extract_heatmap(output)

        # Normalize heatmap to [0, 1] for display
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        label = "Defective" if score > self.threshold else "Good"

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._inference_count += 1

        logger.info(
            "Inference #%d | label=%s | score=%.4f | threshold=%.2f | time=%.1fms",
            self._inference_count,
            label,
            score,
            self.threshold,
            elapsed_ms,
        )

        return {
            "label": label,
            "score": score,
            "heatmap": heatmap,
            "latency_ms": elapsed_ms,
        }

    @staticmethod
    def _extract_score(output) -> float:
        """Extract anomaly score from model output, handling API variations."""
        # Anomalib v2: InferenceBatch with pred_score attribute
        if hasattr(output, "pred_score"):
            val = output.pred_score
            if isinstance(val, torch.Tensor):
                return float(val.cpu().item() if val.numel() == 1 else val.cpu().squeeze().item())
            return float(val)

        # Dict-style output (older versions)
        if isinstance(output, dict):
            for key in ("pred_score", "pred_scores"):
                if key in output:
                    val = output[key]
                    if isinstance(val, torch.Tensor):
                        return float(val.cpu().item())
                    return float(val)

        logger.warning("Could not extract score from output type: %s", type(output))
        return 0.0

    @staticmethod
    def _extract_heatmap(output) -> np.ndarray:
        """Extract anomaly heatmap from model output, handling API variations."""
        heatmap_tensor = None

        if hasattr(output, "anomaly_map"):
            heatmap_tensor = output.anomaly_map
        elif isinstance(output, dict):
            heatmap_tensor = output.get("anomaly_map") or output.get("anomaly_maps")

        if heatmap_tensor is not None and isinstance(heatmap_tensor, torch.Tensor):
            return heatmap_tensor.squeeze().cpu().numpy()

        logger.warning("Could not extract heatmap from output type: %s", type(output))
        return np.zeros((256, 256), dtype=np.float32)
        
    def predict_from_file(self, image_path: str | Path) -> dict:
        """
        Convenience method to predict from a file path.

        Args:
            image_path: Path to an image file.

        Returns:
            Same dict as predict(), with added 'source_path' key.
        """
        image_path = Path(image_path)

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self.predict(image)
        result["source_path"] = str(image_path)

        return result