
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthEstimate:
    """Depth sampled from an image region.

    distance_m is populated only for explicitly metric depth maps.
    Relative model predictions must never be interpreted as metres.
    """

    distance_m: float | None
    confidence: float
    raw_value: float | None = None
    depth_type: str = "unknown"
    spread: float | None = None
    valid_fraction: float = 0.0
    sample_count: int = 0


class DepthEstimator:
    """Robustly sample a supplied depth map.

    This class does not generate depth predictions or calibrate a model.
    """

    VALID_TYPES = {"metric", "relative", "unknown"}

    def estimate_point(
        self,
        depth_map: np.ndarray | None,
        x: int,
        y: int,
        *,
        depth_type: str = "unknown",
        confidence_map: np.ndarray | None = None,
    ) -> DepthEstimate:
        if depth_map is None:
            return self._empty(depth_type)

        depth = np.asarray(depth_map)
        if depth.ndim != 2:
            raise ValueError("depth_map must be a 2D array.")

        height, width = depth.shape
        if not (0 <= x < width and 0 <= y < height):
            return self._empty(depth_type)

        return self.estimate_region(
            depth,
            x1=x - 2,
            y1=y - 2,
            x2=x + 3,
            y2=y + 3,
            depth_type=depth_type,
            confidence_map=confidence_map,
        )

    def estimate_region(
        self,
        depth_map: np.ndarray | None,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        depth_type: str = "unknown",
        confidence_map: np.ndarray | None = None,
        inset_fraction: float = 0.15,
    ) -> DepthEstimate:
        """Estimate robust depth inside an image bounding box.

        Coordinates use x2/y2 as exclusive bounds. An inset reduces
        contamination from box edges and nearby background pixels.
        Confidence here describes sample validity, not model accuracy.
        """
        if depth_type not in self.VALID_TYPES:
            raise ValueError(
                "depth_type must be 'metric', 'relative', or 'unknown'."
            )

        if not 0.0 <= inset_fraction < 0.5:
            raise ValueError("inset_fraction must be in [0, 0.5).")

        if depth_map is None:
            return self._empty(depth_type)

        depth = np.asarray(depth_map, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth_map must be a non-empty 2D array.")

        height, width = depth.shape

        x1 = int(np.clip(x1, 0, width))
        x2 = int(np.clip(x2, 0, width))
        y1 = int(np.clip(y1, 0, height))
        y2 = int(np.clip(y2, 0, height))

        if x2 <= x1 or y2 <= y1:
            return self._empty(depth_type)

        box_width = x2 - x1
        box_height = y2 - y1

        inset_x = int(box_width * inset_fraction)
        inset_y = int(box_height * inset_fraction)

        sx1 = min(x1 + inset_x, x2 - 1)
        sx2 = max(sx1 + 1, x2 - inset_x)
        sy1 = min(y1 + inset_y, y2 - 1)
        sy2 = max(sy1 + 1, y2 - inset_y)

        region = depth[sy1:sy2, sx1:sx2]
        valid = np.isfinite(region)

        if depth_type == "metric":
            valid &= region > 0

        confidence_values = None
        if confidence_map is not None:
            confidence = np.asarray(confidence_map, dtype=np.float32)
            if confidence.shape != depth.shape:
                raise ValueError(
                    "confidence_map must have the same shape as depth_map."
                )

            confidence_region = confidence[sy1:sy2, sx1:sx2]
            valid &= np.isfinite(confidence_region)
            valid &= confidence_region > 0

            confidence_values = np.clip(
                confidence_region[valid], 0.0, 1.0
            )

        values = region[valid]
        valid_fraction = float(valid.mean())

        if values.size == 0:
            return DepthEstimate(
                distance_m=None,
                confidence=0.0,
                depth_type=depth_type,
                valid_fraction=valid_fraction,
            )

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))

        if confidence_values is None:
            # Sample validity is not a measure of model correctness.
            sample_confidence = 0.0
        else:
            sample_confidence = float(np.median(confidence_values))

        return DepthEstimate(
            distance_m=median if depth_type == "metric" else None,
            confidence=sample_confidence,
            raw_value=median,
            depth_type=depth_type,
            spread=mad,
            valid_fraction=valid_fraction,
            sample_count=int(values.size),
        )

    @staticmethod
    def _empty(depth_type: str) -> DepthEstimate:
        if depth_type not in DepthEstimator.VALID_TYPES:
            raise ValueError(
                "depth_type must be 'metric', 'relative', or 'unknown'."
            )

        return DepthEstimate(
            distance_m=None,
            confidence=0.0,
            depth_type=depth_type,
        )
