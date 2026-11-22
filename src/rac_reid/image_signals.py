from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageSignals:
    """Cheap signals measured directly from image pixels."""

    width: int
    height: int
    aspect_ratio: float
    brightness: float
    contrast: float
    visual_entropy: float
    blur: float
    edge_density: float
    foreground_ratio_estimate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ImageSignalExtractor:
    """CPU-friendly image-quality and complexity extractor."""

    def __init__(
        self,
        max_dimension: int = 512,
        edge_threshold: float = 24.0,
        foreground_threshold: float = 0.10,
        blur_scale: float = 2500.0,
    ) -> None:
        if max_dimension < 32:
            raise ValueError(
                "max_dimension must be at least 32"
            )

        if edge_threshold <= 0.0:
            raise ValueError(
                "edge_threshold must be positive"
            )

        if not 0.0 < foreground_threshold < 1.0:
            raise ValueError(
                "foreground_threshold must be between 0 and 1"
            )

        if blur_scale <= 0.0:
            raise ValueError(
                "blur_scale must be positive"
            )

        self.max_dimension = max_dimension
        self.edge_threshold = edge_threshold
        self.foreground_threshold = foreground_threshold
        self.blur_scale = blur_scale

    def extract(
        self,
        image_path: str | Path,
    ) -> ImageSignals:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(
                opened
            ).convert("RGB")

            width, height = image.size

            if width < 1 or height < 1:
                raise ValueError(
                    "Image has invalid dimensions"
                )

            sample = image.copy()

            sample.thumbnail(
                (
                    self.max_dimension,
                    self.max_dimension,
                ),
                Image.Resampling.LANCZOS,
            )

        rgb = np.asarray(
            sample,
            dtype=np.float32,
        )

        gray = (
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]
        )

        brightness = float(
            np.mean(gray) / 255.0
        )

        contrast = float(
            np.clip(
                np.std(gray) / 127.5,
                0.0,
                1.0,
            )
        )

        return ImageSignals(
            width=width,
            height=height,
            aspect_ratio=float(
                width / height
            ),
            brightness=self._unit(
                brightness
            ),
            contrast=self._unit(
                contrast
            ),
            visual_entropy=self._unit(
                self._entropy(gray)
            ),
            blur=self._unit(
                self._blur_score(gray)
            ),
            edge_density=self._unit(
                self._edge_density(gray)
            ),
            foreground_ratio_estimate=self._unit(
                self._foreground_ratio(rgb)
            ),
        )

    @staticmethod
    def _unit(
        value: float,
    ) -> float:
        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _entropy(
        gray: np.ndarray,
    ) -> float:
        gray_u8 = np.clip(
            np.rint(gray),
            0,
            255,
        ).astype(np.uint8)

        histogram = np.bincount(
            gray_u8.ravel(),
            minlength=256,
        ).astype(np.float64)

        total = histogram.sum()

        if total == 0:
            return 0.0

        probabilities = (
            histogram[histogram > 0]
            / total
        )

        entropy_bits = float(
            -np.sum(
                probabilities
                * np.log2(probabilities)
            )
        )

        return entropy_bits / 8.0

    def _blur_score(
        self,
        gray: np.ndarray,
    ) -> float:
        height, width = gray.shape

        if height < 3 or width < 3:
            return 1.0

        center = gray[1:-1, 1:-1]

        laplacian = (
            -4.0 * center
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )

        variance = float(
            np.var(laplacian)
        )

        return 1.0 / (
            1.0
            + variance / self.blur_scale
        )

    def _edge_density(
        self,
        gray: np.ndarray,
    ) -> float:
        height, width = gray.shape

        edge_pixels = 0
        comparisons = 0

        if width >= 2:
            horizontal = np.abs(
                np.diff(
                    gray,
                    axis=1,
                )
            )

            edge_pixels += int(
                np.count_nonzero(
                    horizontal
                    >= self.edge_threshold
                )
            )

            comparisons += int(
                horizontal.size
            )

        if height >= 2:
            vertical = np.abs(
                np.diff(
                    gray,
                    axis=0,
                )
            )

            edge_pixels += int(
                np.count_nonzero(
                    vertical
                    >= self.edge_threshold
                )
            )

            comparisons += int(
                vertical.size
            )

        if comparisons == 0:
            return 0.0

        return (
            edge_pixels
            / comparisons
        )

    def _foreground_ratio(
        self,
        rgb: np.ndarray,
    ) -> float:
        height, width, _ = rgb.shape

        border_width = max(
            1,
            min(height, width) // 20,
        )

        border = np.concatenate(
            (
                rgb[
                    :border_width,
                    :,
                    :
                ].reshape(-1, 3),
                rgb[
                    -border_width:,
                    :,
                    :
                ].reshape(-1, 3),
                rgb[
                    :,
                    :border_width,
                    :
                ].reshape(-1, 3),
                rgb[
                    :,
                    -border_width:,
                    :
                ].reshape(-1, 3),
            ),
            axis=0,
        )

        background = np.median(
            border,
            axis=0,
        )

        distance = np.linalg.norm(
            rgb - background,
            axis=2,
        )

        max_distance = math.sqrt(
            3.0 * (255.0 ** 2)
        )

        normalized = (
            distance / max_distance
        )

        foreground = (
            normalized
            >= self.foreground_threshold
        )

        return float(
            np.mean(foreground)
        )
