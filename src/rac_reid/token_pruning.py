from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class VisualToken:
    row: int
    column: int
    y_center: float
    x_center: float
    saliency: float
    feature: tuple[float, ...]


@dataclass(frozen=True)
class PreparedImage:
    tokens: tuple[VisualToken, ...]
    rows: int
    columns: int

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class TokenSelection:
    tokens: tuple[VisualToken, ...]
    total_tokens: int
    requested_retention: float

    @property
    def retained_tokens(self) -> int:
        return len(self.tokens)

    @property
    def actual_retention(self) -> float:
        if self.total_tokens == 0:
            return 0.0

        return self.retained_tokens / self.total_tokens


class IdentityAwareTokenPruner:
    """Create patch tokens and preserve identity-relevant body regions."""

    def __init__(
        self,
        width: int = 128,
        height: int = 256,
        patch_size: int = 16,
    ) -> None:
        if width < patch_size or height < patch_size:
            raise ValueError(
                "Canvas must be larger than one patch."
            )

        if width % patch_size != 0:
            raise ValueError(
                "Width must be divisible by patch_size."
            )

        if height % patch_size != 0:
            raise ValueError(
                "Height must be divisible by patch_size."
            )

        self.width = width
        self.height = height
        self.patch_size = patch_size

    def prepare(
        self,
        image_path: str | Path,
    ) -> PreparedImage:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(
                opened
            ).convert("RGB")

            image = ImageOps.fit(
                image,
                (
                    self.width,
                    self.height,
                ),
                method=Image.Resampling.LANCZOS,
            )

        rgb = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        rows = (
            self.height
            // self.patch_size
        )

        columns = (
            self.width
            // self.patch_size
        )

        tokens: list[VisualToken] = []

        for row in range(rows):
            y0 = (
                row
                * self.patch_size
            )

            y1 = (
                y0
                + self.patch_size
            )

            for column in range(columns):
                x0 = (
                    column
                    * self.patch_size
                )

                x1 = (
                    x0
                    + self.patch_size
                )

                patch = rgb[
                    y0:y1,
                    x0:x1,
                    :
                ]

                mean_rgb = np.mean(
                    patch,
                    axis=(0, 1),
                )

                std_rgb = np.std(
                    patch,
                    axis=(0, 1),
                )

                gray = (
                    0.299
                    * patch[:, :, 0]
                    + 0.587
                    * patch[:, :, 1]
                    + 0.114
                    * patch[:, :, 2]
                )

                edge = (
                    self._edge_activity(
                        gray
                    )
                )

                luminance = float(
                    np.mean(gray)
                )

                feature = (
                    float(mean_rgb[0]),
                    float(mean_rgb[1]),
                    float(mean_rgb[2]),
                    float(std_rgb[0]),
                    float(std_rgb[1]),
                    float(std_rgb[2]),
                    luminance,
                    edge,
                )

                x_center = (
                    column + 0.5
                ) / columns

                y_center = (
                    row + 0.5
                ) / rows

                center_prior = max(
                    0.0,
                    1.0
                    - abs(
                        x_center
                        - 0.5
                    )
                    * 2.0,
                )

                texture = float(
                    np.mean(std_rgb)
                )

                saliency = (
                    0.55 * texture
                    + 0.30 * edge
                    + 0.15 * center_prior
                )

                tokens.append(
                    VisualToken(
                        row=row,
                        column=column,
                        y_center=y_center,
                        x_center=x_center,
                        saliency=float(
                            np.clip(
                                saliency,
                                0.0,
                                1.0,
                            )
                        ),
                        feature=feature,
                    )
                )

        return PreparedImage(
            tokens=tuple(tokens),
            rows=rows,
            columns=columns,
        )

    def select(
        self,
        prepared: PreparedImage,
        retention: float,
    ) -> TokenSelection:
        if not 0.0 < retention <= 1.0:
            raise ValueError(
                "retention must be greater than 0 and at most 1"
            )

        if prepared.total_tokens < 3:
            raise ValueError(
                "At least three tokens are required."
            )

        keep_count = max(
            3,
            int(
                math.ceil(
                    prepared.total_tokens
                    * retention
                )
            ),
        )

        keep_count = min(
            keep_count,
            prepared.total_tokens,
        )

        selected: set[int] = set()

        for band in range(3):
            candidates = [
                (
                    index,
                    token,
                )
                for index, token
                in enumerate(
                    prepared.tokens
                )
                if self._band(
                    token.y_center
                ) == band
                and 0.20
                <= token.x_center
                <= 0.80
            ]

            if not candidates:
                candidates = [
                    (
                        index,
                        token,
                    )
                    for index, token
                    in enumerate(
                        prepared.tokens
                    )
                    if self._band(
                        token.y_center
                    ) == band
                ]

            anchor_index, _ = max(
                candidates,
                key=lambda item: (
                    item[1].saliency,
                    -abs(
                        item[1].x_center
                        - 0.5
                    ),
                ),
            )

            selected.add(
                anchor_index
            )

        ranked = sorted(
            enumerate(
                prepared.tokens
            ),
            key=lambda item: (
                item[1].saliency,
                -abs(
                    item[1].x_center
                    - 0.5
                ),
            ),
            reverse=True,
        )

        for index, _ in ranked:
            if len(selected) >= keep_count:
                break

            selected.add(index)

        ordered = tuple(
            prepared.tokens[index]
            for index in sorted(
                selected
            )
        )

        return TokenSelection(
            tokens=ordered,
            total_tokens=(
                prepared.total_tokens
            ),
            requested_retention=(
                retention
            ),
        )

    @staticmethod
    def _band(
        y_center: float,
    ) -> int:
        if y_center < 1.0 / 3.0:
            return 0

        if y_center < 2.0 / 3.0:
            return 1

        return 2

    @staticmethod
    def _edge_activity(
        gray: np.ndarray,
    ) -> float:
        comparisons = 0
        difference_sum = 0.0

        if gray.shape[1] >= 2:
            horizontal = np.abs(
                np.diff(
                    gray,
                    axis=1,
                )
            )

            difference_sum += float(
                np.sum(horizontal)
            )

            comparisons += int(
                horizontal.size
            )

        if gray.shape[0] >= 2:
            vertical = np.abs(
                np.diff(
                    gray,
                    axis=0,
                )
            )

            difference_sum += float(
                np.sum(vertical)
            )

            comparisons += int(
                vertical.size
            )

        if comparisons == 0:
            return 0.0

        return float(
            np.clip(
                difference_sum
                / comparisons,
                0.0,
                1.0,
            )
        )
