from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .token_pruning import (
    IdentityAwareTokenPruner,
    PreparedImage,
    TokenSelection,
)


@dataclass(frozen=True)
class EncodedImage:
    descriptor: np.ndarray
    total_tokens: int
    retained_tokens: int
    actual_retention: float


@dataclass(frozen=True)
class RetrievalCandidate:
    label: str
    path: Path
    score: float


@dataclass(frozen=True)
class GalleryRecord:
    label: str
    path: Path
    descriptor: np.ndarray


class BudgetedVisualEncoder:
    """Create fixed-size descriptors from variable token budgets."""

    def __init__(
        self,
        pruner: IdentityAwareTokenPruner
        | None = None,
    ) -> None:
        if pruner is None:
            pruner = (
                IdentityAwareTokenPruner()
            )

        self.pruner = pruner

    def prepare(
        self,
        image_path: str | Path,
    ) -> PreparedImage:
        return self.pruner.prepare(
            image_path
        )

    def encode(
        self,
        image_path: str | Path,
        retention: float,
    ) -> EncodedImage:
        prepared = self.prepare(
            image_path
        )

        return self.encode_prepared(
            prepared,
            retention,
        )

    def encode_prepared(
        self,
        prepared: PreparedImage,
        retention: float,
    ) -> EncodedImage:
        selection = self.pruner.select(
            prepared,
            retention,
        )

        descriptor = self._descriptor(
            selection
        )

        return EncodedImage(
            descriptor=descriptor,
            total_tokens=(
                selection.total_tokens
            ),
            retained_tokens=(
                selection.retained_tokens
            ),
            actual_retention=(
                selection.actual_retention
            ),
        )

    @staticmethod
    def _descriptor(
        selection: TokenSelection,
    ) -> np.ndarray:
        features = np.asarray(
            [
                token.feature
                for token
                in selection.tokens
            ],
            dtype=np.float32,
        )

        global_mean = np.mean(
            features,
            axis=0,
        )

        global_std = np.std(
            features,
            axis=0,
        )

        band_features: list[
            np.ndarray
        ] = []

        for band in range(3):
            selected = [
                token.feature
                for token
                in selection.tokens
                if BudgetedVisualEncoder._band(
                    token.y_center
                ) == band
            ]

            if selected:
                band_mean = np.mean(
                    np.asarray(
                        selected,
                        dtype=np.float32,
                    ),
                    axis=0,
                )
            else:
                band_mean = np.zeros(
                    features.shape[1],
                    dtype=np.float32,
                )

            band_features.append(
                band_mean
            )

        descriptor = np.concatenate(
            [
                global_mean,
                global_std,
                *band_features,
            ]
        ).astype(
            np.float32
        )

        return descriptor

    @staticmethod
    def _band(
        y_center: float,
    ) -> int:
        if y_center < 1.0 / 3.0:
            return 0

        if y_center < 2.0 / 3.0:
            return 1

        return 2


class GalleryIndex:
    """Precomputed gallery descriptor index."""

    def __init__(
        self,
        records: tuple[
            GalleryRecord,
            ...
        ],
    ) -> None:
        if len(records) < 2:
            raise ValueError(
                "Gallery must contain at least two records."
            )

        self.records = records

    @classmethod
    def build(
        cls,
        images: Mapping[
            str,
            str | Path,
        ],
        encoder: BudgetedVisualEncoder,
    ) -> "GalleryIndex":
        records: list[
            GalleryRecord
        ] = []

        for label, path_value in images.items():
            path = Path(
                path_value
            )

            encoded = encoder.encode(
                path,
                retention=1.0,
            )

            records.append(
                GalleryRecord(
                    label=label,
                    path=path,
                    descriptor=(
                        encoded.descriptor
                    ),
                )
            )

        return cls(
            tuple(records)
        )

    def search(
        self,
        descriptor: np.ndarray,
        top_k: int,
    ) -> list[
        RetrievalCandidate
    ]:
        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        results: list[
            RetrievalCandidate
        ] = []

        for record in self.records:
            if (
                record.descriptor.shape
                != descriptor.shape
            ):
                raise ValueError(
                    "Descriptor shape mismatch."
                )

            difference = (
                descriptor
                - record.descriptor
            )

            rms_distance = float(
                np.sqrt(
                    np.mean(
                        difference
                        * difference
                    )
                )
            )

            score = math.exp(
                -6.0
                * rms_distance
            )

            results.append(
                RetrievalCandidate(
                    label=record.label,
                    path=record.path,
                    score=float(
                        np.clip(
                            score,
                            0.0,
                            1.0,
                        )
                    ),
                )
            )

        results.sort(
            key=lambda candidate: (
                candidate.score,
                candidate.label,
            ),
            reverse=True,
        )

        return results[
            : min(
                top_k,
                len(results),
            )
        ]
