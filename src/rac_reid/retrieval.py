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
    VisualToken,
)


@dataclass(frozen=True)
class EncodedImage:
    descriptor: np.ndarray
    total_tokens: int
    retained_tokens: int
    actual_retention: float
    new_tokens_processed: int


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


class ProgressiveDescriptorState:
    """Incrementally aggregate newly restored visual tokens."""

    FEATURE_SIZE = 8

    def __init__(
        self,
        prepared: PreparedImage,
    ) -> None:
        self.prepared = prepared

        self._processed: set[
            tuple[int, int]
        ] = set()

        self._count = 0

        self._sum = np.zeros(
            self.FEATURE_SIZE,
            dtype=np.float64,
        )

        self._sum_sq = np.zeros(
            self.FEATURE_SIZE,
            dtype=np.float64,
        )

        self._band_count = [
            0,
            0,
            0,
        ]

        self._band_sum = [
            np.zeros(
                self.FEATURE_SIZE,
                dtype=np.float64,
            )
            for _ in range(3)
        ]

    @property
    def processed_tokens(self) -> int:
        return self._count

    def update(
        self,
        selection: TokenSelection,
    ) -> EncodedImage:
        selected_keys = {
            (
                token.row,
                token.column,
            )
            for token in selection.tokens
        }

        if not self._processed.issubset(
            selected_keys
        ):
            raise ValueError(
                "Progressive selections may only add tokens."
            )

        new_tokens = [
            token
            for token in selection.tokens
            if (
                token.row,
                token.column,
            )
            not in self._processed
        ]

        for token in new_tokens:
            self._add_token(
                token
            )

        return EncodedImage(
            descriptor=self._descriptor(),
            total_tokens=selection.total_tokens,
            retained_tokens=selection.retained_tokens,
            actual_retention=selection.actual_retention,
            new_tokens_processed=len(new_tokens),
        )

    def _add_token(
        self,
        token: VisualToken,
    ) -> None:
        key = (
            token.row,
            token.column,
        )

        if key in self._processed:
            return

        feature = np.asarray(
            token.feature,
            dtype=np.float64,
        )

        self._processed.add(
            key
        )

        self._count += 1

        self._sum += feature
        self._sum_sq += feature * feature

        band = self._band(
            token.y_center
        )

        self._band_count[
            band
        ] += 1

        self._band_sum[
            band
        ] += feature

    def _descriptor(
        self,
    ) -> np.ndarray:
        if self._count < 1:
            raise RuntimeError(
                "No tokens have been processed."
            )

        global_mean = (
            self._sum
            / self._count
        )

        variance = (
            self._sum_sq
            / self._count
            - global_mean * global_mean
        )

        global_std = np.sqrt(
            np.maximum(
                variance,
                0.0,
            )
        )

        band_features: list[
            np.ndarray
        ] = []

        for index in range(3):
            count = self._band_count[
                index
            ]

            if count:
                mean = (
                    self._band_sum[
                        index
                    ]
                    / count
                )
            else:
                mean = np.zeros(
                    self.FEATURE_SIZE,
                    dtype=np.float64,
                )

            band_features.append(
                mean
            )

        return np.concatenate(
            [
                global_mean,
                global_std,
                *band_features,
            ]
        ).astype(
            np.float32
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


class BudgetedVisualEncoder:
    """Create adaptive descriptors while reusing prior token computation."""

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

    def start_progressive(
        self,
        prepared: PreparedImage,
    ) -> ProgressiveDescriptorState:
        return ProgressiveDescriptorState(
            prepared
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
        state = (
            self.start_progressive(
                prepared
            )
        )

        return self.encode_progressive(
            state,
            retention,
        )

    def encode_progressive(
        self,
        state: ProgressiveDescriptorState,
        retention: float,
    ) -> EncodedImage:
        selection = (
            self.pruner.select(
                state.prepared,
                retention,
            )
        )

        return state.update(
            selection
        )


class GalleryIndex:
    """Precomputed full-budget gallery descriptor index."""

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
