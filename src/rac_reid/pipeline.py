from __future__ import annotations

import time

from dataclasses import dataclass
from pathlib import Path

from .image_signals import (
    ImageSignalExtractor,
    ImageSignals,
)
from .resource_controller import (
    QuerySignals,
    ResourceController,
)
from .retrieval import (
    BudgetedVisualEncoder,
    GalleryIndex,
    RetrievalCandidate,
)


@dataclass(frozen=True)
class DetectorEvidence:
    confidence: float
    foreground_ratio: float
    occlusion: float
    crowding: float

    def __post_init__(
        self,
    ) -> None:
        values = {
            "confidence":
                self.confidence,
            "foreground_ratio":
                self.foreground_ratio,
            "occlusion":
                self.occlusion,
            "crowding":
                self.crowding,
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )


@dataclass(frozen=True)
class PipelineStage:
    stage_number: int
    difficulty: str
    token_retention: float
    total_tokens: int
    retained_tokens: int
    new_tokens_processed: int
    cumulative_tokens_processed: int
    top_k: int
    top1_score: float
    margin: float
    action: str
    reason: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineResult:
    initial_difficulty: str
    final_action: str
    image_signals: ImageSignals
    query_signals: QuerySignals
    stages: tuple[
        PipelineStage,
        ...
    ]
    candidates: tuple[
        RetrievalCandidate,
        ...
    ]

    @property
    def total_query_tokens_processed(
        self,
    ) -> int:
        return sum(
            stage.new_tokens_processed
            for stage in self.stages
        )

    @property
    def full_token_count(
        self,
    ) -> int:
        if not self.stages:
            return 0

        return self.stages[
            0
        ].total_tokens

    @property
    def token_saving_ratio(
        self,
    ) -> float:
        if self.full_token_count == 0:
            return 0.0

        return max(
            0.0,
            1.0
            - (
                self.total_query_tokens_processed
                / self.full_token_count
            ),
        )


class AdaptiveReIDPipeline:
    """Resource-adaptive ReID with incremental token restoration."""

    def __init__(
        self,
        controller: ResourceController,
        image_extractor: ImageSignalExtractor,
        encoder: BudgetedVisualEncoder,
        gallery: GalleryIndex,
    ) -> None:
        self.controller = controller
        self.image_extractor = (
            image_extractor
        )
        self.encoder = encoder
        self.gallery = gallery

    def run(
        self,
        query_image: str | Path,
        detector: DetectorEvidence,
    ) -> PipelineResult:
        image_signals = (
            self.image_extractor.extract(
                query_image
            )
        )

        query_signals = QuerySignals(
            detector_confidence=(
                detector.confidence
            ),
            foreground_ratio=(
                detector.foreground_ratio
            ),
            blur=image_signals.blur,
            visual_entropy=(
                image_signals.visual_entropy
            ),
            occlusion=(
                detector.occlusion
            ),
            crowding=(
                detector.crowding
            ),
        )

        initial_plan = (
            self.controller.initial_plan(
                query_signals
            )
        )

        plan = initial_plan

        prepared = (
            self.encoder.prepare(
                query_image
            )
        )

        progressive = (
            self.encoder.start_progressive(
                prepared
            )
        )

        stages: list[
            PipelineStage
        ] = []

        final_candidates: list[
            RetrievalCandidate
        ] = []

        cumulative_tokens = 0

        for stage_number in range(
            1,
            5,
        ):
            stage_started = (
                time.perf_counter()
            )

            encoded = (
                self.encoder.encode_progressive(
                    progressive,
                    plan.token_retention,
                )
            )

            cumulative_tokens += (
                encoded.new_tokens_processed
            )

            candidates = (
                self.gallery.search(
                    encoded.descriptor,
                    plan.top_k,
                )
            )

            if len(candidates) < 2:
                raise RuntimeError(
                    "At least two gallery candidates are required."
                )

            decision = (
                self.controller.route_after_retrieval(
                    [
                        candidate.score
                        for candidate
                        in candidates
                    ],
                    plan,
                )
            )

            stage_elapsed = (
                time.perf_counter()
                - stage_started
            )

            stages.append(
                PipelineStage(
                    stage_number=(
                        stage_number
                    ),
                    difficulty=(
                        plan.difficulty.value
                    ),
                    token_retention=(
                        plan.token_retention
                    ),
                    total_tokens=(
                        encoded.total_tokens
                    ),
                    retained_tokens=(
                        encoded.retained_tokens
                    ),
                    new_tokens_processed=(
                        encoded.new_tokens_processed
                    ),
                    cumulative_tokens_processed=(
                        cumulative_tokens
                    ),
                    top_k=(
                        plan.top_k
                    ),
                    top1_score=(
                        decision.top1
                    ),
                    margin=(
                        decision.margin
                    ),
                    action=(
                        decision.action
                    ),
                    reason=(
                        decision.reason
                    ),
                    elapsed_seconds=(
                        stage_elapsed
                    ),
                )
            )

            final_candidates = (
                candidates
            )

            if decision.action == "refine":
                if decision.next_plan is None:
                    raise RuntimeError(
                        "Refinement requested without next plan."
                    )

                plan = (
                    decision.next_plan
                )

                continue

            return PipelineResult(
                initial_difficulty=(
                    initial_plan
                    .difficulty
                    .value
                ),
                final_action=(
                    decision.action
                ),
                image_signals=(
                    image_signals
                ),
                query_signals=(
                    query_signals
                ),
                stages=tuple(
                    stages
                ),
                candidates=tuple(
                    final_candidates
                ),
            )

        raise RuntimeError(
            "Adaptive pipeline exceeded maximum stages."
        )
