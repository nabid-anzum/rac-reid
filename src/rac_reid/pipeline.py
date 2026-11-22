from __future__ import annotations

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
    """Evidence that should originate from a person detector."""

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
    top_k: int
    top1_score: float
    margin: float
    action: str


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
            stage.retained_tokens
            for stage in self.stages
        )


class AdaptiveReIDPipeline:
    """Resource-adaptive query processing and retrieval."""

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

        stages: list[
            PipelineStage
        ] = []

        final_candidates: list[
            RetrievalCandidate
        ] = []

        for stage_number in range(
            1,
            5,
        ):
            encoded = (
                self.encoder.encode_prepared(
                    prepared,
                    plan.token_retention,
                )
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
                )
            )

            final_candidates = candidates

            if (
                decision.action
                == "refine"
            ):
                if (
                    decision.next_plan
                    is None
                ):
                    raise RuntimeError(
                        "Refinement requested without a next plan."
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
