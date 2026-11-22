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
        progress_callback=None,
    ) -> PipelineResult:
        def emit(
            code: str,
            title: str,
            detail: str,
            progress: int,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    code,
                    title,
                    detail,
                    progress,
                )

        emit(
            "adaptive_signal_analysis",
            "Analyzing adaptive crop",
            (
                "Recomputing image signals on the final "
                "adaptive person crop."
            ),
            46,
        )

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

        emit(
            "tokenization",
            "Tokenizing person crop",
            (
                "Splitting the normalized person image into "
                "visual-patch proxy units."
            ),
            49,
        )

        prepared = (
            self.encoder.prepare(
                query_image
            )
        )

        emit(
            "tokenization_complete",
            "Visual patches prepared",
            (
                f"Created {prepared.total_tokens} proxy patches "
                "before adaptive pruning."
            ),
            51,
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
            stage_base = min(
                79,
                52
                + (
                    stage_number - 1
                )
                * 9,
            )

            emit(
                (
                    f"stage_{stage_number}"
                    "_identity_pruning"
                ),
                (
                    f"Stage {stage_number}: "
                    "identity-aware pruning"
                ),
                (
                    f"Selecting the most useful "
                    f"{plan.token_retention:.0%} of proxy patches "
                    "while preserving upper-, middle- and "
                    "lower-body identity anchors."
                ),
                stage_base,
            )

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

            emit(
                (
                    f"stage_{stage_number}"
                    "_pruning_complete"
                ),
                (
                    f"Stage {stage_number}: "
                    "patch selection complete"
                ),
                (
                    f"Retained {encoded.retained_tokens}/"
                    f"{encoded.total_tokens}; "
                    f"processed {encoded.new_tokens_processed} "
                    "new patches; "
                    f"{cumulative_tokens} unique patches so far."
                ),
                min(
                    82,
                    stage_base + 2,
                ),
            )

            emit(
                (
                    f"stage_{stage_number}"
                    "_gallery_search"
                ),
                (
                    f"Stage {stage_number}: "
                    "searching gallery"
                ),
                (
                    f"Comparing the current descriptor against "
                    f"the gallery and requesting Top-{plan.top_k}."
                ),
                min(
                    84,
                    stage_base + 4,
                ),
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

            emit(
                (
                    f"stage_{stage_number}"
                    "_candidate_ranking"
                ),
                (
                    f"Stage {stage_number}: "
                    "ranking candidates"
                ),
                (
                    f"Current top match is "
                    f"{candidates[0].label} "
                    f"with proxy score "
                    f"{candidates[0].score:.3f}."
                ),
                min(
                    85,
                    stage_base + 5,
                ),
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

            emit(
                (
                    f"stage_{stage_number}"
                    "_confidence_routing"
                ),
                (
                    f"Stage {stage_number}: "
                    "confidence decision"
                ),
                (
                    f"Top-1={decision.top1:.3f}; "
                    f"margin={decision.margin:.3f}. "
                    f"{decision.reason}"
                ),
                min(
                    87,
                    stage_base + 7,
                ),
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

            final_candidates = candidates

            if decision.action == "refine":
                if decision.next_plan is None:
                    raise RuntimeError(
                        "Refinement requested without next plan."
                    )

                emit(
                    (
                        f"stage_{stage_number}"
                        "_refinement"
                    ),
                    "Progressive refinement",
                    (
                        "Confidence is insufficient. "
                        "Restoring additional proxy patches "
                        "instead of recomputing previous ones."
                    ),
                    min(
                        88,
                        stage_base + 8,
                    ),
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
