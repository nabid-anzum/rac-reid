from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence


class DifficultyLevel(str, Enum):
    """Query difficulty levels used for resource allocation."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class QuerySignals:
    """Cheap normalized signals measured before expensive ReID processing.

    All values use the inclusive range 0 to 1.

    Higher detector_confidence and foreground_ratio indicate an easier query.
    Higher blur, entropy, occlusion, and crowding indicate a harder query.
    """

    detector_confidence: float
    foreground_ratio: float
    blur: float
    visual_entropy: float
    occlusion: float
    crowding: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1; received {value}"
                )


@dataclass(frozen=True)
class ResourcePlan:
    """Compute budget assigned to a query-processing stage."""

    difficulty: DifficultyLevel
    crop_margin: float
    token_retention: float
    top_k: int
    encoder_depth: int
    vllm_eligible: bool


@dataclass(frozen=True)
class RoutingDecision:
    """Decision taken after initial gallery retrieval."""

    action: str
    reason: str
    top1: float
    margin: float
    next_plan: ResourcePlan | None = None


class ResourceController:
    """Assigns and progressively increases query-specific compute budgets."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self._weights = self._load_weights(config["difficulty_weights"])
        self._thresholds = self._load_thresholds(
            config["difficulty_thresholds"]
        )
        self._budgets = self._load_budgets(config["budgets"])
        self._early_exit = self._load_early_exit(config["early_exit"])

    @classmethod
    def from_json(cls, path: str | Path) -> "ResourceController":
        config_path = Path(path)

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        return cls(config)

    def difficulty_score(self, signals: QuerySignals) -> float:
        """Calculate a normalized query difficulty score."""

        components = {
            "low_detector_confidence":
                1.0 - signals.detector_confidence,
            "low_foreground_ratio":
                1.0 - signals.foreground_ratio,
            "blur":
                signals.blur,
            "visual_entropy":
                signals.visual_entropy,
            "occlusion":
                signals.occlusion,
            "crowding":
                signals.crowding,
        }

        score = sum(
            self._weights[name] * value
            for name, value in components.items()
        )

        return min(1.0, max(0.0, score))

    def classify(self, signals: QuerySignals) -> DifficultyLevel:
        """Classify a query as easy, medium, or hard."""

        score = self.difficulty_score(signals)

        if score <= self._thresholds["easy_max"]:
            return DifficultyLevel.EASY

        if score <= self._thresholds["medium_max"]:
            return DifficultyLevel.MEDIUM

        return DifficultyLevel.HARD

    def initial_plan(self, signals: QuerySignals) -> ResourcePlan:
        """Return the first compute plan for a query."""

        return self.plan_for(self.classify(signals))

    def plan_for(self, difficulty: DifficultyLevel) -> ResourcePlan:
        """Return the configured budget for a difficulty level."""

        budget = self._budgets[difficulty.value]

        return ResourcePlan(
            difficulty=difficulty,
            crop_margin=float(budget["crop_margin"]),
            token_retention=float(budget["token_retention"]),
            top_k=int(budget["top_k"]),
            encoder_depth=int(budget["encoder_depth"]),
            vllm_eligible=bool(budget["vllm_eligible"]),
        )

    def route_after_retrieval(
        self,
        scores: Sequence[float],
        current_plan: ResourcePlan,
    ) -> RoutingDecision:
        """Choose early exit, refinement, or VLLM semantic reasoning."""

        if len(scores) < 2:
            raise ValueError(
                "At least two retrieval scores are required"
            )

        if any(not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError(
                "Retrieval scores must be between 0 and 1"
            )

        ranked = sorted(scores, reverse=True)
        top1 = ranked[0]
        top2 = ranked[1]
        margin = top1 - top2

        confident = (
            top1 >= self._early_exit["min_top1"]
            and margin >= self._early_exit["min_margin"]
        )

        if confident:
            return RoutingDecision(
                action="early_exit",
                reason=(
                    "Top-1 score and Top-1/Top-2 margin "
                    "meet the confidence thresholds."
                ),
                top1=top1,
                margin=margin,
            )

        if current_plan.difficulty is DifficultyLevel.EASY:
            return RoutingDecision(
                action="refine",
                reason=(
                    "Easy-stage retrieval is ambiguous; "
                    "allocate the medium compute budget."
                ),
                top1=top1,
                margin=margin,
                next_plan=self.plan_for(DifficultyLevel.MEDIUM),
            )

        if current_plan.difficulty is DifficultyLevel.MEDIUM:
            return RoutingDecision(
                action="refine",
                reason=(
                    "Medium-stage retrieval is ambiguous; "
                    "restore more tokens and deepen processing."
                ),
                top1=top1,
                margin=margin,
                next_plan=self.plan_for(DifficultyLevel.HARD),
            )

        if current_plan.vllm_eligible:
            return RoutingDecision(
                action="invoke_vllm",
                reason=(
                    "Hard-stage retrieval remains ambiguous; "
                    "request semantic reranking."
                ),
                top1=top1,
                margin=margin,
            )

        return RoutingDecision(
            action="return_best_available",
            reason="No additional compute stage is available.",
            top1=top1,
            margin=margin,
        )

    @staticmethod
    def _load_weights(raw: object) -> dict[str, float]:
        if not isinstance(raw, dict):
            raise TypeError("difficulty_weights must be an object")

        required = {
            "low_detector_confidence",
            "low_foreground_ratio",
            "blur",
            "visual_entropy",
            "occlusion",
            "crowding",
        }

        if set(raw) != required:
            raise ValueError(
                "difficulty_weights has missing or unexpected keys"
            )

        weights = {
            name: float(value)
            for name, value in raw.items()
        }

        if any(value < 0.0 for value in weights.values()):
            raise ValueError("Difficulty weights cannot be negative")

        if abs(sum(weights.values()) - 1.0) > 1e-9:
            raise ValueError("Difficulty weights must sum to 1.0")

        return weights

    @staticmethod
    def _load_thresholds(raw: object) -> dict[str, float]:
        if not isinstance(raw, dict):
            raise TypeError("difficulty_thresholds must be an object")

        thresholds = {
            "easy_max": float(raw["easy_max"]),
            "medium_max": float(raw["medium_max"]),
        }

        if not (
            0.0
            <= thresholds["easy_max"]
            < thresholds["medium_max"]
            <= 1.0
        ):
            raise ValueError(
                "Thresholds must satisfy 0 <= easy < medium <= 1"
            )

        return thresholds

    @staticmethod
    def _load_budgets(
        raw: object,
    ) -> dict[str, dict[str, object]]:
        if not isinstance(raw, dict):
            raise TypeError("budgets must be an object")

        required = {
            level.value
            for level in DifficultyLevel
        }

        if set(raw) != required:
            raise ValueError(
                "budgets must contain easy, medium, and hard"
            )

        budgets: dict[str, dict[str, object]] = {}

        for level, value in raw.items():
            if not isinstance(value, dict):
                raise TypeError(
                    f"Budget {level} must be an object"
                )

            budget = {
                "crop_margin":
                    float(value["crop_margin"]),
                "token_retention":
                    float(value["token_retention"]),
                "top_k":
                    int(value["top_k"]),
                "encoder_depth":
                    int(value["encoder_depth"]),
                "vllm_eligible":
                    bool(value["vllm_eligible"]),
            }

            if not 0.0 <= budget["crop_margin"] <= 0.5:
                raise ValueError(
                    f"{level}.crop_margin is outside its range"
                )

            if not 0.0 < budget["token_retention"] <= 1.0:
                raise ValueError(
                    f"{level}.token_retention is outside its range"
                )

            if budget["top_k"] < 2:
                raise ValueError(
                    f"{level}.top_k must be at least 2"
                )

            if budget["encoder_depth"] < 1:
                raise ValueError(
                    f"{level}.encoder_depth must be at least 1"
                )

            budgets[level] = budget

        return budgets

    @staticmethod
    def _load_early_exit(raw: object) -> dict[str, float]:
        if not isinstance(raw, dict):
            raise TypeError("early_exit must be an object")

        values = {
            "min_top1": float(raw["min_top1"]),
            "min_margin": float(raw["min_margin"]),
        }

        if not 0.0 <= values["min_top1"] <= 1.0:
            raise ValueError(
                "early_exit.min_top1 must be between 0 and 1"
            )

        if not 0.0 <= values["min_margin"] <= 1.0:
            raise ValueError(
                "early_exit.min_margin must be between 0 and 1"
            )

        return values
