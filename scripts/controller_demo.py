from __future__ import annotations

from pathlib import Path

from rac_reid import QuerySignals, ResourceController


ROOT = Path(__file__).resolve().parents[1]

CONTROLLER = ResourceController.from_json(
    ROOT / "configs" / "resource_policy.json"
)


def main() -> None:
    examples = {
        "easy": QuerySignals(
            detector_confidence=0.98,
            foreground_ratio=0.88,
            blur=0.08,
            visual_entropy=0.20,
            occlusion=0.05,
            crowding=0.05,
        ),
        "medium": QuerySignals(
            detector_confidence=0.82,
            foreground_ratio=0.63,
            blur=0.40,
            visual_entropy=0.55,
            occlusion=0.42,
            crowding=0.35,
        ),
        "hard": QuerySignals(
            detector_confidence=0.58,
            foreground_ratio=0.38,
            blur=0.75,
            visual_entropy=0.82,
            occlusion=0.88,
            crowding=0.72,
        ),
    }

    for label, signals in examples.items():
        score = CONTROLLER.difficulty_score(signals)
        plan = CONTROLLER.initial_plan(signals)

        print(
            f"{label:>6} | difficulty={score:.3f} | "
            f"class={plan.difficulty.value:<6} | "
            f"retain={plan.token_retention:.0%} | "
            f"top_k={plan.top_k:>2} | "
            f"depth={plan.encoder_depth} | "
            f"vllm_eligible={plan.vllm_eligible}"
        )

    easy_plan = CONTROLLER.initial_plan(examples["easy"])

    confident = CONTROLLER.route_after_retrieval(
        [0.94, 0.68, 0.61],
        easy_plan,
    )

    print(
        "confident retrieval -> "
        f"action={confident.action}, "
        f"top1={confident.top1:.2f}, "
        f"margin={confident.margin:.2f}"
    )

    ambiguous = CONTROLLER.route_after_retrieval(
        [0.82, 0.79, 0.77],
        easy_plan,
    )

    next_level = (
        ambiguous.next_plan.difficulty.value
        if ambiguous.next_plan
        else "none"
    )

    print(
        "ambiguous retrieval -> "
        f"action={ambiguous.action}, "
        f"next={next_level}"
    )


if __name__ == "__main__":
    main()
