from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rac_reid import (
    AdaptiveReIDPipeline,
    BudgetedVisualEncoder,
    DetectorEvidence,
    GalleryIndex,
    ImageSignalExtractor,
    ResourceController,
)


ROOT = Path(
    __file__
).resolve().parents[1]


def make_person(
    path: Path,
    color: tuple[
        int,
        int,
        int,
    ],
) -> None:
    array = np.full(
        (256, 128, 3),
        245,
        dtype=np.uint8,
    )

    array[
        30:80,
        48:80,
    ] = color

    array[
        80:185,
        38:90,
    ] = color

    darker = tuple(
        max(
            0,
            value - 45,
        )
        for value in color
    )

    array[
        185:235,
        42:62,
    ] = darker

    array[
        185:235,
        68:88,
    ] = darker

    Image.fromarray(
        array
    ).save(path)


def main() -> None:
    with tempfile.TemporaryDirectory(
        dir=ROOT
    ) as directory:
        temp = Path(
            directory
        )

        red = (
            temp
            / "person_red.png"
        )

        blue = (
            temp
            / "person_blue.png"
        )

        green = (
            temp
            / "person_green.png"
        )

        make_person(
            red,
            (190, 45, 45),
        )

        make_person(
            blue,
            (45, 70, 195),
        )

        make_person(
            green,
            (45, 175, 80),
        )

        encoder = (
            BudgetedVisualEncoder()
        )

        gallery = (
            GalleryIndex.build(
                {
                    "person-red":
                        red,
                    "person-blue":
                        blue,
                    "person-green":
                        green,
                },
                encoder,
            )
        )

        controller = (
            ResourceController.from_json(
                ROOT
                / "configs"
                / "resource_policy.json"
            )
        )

        pipeline = (
            AdaptiveReIDPipeline(
                controller=controller,
                image_extractor=(
                    ImageSignalExtractor()
                ),
                encoder=encoder,
                gallery=gallery,
            )
        )

        result = pipeline.run(
            red,
            DetectorEvidence(
                confidence=0.98,
                foreground_ratio=0.55,
                occlusion=0.05,
                crowding=0.05,
            ),
        )

        print(
            "Initial difficulty:",
            result.initial_difficulty,
        )

        print()

        for stage in (
            result.stages
        ):
            print(
                f"Stage {stage.stage_number}: "
                f"class={stage.difficulty} "
                f"retention={stage.token_retention:.0%} "
                f"tokens={stage.retained_tokens}/"
                f"{stage.total_tokens} "
                f"top_k={stage.top_k} "
                f"top1={stage.top1_score:.3f} "
                f"margin={stage.margin:.3f} "
                f"action={stage.action}"
            )

        print()
        print(
            "Final action:",
            result.final_action,
        )

        print(
            "Total query tokens processed:",
            result.total_query_tokens_processed,
        )

        print()
        print("Top candidates:")

        for candidate in (
            result.candidates[:3]
        ):
            print(
                f"  {candidate.label:14} "
                f"score={candidate.score:.3f}"
            )


if __name__ == "__main__":
    main()
