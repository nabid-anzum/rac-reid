from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rac_reid import (
    AdaptiveReIDPipeline,
    BudgetedVisualEncoder,
    DetectorEvidence,
    GalleryIndex,
    IdentityAwareTokenPruner,
    ImageSignalExtractor,
    ResourceController,
)


ROOT = Path(
    __file__
).resolve().parents[1]


def create_person_image(
    path: Path,
    color: tuple[
        int,
        int,
        int,
    ],
    offset: int = 0,
) -> None:
    array = np.full(
        (256, 128, 3),
        245,
        dtype=np.uint8,
    )

    array[
        35:85,
        48 + offset:
        80 + offset,
    ] = color

    array[
        85:185,
        38 + offset:
        90 + offset,
    ] = color

    darker = tuple(
        max(
            0,
            channel - 45,
        )
        for channel in color
    )

    array[
        185:235,
        42 + offset:
        62 + offset,
    ] = darker

    array[
        185:235,
        68 + offset:
        88 + offset,
    ] = darker

    Image.fromarray(
        array
    ).save(path)


class AdaptivePipelineTests(
    unittest.TestCase
):
    def test_token_budget_changes_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            path = (
                Path(directory)
                / "person.png"
            )

            create_person_image(
                path,
                (180, 40, 40),
            )

            pruner = (
                IdentityAwareTokenPruner()
            )

            prepared = (
                pruner.prepare(
                    path
                )
            )

            easy = pruner.select(
                prepared,
                0.30,
            )

            medium = pruner.select(
                prepared,
                0.55,
            )

            hard = pruner.select(
                prepared,
                0.80,
            )

            self.assertEqual(
                prepared.total_tokens,
                128,
            )

            self.assertEqual(
                easy.retained_tokens,
                39,
            )

            self.assertEqual(
                medium.retained_tokens,
                71,
            )

            self.assertEqual(
                hard.retained_tokens,
                103,
            )

    def test_identity_anchors_cover_three_regions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            path = (
                Path(directory)
                / "person.png"
            )

            create_person_image(
                path,
                (40, 80, 190),
            )

            pruner = (
                IdentityAwareTokenPruner()
            )

            prepared = (
                pruner.prepare(
                    path
                )
            )

            selection = (
                pruner.select(
                    prepared,
                    0.05,
                )
            )

            bands = set()

            for token in selection.tokens:
                if (
                    token.y_center
                    < 1.0 / 3.0
                ):
                    bands.add(0)

                elif (
                    token.y_center
                    < 2.0 / 3.0
                ):
                    bands.add(1)

                else:
                    bands.add(2)

            self.assertEqual(
                bands,
                {0, 1, 2},
            )

    def test_gallery_retrieves_matching_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            temp = Path(
                directory
            )

            red = (
                temp / "red.png"
            )

            blue = (
                temp / "blue.png"
            )

            green = (
                temp / "green.png"
            )

            create_person_image(
                red,
                (190, 45, 45),
            )

            create_person_image(
                blue,
                (45, 70, 195),
            )

            create_person_image(
                green,
                (45, 175, 80),
            )

            encoder = (
                BudgetedVisualEncoder()
            )

            gallery = (
                GalleryIndex.build(
                    {
                        "red": red,
                        "blue": blue,
                        "green": green,
                    },
                    encoder,
                )
            )

            query = encoder.encode(
                red,
                0.80,
            )

            results = (
                gallery.search(
                    query.descriptor,
                    3,
                )
            )

            self.assertEqual(
                results[0].label,
                "red",
            )

            self.assertGreater(
                results[0].score,
                results[1].score,
            )

    def test_pipeline_ranks_matching_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            temp = Path(
                directory
            )

            red = (
                temp / "red.png"
            )

            blue = (
                temp / "blue.png"
            )

            green = (
                temp / "green.png"
            )

            create_person_image(
                red,
                (190, 45, 45),
            )

            create_person_image(
                blue,
                (45, 70, 195),
            )

            create_person_image(
                green,
                (45, 175, 80),
            )

            encoder = (
                BudgetedVisualEncoder()
            )

            gallery = (
                GalleryIndex.build(
                    {
                        "red": red,
                        "blue": blue,
                        "green": green,
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

            self.assertEqual(
                result.candidates[0].label,
                "red",
            )

            self.assertGreaterEqual(
                len(
                    result.stages
                ),
                1,
            )

            self.assertLessEqual(
                len(
                    result.stages
                ),
                3,
            )

    def test_detector_evidence_validation(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            DetectorEvidence(
                confidence=1.2,
                foreground_ratio=0.5,
                occlusion=0.0,
                crowding=0.0,
            )


if __name__ == "__main__":
    unittest.main()
