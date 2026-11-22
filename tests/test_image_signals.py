from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rac_reid.image_signals import (
    ImageSignalExtractor,
)


ROOT = Path(
    __file__
).resolve().parents[1]


class ImageSignalExtractorTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.extractor = (
            ImageSignalExtractor()
        )

    def test_solid_image(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            path = (
                Path(directory)
                / "solid.png"
            )

            Image.new(
                "RGB",
                (64, 32),
                (128, 128, 128),
            ).save(path)

            signals = (
                self.extractor.extract(
                    path
                )
            )

            self.assertEqual(
                signals.width,
                64,
            )

            self.assertEqual(
                signals.height,
                32,
            )

            self.assertAlmostEqual(
                signals.aspect_ratio,
                2.0,
            )

            self.assertLess(
                signals.contrast,
                0.01,
            )

            self.assertLess(
                signals.visual_entropy,
                0.01,
            )

            self.assertGreater(
                signals.blur,
                0.99,
            )

            self.assertLess(
                signals.edge_density,
                0.01,
            )

            self.assertLess(
                signals.foreground_ratio_estimate,
                0.01,
            )

    def test_checkerboard_has_edges(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            path = (
                Path(directory)
                / "checker.png"
            )

            array = (
                np.indices(
                    (64, 64)
                ).sum(axis=0)
                % 2
            )

            rgb = np.repeat(
                (
                    array * 255
                ).astype(
                    np.uint8
                )[:, :, None],
                3,
                axis=2,
            )

            Image.fromarray(
                rgb
            ).save(path)

            signals = (
                self.extractor.extract(
                    path
                )
            )

            self.assertGreater(
                signals.edge_density,
                0.90,
            )

            self.assertLess(
                signals.blur,
                0.10,
            )

            self.assertGreater(
                signals.visual_entropy,
                0.10,
            )

    def test_center_subject_estimate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            path = (
                Path(directory)
                / "subject.png"
            )

            array = np.full(
                (64, 64, 3),
                255,
                dtype=np.uint8,
            )

            array[
                16:48,
                16:48,
            ] = 0

            Image.fromarray(
                array
            ).save(path)

            signals = (
                self.extractor.extract(
                    path
                )
            )

            self.assertGreater(
                signals.foreground_ratio_estimate,
                0.20,
            )

            self.assertLess(
                signals.foreground_ratio_estimate,
                0.30,
            )

    def test_values_are_normalized(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            dir=ROOT
        ) as directory:
            path = (
                Path(directory)
                / "random.png"
            )

            rng = (
                np.random.default_rng(
                    42
                )
            )

            array = rng.integers(
                0,
                256,
                size=(80, 50, 3),
                dtype=np.uint8,
            )

            Image.fromarray(
                array
            ).save(path)

            signals = (
                self.extractor.extract(
                    path
                )
            )

            values = (
                signals.brightness,
                signals.contrast,
                signals.visual_entropy,
                signals.blur,
                signals.edge_density,
                signals.foreground_ratio_estimate,
            )

            for value in values:
                self.assertGreaterEqual(
                    value,
                    0.0,
                )

                self.assertLessEqual(
                    value,
                    1.0,
                )

    def test_missing_image_is_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            FileNotFoundError
        ):
            self.extractor.extract(
                ROOT
                / "does-not-exist.png"
            )


if __name__ == "__main__":
    unittest.main()
