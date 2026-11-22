from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rac_reid.learned_reid import (
    MultiResolutionOSNetReIDEncoder,
)


class MultiResolutionEncoderTests(
    unittest.TestCase
):

    def test_modes_are_fixed(
        self,
    ) -> None:
        self.assertEqual(
            MultiResolutionOSNetReIDEncoder
            .resolution_for_mode(
                "efficient"
            ),
            (160, 80),
        )

        self.assertEqual(
            MultiResolutionOSNetReIDEncoder
            .resolution_for_mode(
                "full"
            ),
            (256, 128),
        )

    def test_invalid_mode_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            (
                MultiResolutionOSNetReIDEncoder
                .resolution_for_mode(
                    "unknown"
                )
            )

    def test_constructor_is_lazy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            encoder = (
                MultiResolutionOSNetReIDEncoder(
                    cache_dir=Path(
                        temp
                    ),
                    checkpoint_path=(
                        Path(temp)
                        / "not-loaded.pt"
                    ),
                    expected_sha256=(
                        "0" * 64
                    ),
                )
            )

            self.assertFalse(
                encoder.loaded
            )

            self.assertEqual(
                encoder.query_resolution,
                (160, 80),
            )

            self.assertEqual(
                encoder.gallery_resolution,
                (256, 128),
            )


if __name__ == "__main__":
    unittest.main()
