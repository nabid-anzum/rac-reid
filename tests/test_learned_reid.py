from __future__ import annotations

import unittest

import numpy as np

from rac_reid.learned_reid import (
    OSNetReIDEncoder,
)


class LearnedReIDTests(
    unittest.TestCase
):
    def test_normalization(
        self,
    ) -> None:
        vector = np.asarray(
            [
                3.0,
                4.0,
            ],
            dtype=np.float32,
        )

        result = (
            OSNetReIDEncoder
            .normalize(
                vector
            )
        )

        self.assertAlmostEqual(
            float(
                np.linalg.norm(
                    result
                )
            ),
            1.0,
            places=6,
        )

    def test_identical_cosine(
        self,
    ) -> None:
        vector = np.asarray(
            [
                1.0,
                2.0,
                3.0,
            ],
            dtype=np.float32,
        )

        score = (
            OSNetReIDEncoder
            .cosine_similarity(
                vector,
                vector,
            )
        )

        self.assertAlmostEqual(
            score,
            1.0,
            places=6,
        )

    def test_ranking(
        self,
    ) -> None:
        query = np.asarray(
            [
                1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        gallery = {
            "same":
                np.asarray(
                    [
                        1.0,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            "different":
                np.asarray(
                    [
                        0.0,
                        1.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
        }

        result = (
            OSNetReIDEncoder
            .rank_from_embeddings(
                query_embedding=query,
                gallery_embeddings=(
                    gallery
                ),
                cheap_scores={
                    "same":
                        0.1,
                    "different":
                        0.9,
                },
                model_name="test",
            )
        )

        self.assertEqual(
            result.candidates[
                0
            ].label,
            "same",
        )

        self.assertGreater(
            result.candidates[
                0
            ].cosine_similarity,
            result.candidates[
                1
            ].cosine_similarity,
        )

    def test_zero_vector_rejected(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            (
                OSNetReIDEncoder
                .normalize(
                    np.zeros(
                        4,
                        dtype=np.float32,
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
