from __future__ import annotations

import unittest

from rac_reid import (
    PersonDetection,
    PersonDetectionResult,
    YOLOPersonDetector,
)


class PersonDetectorGeometryTests(
    unittest.TestCase
):
    def test_expanded_box_stays_inside_image(
        self,
    ) -> None:
        detection = PersonDetection(
            x1=10.0,
            y1=20.0,
            x2=50.0,
            y2=100.0,
            confidence=0.9,
        )

        box = (
            YOLOPersonDetector.expanded_box(
                detection,
                image_width=100,
                image_height=200,
                margin=0.20,
            )
        )

        x1, y1, x2, y2 = box

        self.assertGreaterEqual(
            x1,
            0,
        )

        self.assertGreaterEqual(
            y1,
            0,
        )

        self.assertLessEqual(
            x2,
            100,
        )

        self.assertLessEqual(
            y2,
            200,
        )

        self.assertLess(
            x1,
            10,
        )

        self.assertGreater(
            x2,
            50,
        )

    def test_detector_evidence_is_normalized(
        self,
    ) -> None:
        result = (
            PersonDetectionResult(
                image_width=200,
                image_height=300,
                detections=(
                    PersonDetection(
                        20,
                        20,
                        120,
                        280,
                        0.91,
                    ),
                    PersonDetection(
                        90,
                        40,
                        180,
                        270,
                        0.82,
                    ),
                ),
                primary_index=0,
            )
        )

        evidence = (
            result.evidence()
        )

        values = (
            evidence.confidence,
            evidence.foreground_ratio,
            evidence.occlusion,
            evidence.crowding,
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

        self.assertGreater(
            evidence.occlusion,
            0.0,
        )

        self.assertGreater(
            evidence.crowding,
            0.0,
        )

    def test_person_count(
        self,
    ) -> None:
        result = (
            PersonDetectionResult(
                image_width=100,
                image_height=100,
                detections=(
                    PersonDetection(
                        10,
                        10,
                        50,
                        90,
                        0.9,
                    ),
                    PersonDetection(
                        55,
                        10,
                        95,
                        90,
                        0.8,
                    ),
                ),
                primary_index=0,
            )
        )

        self.assertEqual(
            result.person_count,
            2,
        )


if __name__ == "__main__":
    unittest.main()
