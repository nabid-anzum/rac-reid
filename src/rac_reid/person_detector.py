from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from .image_signals import ImageSignalExtractor
from .pipeline import DetectorEvidence
from .resource_controller import (
    QuerySignals,
    ResourceController,
)


class NoPersonDetectedError(RuntimeError):
    """Raised when no person is detected in the supplied scene."""


@dataclass(frozen=True)
class PersonDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.x2 - self.x1,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.y2 - self.y1,
        )

    @property
    def area(self) -> float:
        return (
            self.width
            * self.height
        )


@dataclass(frozen=True)
class PersonDetectionResult:
    image_width: int
    image_height: int
    detections: tuple[
        PersonDetection,
        ...
    ]
    primary_index: int

    @property
    def primary(
        self,
    ) -> PersonDetection:
        return self.detections[
            self.primary_index
        ]

    @property
    def person_count(
        self,
    ) -> int:
        return len(
            self.detections
        )

    def evidence(
        self,
    ) -> DetectorEvidence:
        primary = self.primary

        image_area = max(
            1.0,
            float(
                self.image_width
                * self.image_height
            ),
        )

        foreground_ratio = min(
            1.0,
            primary.area
            / image_area,
        )

        overlap_proxy = 0.0

        for index, other in enumerate(
            self.detections
        ):
            if index == self.primary_index:
                continue

            overlap_proxy = max(
                overlap_proxy,
                self._intersection_over_primary(
                    primary,
                    other,
                ),
            )

        boundary_hits = 0

        x_margin = (
            self.image_width
            * 0.02
        )

        y_margin = (
            self.image_height
            * 0.02
        )

        if primary.x1 <= x_margin:
            boundary_hits += 1

        if primary.y1 <= y_margin:
            boundary_hits += 1

        if (
            self.image_width
            - primary.x2
            <= x_margin
        ):
            boundary_hits += 1

        if (
            self.image_height
            - primary.y2
            <= y_margin
        ):
            boundary_hits += 1

        boundary_proxy = (
            boundary_hits
            / 4.0
        )

        occlusion_proxy = min(
            1.0,
            0.75
            * overlap_proxy
            + 0.25
            * boundary_proxy,
        )

        extra_people = max(
            0,
            self.person_count - 1,
        )

        crowd_count_proxy = min(
            1.0,
            extra_people / 4.0,
        )

        crowding_proxy = min(
            1.0,
            0.65
            * crowd_count_proxy
            + 0.35
            * overlap_proxy,
        )

        return DetectorEvidence(
            confidence=(
                primary.confidence
            ),
            foreground_ratio=(
                foreground_ratio
            ),
            occlusion=(
                occlusion_proxy
            ),
            crowding=(
                crowding_proxy
            ),
        )

    @staticmethod
    def _intersection_over_primary(
        primary: PersonDetection,
        other: PersonDetection,
    ) -> float:
        ix1 = max(
            primary.x1,
            other.x1,
        )

        iy1 = max(
            primary.y1,
            other.y1,
        )

        ix2 = min(
            primary.x2,
            other.x2,
        )

        iy2 = min(
            primary.y2,
            other.y2,
        )

        width = max(
            0.0,
            ix2 - ix1,
        )

        height = max(
            0.0,
            iy2 - iy1,
        )

        intersection = (
            width
            * height
        )

        if primary.area <= 0.0:
            return 0.0

        return min(
            1.0,
            intersection
            / primary.area,
        )


@dataclass(frozen=True)
class PreparedPersonQuery:
    crop_path: Path
    evidence: DetectorEvidence
    initial_difficulty: str
    crop_margin: float
    person_count: int
    primary_confidence: float
    bbox: tuple[
        int,
        int,
        int,
        int,
    ]


class YOLOPersonDetector:
    """CPU-only lightweight YOLO detector restricted to person class."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        image_size: int = 640,
        torch_threads: int = 2,
    ) -> None:
        path = Path(
            model_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {path}"
            )

        if not 0.0 < confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1"
            )

        if image_size < 160:
            raise ValueError(
                "image_size is too small"
            )

        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES",
            "",
        )

        import cv2
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(
            max(
                1,
                int(torch_threads),
            )
        )

        cv2.setNumThreads(1)

        self.model_path = path
        self.confidence = confidence
        self.image_size = image_size
        self.device = "cpu"

        self._model = YOLO(
            str(path)
        )

    def detect(
        self,
        image_path: str | Path,
    ) -> PersonDetectionResult:
        path = Path(
            image_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        with Image.open(path) as opened:
            image = (
                ImageOps.exif_transpose(
                    opened
                )
            )

            image_width, image_height = (
                image.size
            )

        prediction = (
            self._model.predict(
                source=str(path),
                classes=[0],
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
                save=False,
            )[0]
        )

        detections: list[
            PersonDetection
        ] = []

        boxes = prediction.boxes

        if boxes is not None:
            xyxy = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            confidences = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, confidence in zip(
                xyxy,
                confidences,
                strict=True,
            ):
                detections.append(
                    PersonDetection(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(
                            confidence
                        ),
                    )
                )

        if not detections:
            raise NoPersonDetectedError(
                "No person was detected in the image."
            )

        image_area = max(
            1.0,
            float(
                image_width
                * image_height
            ),
        )

        primary_index = max(
            range(
                len(detections)
            ),
            key=lambda index: (
                detections[
                    index
                ].confidence
                * math.sqrt(
                    max(
                        detections[
                            index
                        ].area
                        / image_area,
                        1e-9,
                    )
                ),
                detections[
                    index
                ].confidence,
                detections[
                    index
                ].area,
            ),
        )

        return PersonDetectionResult(
            image_width=(
                image_width
            ),
            image_height=(
                image_height
            ),
            detections=tuple(
                detections
            ),
            primary_index=(
                primary_index
            ),
        )

    @staticmethod
    def expanded_box(
        detection: PersonDetection,
        image_width: int,
        image_height: int,
        margin: float,
    ) -> tuple[
        int,
        int,
        int,
        int,
    ]:
        if not 0.0 <= margin <= 0.5:
            raise ValueError(
                "margin must be between 0 and 0.5"
            )

        dx = (
            detection.width
            * margin
        )

        dy = (
            detection.height
            * margin
        )

        x1 = max(
            0,
            int(
                math.floor(
                    detection.x1 - dx
                )
            ),
        )

        y1 = max(
            0,
            int(
                math.floor(
                    detection.y1 - dy
                )
            ),
        )

        x2 = min(
            image_width,
            int(
                math.ceil(
                    detection.x2 + dx
                )
            ),
        )

        y2 = min(
            image_height,
            int(
                math.ceil(
                    detection.y2 + dy
                )
            ),
        )

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "Expanded crop has invalid dimensions."
            )

        return (
            x1,
            y1,
            x2,
            y2,
        )

    def crop_primary(
        self,
        image_path: str | Path,
        result: PersonDetectionResult,
        output_path: str | Path,
        margin: float,
    ) -> Path:
        source = Path(
            image_path
        )

        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        box = self.expanded_box(
            result.primary,
            result.image_width,
            result.image_height,
            margin,
        )

        with Image.open(source) as opened:
            image = (
                ImageOps.exif_transpose(
                    opened
                )
                .convert("RGB")
            )

            crop = image.crop(
                box
            )

            crop.save(
                output,
                format="JPEG",
                quality=92,
                optimize=True,
            )

        return output


class PersonQueryPreprocessor:
    """Detect, estimate difficulty, and apply adaptive context crop."""

    def __init__(
        self,
        detector: YOLOPersonDetector,
        controller: ResourceController,
        image_extractor: ImageSignalExtractor,
    ) -> None:
        self.detector = detector
        self.controller = controller
        self.image_extractor = (
            image_extractor
        )

    def prepare(
        self,
        scene_path: str | Path,
        workspace: str | Path,
    ) -> PreparedPersonQuery:
        workspace_path = Path(
            workspace
        )

        workspace_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        detection = (
            self.detector.detect(
                scene_path
            )
        )

        evidence = (
            detection.evidence()
        )

        tight_crop = (
            workspace_path
            / "tight-person.jpg"
        )

        self.detector.crop_primary(
            scene_path,
            detection,
            tight_crop,
            margin=0.0,
        )

        signals = (
            self.image_extractor.extract(
                tight_crop
            )
        )

        query_signals = QuerySignals(
            detector_confidence=(
                evidence.confidence
            ),
            foreground_ratio=(
                evidence.foreground_ratio
            ),
            blur=signals.blur,
            visual_entropy=(
                signals.visual_entropy
            ),
            occlusion=(
                evidence.occlusion
            ),
            crowding=(
                evidence.crowding
            ),
        )

        plan = (
            self.controller.initial_plan(
                query_signals
            )
        )

        adaptive_crop = (
            workspace_path
            / "person.jpg"
        )

        self.detector.crop_primary(
            scene_path,
            detection,
            adaptive_crop,
            margin=plan.crop_margin,
        )

        bbox = (
            self.detector.expanded_box(
                detection.primary,
                detection.image_width,
                detection.image_height,
                plan.crop_margin,
            )
        )

        return PreparedPersonQuery(
            crop_path=adaptive_crop,
            evidence=evidence,
            initial_difficulty=(
                plan.difficulty.value
            ),
            crop_margin=(
                plan.crop_margin
            ),
            person_count=(
                detection.person_count
            ),
            primary_confidence=(
                detection
                .primary
                .confidence
            ),
            bbox=bbox,
        )
