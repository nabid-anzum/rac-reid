from __future__ import annotations

import time
import urllib.request
from pathlib import Path

from rac_reid import (
    ImageSignalExtractor,
    PersonQueryPreprocessor,
    ResourceController,
    YOLOPersonDetector,
)


ROOT = Path(
    __file__
).resolve().parents[1]

MODEL = (
    ROOT
    / "models"
    / "yolo26n.pt"
)

SAMPLE_DIR = (
    ROOT
    / ".runtime"
    / "samples"
)

SAMPLE = (
    SAMPLE_DIR
    / "bus.jpg"
)

WORK = (
    ROOT
    / ".runtime"
    / "detector-smoke"
)


def main() -> None:
    SAMPLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SAMPLE.is_file():
        urllib.request.urlretrieve(
            "https://ultralytics.com/images/bus.jpg",
            SAMPLE,
        )

    controller = (
        ResourceController.from_json(
            ROOT
            / "configs"
            / "resource_policy.json"
        )
    )

    extractor = (
        ImageSignalExtractor()
    )

    detector = (
        YOLOPersonDetector(
            MODEL,
            torch_threads=2,
        )
    )

    started = (
        time.perf_counter()
    )

    detection = detector.detect(
        SAMPLE
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    if detection.person_count < 1:
        raise SystemExit(
            "ERROR: No persons detected in smoke image."
        )

    evidence = (
        detection.evidence()
    )

    preprocessor = (
        PersonQueryPreprocessor(
            detector=detector,
            controller=controller,
            image_extractor=extractor,
        )
    )

    prepared = (
        preprocessor.prepare(
            SAMPLE,
            WORK,
        )
    )

    if not prepared.crop_path.is_file():
        raise SystemExit(
            "ERROR: Adaptive crop was not produced."
        )

    print(
        "People detected:",
        detection.person_count,
    )

    print(
        "Primary confidence:",
        f"{prepared.primary_confidence:.3f}",
    )

    print(
        "Detector foreground ratio:",
        f"{evidence.foreground_ratio:.3f}",
    )

    print(
        "Occlusion proxy:",
        f"{evidence.occlusion:.3f}",
    )

    print(
        "Crowding proxy:",
        f"{evidence.crowding:.3f}",
    )

    print(
        "Initial difficulty:",
        prepared.initial_difficulty,
    )

    print(
        "Adaptive crop margin:",
        f"{prepared.crop_margin:.2f}",
    )

    print(
        "Crop:",
        prepared.crop_path,
    )

    print(
        "CPU detection seconds:",
        f"{elapsed:.3f}",
    )


if __name__ == "__main__":
    main()
