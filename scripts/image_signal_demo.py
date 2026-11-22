from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from rac_reid.image_signals import (
    ImageSignalExtractor,
)


ROOT = Path(
    __file__
).resolve().parents[1]


def show(
    name: str,
    path: Path,
    extractor: ImageSignalExtractor,
) -> None:
    signals = extractor.extract(
        path
    )

    print(
        f"{name:10} "
        f"brightness={signals.brightness:.3f} "
        f"contrast={signals.contrast:.3f} "
        f"entropy={signals.visual_entropy:.3f} "
        f"blur={signals.blur:.3f} "
        f"edges={signals.edge_density:.3f} "
        f"foreground="
        f"{signals.foreground_ratio_estimate:.3f}"
    )


def main() -> None:
    extractor = ImageSignalExtractor()

    with tempfile.TemporaryDirectory(
        dir=ROOT
    ) as directory:
        temp = Path(directory)

        solid = temp / "solid.png"

        Image.new(
            "RGB",
            (128, 256),
            (128, 128, 128),
        ).save(solid)

        checker = (
            temp / "checker.png"
        )

        pattern = (
            np.indices(
                (128, 128)
            ).sum(axis=0)
            % 2
        )

        checker_array = np.repeat(
            (
                pattern * 255
            ).astype(
                np.uint8
            )[:, :, None],
            3,
            axis=2,
        )

        Image.fromarray(
            checker_array
        ).save(checker)

        subject = (
            temp / "subject.png"
        )

        subject_array = np.full(
            (256, 128, 3),
            245,
            dtype=np.uint8,
        )

        subject_array[
            48:220,
            38:90,
        ] = (
            40,
            80,
            160,
        )

        Image.fromarray(
            subject_array
        ).save(subject)

        show(
            "solid",
            solid,
            extractor,
        )

        show(
            "checker",
            checker,
            extractor,
        )

        show(
            "subject",
            subject,
            extractor,
        )


if __name__ == "__main__":
    main()
