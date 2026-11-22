from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from rac_reid import (
    OSNetReIDEncoder,
)


ROOT = Path(
    __file__
).resolve().parents[1]

RUNTIME = (
    ROOT
    / ".runtime"
)

SAMPLE = (
    RUNTIME
    / "detector-smoke"
    / "person.jpg"
)

CACHE = (
    RUNTIME
    / "reid-cache"
)


def main() -> None:
    if not SAMPLE.is_file():
        raise SystemExit(
            "ERROR: Detector smoke crop "
            "does not exist."
        )

    encoder = (
        OSNetReIDEncoder(
            cache_dir=CACHE,
            torch_threads=2,
        )
    )

    (
        embedding,
        inference_seconds,
        load_seconds,
    ) = (
        encoder.embed_path_timed(
            SAMPLE
        )
    )

    print(
        "Model:",
        encoder.model_name,
    )

    print(
        "Device: cpu"
    )

    print(
        "Embedding shape:",
        embedding.shape,
    )

    print(
        "Embedding norm:",
        f"{np.linalg.norm(embedding):.6f}",
    )

    print(
        "Model load seconds:",
        f"{load_seconds:.3f}",
    )

    print(
        "Inference seconds:",
        f"{inference_seconds:.3f}",
    )

    print(
        "REID_CACHE_DIR:",
        os.environ.get(
            "REID_CACHE_DIR"
        ),
    )

    print(
        "HF_HOME:",
        os.environ.get(
            "HF_HOME"
        ),
    )

    if embedding.shape != (
        512,
    ):
        raise SystemExit(
            "ERROR: Expected 512 dimensions."
        )

    if not np.all(
        np.isfinite(
            embedding
        )
    ):
        raise SystemExit(
            "ERROR: Non-finite embedding."
        )

    norm = float(
        np.linalg.norm(
            embedding
        )
    )

    if not 0.999 <= norm <= 1.001:
        raise SystemExit(
            "ERROR: Embedding is not normalized."
        )

    print(
        "PASS: Real OSNet embedding generated."
    )


if __name__ == "__main__":
    main()
