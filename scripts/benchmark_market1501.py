from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from reid import (
    ReIDModel,
    load_market1501,
)
from reid.eval import (
    RetrievalProtocol,
)


DEFAULT_MODEL = (
    "osnet_x1_0_msmt17_combineall"
)


def sha256_file(
    path: Path,
) -> str:
    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return (
        digest.hexdigest()
    )


def count_images(
    root: Path,
    directory: str,
) -> int:
    return len(
        list(
            (
                root
                / directory
            ).glob(
                "*.jpg"
            )
        )
    )


def main() -> None:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = (
        parser.parse_args()
    )

    dataset = (
        args.dataset.resolve()
    )

    archive = (
        args.archive.resolve()
    )

    output = (
        args.output.resolve()
    )

    if not dataset.is_dir():
        raise SystemExit(
            f"Dataset missing: "
            f"{dataset}"
        )

    if not archive.is_file():
        raise SystemExit(
            f"Archive missing: "
            f"{archive}"
        )

    expected_counts = {
        "training_images": 12936,
        "gallery_images": 19732,
        "query_images": 3368,
    }

    counts = {
        "training_images":
            count_images(
                dataset,
                "bounding_box_train",
            ),
        "gallery_images":
            count_images(
                dataset,
                "bounding_box_test",
            ),
        "query_images":
            count_images(
                dataset,
                "query",
            ),
    }

    if counts != expected_counts:
        raise SystemExit(
            "ERROR: Non-standard "
            f"Market-1501 counts: {counts}"
        )

    threads = max(
        1,
        int(
            args.threads
        ),
    )

    torch.set_num_threads(
        threads
    )

    try:
        torch.set_num_interop_threads(
            1
        )

    except RuntimeError:
        pass

    if torch.cuda.is_available():
        raise SystemExit(
            "ERROR: This reproducible "
            "VPS benchmark is CPU-only."
        )

    query, gallery = (
        load_market1501(
            str(
                dataset
            )
        )
    )

    if len(query) != 3368:
        raise SystemExit(
            "ERROR: Unexpected query count."
        )

    if len(gallery) != 19732:
        raise SystemExit(
            "ERROR: Unexpected gallery count."
        )

    print(
        "Dataset:",
        dataset,
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Device: CPU"
    )

    print(
        "Query images:",
        len(query),
    )

    print(
        "Gallery images:",
        len(gallery),
    )

    started = (
        time.perf_counter()
    )

    model = (
        ReIDModel.from_pretrained(
            args.model,
            device="cpu",
        )
    )

    model_load_seconds = (
        time.perf_counter()
        - started
    )

    protocol = (
        RetrievalProtocol(
            distance="euclidean"
        )
    )

    evaluation_started = (
        time.perf_counter()
    )

    result = (
        model.evaluate(
            query,
            gallery,
            protocol=protocol,
        )
    )

    evaluation_seconds = (
        time.perf_counter()
        - evaluation_started
    )

    metrics = {
        str(key):
            float(value)
        for key, value
        in result.metrics.items()
    }

    required = {
        "map",
        "rank1",
        "rank5",
        "rank10",
    }

    missing = (
        required
        - set(
            metrics
        )
    )

    if missing:
        raise SystemExit(
            "ERROR: Missing metrics: "
            f"{sorted(missing)}"
        )

    if int(
        result.num_queries
    ) != 3368:
        raise SystemExit(
            "ERROR: Not all queries "
            "were scored."
        )

    payload = {
        "benchmark":
            (
                "Market-1501 standard "
                "cross-domain retrieval"
            ),
        "research_scope":
            (
                "OSNet learned baseline; "
                "not RAC adaptive ablation"
            ),
        "dataset": {
            "name":
                "Market-1501-v15.09.15",
            "archive_sha256":
                sha256_file(
                    archive
                ),
            **counts,
            "scored_queries":
                int(
                    result.num_queries
                ),
        },
        "model": {
            "alias":
                args.model,
            "architecture":
                "OSNet x1.0",
            "training_source":
                (
                    "MSMT17 all splits combined"
                ),
            "evaluation_relationship":
                (
                    "cross-domain on "
                    "Market-1501"
                ),
            "device":
                "cpu",
        },
        "protocol": {
            "distance":
                "euclidean",
            "junk_filtering":
                (
                    "reid standard "
                    "Market-1501 protocol"
                ),
        },
        "metrics_percent":
            metrics,
        "runtime": {
            "torch_threads":
                threads,
            "model_load_seconds":
                model_load_seconds,
            "evaluation_wall_seconds":
                evaluation_seconds,
            "python":
                platform.python_version(),
            "torch":
                torch.__version__,
            "machine":
                platform.machine(),
        },
        "scientific_notes": [
            (
                "This is the learned OSNet "
                "baseline, not the final "
                "RAC-ReID result."
            ),
            (
                "The checkpoint is used "
                "cross-domain on Market-1501."
            ),
            (
                "Do not use this checkpoint "
                "for unbiased MSMT17 evaluation."
            ),
        ],
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "========================================"
    )

    print(
        " MARKET-1501 BASELINE RESULTS"
    )

    print(
        "========================================"
    )

    for key in (
        "map",
        "rank1",
        "rank5",
        "rank10",
        "minp",
    ):
        if key in metrics:
            print(
                f"{key:>8}: "
                f"{metrics[key]:.3f}%"
            )

    print(
        "Queries:",
        result.num_queries,
    )

    print(
        "Evaluation wall seconds:",
        f"{evaluation_seconds:.3f}",
    )

    print(
        "Result:",
        output,
    )


if __name__ == "__main__":
    main()
