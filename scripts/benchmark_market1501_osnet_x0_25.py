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


EXPECTED_QUERY = 3368
EXPECTED_GALLERY = 19732

REFERENCE_RANK1 = 59.9
REFERENCE_MAP = 31.0


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
        "--weights",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
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
        args.dataset
        .resolve()
    )

    weights = (
        args.weights
        .resolve()
    )

    output = (
        args.output
        .resolve()
    )

    if not dataset.is_dir():
        raise SystemExit(
            f"Dataset missing: {dataset}"
        )

    if not weights.is_file():
        raise SystemExit(
            f"Weights missing: {weights}"
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
            "ERROR: This VPS benchmark is CPU-only."
        )

    query, gallery = (
        load_market1501(
            str(
                dataset
            )
        )
    )

    if len(query) != EXPECTED_QUERY:
        raise SystemExit(
            "ERROR: Unexpected query count."
        )

    if len(gallery) != EXPECTED_GALLERY:
        raise SystemExit(
            "ERROR: Unexpected gallery count."
        )

    print(
        "Architecture: osnet_x0_25",
        flush=True,
    )

    print(
        "Weights:",
        weights,
        flush=True,
    )

    print(
        "Device: CPU",
        flush=True,
    )

    print(
        "Queries:",
        len(query),
        flush=True,
    )

    print(
        "Gallery:",
        len(gallery),
        flush=True,
    )

    model_started = (
        time.perf_counter()
    )

    model = (
        ReIDModel.from_pretrained(
            str(
                weights
            ),
            architecture=(
                "osnet_x0_25"
            ),
            device="cpu",
        )
    )

    model_load_seconds = (
        time.perf_counter()
        - model_started
    )

    print(
        "Model load seconds:",
        f"{model_load_seconds:.4f}",
        flush=True,
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
        for (
            key,
            value,
        )
        in result.metrics.items()
    }

    for required in (
        "map",
        "rank1",
        "rank5",
        "rank10",
        "minp",
    ):
        if required not in metrics:
            raise SystemExit(
                f"ERROR: Metric missing: {required}"
            )

    if int(
        result.num_queries
    ) != EXPECTED_QUERY:
        raise SystemExit(
            "ERROR: Not all queries were scored."
        )

    rank1_delta = (
        metrics[
            "rank1"
        ]
        - REFERENCE_RANK1
    )

    map_delta = (
        metrics[
            "map"
        ]
        - REFERENCE_MAP
    )

    payload = {
        "benchmark":
            (
                "Market-1501 lightweight "
                "cross-domain learned baseline"
            ),
        "dataset": {
            "name":
                "Market-1501-v15.09.15",
            "query_images":
                EXPECTED_QUERY,
            "gallery_images":
                EXPECTED_GALLERY,
            "scored_queries":
                int(
                    result.num_queries
                ),
        },
        "model": {
            "architecture":
                "osnet_x0_25",
            "training_source":
                "MSMT17 combineall",
            "weights_file":
                weights.name,
            "weights_sha256":
                sha256_file(
                    weights
                ),
            "parameters_reference_millions":
                0.2,
            "gflops_reference":
                0.08,
            "device":
                "cpu",
        },
        "protocol": {
            "distance":
                "euclidean",
            "standard_market1501":
                True,
        },
        "metrics_percent":
            metrics,
        "official_reference_percent": {
            "rank1":
                REFERENCE_RANK1,
            "map":
                REFERENCE_MAP,
        },
        "delta_vs_official_reference_pp": {
            "rank1":
                rank1_delta,
            "map":
                map_delta,
        },
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
                "This is the lightweight learned "
                "coarse-stage baseline."
            ),
            (
                "It is not yet the final "
                "coarse-to-fine RAC-ReID cascade."
            ),
            (
                "The checkpoint is MSMT17 combineall "
                "evaluated cross-domain on Market-1501."
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
        " OSNET X0.25 MARKET-1501 RESULTS"
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
        print(
            f"{key:>8}: "
            f"{metrics[key]:.3f}%"
        )

    print(
        "Reference Rank-1:",
        f"{REFERENCE_RANK1:.3f}%"
    )

    print(
        "Reference mAP:",
        f"{REFERENCE_MAP:.3f}%"
    )

    print(
        "Rank-1 delta:",
        f"{rank1_delta:+.3f} pp"
    )

    print(
        "mAP delta:",
        f"{map_delta:+.3f} pp"
    )

    print(
        "Evaluation wall seconds:",
        f"{evaluation_seconds:.3f}"
    )

    print(
        "Result:",
        output
    )


if __name__ == "__main__":
    main()
