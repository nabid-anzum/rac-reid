from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time

from pathlib import Path

import numpy as np
import torch
from torch import nn


EXPECTED_IMAGES = 12936
EXPECTED_IDS = 751

CALIBRATION_PRECISION_TARGET = 0.97
VALIDATION_PRECISION_MINIMUM = 0.95
VALIDATION_COVERAGE_MINIMUM = 0.10

X025_GFLOPS_REFERENCE = 0.08
X10_GFLOPS_REFERENCE = 0.98

PATTERN = re.compile(
    r"^(-?\d+)_c(\d+)s\d+_"
)


FEATURE_NAMES = [
    "top1",
    "top2",
    "top3",
    "top5",
    "top10",
    "margin_1_2",
    "margin_1_3",
    "margin_1_5",
    "top5_mean",
    "top5_std",
    "top10_mean",
    "top10_std",
    "top1_minus_top5_mean",
    "top1_minus_top10_mean",
    "top5_softmax_peak",
    "top10_softmax_peak",
    "top5_entropy",
    "top10_entropy",
]


def atomic_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(
        path
    )


def parse_records(
    root: Path,
):
    records = []

    for path in sorted(
        root.glob(
            "*.jpg"
        )
    ):
        match = PATTERN.match(
            path.name
        )

        if match is None:
            raise RuntimeError(
                "Unexpected filename: "
                + path.name
            )

        pid = int(
            match.group(
                1
            )
        )

        camid = int(
            match.group(
                2
            )
        )

        if pid == -1:
            continue

        records.append(
            (
                path,
                pid,
                camid,
            )
        )

    return records


def entropy_and_peak(
    values,
):
    shifted = (
        values
        - values[
            :,
            :1
        ]
    )

    # Temperature sharpens the very narrow cosine range
    # without changing candidate order.
    logits = (
        shifted
        * 20.0
    )

    logits -= np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_values = np.exp(
        logits
    )

    probabilities = (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )
    )

    peak = (
        probabilities[
            :,
            0
        ]
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
            + 1e-12
        ),
        axis=1,
    )

    entropy /= math.log(
        values.shape[
            1
        ]
    )

    return (
        peak,
        entropy,
    )


def build_features(
    embeddings,
    pids,
    camids,
    cache_path: Path,
):
    if cache_path.is_file():
        cached = np.load(
            cache_path
        )

        features = (
            cached[
                "features"
            ]
        )

        labels = (
            cached[
                "labels"
            ]
            .astype(
                np.float32
            )
        )

        cached_pids = (
            cached[
                "pids"
            ]
        )

        cached_camids = (
            cached[
                "camids"
            ]
        )

        if (
            features.shape[
                0
            ]
            == embeddings.shape[
                0
            ]
            and np.array_equal(
                cached_pids,
                pids,
            )
            and np.array_equal(
                cached_camids,
                camids,
            )
        ):
            print(
                "Feature cache: REUSED",
                flush=True,
            )

            return (
                features,
                labels,
                True,
            )

        raise RuntimeError(
            "Existing feature cache does not match dataset."
        )

    count = int(
        embeddings.shape[
            0
        ]
    )

    features = np.empty(
        (
            count,
            len(
                FEATURE_NAMES
            ),
        ),
        dtype=np.float32,
    )

    labels = np.zeros(
        count,
        dtype=np.float32,
    )

    chunk_size = 160

    for start in range(
        0,
        count,
        chunk_size,
    ):
        end = min(
            start
            + chunk_size,
            count,
        )

        queries = np.asarray(
            embeddings[
                start:end
            ],
            dtype=np.float32,
        )

        similarities = (
            queries
            @ embeddings.T
        )

        query_pids = (
            pids[
                start:end
            ]
        )

        query_camids = (
            camids[
                start:end
            ]
        )

        invalid = (
            (
                query_pids[
                    :,
                    None
                ]
                == pids[
                    None,
                    :
                ]
            )
            &
            (
                query_camids[
                    :,
                    None
                ]
                == camids[
                    None,
                    :
                ]
            )
        )

        similarities[
            invalid
        ] = -np.inf

        top_indices = np.argpartition(
            similarities,
            kth=(
                similarities.shape[
                    1
                ]
                - 10
            ),
            axis=1,
        )[
            :,
            -10:
        ]

        top_values = np.take_along_axis(
            similarities,
            top_indices,
            axis=1,
        )

        order = np.argsort(
            top_values,
            axis=1,
        )[
            :,
            ::-1
        ]

        top_indices = np.take_along_axis(
            top_indices,
            order,
            axis=1,
        )

        top_values = np.take_along_axis(
            top_values,
            order,
            axis=1,
        )

        predicted_pids = (
            pids[
                top_indices[
                    :,
                    0
                ]
            ]
        )

        labels[
            start:end
        ] = (
            predicted_pids
            == query_pids
        ).astype(
            np.float32
        )

        top5 = (
            top_values[
                :,
                :5
            ]
        )

        top10 = (
            top_values
        )

        top5_peak, top5_entropy = (
            entropy_and_peak(
                top5
            )
        )

        top10_peak, top10_entropy = (
            entropy_and_peak(
                top10
            )
        )

        feature_matrix = np.column_stack(
            [
                top_values[
                    :,
                    0
                ],
                top_values[
                    :,
                    1
                ],
                top_values[
                    :,
                    2
                ],
                top_values[
                    :,
                    4
                ],
                top_values[
                    :,
                    9
                ],
                (
                    top_values[
                        :,
                        0
                    ]
                    - top_values[
                        :,
                        1
                    ]
                ),
                (
                    top_values[
                        :,
                        0
                    ]
                    - top_values[
                        :,
                        2
                    ]
                ),
                (
                    top_values[
                        :,
                        0
                    ]
                    - top_values[
                        :,
                        4
                    ]
                ),
                np.mean(
                    top5,
                    axis=1,
                ),
                np.std(
                    top5,
                    axis=1,
                ),
                np.mean(
                    top10,
                    axis=1,
                ),
                np.std(
                    top10,
                    axis=1,
                ),
                (
                    top_values[
                        :,
                        0
                    ]
                    - np.mean(
                        top5,
                        axis=1,
                    )
                ),
                (
                    top_values[
                        :,
                        0
                    ]
                    - np.mean(
                        top10,
                        axis=1,
                    )
                ),
                top5_peak,
                top10_peak,
                top5_entropy,
                top10_entropy,
            ]
        ).astype(
            np.float32
        )

        features[
            start:end
        ] = feature_matrix

        print(
            "Feature extraction:",
            end,
            "/",
            count,
            flush=True,
        )

    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        pids=pids,
        camids=camids,
    )

    return (
        features,
        labels,
        False,
    )


def evaluate_threshold(
    probabilities,
    labels,
    mask,
    threshold,
):
    exits = (
        mask
        & (
            probabilities
            >= threshold
        )
    )

    count = int(
        np.sum(
            exits
        )
    )

    population = int(
        np.sum(
            mask
        )
    )

    if count == 0:
        precision = None
    else:
        precision = float(
            np.mean(
                labels[
                    exits
                ]
            )
        )

    return {
        "exits":
            count,
        "coverage":
            (
                count
                / population
            ),
        "precision":
            precision,
    }


def select_threshold(
    probabilities,
    labels,
    mask,
):
    values = np.unique(
        probabilities[
            mask
        ]
    )

    values = np.sort(
        values
    )[
        ::-1
    ]

    best = None

    for threshold in values:
        result = (
            evaluate_threshold(
                probabilities,
                labels,
                mask,
                float(
                    threshold
                ),
            )
        )

        if (
            result[
                "precision"
            ]
            is None
        ):
            continue

        if (
            result[
                "precision"
            ]
            < CALIBRATION_PRECISION_TARGET
        ):
            continue

        candidate = {
            "threshold":
                float(
                    threshold
                ),
            **result,
        }

        if (
            best is None
            or candidate[
                "exits"
            ]
            > best[
                "exits"
            ]
        ):
            best = candidate

    return best


def frontier(
    probabilities,
    labels,
    mask,
):
    output = {}

    for target in (
        0.90,
        0.95,
        0.97,
        0.98,
        0.99,
    ):
        values = np.sort(
            np.unique(
                probabilities[
                    mask
                ]
            )
        )[
            ::-1
        ]

        best = None

        for threshold in values:
            result = (
                evaluate_threshold(
                    probabilities,
                    labels,
                    mask,
                    float(
                        threshold
                    ),
                )
            )

            precision = (
                result[
                    "precision"
                ]
            )

            if (
                precision is None
                or precision
                < target
            ):
                continue

            candidate = {
                "threshold":
                    float(
                        threshold
                    ),
                **result,
            }

            if (
                best is None
                or candidate[
                    "exits"
                ]
                > best[
                    "exits"
                ]
            ):
                best = candidate

        output[
            str(
                target
            )
        ] = best

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--feature-cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--runtime-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    records = parse_records(
        args.train
    )

    if len(
        records
    ) != EXPECTED_IMAGES:
        raise SystemExit(
            "ERROR: Unexpected training image count."
        )

    pids = np.asarray(
        [
            item[
                1
            ]
            for item
            in records
        ],
        dtype=np.int64,
    )

    camids = np.asarray(
        [
            item[
                2
            ]
            for item
            in records
        ],
        dtype=np.int64,
    )

    unique_ids = np.unique(
        pids
    )

    if len(
        unique_ids
    ) != EXPECTED_IDS:
        raise SystemExit(
            "ERROR: Unexpected identity count."
        )

    embeddings = np.load(
        args.embeddings,
        mmap_mode="r",
    )

    if embeddings.shape != (
        EXPECTED_IMAGES,
        512,
    ):
        raise SystemExit(
            "ERROR: Unexpected embedding cache shape."
        )

    (
        features,
        labels,
        feature_cache_reused,
    ) = build_features(
        embeddings,
        pids,
        camids,
        args.feature_cache,
    )

    train_mask = np.isin(
        pids % 5,
        [
            0,
            1,
            2,
        ],
    )

    calibration_mask = (
        (
            pids
            % 5
        )
        == 3
    )

    validation_mask = (
        (
            pids
            % 5
        )
        == 4
    )

    for (
        name,
        mask,
    ) in (
        (
            "train",
            train_mask,
        ),
        (
            "calibration",
            calibration_mask,
        ),
        (
            "validation",
            validation_mask,
        ),
    ):
        print(
            name,
            "images:",
            int(
                np.sum(
                    mask
                )
            ),
        )

        print(
            name,
            "identities:",
            len(
                np.unique(
                    pids[
                        mask
                    ]
                )
            ),
        )

        print(
            name,
            "base correctness:",
            f"{float(np.mean(labels[mask])):.3%}",
        )

    mean = np.mean(
        features[
            train_mask
        ],
        axis=0,
    )

    std = np.std(
        features[
            train_mask
        ],
        axis=0,
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std,
    )

    normalized = (
        (
            features
            - mean
        )
        / std
    ).astype(
        np.float32
    )

    torch.manual_seed(
        2903
    )

    x_train = torch.from_numpy(
        normalized[
            train_mask
        ]
    )

    y_train = torch.from_numpy(
        labels[
            train_mask
        ]
    ).reshape(
        -1,
        1,
    )

    model = nn.Linear(
        len(
            FEATURE_NAMES
        ),
        1,
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=0.015,
            weight_decay=0.002,
        )
    )

    loss_fn = (
        nn.BCEWithLogitsLoss()
    )

    model.train()

    started = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        501,
    ):
        optimizer.zero_grad()

        logits = model(
            x_train
        )

        loss = loss_fn(
            logits,
            y_train,
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 1
            or epoch % 100 == 0
            or epoch == 500
        ):
            print(
                "Epoch:",
                epoch,
                "loss:",
                f"{float(loss):.6f}",
                flush=True,
            )

    training_seconds = (
        time.perf_counter()
        - started
    )

    model.eval()

    with torch.no_grad():
        probabilities = (
            torch.sigmoid(
                model(
                    torch.from_numpy(
                        normalized
                    )
                )
            )
            .reshape(
                -1
            )
            .numpy()
            .astype(
                np.float64
            )
        )

    selected = select_threshold(
        probabilities,
        labels,
        calibration_mask,
    )

    if selected is None:
        raise SystemExit(
            "ERROR: Learned controller could not "
            "reach calibration precision target."
        )

    validation = evaluate_threshold(
        probabilities,
        labels,
        validation_mask,
        selected[
            "threshold"
        ],
    )

    calibration_frontier = frontier(
        probabilities,
        labels,
        calibration_mask,
    )

    validation_frontier = {}

    for (
        target,
        operating_point,
    ) in calibration_frontier.items():
        if operating_point is None:
            validation_frontier[
                target
            ] = None
            continue

        validation_frontier[
            target
        ] = evaluate_threshold(
            probabilities,
            labels,
            validation_mask,
            operating_point[
                "threshold"
            ],
        )

    break_even_coverage = (
        X025_GFLOPS_REFERENCE
        / X10_GFLOPS_REFERENCE
    )

    validation_precision = (
        validation[
            "precision"
        ]
    )

    validation_coverage = float(
        validation[
            "coverage"
        ]
    )

    theoretical_relative_compute = (
        X025_GFLOPS_REFERENCE
        + (
            1.0
            - validation_coverage
        )
        * X10_GFLOPS_REFERENCE
    )

    full_compute = (
        X10_GFLOPS_REFERENCE
    )

    theoretical_compute_reduction = (
        1.0
        - (
            theoretical_relative_compute
            / full_compute
        )
    )

    passed = (
        validation_precision is not None
        and validation_precision
        >= VALIDATION_PRECISION_MINIMUM
        and validation_coverage
        >= VALIDATION_COVERAGE_MINIMUM
    )

    weights = (
        model.weight
        .detach()
        .cpu()
        .numpy()
        .reshape(
            -1
        )
    )

    bias = float(
        model.bias
        .detach()
        .cpu()
        .numpy()[
            0
        ]
    )

    payload = {
        "status":
            (
                "passed"
                if passed
                else "failed_validation"
            ),
        "controller": {
            "type":
                "logistic regression",
            "purpose":
                (
                    "predict whether OSNet x0.25 "
                    "Top-1 identity is safe to early-exit"
                ),
            "feature_names":
                FEATURE_NAMES,
            "feature_mean":
                [
                    float(
                        value
                    )
                    for value
                    in mean
                ],
            "feature_std":
                [
                    float(
                        value
                    )
                    for value
                    in std
                ],
            "weights":
                [
                    float(
                        value
                    )
                    for value
                    in weights
                ],
            "bias":
                bias,
            "probability_threshold":
                float(
                    selected[
                        "threshold"
                    ]
                ),
        },
        "split": {
            "train":
                "PID % 5 in {0,1,2}",
            "threshold_calibration":
                "PID % 5 == 3",
            "held_out_validation":
                "PID % 5 == 4",
            "test_set_used":
                False,
        },
        "criteria": {
            "calibration_precision_target":
                CALIBRATION_PRECISION_TARGET,
            "validation_precision_minimum":
                VALIDATION_PRECISION_MINIMUM,
            "validation_coverage_minimum":
                VALIDATION_COVERAGE_MINIMUM,
            "theoretical_break_even_coverage":
                break_even_coverage,
        },
        "calibration":
            selected,
        "validation":
            validation,
        "calibration_frontier":
            calibration_frontier,
        "validation_frontier":
            validation_frontier,
        "compute_projection": {
            "x0_25_reference_gflops":
                X025_GFLOPS_REFERENCE,
            "x1_0_reference_gflops":
                X10_GFLOPS_REFERENCE,
            "break_even_early_exit_rate":
                break_even_coverage,
            "validation_early_exit_rate":
                validation_coverage,
            "relative_compute_vs_x1_only":
                (
                    theoretical_relative_compute
                    / full_compute
                ),
            "projected_compute_reduction":
                theoretical_compute_reduction,
        },
        "runtime": {
            "feature_cache_reused":
                feature_cache_reused,
            "training_seconds":
                training_seconds,
        },
        "scientific_notes": [
            (
                "All controller development uses "
                "Market-1501 training identities only."
            ),
            (
                "Controller training, threshold calibration "
                "and validation use identity-disjoint subsets."
            ),
            (
                "The classifier predicts x0.25 Top-1 "
                "correctness; it does not perform ReID."
            ),
            (
                "The production cascade must still be "
                "evaluated on the untouched Market-1501 "
                "query/gallery test split."
            ),
            (
                "Projected GFLOP reduction is a reference "
                "complexity estimate; measured CPU latency "
                "must also be reported."
            ),
        ],
    }

    atomic_json(
        args.runtime_output,
        payload,
    )

    print()
    print(
        "================================================="
    )

    print(
        " LEARNED CONFIDENCE CONTROLLER RESULT"
    )

    print(
        "================================================="
    )

    print(
        "Probability threshold:",
        f"{selected['threshold']:.6f}",
    )

    print(
        "Calibration exits:",
        selected[
            "exits"
        ],
    )

    print(
        "Calibration coverage:",
        f"{selected['coverage']:.3%}",
    )

    print(
        "Calibration precision:",
        f"{selected['precision']:.3%}",
    )

    print()

    print(
        "Validation exits:",
        validation[
            "exits"
        ],
    )

    print(
        "Validation coverage:",
        f"{validation['coverage']:.3%}",
    )

    if (
        validation[
            "precision"
        ]
        is None
    ):
        print(
            "Validation precision: n/a"
        )

    else:
        print(
            "Validation precision:",
            f"{validation['precision']:.3%}",
        )

    print()

    print(
        "Break-even exit coverage:",
        f"{break_even_coverage:.3%}",
    )

    print(
        "Projected learned-compute reduction:",
        f"{theoretical_compute_reduction:.3%}",
    )

    print()

    print(
        "Calibration frontier:"
    )

    for target in (
        "0.9",
        "0.95",
        "0.97",
        "0.98",
        "0.99",
    ):
        cal = (
            calibration_frontier[
                target
            ]
        )

        val = (
            validation_frontier[
                target
            ]
        )

        print()

        print(
            "Target:",
            target,
        )

        if cal is None:
            print(
                "  no operating point"
            )
            continue

        print(
            "  threshold:",
            f"{cal['threshold']:.6f}",
        )

        print(
            "  CAL coverage:",
            f"{cal['coverage']:.3%}",
        )

        print(
            "  CAL precision:",
            f"{cal['precision']:.3%}",
        )

        print(
            "  VAL coverage:",
            f"{val['coverage']:.3%}",
        )

        if (
            val[
                "precision"
            ]
            is None
        ):
            value = "n/a"
        else:
            value = (
                f"{val['precision']:.3%}"
            )

        print(
            "  VAL precision:",
            value,
        )

    if not passed:
        raise SystemExit(
            "ERROR: Learned confidence controller "
            "did not meet held-out safety/coverage gates."
        )

    config = {
        "version":
            2,
        "coarse_model": {
            "architecture":
                "osnet_x0_25",
            "training_source":
                "MSMT17 combineall",
            "weights_sha256":
                (
                    "cf55163d78fc44c62c82f85ab62d39f"
                    "10438679b5abe8c698ae08cfa84aa6e18"
                ),
        },
        "heavy_model": {
            "architecture":
                "osnet_x1_0",
            "source":
                "osnet_x1_0_msmt17_combineall",
        },
        "confidence_controller":
            payload[
                "controller"
            ],
        "validation": {
            "precision":
                validation[
                    "precision"
                ],
            "coverage":
                validation[
                    "coverage"
                ],
            "projected_compute_reduction":
                theoretical_compute_reduction,
        },
        "development_data": {
            "source":
                "Market-1501 training split only",
            "test_set_used":
                False,
        },
    }

    atomic_json(
        args.config_output,
        config,
    )

    print()
    print(
        "PASS: Learned confidence controller "
        "passed held-out validation."
    )


if __name__ == "__main__":
    main()
