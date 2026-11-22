from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 29603

MAX_HYBRID_RANK1_DROP = 0.005
MIN_PROJECTED_CPU_SAVING = 0.05
MIN_GAIN_VS_LOW = 0.0


def write_json(path: Path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location(
        "rac_lowres_router_source",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not load previous router module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


class LinearBenefit(nn.Module):
    def __init__(self, dimensions):
        super().__init__()

        self.net = nn.Linear(
            dimensions,
            3,
        )

    def forward(self, x):
        return self.net(x)


class MLPBenefit(nn.Module):
    def __init__(self, dimensions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dimensions,
                24,
            ),
            nn.ReLU(),
            nn.Linear(
                24,
                12,
            ),
            nn.ReLU(),
            nn.Linear(
                12,
                3,
            ),
        )

    def forward(self, x):
        return self.net(x)


def make_classes(
    low_correct,
    full_correct,
):
    low = np.asarray(
        low_correct
    ).astype(bool)

    full = np.asarray(
        full_correct
    ).astype(bool)

    # 0 = refinement hurts:
    #     low correct, full wrong
    #
    # 1 = refinement does not change correctness:
    #     both right or both wrong
    #
    # 2 = refinement rescues:
    #     low wrong, full correct

    labels = np.ones(
        len(low),
        dtype=np.int64,
    )

    labels[
        low & ~full
    ] = 0

    labels[
        ~low & full
    ] = 2

    return labels


def standardize(
    train_x,
    other_x,
):
    mean = np.mean(
        train_x,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.std(
        train_x,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.maximum(
        std,
        1e-6,
    )

    train_z = (
        (
            train_x
            - mean
        )
        / std
    ).astype(
        np.float32
    )

    other_z = (
        (
            other_x
            - mean
        )
        / std
    ).astype(
        np.float32
    )

    return (
        train_z,
        other_z,
        mean,
        std,
    )


def fit_model(
    kind,
    x,
    y,
):
    torch.manual_seed(
        SEED
    )

    dimensions = x.shape[1]

    if kind == "linear":
        model = LinearBenefit(
            dimensions
        )

        epochs = 700
        lr = 0.01

    elif kind == "mlp":
        model = MLPBenefit(
            dimensions
        )

        epochs = 1000
        lr = 0.003

    else:
        raise ValueError(kind)

    counts = np.bincount(
        y,
        minlength=3,
    ).astype(
        np.float64
    )

    frequencies = (
        counts
        / np.sum(
            counts
        )
    )

    # Moderate balancing rather than full inverse-frequency
    # weighting so the rare rescue/harm classes matter without
    # completely overwhelming the dominant neutral class.
    weights = (
        1.0
        / np.sqrt(
            np.maximum(
                frequencies,
                1e-8,
            )
        )
    )

    weights /= np.mean(
        weights
    )

    weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    tx = torch.from_numpy(
        x
    )

    ty = torch.from_numpy(
        y
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    model.train()

    last_loss = None

    for epoch in range(
        epochs
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            tx
        )

        loss = F.cross_entropy(
            logits,
            ty,
            weight=weight_tensor,
        )

        loss.backward()

        optimizer.step()

        last_loss = float(
            loss.detach()
        )

    model.eval()

    return (
        model,
        last_loss,
        counts,
        weights,
    )


def predict_score(
    model,
    x,
):
    with torch.no_grad():
        logits = model(
            torch.from_numpy(
                x
            )
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    probs = (
        probabilities
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )

    # Expected benefit score:
    # high rescue probability increases the score;
    # high harm probability decreases it.
    score = (
        probs[:, 2]
        - probs[:, 0]
    )

    return (
        score,
        probs,
    )


def benchmark_router(
    model,
    sample,
):
    tensor = torch.from_numpy(
        sample[
            None,
            :
        ]
    )

    timings = []

    model.eval()

    with torch.no_grad():
        for _ in range(100):
            model(tensor)

        for _ in range(5000):
            started = (
                time.perf_counter()
            )

            model(tensor)

            timings.append(
                time.perf_counter()
                - started
            )

    return float(
        statistics.median(
            timings
        )
    )


def evaluate(
    score,
    threshold,
    low_correct,
    full_correct,
    low_full_ratio,
    router_full_ratio,
):
    low = np.asarray(
        low_correct
    ).astype(bool)

    full = np.asarray(
        full_correct
    ).astype(bool)

    refine = (
        score
        >= threshold
    )

    rescue = (
        ~low
        & full
    )

    harm = (
        low
        & ~full
    )

    hybrid = np.where(
        refine,
        full,
        low,
    )

    refine_count = int(
        np.sum(
            refine
        )
    )

    rescue_total = int(
        np.sum(
            rescue
        )
    )

    harm_total = int(
        np.sum(
            harm
        )
    )

    refined_rescues = int(
        np.sum(
            refine
            & rescue
        )
    )

    refined_harms = int(
        np.sum(
            refine
            & harm
        )
    )

    refine_rate = float(
        np.mean(
            refine
        )
    )

    exit_rate = (
        1.0
        - refine_rate
    )

    low_rank1 = float(
        np.mean(
            low
        )
    )

    full_rank1 = float(
        np.mean(
            full
        )
    )

    hybrid_rank1 = float(
        np.mean(
            hybrid
        )
    )

    normalized_cost = (
        low_full_ratio
        + refine_rate
        + router_full_ratio
    )

    saving = (
        1.0
        - normalized_cost
    )

    if refine_count:
        rescue_precision = (
            refined_rescues
            / refine_count
        )

        harm_precision = (
            refined_harms
            / refine_count
        )

    else:
        rescue_precision = 0.0
        harm_precision = 0.0

    rescue_capture = (
        refined_rescues
        / rescue_total
        if rescue_total
        else 0.0
    )

    harm_capture = (
        refined_harms
        / harm_total
        if harm_total
        else 0.0
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "refine_rate":
            refine_rate,

        "exit_rate":
            exit_rate,

        "low_rank1":
            low_rank1,

        "full_rank1":
            full_rank1,

        "hybrid_rank1":
            hybrid_rank1,

        "hybrid_rank1_delta_vs_full":
            hybrid_rank1
            - full_rank1,

        "hybrid_rank1_delta_vs_low":
            hybrid_rank1
            - low_rank1,

        "refined_queries":
            refine_count,

        "refined_rescues":
            refined_rescues,

        "refined_harms":
            refined_harms,

        "net_correctness_gain_count":
            refined_rescues
            - refined_harms,

        "refinement_rescue_precision":
            rescue_precision,

        "refinement_harm_rate":
            harm_precision,

        "rescue_capture_rate":
            rescue_capture,

        "harm_capture_rate":
            harm_capture,

        "projected_cpu_saving":
            saving,

        "normalized_cost":
            normalized_cost,
    }


def choose_threshold(
    score,
    low_correct,
    full_correct,
    low_full_ratio,
    router_full_ratio,
):
    quantiles = np.linspace(
        0.0,
        1.0,
        3001,
    )

    thresholds = np.unique(
        np.quantile(
            score,
            quantiles,
        )
    )

    thresholds = np.concatenate(
        [
            [
                np.min(score)
                - 1e-9
            ],
            thresholds,
            [
                np.max(score)
                + 1e-9
            ],
        ]
    )

    passing = []
    all_metrics = []

    for threshold in thresholds:
        metrics = evaluate(
            score,
            threshold,
            low_correct,
            full_correct,
            low_full_ratio,
            router_full_ratio,
        )

        all_metrics.append(
            metrics
        )

        if (
            metrics[
                "projected_cpu_saving"
            ]
            >= MIN_PROJECTED_CPU_SAVING

            and metrics[
                "hybrid_rank1_delta_vs_full"
            ]
            >= -MAX_HYBRID_RANK1_DROP

            and metrics[
                "hybrid_rank1_delta_vs_low"
            ]
            >= MIN_GAIN_VS_LOW

            and metrics[
                "refined_rescues"
            ]
            > metrics[
                "refined_harms"
            ]
        ):
            passing.append(
                metrics
            )

    passing.sort(
        key=lambda item: (
            item[
                "projected_cpu_saving"
            ],
            item[
                "hybrid_rank1"
            ],
            item[
                "net_correctness_gain_count"
            ],
        ),
        reverse=True,
    )

    diagnostic = sorted(
        all_metrics,
        key=lambda item: (
            item[
                "hybrid_rank1_delta_vs_full"
            ]
            >= -MAX_HYBRID_RANK1_DROP,
            item[
                "hybrid_rank1_delta_vs_low"
            ]
            >= 0.0,
            item[
                "projected_cpu_saving"
            ],
            item[
                "hybrid_rank1"
            ],
        ),
        reverse=True,
    )[0]

    return (
        passing[0]
        if passing
        else None,
        diagnostic,
    )


def export_state(model):
    return {
        name:
            tensor
            .detach()
            .cpu()
            .numpy()
            .tolist()

        for name, tensor
        in model.state_dict().items()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--source-script",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--multires-result",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--result",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    torch.set_num_threads(4)

    try:
        torch.set_num_interop_threads(1)

    except RuntimeError:
        pass

    source = load_source(
        args.source_script
    )

    records = source.parse_records(
        args.train_dir
    )

    pids = np.asarray(
        [
            item[1]
            for item
            in records
        ],
        dtype=np.int64,
    )

    camids = np.asarray(
        [
            item[2]
            for item
            in records
        ],
        dtype=np.int64,
    )

    train_indices = np.flatnonzero(
        np.isin(
            pids % 5,
            [
                0,
                1,
                2,
            ],
        )
    )

    cal_indices = np.flatnonzero(
        (
            pids % 5
        )
        == 3
    )

    val_indices = np.flatnonzero(
        (
            pids % 5
        )
        == 4
    )

    print(
        "Development split:"
    )

    print(
        " TRAIN:",
        len(train_indices),
        "images /",
        len(
            np.unique(
                pids[
                    train_indices
                ]
            )
        ),
        "identities",
    )

    print(
        " CAL:",
        len(cal_indices),
        "images /",
        len(
            np.unique(
                pids[
                    cal_indices
                ]
            )
        ),
        "identities",
    )

    print(
        " HELDOUT:",
        len(val_indices),
        "images /",
        len(
            np.unique(
                pids[
                    val_indices
                ]
            )
        ),
        "identities",
    )

    multires = json.loads(
        args.multires_result.read_text(
            encoding="utf-8"
        )
    )

    assert (
        multires[
            "standard_market_test_used"
        ]
        is False
    )

    timing = (
        multires[
            "timing"
        ][
            "4"
        ]
    )

    full_seconds = float(
        timing[
            "full_median_seconds"
        ]
    )

    low_seconds = float(
        timing[
            "low_160x80_including_resize_median_seconds"
        ]
    )

    low_full_ratio = (
        low_seconds
        / full_seconds
    )

    print()
    print(
        "Measured low/full compute ratio:",
        f"{low_full_ratio:.3%}",
    )

    print(
        "Fixed-low measured saving:",
        f"{1.0 - low_full_ratio:.3%}",
    )

    low_all = np.load(
        args.cache
        / "trained_160x80.npy",
        mmap_mode="r",
    )

    full_all = np.load(
        args.cache
        / "trained_256x128.npy",
        mmap_mode="r",
    )

    train_npz = np.load(
        args.cache
        / "train_router_features.npz"
    )

    cal_npz = np.load(
        args.cache
        / "calibration_router_features.npz"
    )

    train_x_raw = np.asarray(
        train_npz[
            "features"
        ],
        dtype=np.float32,
    )

    cal_x_raw = np.asarray(
        cal_npz[
            "features"
        ],
        dtype=np.float32,
    )

    (
        train_x,
        cal_x,
        mean,
        std,
    ) = standardize(
        train_x_raw,
        cal_x_raw,
    )

    train_low = np.asarray(
        train_npz[
            "low_correct"
        ]
    ).astype(bool)

    train_full = np.asarray(
        train_npz[
            "full_correct"
        ]
    ).astype(bool)

    cal_low = np.asarray(
        cal_npz[
            "low_correct"
        ]
    ).astype(bool)

    cal_full = np.asarray(
        cal_npz[
            "full_correct"
        ]
    ).astype(bool)

    train_y = make_classes(
        train_low,
        train_full,
    )

    cal_y = make_classes(
        cal_low,
        cal_full,
    )

    def describe_labels(
        name,
        labels,
    ):
        counts = np.bincount(
            labels,
            minlength=3,
        )

        total = len(labels)

        print()
        print(name)
        print(
            "  harm:",
            int(counts[0]),
            f"({counts[0] / total:.3%})",
        )

        print(
            "  neutral:",
            int(counts[1]),
            f"({counts[1] / total:.3%})",
        )

        print(
            "  rescue:",
            int(counts[2]),
            f"({counts[2] / total:.3%})",
        )

    describe_labels(
        "TRAIN utility labels:",
        train_y,
    )

    describe_labels(
        "CAL utility labels:",
        cal_y,
    )

    candidate_results = []
    candidate_models = {}

    print()
    print(
        "Training benefit-router candidates..."
    )

    for kind in (
        "linear",
        "mlp",
    ):
        print()
        print(
            "Candidate:",
            kind,
        )

        (
            model,
            loss,
            counts,
            class_weights,
        ) = fit_model(
            kind,
            train_x,
            train_y,
        )

        cal_score, cal_probs = (
            predict_score(
                model,
                cal_x,
            )
        )

        router_seconds = (
            benchmark_router(
                model,
                cal_x[0],
            )
        )

        router_ratio = (
            router_seconds
            / full_seconds
        )

        selected, diagnostic = (
            choose_threshold(
                cal_score,
                cal_low,
                cal_full,
                low_full_ratio,
                router_ratio,
            )
        )

        candidate = {
            "model":
                kind,

            "training_loss":
                loss,

            "training_class_counts":
                counts.tolist(),

            "training_class_weights":
                class_weights.tolist(),

            "router_median_seconds":
                router_seconds,

            "router_full_cost_ratio":
                router_ratio,

            "selected_threshold":
                selected,

            "diagnostic_threshold":
                diagnostic,
        }

        candidate_results.append(
            candidate
        )

        candidate_models[
            kind
        ] = model

        if selected is None:
            print(
                "  No threshold passed all calibration gates."
            )

            chosen = diagnostic

        else:
            print(
                "  PASSED calibration gates."
            )

            chosen = selected

        print(
            "  Refine rate:",
            f"{chosen['refine_rate']:.3%}",
        )

        print(
            "  Rescue capture:",
            f"{chosen['rescue_capture_rate']:.3%}",
        )

        print(
            "  Refined rescues:",
            chosen[
                "refined_rescues"
            ],
        )

        print(
            "  Refined harms:",
            chosen[
                "refined_harms"
            ],
        )

        print(
            "  Adaptive R1:",
            f"{chosen['hybrid_rank1']:.3%}",
        )

        print(
            "  Delta vs full:",
            f"{chosen['hybrid_rank1_delta_vs_full']:+.3%}",
        )

        print(
            "  Delta vs low:",
            f"{chosen['hybrid_rank1_delta_vs_low']:+.3%}",
        )

        print(
            "  CPU saving:",
            f"{chosen['projected_cpu_saving']:+.3%}",
        )

    passing = [
        item
        for item
        in candidate_results
        if item[
            "selected_threshold"
        ]
        is not None
    ]

    selected_name = None
    selected_cal = None
    heldout = None
    deployable = False
    status = (
        "failed_calibration"
    )

    if passing:
        passing.sort(
            key=lambda item: (
                item[
                    "selected_threshold"
                ][
                    "projected_cpu_saving"
                ],
                item[
                    "selected_threshold"
                ][
                    "hybrid_rank1"
                ],
                item[
                    "selected_threshold"
                ][
                    "net_correctness_gain_count"
                ],
            ),
            reverse=True,
        )

        winner = passing[0]

        selected_name = (
            winner[
                "model"
            ]
        )

        selected_cal = (
            winner[
                "selected_threshold"
            ]
        )

        selected_model = (
            candidate_models[
                selected_name
            ]
        )

        print()
        print(
            "SELECTED ON CALIBRATION:",
            selected_name,
        )

        print(
            "Threshold:",
            f"{selected_cal['threshold']:.6f}",
        )

        print()
        print(
            "FREEZING MODEL + THRESHOLD BEFORE HELD-OUT."
        )

        print()
        print(
            "Building held-out features..."
        )

        val_data = (
            source.build_split_data(
                "validation",
                val_indices,
                low_all,
                full_all,
                pids,
                camids,
                args.cache,
            )
        )

        val_x = (
            (
                np.asarray(
                    val_data[
                        "features"
                    ],
                    dtype=np.float32,
                )
                - mean
            )
            / std
        ).astype(
            np.float32
        )

        val_low = np.asarray(
            val_data[
                "low_correct"
            ]
        ).astype(bool)

        val_full = np.asarray(
            val_data[
                "full_correct"
            ]
        ).astype(bool)

        val_score, val_probs = (
            predict_score(
                selected_model,
                val_x,
            )
        )

        selected_router_seconds = (
            winner[
                "router_median_seconds"
            ]
        )

        heldout = evaluate(
            val_score,
            selected_cal[
                "threshold"
            ],
            val_low,
            val_full,
            low_full_ratio,
            selected_router_seconds
            / full_seconds,
        )

        heldout_pass = (
            heldout[
                "projected_cpu_saving"
            ]
            >= MIN_PROJECTED_CPU_SAVING

            and heldout[
                "hybrid_rank1_delta_vs_full"
            ]
            >= -MAX_HYBRID_RANK1_DROP

            and heldout[
                "hybrid_rank1_delta_vs_low"
            ]
            >= MIN_GAIN_VS_LOW

            and heldout[
                "refined_rescues"
            ]
            > heldout[
                "refined_harms"
            ]
        )

        if heldout_pass:
            deployable = True
            status = "passed"

        else:
            status = (
                "failed_heldout_validation"
            )

        print()
        print(
            "=" * 72
        )

        print(
            "HELD-OUT VALIDATION"
        )

        print(
            "=" * 72
        )

        for label, value in [
            (
                "Low Rank-1",
                f"{heldout['low_rank1']:.3%}",
            ),
            (
                "Full Rank-1",
                f"{heldout['full_rank1']:.3%}",
            ),
            (
                "Adaptive Rank-1",
                f"{heldout['hybrid_rank1']:.3%}",
            ),
            (
                "Delta vs full",
                f"{heldout['hybrid_rank1_delta_vs_full']:+.3%}",
            ),
            (
                "Delta vs low",
                f"{heldout['hybrid_rank1_delta_vs_low']:+.3%}",
            ),
            (
                "Refine rate",
                f"{heldout['refine_rate']:.3%}",
            ),
            (
                "Exit rate",
                f"{heldout['exit_rate']:.3%}",
            ),
            (
                "Rescue capture",
                f"{heldout['rescue_capture_rate']:.3%}",
            ),
            (
                "CPU saving",
                f"{heldout['projected_cpu_saving']:+.3%}",
            ),
        ]:
            print(
                f"{label}: {value}"
            )

        print(
            "Refined rescues:",
            heldout[
                "refined_rescues"
            ],
        )

        print(
            "Refined harms:",
            heldout[
                "refined_harms"
            ],
        )

        print(
            "Deployable:",
            deployable,
        )

    controller = None

    if selected_name is not None:
        model = candidate_models[
            selected_name
        ]

        controller = {
            "model_type":
                selected_name,

            "classes": {
                "0":
                    "refinement_harms",

                "1":
                    "refinement_neutral",

                "2":
                    "refinement_rescues",
            },

            "score":
                "P(rescue) - P(harm)",

            "threshold":
                float(
                    selected_cal[
                        "threshold"
                    ]
                ),

            "feature_names":
                source.FEATURE_NAMES,

            "normalization_mean":
                mean.tolist(),

            "normalization_std":
                std.tolist(),

            "state_dict":
                export_state(
                    model
                ),

            "uses_low_resolution_retrieval_only":
                True,

            "uses_ground_truth_at_inference":
                False,

            "uses_full_resolution_query_before_routing":
                False,
        }

    result = {
        "status":
            status,

        "deployable":
            deployable,

        "experiment":
            "refinement-benefit router",

        "date":
            "2026-11-22",

        "decision_target":
            (
                "Predict whether 256x128 refinement will "
                "improve correctness relative to the "
                "160x80 retrieval result."
            ),

        "training_labels": {
            "harm":
                (
                    "160x80 correct and "
                    "256x128 wrong"
                ),

            "neutral":
                (
                    "both resolutions have the "
                    "same correctness outcome"
                ),

            "rescue":
                (
                    "160x80 wrong and "
                    "256x128 correct"
                ),
        },

        "production_observable_features_only":
            True,

        "feature_names":
            source.FEATURE_NAMES,

        "gates": {
            "minimum_projected_cpu_saving":
                MIN_PROJECTED_CPU_SAVING,

            "maximum_rank1_drop_vs_full":
                MAX_HYBRID_RANK1_DROP,

            "minimum_gain_vs_low":
                MIN_GAIN_VS_LOW,

            "refined_rescues_must_exceed_harms":
                True,
        },

        "compute": {
            "full_seconds":
                full_seconds,

            "low_seconds":
                low_seconds,

            "low_full_ratio":
                low_full_ratio,

            "fixed_low_saving":
                1.0
                - low_full_ratio,
        },

        "candidate_models":
            candidate_results,

        "selected_model":
            selected_name,

        "selected_calibration":
            selected_cal,

        "heldout_validation":
            heldout,

        "controller":
            controller,

        "standard_market_test_used":
            False,

        "production_changed":
            False,
    }

    write_json(
        args.result,
        result,
    )

    config = {
        "version":
            1,

        "status":
            status,

        "deployable":
            deployable,

        "controller":
            controller,

        "gates":
            result[
                "gates"
            ],

        "standard_market_test_used":
            False,
    }

    write_json(
        args.config,
        config,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "STEP 29G2 RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "Status:",
        status,
    )

    print(
        "Deployable:",
        deployable,
    )

    print(
        "Selected router:",
        selected_name,
    )

    if heldout is not None:
        print(
            "Held-out adaptive Rank-1:",
            f"{heldout['hybrid_rank1']:.3%}",
        )

        print(
            "Held-out full Rank-1:",
            f"{heldout['full_rank1']:.3%}",
        )

        print(
            "Held-out CPU saving:",
            f"{heldout['projected_cpu_saving']:+.3%}",
        )

    print(
        "Standard Market test used:",
        False,
    )


if __name__ == "__main__":
    main()
