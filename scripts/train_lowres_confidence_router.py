from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid import ReIDModel


MODEL_ALIAS = "osnet_x1_0_msmt17_combineall"

LOW_SIZE = (160, 80)

EXPECTED_IMAGES = 12936
EXPECTED_IDS = 751

SEED = 29602

MIN_EXIT_PRECISION = 0.97
MAX_HYBRID_RANK1_DROP = 0.005
MIN_PROJECTED_CPU_SAVING = 0.05

TOP_K = 20

PATTERN = re.compile(
    r"^(-?\d+)_c(\d+)s\d+_"
)


FEATURE_NAMES = [
    "top1_similarity",
    "top2_similarity",
    "top1_top2_margin",
    "top1_top3_margin",
    "top1_top5_margin",
    "top1_top10_margin",
    "top5_mean",
    "top5_std",
    "top10_mean",
    "top10_std",
    "top20_mean",
    "top20_std",
    "top1_zscore_top20",
    "normalized_entropy_top20",
    "top1_softmax_share_top20",
    "near_top1_count_002",
    "near_top1_count_005",
    "top5_top10_mean_gap",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def parse_records(root: Path):
    records = []

    for path in sorted(
        root.glob("*.jpg")
    ):
        match = PATTERN.match(
            path.name
        )

        if match is None:
            raise RuntimeError(
                f"Unexpected Market filename: {path.name}"
            )

        pid = int(
            match.group(1)
        )

        camid = int(
            match.group(2)
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


class CaptureInput(RuntimeError):
    pass


def capture_preprocessed(
    model,
    backbone,
    paths,
):
    captured = {}

    def hook(
        module,
        inputs,
    ):
        captured[
            "array"
        ] = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        raise CaptureInput()

    handle = (
        backbone
        .register_forward_pre_hook(
            hook
        )
    )

    try:
        try:
            model.extract_embeddings_from_paths(
                [
                    str(path)
                    for path
                    in paths
                ]
            )

        except CaptureInput:
            pass

    finally:
        handle.remove()

    if "array" not in captured:
        raise RuntimeError(
            "Failed to capture preprocessing tensor."
        )

    return torch.from_numpy(
        np.array(
            captured[
                "array"
            ],
            dtype=np.float32,
            copy=True,
        )
    ).contiguous()


def build_embedding_cache(
    records,
    checkpoint_path: Path,
    cache_dir: Path,
):
    low_path = (
        cache_dir
        / "trained_160x80.npy"
    )

    full_path = (
        cache_dir
        / "trained_256x128.npy"
    )

    meta_path = (
        cache_dir
        / "embedding_cache.json"
    )

    checkpoint_sha = sha256_file(
        checkpoint_path
    )

    if (
        low_path.is_file()
        and full_path.is_file()
        and meta_path.is_file()
    ):
        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            meta.get(
                "checkpoint_sha256"
            )
            == checkpoint_sha
            and meta.get(
                "images"
            )
            == len(
                records
            )
        ):
            low = np.load(
                low_path,
                mmap_mode="r",
            )

            full = np.load(
                full_path,
                mmap_mode="r",
            )

            if (
                low.shape
                == (
                    len(
                        records
                    ),
                    512,
                )
                and full.shape
                == (
                    len(
                        records
                    ),
                    512,
                )
            ):
                print(
                    "Reusing verified embedding cache."
                )

                return (
                    np.asarray(
                        low,
                        dtype=np.float32,
                    ),
                    np.asarray(
                        full,
                        dtype=np.float32,
                    ),
                    meta,
                )

    print(
        "Building embeddings with trained epoch-8 model..."
    )

    capture_model = (
        ReIDModel.from_pretrained(
            MODEL_ALIAS,
            device="cpu",
        )
    )

    capture_backbone = (
        capture_model._backbone
    )

    trained_model = (
        ReIDModel.from_pretrained(
            MODEL_ALIAS,
            device="cpu",
        )
    )

    backbone = (
        trained_model._backbone
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if int(
        checkpoint.get(
            "epoch",
            -1,
        )
    ) != 8:
        raise RuntimeError(
            "Expected epoch-8 checkpoint."
        )

    backbone.load_state_dict(
        checkpoint[
            "backbone_state_dict"
        ]
    )

    backbone.eval()

    low_outputs = []
    full_outputs = []

    started = time.perf_counter()

    with torch.no_grad():
        for offset in range(
            0,
            len(
                records
            ),
            32,
        ):
            batch = records[
                offset:
                offset + 32
            ]

            paths = [
                item[0]
                for item
                in batch
            ]

            full_input = (
                capture_preprocessed(
                    capture_model,
                    capture_backbone,
                    paths,
                )
            )

            low_input = F.interpolate(
                full_input,
                size=LOW_SIZE,
                mode="bilinear",
                align_corners=False,
            )

            full_embedding = F.normalize(
                backbone(
                    full_input
                ),
                p=2,
                dim=1,
            )

            low_embedding = F.normalize(
                backbone(
                    low_input
                ),
                p=2,
                dim=1,
            )

            full_outputs.append(
                full_embedding
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            low_outputs.append(
                low_embedding
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            if (
                offset == 0
                or (
                    offset
                    + len(
                        batch
                    )
                )
                % 1024
                < 32
            ):
                print(
                    "Embedded:",
                    min(
                        offset
                        + len(
                            batch
                        ),
                        len(
                            records
                        ),
                    ),
                    "/",
                    len(
                        records
                    ),
                )

    low = np.concatenate(
        low_outputs,
        axis=0,
    )

    full = np.concatenate(
        full_outputs,
        axis=0,
    )

    np.save(
        low_path,
        low,
    )

    np.save(
        full_path,
        full,
    )

    meta = {
        "images":
            len(
                records
            ),

        "embedding_dim":
            int(
                low.shape[1]
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "low_resolution":
            [160, 80],

        "full_resolution":
            [256, 128],

        "seconds":
            float(
                time.perf_counter()
                - started
            ),
    }

    write_json(
        meta_path,
        meta,
    )

    return (
        low,
        full,
        meta,
    )


def top_features(
    values,
):
    top1 = values[:, 0]
    top2 = values[:, 1]

    top5 = values[:, :5]
    top10 = values[:, :10]
    top20 = values[:, :20]

    mean5 = np.mean(
        top5,
        axis=1,
    )

    std5 = np.std(
        top5,
        axis=1,
    )

    mean10 = np.mean(
        top10,
        axis=1,
    )

    std10 = np.std(
        top10,
        axis=1,
    )

    mean20 = np.mean(
        top20,
        axis=1,
    )

    std20 = np.std(
        top20,
        axis=1,
    )

    shifted = (
        top20
        - top20[:, :1]
    ) / 0.05

    shifted = np.clip(
        shifted,
        -80.0,
        0.0,
    )

    weights = np.exp(
        shifted
    )

    probs = (
        weights
        / np.sum(
            weights,
            axis=1,
            keepdims=True,
        )
    )

    entropy = -np.sum(
        probs
        * np.log(
            np.maximum(
                probs,
                1e-12,
            )
        ),
        axis=1,
    )

    entropy /= math.log(
        TOP_K
    )

    near002 = np.sum(
        top20
        >= (
            top1[:, None]
            - 0.02
        ),
        axis=1,
    )

    near005 = np.sum(
        top20
        >= (
            top1[:, None]
            - 0.05
        ),
        axis=1,
    )

    features = np.stack(
        [
            top1,
            top2,
            top1 - top2,
            top1 - values[:, 2],
            top1 - values[:, 4],
            top1 - values[:, 9],
            mean5,
            std5,
            mean10,
            std10,
            mean20,
            std20,
            (
                top1
                - mean20
            )
            / (
                std20
                + 1e-6
            ),
            entropy,
            probs[:, 0],
            near002.astype(
                np.float32
            )
            / TOP_K,
            near005.astype(
                np.float32
            )
            / TOP_K,
            mean5 - mean10,
        ],
        axis=1,
    )

    return features.astype(
        np.float32
    )


def build_split_data(
    name,
    indices,
    low_all,
    full_all,
    pids_all,
    camids_all,
    cache_dir,
):
    cache_path = (
        cache_dir
        / f"{name}_router_features.npz"
    )

    if cache_path.is_file():
        loaded = np.load(
            cache_path
        )

        print(
            "Reusing feature cache:",
            name,
        )

        return {
            key:
                np.asarray(
                    loaded[
                        key
                    ]
                )

            for key
            in loaded.files
        }

    low = np.asarray(
        low_all[
            indices
        ],
        dtype=np.float32,
    )

    full = np.asarray(
        full_all[
            indices
        ],
        dtype=np.float32,
    )

    pids = pids_all[
        indices
    ]

    camids = camids_all[
        indices
    ]

    n = len(
        indices
    )

    feature_parts = []
    low_correct_parts = []
    full_correct_parts = []
    agreement_parts = []

    for start in range(
        0,
        n,
        128,
    ):
        end = min(
            start + 128,
            n,
        )

        low_scores = (
            low[
                start:end
            ]
            @ full.T
        )

        full_scores = (
            full[
                start:end
            ]
            @ full.T
        )

        rows = np.arange(
            end - start
        )

        cols = np.arange(
            start,
            end,
        )

        low_scores[
            rows,
            cols,
        ] = -np.inf

        full_scores[
            rows,
            cols,
        ] = -np.inf

        low_raw_top = np.argmax(
            low_scores,
            axis=1,
        )

        full_raw_top = np.argmax(
            full_scores,
            axis=1,
        )

        agreement = (
            pids[
                low_raw_top
            ]
            ==
            pids[
                full_raw_top
            ]
        )

        candidate_idx = np.argpartition(
            low_scores,
            -TOP_K,
            axis=1,
        )[
            :,
            -TOP_K:
        ]

        candidate_values = (
            np.take_along_axis(
                low_scores,
                candidate_idx,
                axis=1,
            )
        )

        candidate_values.sort(
            axis=1
        )

        candidate_values = (
            candidate_values[
                :,
                ::-1
            ]
        )

        features = top_features(
            candidate_values
        )

        query_pids = pids[
            start:end
        ]

        query_camids = camids[
            start:end
        ]

        invalid = (
            (
                query_pids[
                    :,
                    None
                ]
                ==
                pids[
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
                ==
                camids[
                    None,
                    :
                ]
            )
        )

        low_metric = (
            low_scores.copy()
        )

        full_metric = (
            full_scores.copy()
        )

        low_metric[
            invalid
        ] = -np.inf

        full_metric[
            invalid
        ] = -np.inf

        low_top = np.argmax(
            low_metric,
            axis=1,
        )

        full_top = np.argmax(
            full_metric,
            axis=1,
        )

        low_correct = (
            pids[
                low_top
            ]
            ==
            query_pids
        )

        full_correct = (
            pids[
                full_top
            ]
            ==
            query_pids
        )

        feature_parts.append(
            features
        )

        low_correct_parts.append(
            low_correct
        )

        full_correct_parts.append(
            full_correct
        )

        agreement_parts.append(
            agreement
        )

    data = {
        "features":
            np.concatenate(
                feature_parts,
                axis=0,
            ).astype(
                np.float32
            ),

        "low_correct":
            np.concatenate(
                low_correct_parts
            ).astype(
                np.int8
            ),

        "full_correct":
            np.concatenate(
                full_correct_parts
            ).astype(
                np.int8
            ),

        "raw_agreement":
            np.concatenate(
                agreement_parts
            ).astype(
                np.int8
            ),

        "indices":
            np.asarray(
                indices,
                dtype=np.int64,
            ),
    }

    np.savez_compressed(
        cache_path,
        **data,
    )

    return data


class LinearRouter(nn.Module):
    def __init__(
        self,
        dimensions,
    ):
        super().__init__()

        self.net = nn.Linear(
            dimensions,
            1,
        )

    def forward(
        self,
        x,
    ):
        return (
            self.net(
                x
            )
            .squeeze(
                1
            )
        )


class MLPRouter(nn.Module):
    def __init__(
        self,
        dimensions,
    ):
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
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return (
            self.net(
                x
            )
            .squeeze(
                1
            )
        )


def fit_router(
    kind,
    train_x,
    train_y,
):
    torch.manual_seed(
        SEED
    )

    dimensions = (
        train_x.shape[1]
    )

    if kind == "linear":
        model = LinearRouter(
            dimensions
        )

        learning_rate = 0.02
        epochs = 600

    elif kind == "mlp":
        model = MLPRouter(
            dimensions
        )

        learning_rate = 0.005
        epochs = 800

    else:
        raise ValueError(
            kind
        )

    x = torch.from_numpy(
        train_x
    )

    y = torch.from_numpy(
        train_y.astype(
            np.float32
        )
    )

    positive_rate = float(
        np.mean(
            train_y
        )
    )

    positive_weight = (
        0.5
        / max(
            positive_rate,
            1e-6,
        )
    )

    negative_weight = (
        0.5
        / max(
            1.0
            - positive_rate,
            1e-6,
        )
    )

    sample_weight = torch.where(
        y > 0.5,
        torch.tensor(
            positive_weight,
            dtype=torch.float32,
        ),
        torch.tensor(
            negative_weight,
            dtype=torch.float32,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
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
            x
        )

        loss_vector = (
            F.binary_cross_entropy_with_logits(
                logits,
                y,
                reduction="none",
            )
        )

        loss = torch.mean(
            loss_vector
            * sample_weight
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
    )


def predict(
    model,
    x,
):
    with torch.no_grad():
        values = torch.sigmoid(
            model(
                torch.from_numpy(
                    x
                )
            )
        )

    return (
        values
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
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

    values = []

    model.eval()

    with torch.no_grad():
        for _ in range(
            100
        ):
            model(
                tensor
            )

        for _ in range(
            5000
        ):
            started = (
                time.perf_counter()
            )

            model(
                tensor
            )

            values.append(
                time.perf_counter()
                - started
            )

    return float(
        statistics.median(
            values
        )
    )


def evaluate_threshold(
    probabilities,
    threshold,
    low_correct,
    full_correct,
    agreement,
    low_full_ratio,
    router_full_ratio,
):
    exit_mask = (
        probabilities
        >= threshold
    )

    exit_rate = float(
        np.mean(
            exit_mask
        )
    )

    exited = int(
        np.sum(
            exit_mask
        )
    )

    if exited == 0:
        precision = 1.0
        agreement_precision = 1.0

    else:
        precision = float(
            np.mean(
                low_correct[
                    exit_mask
                ]
            )
        )

        agreement_precision = float(
            np.mean(
                agreement[
                    exit_mask
                ]
            )
        )

    hybrid_correct = np.where(
        exit_mask,
        low_correct,
        full_correct,
    )

    hybrid_rank1 = float(
        np.mean(
            hybrid_correct
        )
    )

    full_rank1 = float(
        np.mean(
            full_correct
        )
    )

    low_rank1 = float(
        np.mean(
            low_correct
        )
    )

    normalized_cost = (
        low_full_ratio
        + (
            1.0
            - exit_rate
        )
        + router_full_ratio
    )

    projected_saving = (
        1.0
        - normalized_cost
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "exit_rate":
            exit_rate,

        "refine_rate":
            1.0
            - exit_rate,

        "exit_precision":
            precision,

        "exit_agreement_precision":
            agreement_precision,

        "low_rank1":
            low_rank1,

        "full_rank1":
            full_rank1,

        "hybrid_rank1":
            hybrid_rank1,

        "hybrid_rank1_delta_vs_full":
            hybrid_rank1
            - full_rank1,

        "projected_cpu_saving":
            projected_saving,

        "normalized_progressive_cost":
            normalized_cost,
    }


def select_threshold(
    probabilities,
    low_correct,
    full_correct,
    agreement,
    low_full_ratio,
    router_full_ratio,
):
    quantiles = np.linspace(
        0.0,
        1.0,
        2001,
    )

    thresholds = np.unique(
        np.quantile(
            probabilities,
            quantiles,
        )
    )

    thresholds = np.concatenate(
        [
            [
                float(
                    np.min(
                        probabilities
                    )
                    - 1e-9
                )
            ],
            thresholds,
            [
                float(
                    np.max(
                        probabilities
                    )
                    + 1e-9
                )
            ],
        ]
    )

    passing = []
    frontier = []

    for threshold in thresholds:
        metrics = evaluate_threshold(
            probabilities,
            float(
                threshold
            ),
            low_correct,
            full_correct,
            agreement,
            low_full_ratio,
            router_full_ratio,
        )

        if (
            metrics[
                "exit_precision"
            ]
            >= MIN_EXIT_PRECISION
            and metrics[
                "hybrid_rank1_delta_vs_full"
            ]
            >= -MAX_HYBRID_RANK1_DROP
            and metrics[
                "projected_cpu_saving"
            ]
            >= MIN_PROJECTED_CPU_SAVING
        ):
            passing.append(
                metrics
            )

        frontier.append(
            metrics
        )

    passing.sort(
        key=lambda item: (
            item[
                "projected_cpu_saving"
            ],
            item[
                "exit_precision"
            ],
            item[
                "hybrid_rank1"
            ],
        ),
        reverse=True,
    )

    diagnostic = sorted(
        frontier,
        key=lambda item: (
            item[
                "exit_precision"
            ]
            >= 0.95,
            item[
                "hybrid_rank1_delta_vs_full"
            ]
            >= -0.01,
            item[
                "projected_cpu_saving"
            ],
            item[
                "exit_precision"
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


def export_model(
    model,
):
    state = {}

    for name, tensor in (
        model.state_dict().items()
    ):
        state[
            name
        ] = (
            tensor
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    return state


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--multires-result",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache",
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

    torch.set_num_threads(
        4
    )

    try:
        torch.set_num_interop_threads(
            1
        )

    except RuntimeError:
        pass

    args.cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = parse_records(
        args.train
    )

    if len(
        records
    ) != EXPECTED_IMAGES:
        raise RuntimeError(
            "Unexpected Market image count."
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

    if len(
        np.unique(
            pids
        )
    ) != EXPECTED_IDS:
        raise RuntimeError(
            "Unexpected identity count."
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
        "Identity-disjoint controller split:"
    )

    for label, indices in [
        (
            "TRAIN",
            train_indices,
        ),
        (
            "CALIBRATION",
            cal_indices,
        ),
        (
            "HELD-OUT VALIDATION",
            val_indices,
        ),
    ]:
        print(
            f"  {label}: "
            f"{len(indices)} images / "
            f"{len(np.unique(pids[indices]))} identities"
        )

    multires = json.loads(
        args.multires_result.read_text(
            encoding="utf-8"
        )
    )

    if (
        multires.get(
            "status"
        )
        != "passed"
    ):
        raise RuntimeError(
            "Multi-resolution model is not validated."
        )

    if (
        multires.get(
            "standard_market_test_used"
        )
        is not False
    ):
        raise RuntimeError(
            "Unexpected test-set state."
        )

    expected_sha = (
        multires[
            "checkpoint"
        ][
            "sha256"
        ]
    )

    actual_sha = sha256_file(
        args.checkpoint
    )

    if (
        actual_sha
        != expected_sha
    ):
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    timing4 = (
        multires[
            "timing"
        ][
            "4"
        ]
    )

    full_seconds = float(
        timing4[
            "full_median_seconds"
        ]
    )

    low_seconds = float(
        timing4[
            "low_160x80_including_resize_median_seconds"
        ]
    )

    low_full_ratio = (
        low_seconds
        / full_seconds
    )

    print()
    print(
        "Measured 4-thread low/full ratio:",
        f"{low_full_ratio:.3%}",
    )

    print(
        "Low-only measured saving:",
        f"{1.0 - low_full_ratio:.3%}",
    )

    low_all, full_all, embedding_meta = (
        build_embedding_cache(
            records,
            args.checkpoint,
            args.cache,
        )
    )

    print()
    print(
        "Building production-observable retrieval features..."
    )

    train_data = build_split_data(
        "train",
        train_indices,
        low_all,
        full_all,
        pids,
        camids,
        args.cache,
    )

    cal_data = build_split_data(
        "calibration",
        cal_indices,
        low_all,
        full_all,
        pids,
        camids,
        args.cache,
    )

    print()
    print(
        "Training controller candidates..."
    )

    train_x_raw = (
        train_data[
            "features"
        ]
    )

    mean = np.mean(
        train_x_raw,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.std(
        train_x_raw,
        axis=0,
    ).astype(
        np.float32
    )

    std = np.maximum(
        std,
        1e-6,
    )

    train_x = (
        (
            train_x_raw
            - mean
        )
        / std
    ).astype(
        np.float32
    )

    cal_x = (
        (
            cal_data[
                "features"
            ]
            - mean
        )
        / std
    ).astype(
        np.float32
    )

    train_y = (
        train_data[
            "low_correct"
        ]
        .astype(
            np.float32
        )
    )

    candidate_results = []
    candidate_models = {}

    for kind in (
        "linear",
        "mlp",
    ):
        print()
        print(
            "Candidate:",
            kind,
        )

        model, train_loss = fit_router(
            kind,
            train_x,
            train_y,
        )

        probabilities = predict(
            model,
            cal_x,
        )

        router_seconds = benchmark_router(
            model,
            cal_x[0],
        )

        router_full_ratio = (
            router_seconds
            / full_seconds
        )

        selected, diagnostic = (
            select_threshold(
                probabilities,
                cal_data[
                    "low_correct"
                ].astype(
                    bool
                ),
                cal_data[
                    "full_correct"
                ].astype(
                    bool
                ),
                cal_data[
                    "raw_agreement"
                ].astype(
                    bool
                ),
                low_full_ratio,
                router_full_ratio,
            )
        )

        summary = {
            "model":
                kind,

            "training_loss":
                train_loss,

            "router_median_seconds":
                router_seconds,

            "router_full_cost_ratio":
                router_full_ratio,

            "calibration_low_correct_rate":
                float(
                    np.mean(
                        cal_data[
                            "low_correct"
                        ]
                    )
                ),

            "calibration_full_correct_rate":
                float(
                    np.mean(
                        cal_data[
                            "full_correct"
                        ]
                    )
                ),

            "calibration_raw_low_full_agreement":
                float(
                    np.mean(
                        cal_data[
                            "raw_agreement"
                        ]
                    )
                ),

            "selected_threshold":
                selected,

            "best_diagnostic_threshold":
                diagnostic,
        }

        candidate_results.append(
            summary
        )

        candidate_models[
            kind
        ] = (
            model,
            router_seconds,
        )

        if selected is None:
            print(
                "  No calibration threshold passed all gates."
            )

            print(
                "  Diagnostic exit:",
                f"{diagnostic['exit_rate']:.3%}",
            )

            print(
                "  Diagnostic precision:",
                f"{diagnostic['exit_precision']:.3%}",
            )

            print(
                "  Diagnostic hybrid delta:",
                f"{diagnostic['hybrid_rank1_delta_vs_full']:+.3%}",
            )

            print(
                "  Diagnostic saving:",
                f"{diagnostic['projected_cpu_saving']:+.3%}",
            )

        else:
            print(
                "  Calibration threshold:",
                f"{selected['threshold']:.6f}",
            )

            print(
                "  Exit rate:",
                f"{selected['exit_rate']:.3%}",
            )

            print(
                "  Exit precision:",
                f"{selected['exit_precision']:.3%}",
            )

            print(
                "  Hybrid Rank-1:",
                f"{selected['hybrid_rank1']:.3%}",
            )

            print(
                "  Delta vs full:",
                f"{selected['hybrid_rank1_delta_vs_full']:+.3%}",
            )

            print(
                "  Projected CPU saving:",
                f"{selected['projected_cpu_saving']:+.3%}",
            )

    passing_candidates = [
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
    validation_metrics = None
    deployable = False
    status = (
        "failed_calibration"
    )

    if passing_candidates:
        passing_candidates.sort(
            key=lambda item: (
                item[
                    "selected_threshold"
                ][
                    "projected_cpu_saving"
                ],
                item[
                    "selected_threshold"
                ][
                    "exit_precision"
                ],
                item[
                    "selected_threshold"
                ][
                    "hybrid_rank1"
                ],
            ),
            reverse=True,
        )

        winner = (
            passing_candidates[0]
        )

        selected_name = winner[
            "model"
        ]

        selected_cal = winner[
            "selected_threshold"
        ]

        print()
        print(
            "Calibration selected model:",
            selected_name,
        )

        print(
            "Freezing controller and threshold before held-out validation."
        )

        print()
        print(
            "Building HELD-OUT validation features..."
        )

        val_data = build_split_data(
            "validation",
            val_indices,
            low_all,
            full_all,
            pids,
            camids,
            args.cache,
        )

        val_x = (
            (
                val_data[
                    "features"
                ]
                - mean
            )
            / std
        ).astype(
            np.float32
        )

        selected_model, router_seconds = (
            candidate_models[
                selected_name
            ]
        )

        val_probabilities = predict(
            selected_model,
            val_x,
        )

        validation_metrics = evaluate_threshold(
            val_probabilities,
            selected_cal[
                "threshold"
            ],
            val_data[
                "low_correct"
            ].astype(
                bool
            ),
            val_data[
                "full_correct"
            ].astype(
                bool
            ),
            val_data[
                "raw_agreement"
            ].astype(
                bool
            ),
            low_full_ratio,
            router_seconds
            / full_seconds,
        )

        heldout_pass = (
            validation_metrics[
                "exit_precision"
            ]
            >= MIN_EXIT_PRECISION
            and validation_metrics[
                "hybrid_rank1_delta_vs_full"
            ]
            >= -MAX_HYBRID_RANK1_DROP
            and validation_metrics[
                "projected_cpu_saving"
            ]
            >= MIN_PROJECTED_CPU_SAVING
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

        print(
            "Low-only Rank-1:",
            f"{validation_metrics['low_rank1']:.3%}",
        )

        print(
            "Full Rank-1:",
            f"{validation_metrics['full_rank1']:.3%}",
        )

        print(
            "Adaptive hybrid Rank-1:",
            f"{validation_metrics['hybrid_rank1']:.3%}",
        )

        print(
            "Adaptive delta vs full:",
            f"{validation_metrics['hybrid_rank1_delta_vs_full']:+.3%}",
        )

        print(
            "Early-exit rate:",
            f"{validation_metrics['exit_rate']:.3%}",
        )

        print(
            "Refinement rate:",
            f"{validation_metrics['refine_rate']:.3%}",
        )

        print(
            "Early-exit precision:",
            f"{validation_metrics['exit_precision']:.3%}",
        )

        print(
            "Exit low/full agreement:",
            f"{validation_metrics['exit_agreement_precision']:.3%}",
        )

        print(
            "Projected CPU saving:",
            f"{validation_metrics['projected_cpu_saving']:+.3%}",
        )

        print(
            "Deployable:",
            deployable,
        )

    selected_model_payload = None

    if selected_name is not None:
        selected_model = (
            candidate_models[
                selected_name
            ][0]
        )

        selected_model_payload = {
            "model_type":
                selected_name,

            "feature_names":
                FEATURE_NAMES,

            "normalization_mean":
                mean.tolist(),

            "normalization_std":
                std.tolist(),

            "threshold":
                float(
                    selected_cal[
                        "threshold"
                    ]
                ),

            "state_dict":
                export_model(
                    selected_model
                ),

            "production_inputs_only":
                True,

            "uses_pid_at_inference":
                False,

            "uses_camera_id_at_inference":
                False,

            "uses_full_resolution_query_at_inference_before_routing":
                False,
        }

    result = {
        "status":
            status,

        "deployable":
            deployable,

        "experiment":
            "production-observable 160x80 confidence controller",

        "date":
            "2026-11-22",

        "seed":
            SEED,

        "base_checkpoint_sha256":
            actual_sha,

        "architecture": {
            "low_resolution":
                [160, 80],

            "refinement_resolution":
                [256, 128],

            "gallery_embedding_resolution":
                [256, 128],

            "same_multiresolution_backbone":
                True,
        },

        "features": {
            "count":
                len(
                    FEATURE_NAMES
                ),

            "names":
                FEATURE_NAMES,

            "production_observable_only":
                True,

            "description":
                (
                    "Features are derived only from the "
                    "160x80 query retrieval score distribution. "
                    "Ground-truth identity and camera labels are "
                    "used only offline for training targets and "
                    "evaluation masks."
                ),
        },

        "development_split": {
            "gradient_training": {
                "images":
                    int(
                        len(
                            train_indices
                        )
                    ),

                "identities":
                    int(
                        len(
                            np.unique(
                                pids[
                                    train_indices
                                ]
                            )
                        )
                    ),

                "rule":
                    "PID % 5 in {0,1,2}",
            },

            "calibration": {
                "images":
                    int(
                        len(
                            cal_indices
                        )
                    ),

                "identities":
                    int(
                        len(
                            np.unique(
                                pids[
                                    cal_indices
                                ]
                            )
                        )
                    ),

                "rule":
                    "PID % 5 == 3",
            },

            "heldout_validation": {
                "images":
                    int(
                        len(
                            val_indices
                        )
                    ),

                "identities":
                    int(
                        len(
                            np.unique(
                                pids[
                                    val_indices
                                ]
                            )
                        )
                    ),

                "rule":
                    "PID % 5 == 4",
            },

            "standard_market_test_used":
                False,
        },

        "gates": {
            "minimum_exit_precision":
                MIN_EXIT_PRECISION,

            "maximum_hybrid_rank1_drop":
                MAX_HYBRID_RANK1_DROP,

            "minimum_projected_cpu_saving":
                MIN_PROJECTED_CPU_SAVING,
        },

        "measured_compute": {
            "four_thread_full_seconds":
                full_seconds,

            "four_thread_low_seconds":
                low_seconds,

            "low_full_ratio":
                low_full_ratio,

            "low_only_saving":
                1.0
                - low_full_ratio,
        },

        "embedding_cache":
            embedding_meta,

        "candidate_models":
            candidate_results,

        "selected_model":
            selected_name,

        "selected_calibration":
            selected_cal,

        "heldout_validation":
            validation_metrics,

        "controller":
            selected_model_payload,

        "standard_market_test_used":
            False,

        "scientific_note":
            (
                "The controller is selected using calibration "
                "identities and evaluated once on held-out "
                "development identities. The standard Market-1501 "
                "query/gallery test remains untouched."
            ),
    }

    write_json(
        args.result,
        result,
    )

    config = {
        "version":
            1,

        "deployable":
            deployable,

        "status":
            status,

        "controller":
            selected_model_payload,

        "gates": {
            "minimum_exit_precision":
                MIN_EXIT_PRECISION,

            "maximum_hybrid_rank1_drop":
                MAX_HYBRID_RANK1_DROP,

            "minimum_projected_cpu_saving":
                MIN_PROJECTED_CPU_SAVING,
        },

        "source_checkpoint_sha256":
            actual_sha,

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
        "STEP 29G RESULT"
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
        "Selected controller:",
        selected_name,
    )

    if validation_metrics is not None:
        print(
            "Held-out exit rate:",
            f"{validation_metrics['exit_rate']:.3%}",
        )

        print(
            "Held-out exit precision:",
            f"{validation_metrics['exit_precision']:.3%}",
        )

        print(
            "Held-out adaptive Rank-1:",
            f"{validation_metrics['hybrid_rank1']:.3%}",
        )

        print(
            "Held-out full Rank-1:",
            f"{validation_metrics['full_rank1']:.3%}",
        )

        print(
            "Held-out Rank-1 delta:",
            f"{validation_metrics['hybrid_rank1_delta_vs_full']:+.3%}",
        )

        print(
            "Projected CPU saving:",
            f"{validation_metrics['projected_cpu_saving']:+.3%}",
        )

    print(
        "Standard Market test used:",
        False,
    )


if __name__ == "__main__":
    main()
