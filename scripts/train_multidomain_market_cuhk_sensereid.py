#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reid import ReIDModel


PROJECT = Path("/root/rac-reid")
RUNTIME = PROJECT / ".runtime"

MARKET_ROOT = (
    PROJECT
    / "data/datasets/market1501/"
    "Market-1501-v15.09.15/"
    "bounding_box_train"
)

CUHK_ROOT = (
    PROJECT
    / "data/datasets/cuhk03/cuhk03-np/"
    "detected/bounding_box_train"
)

MARKET_MANIFEST = (
    "9f413d5102106d9a1ca192a372c118ff"
    "d0ff73039f4fb36ed3b431838853e9af"
)

CUHK_MANIFEST = (
    "daab40687766fc4ccad1c08cce65b64c"
    "e06142c3ed86ed9e7589ca4cc21f3aa6"
)

PREREG_COMMIT = (
    "8be70b58a595bb3ce8fee1992b78e691"
    "257a255e"
)

CONFIG_PATH = (
    PROJECT
    / "configs/"
    "experiment4_multidomain_market_cuhk_sensereid.json"
)

EXP = (
    RUNTIME
    / "experiments/"
    "multidomain-market-cuhk-sensereid"
)

PROGRESS_PATH = (
    EXP
    / "training_progress.pt"
)

HISTORY_PATH = (
    EXP
    / "training_history.json"
)

BASELINE_PATH = (
    EXP
    / "development_baseline.json"
)

COMPLETE_PATH = (
    EXP
    / "TRAINING_COMPLETE.json"
)

MODEL_PATH = (
    RUNTIME
    / "models/multidomain-market-cuhk/"
    "osnet_x1_0_multidomain_market_cuhk.pt"
)

BASE_MODEL = (
    "osnet_x1_0_msmt17_combineall"
)

LOW_SIZE = (
    160,
    80,
)

FULL_SIZE = (
    256,
    128,
)

EMBED_DIM = 512

SEED = 42
EPOCHS = 8
BATCH_SIZE = 16

BACKBONE_LR = 5e-5
CLASSIFIER_LR = 5e-4
WEIGHT_DECAY = 5e-4

W_LOW = 1.00
W_FULL = 0.25
W_REL = 0.50
W_CROSS = 0.25
W_CLS = 0.50

FILENAME_PATTERN = re.compile(
    r"^(-?\d+)_c(\d+)"
)

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


class CaptureInput(RuntimeError):
    pass


def write_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
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

    os.replace(
        temp,
        path,
    )


def sha256_file(
    path: Path,
):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def filename_size_manifest(
    paths,
):
    h = hashlib.sha256()

    for path in paths:
        h.update(
            (
                f"{path.name} "
                f"{path.stat().st_size}\n"
            ).encode("utf-8")
        )

    return h.hexdigest()


def load_domain(
    *,
    name,
    root,
    expected_images,
    expected_ids,
    expected_manifest,
):
    paths = sorted(
        [
            p
            for p in root.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in EXTENSIONS
            )
        ],
        key=lambda p: p.name,
    )

    if len(paths) != expected_images:
        raise RuntimeError(
            f"{name}: expected "
            f"{expected_images} images, "
            f"found {len(paths)}"
        )

    manifest = filename_size_manifest(
        paths
    )

    if manifest != expected_manifest:
        raise RuntimeError(
            f"{name}: manifest changed."
        )

    records = []

    for path in paths:
        match = FILENAME_PATTERN.match(
            path.name
        )

        if match is None:
            raise RuntimeError(
                f"{name}: unexpected filename "
                f"{path.name}"
            )

        pid = int(
            match.group(1)
        )

        camid = int(
            match.group(2)
        )

        if pid < 0:
            continue

        records.append(
            {
                "path":
                    path,

                "pid":
                    pid,

                "camid":
                    camid,

                "domain":
                    name,
            }
        )

    pids = np.asarray(
        [
            row["pid"]
            for row in records
        ],
        dtype=np.int64,
    )

    camids = np.asarray(
        [
            row["camid"]
            for row in records
        ],
        dtype=np.int64,
    )

    identities = set(
        pids.tolist()
    )

    if len(identities) != expected_ids:
        raise RuntimeError(
            f"{name}: expected "
            f"{expected_ids} identities, "
            f"found {len(identities)}"
        )

    gradient = np.flatnonzero(
        np.isin(
            pids % 5,
            [
                0,
                1,
                2,
            ],
        )
    )

    calibration = np.flatnonzero(
        (pids % 5)
        == 3
    )

    held_out = np.flatnonzero(
        (pids % 5)
        == 4
    )

    return {
        "name":
            name,

        "records":
            records,

        "pids":
            pids,

        "camids":
            camids,

        "gradient":
            gradient,

        "calibration":
            calibration,

        "held_out":
            held_out,

        "manifest":
            manifest,
    }


def capture_input(
    capture_model,
    capture_backbone,
    paths,
):
    captured = {}

    def hook(
        module,
        inputs,
    ):
        captured["array"] = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        raise CaptureInput()

    handle = (
        capture_backbone
        .register_forward_pre_hook(
            hook
        )
    )

    try:
        try:
            capture_model.extract_embeddings_from_paths(
                [
                    str(path)
                    for path in paths
                ]
            )

        except CaptureInput:
            pass

    finally:
        handle.remove()

    if "array" not in captured:
        raise RuntimeError(
            "Could not capture preprocessing tensor."
        )

    tensor = torch.from_numpy(
        np.asarray(
            captured["array"],
            dtype=np.float32,
        )
    ).contiguous()

    if tensor.ndim != 4:
        raise RuntimeError(
            f"Unexpected preprocessing tensor "
            f"shape {tuple(tensor.shape)}"
        )

    return tensor


def backbone_embedding(
    backbone,
    tensor,
):
    features = (
        backbone._featuremaps(
            tensor
        )
    )

    vector = (
        backbone.global_avgpool(
            features
        )
        .view(
            features.size(0),
            -1,
        )
    )

    vector = backbone.fc(
        vector
    )

    if (
        vector.ndim != 2
        or vector.shape[1]
        != EMBED_DIM
    ):
        raise RuntimeError(
            f"Unexpected embedding shape "
            f"{tuple(vector.shape)}"
        )

    return vector


def normalized_embedding(
    backbone,
    tensor,
):
    raw = backbone_embedding(
        backbone,
        tensor,
    )

    norm = F.normalize(
        raw,
        p=2,
        dim=1,
    )

    return (
        raw,
        norm,
    )


def cosine_alignment_loss(
    student_norm,
    teacher_norm,
):
    return (
        1.0
        - torch.sum(
            student_norm
            * teacher_norm,
            dim=1,
        ).mean()
    )


def relational_similarity_loss(
    low_norm,
    full_norm,
    teacher_norm,
):
    teacher_rel = (
        teacher_norm
        @ teacher_norm.T
    ).detach()

    low_rel = (
        low_norm
        @ low_norm.T
    )

    full_rel = (
        full_norm
        @ full_norm.T
    )

    return 0.5 * (
        F.mse_loss(
            low_rel,
            teacher_rel,
        )
        +
        F.mse_loss(
            full_rel,
            teacher_rel,
        )
    )


def cross_resolution_loss(
    low_norm,
    full_norm,
):
    return (
        1.0
        - torch.sum(
            low_norm
            * full_norm,
            dim=1,
        ).mean()
    )


def classification_loss(
    classifier,
    low_raw,
    full_raw,
    labels,
):
    return 0.5 * (
        F.cross_entropy(
            classifier(
                low_raw
            ),
            labels,
        )
        +
        F.cross_entropy(
            classifier(
                full_raw
            ),
            labels,
        )
    )


def total_objective(
    *,
    low_norm,
    full_norm,
    teacher_norm,
    classifier,
    low_raw,
    full_raw,
    labels,
):
    low_teacher = (
        cosine_alignment_loss(
            low_norm,
            teacher_norm,
        )
    )

    full_teacher = (
        cosine_alignment_loss(
            full_norm,
            teacher_norm,
        )
    )

    relational = (
        relational_similarity_loss(
            low_norm,
            full_norm,
            teacher_norm,
        )
    )

    cross = (
        cross_resolution_loss(
            low_norm,
            full_norm,
        )
    )

    classification = (
        classification_loss(
            classifier,
            low_raw,
            full_raw,
            labels,
        )
    )

    total = (
        W_LOW
        * low_teacher

        + W_FULL
        * full_teacher

        + W_REL
        * relational

        + W_CROSS
        * cross

        + W_CLS
        * classification
    )

    return {
        "total":
            total,

        "low_teacher_cosine":
            low_teacher,

        "full_teacher_cosine":
            full_teacher,

        "relational_similarity":
            relational,

        "cross_resolution_consistency":
            cross,

        "identity_classification":
            classification,
    }


def evaluate_pair(
    low,
    full,
    pids,
    camids,
):
    n = len(
        pids
    )

    low_scores = (
        low
        @ full.T
    )

    full_scores = (
        full
        @ full.T
    )

    rows = np.arange(
        n
    )

    low_scores[
        rows,
        rows,
    ] = -np.inf

    full_scores[
        rows,
        rows,
    ] = -np.inf

    low_raw = np.argmax(
        low_scores,
        axis=1,
    )

    full_raw = np.argmax(
        full_scores,
        axis=1,
    )

    agreement = float(
        np.mean(
            pids[
                low_raw
            ]
            ==
            pids[
                full_raw
            ]
        )
    )

    invalid = (
        (
            pids[:, None]
            ==
            pids[None, :]
        )
        &
        (
            camids[:, None]
            ==
            camids[None, :]
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

    low_rank1 = float(
        np.mean(
            pids[
                low_top
            ]
            ==
            pids
        )
    )

    full_rank1 = float(
        np.mean(
            pids[
                full_top
            ]
            ==
            pids
        )
    )

    cosine = float(
        np.mean(
            np.sum(
                low
                * full,
                axis=1,
            )
        )
    )

    return {
        "low_rank1":
            low_rank1,

        "full_rank1":
            full_rank1,

        "rank1_gap":
            (
                full_rank1
                - low_rank1
            ),

        "low_full_identity_agreement":
            agreement,

        "low_full_embedding_cosine":
            cosine,
    }


def embed_indices(
    *,
    capture_model,
    capture_backbone,
    backbone,
    domain,
    indices,
):
    low_outputs = []
    full_outputs = []

    backbone.eval()

    with torch.no_grad():

        for offset in range(
            0,
            len(indices),
            32,
        ):
            batch_indices = (
                indices[
                    offset:
                    offset + 32
                ]
            )

            paths = [
                domain[
                    "records"
                ][
                    int(index)
                ][
                    "path"
                ]
                for index
                in batch_indices
            ]

            full_input = capture_input(
                capture_model,
                capture_backbone,
                paths,
            )

            low_input = F.interpolate(
                full_input,
                size=LOW_SIZE,
                mode="bilinear",
                align_corners=False,
            )

            _, full_norm = (
                normalized_embedding(
                    backbone,
                    full_input,
                )
            )

            _, low_norm = (
                normalized_embedding(
                    backbone,
                    low_input,
                )
            )

            full_outputs.append(
                full_norm
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            low_outputs.append(
                low_norm
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

    return (
        np.concatenate(
            low_outputs,
            axis=0,
        ),

        np.concatenate(
            full_outputs,
            axis=0,
        ),
    )


def evaluate_domain(
    *,
    capture_model,
    capture_backbone,
    backbone,
    domain,
    indices,
):
    low, full = embed_indices(
        capture_model=capture_model,
        capture_backbone=capture_backbone,
        backbone=backbone,
        domain=domain,
        indices=indices,
    )

    return evaluate_pair(
        low,
        full,
        domain[
            "pids"
        ][
            indices
        ],
        domain[
            "camids"
        ][
            indices
        ],
    )


def macro_metrics(
    market,
    cuhk,
):
    keys = [
        "low_rank1",
        "full_rank1",
        "low_full_identity_agreement",
        "low_full_embedding_cosine",
    ]

    out = {}

    for key in keys:
        out[
            key
        ] = (
            float(
                market[
                    key
                ]
            )
            +
            float(
                cuhk[
                    key
                ]
            )
        ) / 2.0

    out[
        "rank1_gap"
    ] = (
        out[
            "full_rank1"
        ]
        -
        out[
            "low_rank1"
        ]
    )

    return out


def evaluate_source_pair(
    *,
    capture_model,
    capture_backbone,
    backbone,
    market,
    cuhk,
    split_name,
):
    market_result = evaluate_domain(
        capture_model=capture_model,
        capture_backbone=capture_backbone,
        backbone=backbone,
        domain=market,
        indices=market[
            split_name
        ],
    )

    cuhk_result = evaluate_domain(
        capture_model=capture_model,
        capture_backbone=capture_backbone,
        backbone=backbone,
        domain=cuhk,
        indices=cuhk[
            split_name
        ],
    )

    return {
        "market1501":
            market_result,

        "cuhk03":
            cuhk_result,

        "macro":
            macro_metrics(
                market_result,
                cuhk_result,
            ),
    }


def freeze_early(
    backbone,
):
    for parameter in (
        backbone.conv1.parameters()
    ):
        parameter.requires_grad = (
            False
        )

    for parameter in (
        backbone.conv2.parameters()
    ):
        parameter.requires_grad = (
            False
        )


def set_training_mode(
    backbone,
):
    backbone.train()

    backbone.conv1.eval()
    backbone.conv2.eval()


def self_test():
    torch.manual_seed(
        123
    )

    batch = 6
    classes = 4

    teacher = F.normalize(
        torch.randn(
            batch,
            EMBED_DIM,
        ),
        p=2,
        dim=1,
    )

    low_raw = torch.randn(
        batch,
        EMBED_DIM,
        requires_grad=True,
    )

    full_raw = torch.randn(
        batch,
        EMBED_DIM,
        requires_grad=True,
    )

    low_norm = F.normalize(
        low_raw,
        p=2,
        dim=1,
    )

    full_norm = F.normalize(
        full_raw,
        p=2,
        dim=1,
    )

    classifier = nn.Linear(
        EMBED_DIM,
        classes,
    )

    labels = torch.tensor(
        [
            0,
            1,
            2,
            3,
            0,
            1,
        ],
        dtype=torch.long,
    )

    losses = total_objective(
        low_norm=low_norm,
        full_norm=full_norm,
        teacher_norm=teacher,
        classifier=classifier,
        low_raw=low_raw,
        full_raw=full_raw,
        labels=labels,
    )

    for name, value in (
        losses.items()
    ):
        if not torch.isfinite(
            value
        ):
            raise RuntimeError(
                f"Non-finite loss: "
                f"{name}"
            )

    losses[
        "total"
    ].backward()

    assert low_raw.grad is not None
    assert full_raw.grad is not None
    assert classifier.weight.grad is not None

    print("SELF TEST PASS")

    for name, value in (
        losses.items()
    ):
        print(
            f"  {name}: "
            f"{float(value.detach()):.6f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    torch.set_num_threads(
        4
    )

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        config[
            "status"
        ]
        !=
        "PRE_REGISTERED_BEFORE_TRAINING"
    ):
        raise RuntimeError(
            "Experiment-4 protocol status changed."
        )

    if config["training"]["epochs"] != EPOCHS:
        raise RuntimeError(
            "Epoch count mismatch."
        )

    if config["training"]["batch_size"] != BATCH_SIZE:
        raise RuntimeError(
            "Batch-size mismatch."
        )

    current_head = (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=PROJECT,
            text=True,
        )
        .strip()
    )

    if current_head == PREREG_COMMIT:
        raise RuntimeError(
            "Implementation was not committed "
            "before training."
        )

    market = load_domain(
        name="market1501",
        root=MARKET_ROOT,
        expected_images=12936,
        expected_ids=751,
        expected_manifest=MARKET_MANIFEST,
    )

    cuhk = load_domain(
        name="cuhk03",
        root=CUHK_ROOT,
        expected_images=7365,
        expected_ids=767,
        expected_manifest=CUHK_MANIFEST,
    )

    expected_split = {
        "market1501": {
            "gradient":
                (7811, 449),

            "calibration":
                (2515, 144),

            "held_out":
                (2610, 158),
        },

        "cuhk03": {
            "gradient":
                (4352, 452),

            "calibration":
                (1515, 157),

            "held_out":
                (1498, 158),
        },
    }

    for domain in (
        market,
        cuhk,
    ):
        for split in (
            "gradient",
            "calibration",
            "held_out",
        ):
            indices = domain[
                split
            ]

            pids = {
                int(
                    domain[
                        "pids"
                    ][
                        int(index)
                    ]
                )
                for index
                in indices
            }

            expected_images, expected_ids = (
                expected_split[
                    domain[
                        "name"
                    ]
                ][
                    split
                ]
            )

            assert (
                len(indices)
                == expected_images
            )

            assert (
                len(pids)
                == expected_ids
            )

    gradient_population = []

    for domain in (
        market,
        cuhk,
    ):
        for index in domain[
            "gradient"
        ]:
            gradient_population.append(
                (
                    domain[
                        "name"
                    ],
                    int(
                        index
                    ),
                )
            )

    if len(
        gradient_population
    ) != (
        7811
        + 4352
    ):
        raise RuntimeError(
            "Combined gradient population mismatch."
        )

    train_pid_keys = []

    for domain in (
        market,
        cuhk,
    ):
        pids = sorted({
            int(
                domain[
                    "pids"
                ][
                    int(index)
                ]
            )
            for index
            in domain[
                "gradient"
            ]
        })

        train_pid_keys.extend(
            [
                (
                    domain[
                        "name"
                    ],
                    pid,
                )
                for pid in pids
            ]
        )

    if len(
        train_pid_keys
    ) != (
        449
        + 452
    ):
        raise RuntimeError(
            "Classifier identity count mismatch."
        )

    class_map = {
        key:
            class_index
        for class_index, key
        in enumerate(
            train_pid_keys
        )
    }

    domain_by_name = {
        "market1501":
            market,

        "cuhk03":
            cuhk,
    }

    EXP.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading original pretrained "
        "OSNet models..."
    )

    capture_model = (
        ReIDModel.from_pretrained(
            BASE_MODEL,
            device="cpu",
        )
    )

    capture_backbone = (
        capture_model._backbone
    )

    capture_backbone.eval()

    teacher_model = (
        ReIDModel.from_pretrained(
            BASE_MODEL,
            device="cpu",
        )
    )

    teacher_backbone = (
        teacher_model._backbone
    )

    teacher_backbone.eval()

    student_model = (
        ReIDModel.from_pretrained(
            BASE_MODEL,
            device="cpu",
        )
    )

    student_backbone = (
        student_model._backbone
    )

    freeze_early(
        student_backbone
    )

    classifier = nn.Linear(
        EMBED_DIM,
        len(
            class_map
        ),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameter
                    for parameter
                    in student_backbone.parameters()
                    if parameter.requires_grad
                ],
                "lr":
                    BACKBONE_LR,
            },

            {
                "params":
                    classifier.parameters(),

                "lr":
                    CLASSIFIER_LR,
            },
        ],
        weight_decay=(
            WEIGHT_DECAY
        ),
    )

    if BASELINE_PATH.exists():
        baseline = json.loads(
            BASELINE_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:
        print()
        print(
            "Computing source-domain "
            "pre-training calibration baseline..."
        )

        baseline_calibration = (
            evaluate_source_pair(
                capture_model=capture_model,
                capture_backbone=capture_backbone,
                backbone=teacher_backbone,
                market=market,
                cuhk=cuhk,
                split_name="calibration",
            )
        )

        baseline = {
            "source":
                BASE_MODEL,

            "gradient_updates":
                0,

            "calibration":
                baseline_calibration,

            "held_out":
                None,

            "held_out_evaluated_before_selection":
                False,
        }

        write_json(
            BASELINE_PATH,
            baseline,
        )

        print(
            "Baseline macro Efficient R1:",
            f"{baseline_calibration['macro']['low_rank1']:.3%}",
        )

        print(
            "Baseline macro Full R1:",
            f"{baseline_calibration['macro']['full_rank1']:.3%}",
        )

    history = []
    start_epoch = 1

    best_key = None
    best_epoch = None

    best_backbone_state = None
    best_classifier_state = None

    if PROGRESS_PATH.exists():
        progress = torch.load(
            PROGRESS_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if (
            progress[
                "implementation_commit"
            ]
            != current_head
        ):
            raise RuntimeError(
                "Training progress belongs to "
                "another implementation commit."
            )

        student_backbone.load_state_dict(
            progress[
                "backbone_state_dict"
            ]
        )

        classifier.load_state_dict(
            progress[
                "classifier_state_dict"
            ]
        )

        optimizer.load_state_dict(
            progress[
                "optimizer_state_dict"
            ]
        )

        history = progress[
            "epoch_history"
        ]

        start_epoch = (
            int(
                progress[
                    "completed_epoch"
                ]
            )
            + 1
        )

        best_epoch = (
            progress.get(
                "best_epoch"
            )
        )

        raw_key = progress.get(
            "best_key"
        )

        if raw_key is not None:
            best_key = tuple(
                raw_key
            )

        best_backbone_state = (
            progress.get(
                "best_backbone_state_dict"
            )
        )

        best_classifier_state = (
            progress.get(
                "best_classifier_state_dict"
            )
        )

        print(
            "Resuming after epoch:",
            start_epoch - 1,
        )

    print()
    print(
        "========================================"
    )

    print(
        " EXPERIMENT 4 TRAINING"
    )

    print(
        "========================================"
    )

    print(
        "Market gradient images:",
        len(
            market[
                "gradient"
            ]
        ),
    )

    print(
        "CUHK gradient images:",
        len(
            cuhk[
                "gradient"
            ]
        ),
    )

    print(
        "Combined gradient images:",
        len(
            gradient_population
        ),
    )

    print(
        "Classifier classes:",
        len(
            class_map
        ),
    )

    print(
        "Implementation commit:",
        current_head,
    )

    print()

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        epoch_started = (
            time.perf_counter()
        )

        set_training_mode(
            student_backbone
        )

        classifier.train()

        rng = np.random.default_rng(
            SEED + epoch
        )

        order = np.arange(
            len(
                gradient_population
            )
        )

        rng.shuffle(
            order
        )

        totals = {
            "total": 0.0,
            "low_teacher_cosine": 0.0,
            "full_teacher_cosine": 0.0,
            "relational_similarity": 0.0,
            "cross_resolution_consistency": 0.0,
            "identity_classification": 0.0,
        }

        samples_seen = 0

        for offset in range(
            0,
            len(order),
            BATCH_SIZE,
        ):
            batch_positions = order[
                offset:
                offset + BATCH_SIZE
            ]

            batch_refs = [
                gradient_population[
                    int(position)
                ]
                for position
                in batch_positions
            ]

            rows = []

            for domain_name, index in batch_refs:
                domain = domain_by_name[
                    domain_name
                ]

                row = domain[
                    "records"
                ][
                    index
                ]

                rows.append(
                    row
                )

            paths = [
                row[
                    "path"
                ]
                for row in rows
            ]

            labels = torch.tensor(
                [
                    class_map[
                        (
                            row[
                                "domain"
                            ],
                            row[
                                "pid"
                            ],
                        )
                    ]
                    for row in rows
                ],
                dtype=torch.long,
            )

            full_input = capture_input(
                capture_model,
                capture_backbone,
                paths,
            )

            low_input = F.interpolate(
                full_input,
                size=LOW_SIZE,
                mode="bilinear",
                align_corners=False,
            )

            with torch.no_grad():
                _, teacher_norm = (
                    normalized_embedding(
                        teacher_backbone,
                        full_input,
                    )
                )

            low_raw, low_norm = (
                normalized_embedding(
                    student_backbone,
                    low_input,
                )
            )

            full_raw, full_norm = (
                normalized_embedding(
                    student_backbone,
                    full_input,
                )
            )

            losses = total_objective(
                low_norm=low_norm,
                full_norm=full_norm,
                teacher_norm=teacher_norm,
                classifier=classifier,
                low_raw=low_raw,
                full_raw=full_raw,
                labels=labels,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            losses[
                "total"
            ].backward()

            optimizer.step()

            batch_size = len(
                rows
            )

            samples_seen += (
                batch_size
            )

            for name, value in (
                losses.items()
            ):
                totals[
                    name
                ] += (
                    float(
                        value
                        .detach()
                        .cpu()
                    )
                    * batch_size
                )

            batch_number = (
                offset
                // BATCH_SIZE
                + 1
            )

            if (
                batch_number % 100
                == 0
                or samples_seen
                == len(order)
            ):
                print(
                    f"Epoch {epoch}/{EPOCHS}: "
                    f"{samples_seen}/"
                    f"{len(order)} "
                    f"loss="
                    f"{totals['total']/samples_seen:.5f}",
                    flush=True,
                )

        epoch_losses = {
            name:
                value
                / samples_seen
            for name, value
            in totals.items()
        }

        print()
        print(
            f"Epoch {epoch}: "
            "source calibration..."
        )

        calibration = (
            evaluate_source_pair(
                capture_model=capture_model,
                capture_backbone=capture_backbone,
                backbone=student_backbone,
                market=market,
                cuhk=cuhk,
                split_name="calibration",
            )
        )

        macro = calibration[
            "macro"
        ]

        selection_key = (
            macro[
                "low_rank1"
            ],

            macro[
                "full_rank1"
            ],

            macro[
                "low_full_embedding_cosine"
            ],
        )

        selected = (
            best_key is None
            or selection_key
            > best_key
        )

        if selected:
            best_key = (
                selection_key
            )

            best_epoch = (
                epoch
            )

            best_backbone_state = {
                name:
                    tensor
                    .detach()
                    .cpu()
                    .clone()
                for name, tensor
                in student_backbone
                .state_dict()
                .items()
            }

            best_classifier_state = {
                name:
                    tensor
                    .detach()
                    .cpu()
                    .clone()
                for name, tensor
                in classifier
                .state_dict()
                .items()
            }

        record = {
            "epoch":
                epoch,

            "wall_seconds":
                (
                    time.perf_counter()
                    - epoch_started
                ),

            "training_loss":
                epoch_losses,

            "calibration":
                calibration,

            "selection_key": {
                "macro_efficient_rank1":
                    selection_key[0],

                "macro_full_rank1":
                    selection_key[1],

                "macro_low_full_cosine":
                    selection_key[2],
            },

            "selected_after_epoch":
                bool(
                    selected
                ),

            "best_epoch_after_epoch":
                int(
                    best_epoch
                ),
        }

        history.append(
            record
        )

        print(
            f"Epoch {epoch}: "
            f"Market low="
            f"{calibration['market1501']['low_rank1']:.3%} "
            f"full="
            f"{calibration['market1501']['full_rank1']:.3%}"
        )

        print(
            f"Epoch {epoch}: "
            f"CUHK low="
            f"{calibration['cuhk03']['low_rank1']:.3%} "
            f"full="
            f"{calibration['cuhk03']['full_rank1']:.3%}"
        )

        print(
            f"Epoch {epoch}: "
            f"Macro low="
            f"{macro['low_rank1']:.3%} "
            f"full="
            f"{macro['full_rank1']:.3%} "
            f"cosine="
            f"{macro['low_full_embedding_cosine']:.4f} "
            f"{'[SELECTED]' if selected else ''}"
        )

        progress = {
            "experiment":
                4,

            "implementation_commit":
                current_head,

            "completed_epoch":
                epoch,

            "backbone_state_dict":
                student_backbone
                .state_dict(),

            "classifier_state_dict":
                classifier
                .state_dict(),

            "optimizer_state_dict":
                optimizer
                .state_dict(),

            "epoch_history":
                history,

            "best_epoch":
                best_epoch,

            "best_key":
                list(
                    best_key
                ),

            "best_backbone_state_dict":
                best_backbone_state,

            "best_classifier_state_dict":
                best_classifier_state,
        }

        temp_progress = (
            PROGRESS_PATH
            .with_suffix(
                ".tmp"
            )
        )

        torch.save(
            progress,
            temp_progress,
        )

        os.replace(
            temp_progress,
            PROGRESS_PATH,
        )

        write_json(
            HISTORY_PATH,
            {
                "experiment":
                    4,

                "implementation_commit":
                    current_head,

                "completed_epoch":
                    epoch,

                "best_epoch":
                    best_epoch,

                "epoch_history":
                    history,
            },
        )

        print()

    if best_epoch is None:
        raise RuntimeError(
            "No selected epoch."
        )

    print(
        "Loading selected epoch:",
        best_epoch,
    )

    student_backbone.load_state_dict(
        best_backbone_state
    )

    classifier.load_state_dict(
        best_classifier_state
    )

    student_backbone.eval()
    classifier.eval()

    print()
    print(
        "Running source held-out "
        "validation AFTER checkpoint selection..."
    )

    held_out = (
        evaluate_source_pair(
            capture_model=capture_model,
            capture_backbone=capture_backbone,
            backbone=student_backbone,
            market=market,
            cuhk=cuhk,
            split_name="held_out",
        )
    )

    print()
    print(
        "Running pretrained-base "
        "held-out comparison AFTER selection..."
    )

    baseline_held_out = (
        evaluate_source_pair(
            capture_model=capture_model,
            capture_backbone=capture_backbone,
            backbone=teacher_backbone,
            market=market,
            cuhk=cuhk,
            split_name="held_out",
        )
    )

    baseline[
        "held_out"
    ] = baseline_held_out

    baseline[
        "held_out_evaluated_before_selection"
    ] = False

    write_json(
        BASELINE_PATH,
        baseline,
    )

    selected_calibration = next(
        row[
            "calibration"
        ]
        for row in history
        if row[
            "epoch"
        ] == best_epoch
    )

    checkpoint = {
        "experiment":
            4,

        "epoch":
            int(
                best_epoch
            ),

        "selected_epoch":
            int(
                best_epoch
            ),

        "model":
            (
                "osnet_x1_0_multidomain_"
                "market_cuhk"
            ),

        "base_model":
            BASE_MODEL,

        "implementation_commit":
            current_head,

        "preregistration_commit":
            PREREG_COMMIT,

        "resolution_low":
            list(
                LOW_SIZE
            ),

        "resolution_full":
            list(
                FULL_SIZE
            ),

        "frozen_modules":
            [
                "conv1",
                "conv2",
            ],

        "seed":
            SEED,

        "classifier_identity_namespace":
            "dataset-scoped",

        "classifier_classes":
            len(
                class_map
            ),

        "classifier_pid_order":
            [
                [
                    domain,
                    int(
                        pid
                    ),
                ]
                for domain, pid
                in train_pid_keys
            ],

        "backbone_state_dict":
            best_backbone_state,

        "classifier_state_dict":
            best_classifier_state,

        "calibration":
            selected_calibration,

        "held_out":
            held_out,
    }

    temp_model = (
        MODEL_PATH
        .with_suffix(
            ".tmp"
        )
    )

    torch.save(
        checkpoint,
        temp_model,
    )

    os.replace(
        temp_model,
        MODEL_PATH,
    )

    model_sha = sha256_file(
        MODEL_PATH
    )

    complete = {
        "status":
            "TRAINING_COMPLETE",

        "experiment":
            4,

        "implementation_commit":
            current_head,

        "preregistration_commit":
            PREREG_COMMIT,

        "selected_epoch":
            int(
                best_epoch
            ),

        "checkpoint":
            str(
                MODEL_PATH
            ),

        "checkpoint_sha256":
            model_sha,

        "source_domains":
            [
                "Market-1501",
                "CUHK03-NP Detected",
            ],

        "gradient_images":
            {
                "market1501":
                    7811,

                "cuhk03":
                    4352,

                "combined":
                    12163,
            },

        "classifier_classes":
            len(
                class_map
            ),

        "calibration":
            selected_calibration,

        "held_out":
            held_out,

        "pretrained_baseline":
            baseline,

        "sense_reid_model_inference":
            False,

        "target_used_for_training":
            False,

        "target_used_for_selection":
            False,

        "epoch_history":
            history,

        "post_target_tuning_allowed":
            False,
    }

    write_json(
        COMPLETE_PATH,
        complete,
    )

    print()
    print(
        "========================================"
    )

    print(
        " EXPERIMENT 4 TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "Selected epoch:",
        best_epoch,
    )

    print(
        "Calibration macro Efficient R1:",
        f"{selected_calibration['macro']['low_rank1']:.3%}",
    )

    print(
        "Calibration macro Full R1:",
        f"{selected_calibration['macro']['full_rank1']:.3%}",
    )

    print(
        "Held-out Market Efficient R1:",
        f"{held_out['market1501']['low_rank1']:.3%}",
    )

    print(
        "Held-out Market Full R1:",
        f"{held_out['market1501']['full_rank1']:.3%}",
    )

    print(
        "Held-out CUHK Efficient R1:",
        f"{held_out['cuhk03']['low_rank1']:.3%}",
    )

    print(
        "Held-out CUHK Full R1:",
        f"{held_out['cuhk03']['full_rank1']:.3%}",
    )

    print(
        "Held-out macro Efficient R1:",
        f"{held_out['macro']['low_rank1']:.3%}",
    )

    print(
        "Held-out macro Full R1:",
        f"{held_out['macro']['full_rank1']:.3%}",
    )

    print(
        "Checkpoint SHA256:",
        model_sha,
    )

    print(
        "SenseReID inference performed:",
        False,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
