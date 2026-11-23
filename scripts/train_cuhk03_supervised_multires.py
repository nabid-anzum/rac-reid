#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
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

TRAIN_ROOT = (
    PROJECT
    / "data/datasets/cuhk03/cuhk03-np/"
    "detected/bounding_box_train"
)

CONFIG_PATH = (
    PROJECT
    / "configs/"
    "cuhk03_supervised_multires_experiment.json"
)

EXP = (
    RUNTIME
    / "experiments/cuhk03-supervised"
)

PROGRESS_PATH = EXP / "training_progress.pt"
HISTORY_PATH = EXP / "training_history.json"
BASELINE_PATH = EXP / "development_baseline.json"
COMPLETE_PATH = EXP / "TRAINING_COMPLETE.json"

MODEL_PATH = (
    RUNTIME
    / "models/cuhk03-supervised/"
    "osnet_x1_0_cuhk03_multires.pt"
)

MODEL_ALIAS = (
    "osnet_x1_0_msmt17_combineall"
)

LOW_SIZE = (160, 80)
FULL_SIZE = (256, 128)

EXPECTED_IMAGES = 7365
EXPECTED_IDS = 767

EXPECTED_MANIFEST = (
    "daab40687766fc4ccad1c08cce65b64c"
    "e06142c3ed86ed9e7589ca4cc21f3aa6"
)

PREREG_COMMIT = (
    "78971c77998e7765fbb6fc0660a7bad6"
    "e85aa13a"
)

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

EMBED_DIM = 512

PATTERN = re.compile(
    r"^(-?\d+)_c(\d+)"
)

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def write_json(
    path: Path,
    payload,
) -> None:
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
) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def dataset_manifest(
    paths: list[Path],
) -> str:
    h = hashlib.sha256()

    for path in paths:
        h.update(
            (
                f"{path.name} "
                f"{path.stat().st_size}\n"
            ).encode("utf-8")
        )

    return h.hexdigest()


def parse_record(
    path: Path,
) -> tuple[int, int]:
    match = PATTERN.match(
        path.name
    )

    if match is None:
        raise RuntimeError(
            f"Unexpected CUHK03 filename: "
            f"{path.name}"
        )

    return (
        int(match.group(1)),
        int(match.group(2)),
    )


def load_records():
    paths = sorted(
        [
            path
            for path in TRAIN_ROOT.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in EXTENSIONS
            )
        ],
        key=lambda path: path.name,
    )

    if len(paths) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images, "
            f"found {len(paths)}"
        )

    if (
        dataset_manifest(paths)
        != EXPECTED_MANIFEST
    ):
        raise RuntimeError(
            "CUHK03 training manifest changed."
        )

    records = []

    for path in paths:
        pid, camid = parse_record(
            path
        )

        if pid < 0:
            continue

        records.append(
            (
                path,
                pid,
                camid,
            )
        )

    pids = np.asarray(
        [
            item[1]
            for item in records
        ],
        dtype=np.int64,
    )

    camids = np.asarray(
        [
            item[2]
            for item in records
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
            [0, 1, 2],
        )
    )

    cal_indices = np.flatnonzero(
        (pids % 5) == 3
    )

    val_indices = np.flatnonzero(
        (pids % 5) == 4
    )

    if len(train_indices) != 4352:
        raise RuntimeError(
            "Unexpected gradient-training "
            "image count."
        )

    if len(cal_indices) != 1515:
        raise RuntimeError(
            "Unexpected calibration "
            "image count."
        )

    if len(val_indices) != 1498:
        raise RuntimeError(
            "Unexpected held-out "
            "image count."
        )

    return (
        records,
        pids,
        camids,
        train_indices,
        cal_indices,
        val_indices,
    )


class CaptureInput(
    RuntimeError
):
    pass


def capture_input(
    capture_model,
    capture_backbone,
    paths: list[Path],
) -> torch.Tensor:
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
            "Could not capture ReID "
            "preprocessing tensor."
        )

    tensor = torch.from_numpy(
        np.asarray(
            captured["array"],
            dtype=np.float32,
        )
    ).contiguous()

    if tensor.ndim != 4:
        raise RuntimeError(
            f"Unexpected input tensor: "
            f"{tensor.shape}"
        )

    return tensor


def backbone_embedding(
    backbone,
    tensor: torch.Tensor,
) -> torch.Tensor:
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

    if vector.ndim != 2:
        raise RuntimeError(
            "Unexpected backbone embedding "
            f"shape: {vector.shape}"
        )

    if vector.shape[1] != EMBED_DIM:
        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{vector.shape}"
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

    return raw, norm


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
    low_logits = classifier(
        low_raw
    )

    full_logits = classifier(
        full_raw
    )

    return 0.5 * (
        F.cross_entropy(
            low_logits,
            labels,
        )
        +
        F.cross_entropy(
            full_logits,
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
        [0, 1, 2, 3, 0, 1],
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

    for name, value in losses.items():
        if not torch.isfinite(
            value
        ):
            raise RuntimeError(
                f"Non-finite self-test loss: "
                f"{name}"
            )

    losses["total"].backward()

    if low_raw.grad is None:
        raise RuntimeError(
            "Low-resolution gradient missing."
        )

    if full_raw.grad is None:
        raise RuntimeError(
            "Full-resolution gradient missing."
        )

    if classifier.weight.grad is None:
        raise RuntimeError(
            "Classifier gradient missing."
        )

    # Perfectly aligned tensors should
    # have approximately zero cosine loss.
    aligned = cosine_alignment_loss(
        teacher,
        teacher,
    )

    if abs(
        float(
            aligned.detach()
        )
    ) > 1e-5:
        raise RuntimeError(
            "Cosine alignment self-test "
            "failed."
        )

    print(
        "SELF TEST PASS"
    )

    for name, value in losses.items():
        print(
            f"  {name}: "
            f"{float(value.detach()):.6f}"
        )


def evaluate_pair(
    low: np.ndarray,
    full: np.ndarray,
    local_pids: np.ndarray,
    local_camids: np.ndarray,
) -> dict:
    n = len(
        local_pids
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
            local_pids[
                low_raw
            ]
            ==
            local_pids[
                full_raw
            ]
        )
    )

    invalid = (
        (
            local_pids[:, None]
            ==
            local_pids[None, :]
        )
        &
        (
            local_camids[:, None]
            ==
            local_camids[None, :]
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
            local_pids[
                low_top
            ]
            ==
            local_pids
        )
    )

    full_rank1 = float(
        np.mean(
            local_pids[
                full_top
            ]
            ==
            local_pids
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
    records,
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
                records[
                    int(index)
                ][0]
                for index in batch_indices
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


def evaluate_indices(
    *,
    capture_model,
    capture_backbone,
    backbone,
    records,
    pids,
    camids,
    indices,
):
    low, full = embed_indices(
        capture_model=capture_model,
        capture_backbone=(
            capture_backbone
        ),
        backbone=backbone,
        records=records,
        indices=indices,
    )

    return evaluate_pair(
        low,
        full,
        pids[
            indices
        ],
        camids[
            indices
        ],
    )


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

    # Frozen early modules must not
    # update BatchNorm statistics.
    backbone.conv1.eval()
    backbone.conv2.eval()


def main():
    parser = (
        argparse.ArgumentParser()
    )

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
        config["status"]
        !=
        "PRE_REGISTERED_BEFORE_TRAINING"
    ):
        raise RuntimeError(
            "Experiment-3 protocol status "
            "changed."
        )

    if (
        config["training"]["epochs"]
        != EPOCHS
    ):
        raise RuntimeError(
            "Epoch count mismatch."
        )

    if (
        config["training"]["batch_size"]
        != BATCH_SIZE
    ):
        raise RuntimeError(
            "Batch size mismatch."
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

    # Training is allowed only after
    # preregistration plus trainer-source
    # commit. The exact commit is recorded.
    if current_head == PREREG_COMMIT:
        raise RuntimeError(
            "Trainer source has not been "
            "committed before training."
        )

    (
        records,
        pids,
        camids,
        train_indices,
        cal_indices,
        val_indices,
    ) = load_records()

    train_pid_values = sorted({
        int(
            pids[index]
        )
        for index in train_indices
    })

    pid_to_class = {
        pid: class_index
        for class_index, pid
        in enumerate(
            train_pid_values
        )
    }

    if len(
        pid_to_class
    ) != 452:
        raise RuntimeError(
            "Unexpected training class count."
        )

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
        "OSNet base..."
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

    teacher_model = (
        ReIDModel.from_pretrained(
            MODEL_ALIAS,
            device="cpu",
        )
    )

    teacher_backbone = (
        teacher_model._backbone
    )

    teacher_backbone.eval()

    student_model = (
        ReIDModel.from_pretrained(
            MODEL_ALIAS,
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
            pid_to_class
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

    # Establish development baseline
    # before any parameter update.
    if BASELINE_PATH.exists():
        baseline = json.loads(
            BASELINE_PATH.read_text(
                encoding="utf-8"
            )
        )

    else:
        print()
        print(
            "Computing pre-training "
            "development baseline..."
        )

        cal_before = evaluate_indices(
            capture_model=capture_model,
            capture_backbone=(
                capture_backbone
            ),
            backbone=student_backbone,
            records=records,
            pids=pids,
            camids=camids,
            indices=cal_indices,
        )

        val_before = evaluate_indices(
            capture_model=capture_model,
            capture_backbone=(
                capture_backbone
            ),
            backbone=student_backbone,
            records=records,
            pids=pids,
            camids=camids,
            indices=val_indices,
        )

        baseline = {
            "calibration":
                cal_before,

            "held_out_validation":
                val_before,

            "source":
                MODEL_ALIAS,

            "gradient_updates":
                0,
        }

        write_json(
            BASELINE_PATH,
            baseline,
        )

        print(
            "Baseline calibration low R1:",
            f"{cal_before['low_rank1']:.3%}",
        )

        print(
            "Baseline calibration full R1:",
            f"{cal_before['full_rank1']:.3%}",
        )

    history = []
    start_epoch = 1

    best_key = None
    best_epoch = None
    best_backbone_state = None
    best_classifier_state = None

    if PROGRESS_PATH.exists():
        print()
        print(
            "Resuming Experiment 3 "
            "from epoch-boundary checkpoint..."
        )

        progress = torch.load(
            PROGRESS_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if progress[
            "trainer_commit"
        ] != current_head:
            raise RuntimeError(
                "Progress belongs to another "
                "trainer commit."
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

        best_epoch = progress.get(
            "best_epoch"
        )

        raw_best_key = progress.get(
            "best_key"
        )

        if raw_best_key is not None:
            best_key = tuple(
                raw_best_key
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
            "Resume epoch:",
            start_epoch,
        )

    print()
    print(
        "========================================"
    )

    print(
        " EXPERIMENT 3 TRAINING START"
    )

    print(
        "========================================"
    )

    print(
        "Gradient images:",
        len(train_indices),
    )

    print(
        "Training identities:",
        len(pid_to_class),
    )

    print(
        "Calibration images:",
        len(cal_indices),
    )

    print(
        "Held-out images:",
        len(val_indices),
    )

    print(
        "Trainer commit:",
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

        shuffled = np.array(
            train_indices,
            copy=True,
        )

        rng.shuffle(
            shuffled
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
            len(shuffled),
            BATCH_SIZE,
        ):
            batch_indices = (
                shuffled[
                    offset:
                    offset + BATCH_SIZE
                ]
            )

            paths = [
                records[
                    int(index)
                ][0]
                for index in batch_indices
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

            labels = torch.tensor(
                [
                    pid_to_class[
                        int(
                            pids[
                                int(index)
                            ]
                        )
                    ]
                    for index
                    in batch_indices
                ],
                dtype=torch.long,
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
                teacher_norm=(
                    teacher_norm
                ),
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
                batch_indices
            )

            samples_seen += (
                batch_size
            )

            for name, value in (
                losses.items()
            ):
                totals[name] += (
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
                batch_number % 50 == 0
                or samples_seen
                == len(shuffled)
            ):
                print(
                    f"Epoch {epoch}/"
                    f"{EPOCHS}: "
                    f"{samples_seen}/"
                    f"{len(shuffled)} "
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
            "calibration evaluation..."
        )

        calibration = evaluate_indices(
            capture_model=capture_model,
            capture_backbone=(
                capture_backbone
            ),
            backbone=student_backbone,
            records=records,
            pids=pids,
            camids=camids,
            indices=cal_indices,
        )

        baseline_cal = (
            baseline[
                "calibration"
            ]
        )

        calibration[
            "low_rank1_gain_vs_original"
        ] = (
            calibration[
                "low_rank1"
            ]
            -
            baseline_cal[
                "low_rank1"
            ]
        )

        calibration[
            "full_rank1_delta_vs_original"
        ] = (
            calibration[
                "full_rank1"
            ]
            -
            baseline_cal[
                "full_rank1"
            ]
        )

        selection_key = (
            calibration[
                "low_rank1"
            ],
            calibration[
                "full_rank1"
            ],
            calibration[
                "low_full_embedding_cosine"
            ],
        )

        selected_this_epoch = (
            best_key is None
            or selection_key
            > best_key
        )

        if selected_this_epoch:
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

        epoch_record = {
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

            "selected_after_epoch":
                bool(
                    selected_this_epoch
                ),

            "best_epoch_after_epoch":
                int(
                    best_epoch
                ),
        }

        history.append(
            epoch_record
        )

        print(
            f"Epoch {epoch}: "
            f"low R1="
            f"{calibration['low_rank1']:.3%} "
            f"full R1="
            f"{calibration['full_rank1']:.3%} "
            f"agreement="
            f"{calibration['low_full_identity_agreement']:.3%} "
            f"cosine="
            f"{calibration['low_full_embedding_cosine']:.4f} "
            f"{'[SELECTED]' if selected_this_epoch else ''}"
        )

        progress = {
            "experiment":
                3,

            "trainer_commit":
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
            .with_suffix(".tmp")
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
                "trainer_commit":
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
        "Running held-out development "
        "validation ONCE after selection..."
    )

    held_out = evaluate_indices(
        capture_model=capture_model,
        capture_backbone=(
            capture_backbone
        ),
        backbone=student_backbone,
        records=records,
        pids=pids,
        camids=camids,
        indices=val_indices,
    )

    baseline_val = (
        baseline[
            "held_out_validation"
        ]
    )

    held_out[
        "low_rank1_gain_vs_original"
    ] = (
        held_out[
            "low_rank1"
        ]
        -
        baseline_val[
            "low_rank1"
        ]
    )

    held_out[
        "full_rank1_delta_vs_original"
    ] = (
        held_out[
            "full_rank1"
        ]
        -
        baseline_val[
            "full_rank1"
        ]
    )

    selected_calibration = next(
        item[
            "calibration"
        ]
        for item in history
        if item[
            "epoch"
        ] == best_epoch
    )

    checkpoint = {
        "experiment":
            3,

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
                "osnet_x1_0_cuhk03_"
                "supervised_multires"
            ),

        "base_model":
            MODEL_ALIAS,

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

        "trainer_commit":
            current_head,

        "preregistration_commit":
            PREREG_COMMIT,

        "backbone_state_dict":
            best_backbone_state,

        "classifier_state_dict":
            best_classifier_state,

        "classifier_pid_order":
            train_pid_values,

        "calibration":
            selected_calibration,

        "held_out_validation":
            held_out,
    }

    temp_model = (
        MODEL_PATH
        .with_suffix(".tmp")
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
            3,

        "trainer_commit":
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

        "calibration":
            selected_calibration,

        "held_out_validation":
            held_out,

        "official_cuhk03_query_gallery_used":
            False,

        "production_model_changed":
            False,

        "epoch_history":
            history,

        "loss_definitions": {
            "low_teacher_cosine":
                (
                    "1 - mean cosine("
                    "student_low, teacher_full)"
                ),

            "full_teacher_cosine":
                (
                    "1 - mean cosine("
                    "student_full, teacher_full)"
                ),

            "relational_similarity":
                (
                    "0.5*MSE(low pairwise cosine,"
                    " teacher pairwise cosine)"
                    " + 0.5*MSE(full pairwise cosine,"
                    " teacher pairwise cosine)"
                ),

            "cross_resolution_consistency":
                (
                    "1 - mean cosine("
                    "student_low, student_full)"
                ),

            "identity_classification":
                (
                    "0.5*CE(low embedding classifier)"
                    " + 0.5*CE(full embedding classifier)"
                ),
        },

        "loss_weights": {
            "low_teacher_cosine":
                W_LOW,

            "full_teacher_cosine":
                W_FULL,

            "relational_similarity":
                W_REL,

            "cross_resolution_consistency":
                W_CROSS,

            "identity_classification":
                W_CLS,
        },
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
        " EXPERIMENT 3 TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "Selected epoch:",
        best_epoch,
    )

    print(
        "Calibration Efficient R1:",
        f"{selected_calibration['low_rank1']:.3%}",
    )

    print(
        "Calibration Full R1:",
        f"{selected_calibration['full_rank1']:.3%}",
    )

    print(
        "Held-out Efficient R1:",
        f"{held_out['low_rank1']:.3%}",
    )

    print(
        "Held-out Full R1:",
        f"{held_out['full_rank1']:.3%}",
    )

    print(
        "Held-out agreement:",
        f"{held_out['low_full_identity_agreement']:.3%}",
    )

    print(
        "Held-out cosine:",
        f"{held_out['low_full_embedding_cosine']:.4f}",
    )

    print(
        "Checkpoint:",
        MODEL_PATH,
    )

    print(
        "SHA256:",
        model_sha,
    )

    print(
        "Official CUHK03 query/gallery used:",
        False,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
