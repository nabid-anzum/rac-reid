from __future__ import annotations

import hashlib
import json
import re
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reid import ReIDModel


PROJECT = Path("/root/rac-reid")
RUNTIME = PROJECT / ".runtime"

TRAIN = (
    PROJECT
    / "data/datasets/market1501/"
      "Market-1501-v15.09.15/"
      "bounding_box_train"
)

TEACHER_PATH = (
    RUNTIME
    / "benchmarks/market1501-learned-cascade-calibration/"
      "train_x1_0.npy"
)

HISTORY_PATH = (
    RUNTIME
    / "benchmarks/market1501-multires-training/"
      "multiresolution_training.json"
)

MODEL_PATH = (
    RUNTIME
    / "models/multires/"
      "osnet_x1_0_multires_160x80.pt"
)

RESULT_PATH = (
    PROJECT
    / "results/market1501_multiresolution_training.json"
)

CONFIG_PATH = (
    PROJECT
    / "configs/multiresolution_osnet.json"
)

UPDATE_PATH = (
    PROJECT
    / "docs/PROJECT_UPDATE_2026-11-22.md"
)

HISTORY_DOC = (
    PROJECT
    / "docs/EXPERIMENT_HISTORY.md"
)

PROTOCOL_PATH = (
    PROJECT
    / "docs/MULTIRESOLUTION_TRAINING_PROTOCOL.md"
)

RUNTIME_HISTORY_DIR = (
    PROJECT
    / "results/runtime_history"
)

RECOVERY_SCRIPT = (
    PROJECT
    / "scripts/recover_multiresolution_training.py"
)

README = PROJECT / "README.md"

METHODOLOGY = (
    PROJECT
    / "docs/METHODOLOGY.md"
)

MODEL_ALIAS = "osnet_x1_0_msmt17_combineall"

LOW_SIZE = (160, 80)
FULL_SIZE = (256, 128)

EXPECTED_IMAGES = 12936
EXPECTED_IDS = 751

PATTERN = re.compile(
    r"^(-?\d+)_c(\d+)s\d+_"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def normalize(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        values,
        axis=1,
        keepdims=True,
    )

    return values / np.maximum(
        norms,
        1e-12,
    )


records = []

for path in sorted(TRAIN.glob("*.jpg")):
    match = PATTERN.match(path.name)

    if match is None:
        raise RuntimeError(
            f"Unexpected Market filename: {path.name}"
        )

    pid = int(match.group(1))
    camid = int(match.group(2))

    if pid == -1:
        continue

    records.append(
        (path, pid, camid)
    )


pids = np.asarray(
    [item[1] for item in records],
    dtype=np.int64,
)

camids = np.asarray(
    [item[2] for item in records],
    dtype=np.int64,
)


assert len(records) == EXPECTED_IMAGES
assert len(np.unique(pids)) == EXPECTED_IDS


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


history = json.loads(
    HISTORY_PATH.read_text(
        encoding="utf-8"
    )
)

epoch_history = history.get(
    "epoch_history",
    []
)

if len(epoch_history) != 8:
    raise RuntimeError(
        f"Expected 8 epoch records, found {len(epoch_history)}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)


if int(checkpoint.get("epoch", -1)) != 8:
    raise RuntimeError(
        "Preserved checkpoint is not epoch 8."
    )


teacher = normalize(
    np.array(
        np.load(
            TEACHER_PATH,
            mmap_mode="r",
        ),
        dtype=np.float32,
        copy=True,
    )
)


class CaptureInput(RuntimeError):
    pass


def capture_input(
    model,
    backbone,
    paths,
):
    captured = {}

    def hook(module, inputs):
        captured["array"] = (
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
                [str(path) for path in paths]
            )
        except CaptureInput:
            pass
    finally:
        handle.remove()

    if "array" not in captured:
        raise RuntimeError(
            "Could not capture preprocessing tensor."
        )

    return torch.from_numpy(
        np.array(
            captured["array"],
            dtype=np.float32,
            copy=True,
        )
    ).contiguous()


print("Loading original OSNet...")

capture_model = ReIDModel.from_pretrained(
    MODEL_ALIAS,
    device="cpu",
)

capture_backbone = capture_model._backbone


original_model = ReIDModel.from_pretrained(
    MODEL_ALIAS,
    device="cpu",
)

original_backbone = original_model._backbone
original_backbone.eval()


trained_model = ReIDModel.from_pretrained(
    MODEL_ALIAS,
    device="cpu",
)

trained_backbone = trained_model._backbone

trained_backbone.load_state_dict(
    checkpoint["backbone_state_dict"]
)

trained_backbone.eval()


def embed_split(
    backbone,
    indices,
):
    low_outputs = []
    full_outputs = []

    with torch.no_grad():
        for offset in range(
            0,
            len(indices),
            32,
        ):
            batch_indices = indices[
                offset:
                offset + 32
            ]

            paths = [
                records[int(i)][0]
                for i in batch_indices
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

            full_embedding = F.normalize(
                backbone(full_input),
                p=2,
                dim=1,
            )

            low_embedding = F.normalize(
                backbone(low_input),
                p=2,
                dim=1,
            )

            full_outputs.append(
                full_embedding
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            low_outputs.append(
                low_embedding
                .cpu()
                .numpy()
                .astype(np.float32)
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


def evaluate_pair(
    low,
    full,
    local_pids,
    local_camids,
):
    n = len(local_pids)

    low_scores = low @ full.T
    full_scores = full @ full.T

    rows = np.arange(n)

    low_scores[rows, rows] = -np.inf
    full_scores[rows, rows] = -np.inf

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
            local_pids[low_raw]
            ==
            local_pids[full_raw]
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

    low_metric = low_scores.copy()
    full_metric = full_scores.copy()

    low_metric[invalid] = -np.inf
    full_metric[invalid] = -np.inf

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
            local_pids[low_top]
            ==
            local_pids
        )
    )

    full_rank1 = float(
        np.mean(
            local_pids[full_top]
            ==
            local_pids
        )
    )

    cosine = float(
        np.mean(
            np.sum(
                low * full,
                axis=1,
            )
        )
    )

    return {
        "low_rank1": low_rank1,
        "full_rank1": full_rank1,
        "rank1_gap": (
            full_rank1
            - low_rank1
        ),
        "low_full_identity_agreement": agreement,
        "low_full_embedding_cosine": cosine,
    }


def teacher_rank1(indices):
    emb = teacher[indices]

    local_pids = pids[indices]
    local_camids = camids[indices]

    scores = emb @ emb.T

    rows = np.arange(len(indices))
    scores[rows, rows] = -np.inf

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

    scores[invalid] = -np.inf

    top = np.argmax(
        scores,
        axis=1,
    )

    return float(
        np.mean(
            local_pids[top]
            ==
            local_pids
        )
    )


def evaluate_model(
    backbone,
    indices,
):
    low, full = embed_split(
        backbone,
        indices,
    )

    return evaluate_pair(
        low,
        full,
        pids[indices],
        camids[indices],
    )


print()
print("Re-evaluating calibration identities...")

trained_cal = evaluate_model(
    trained_backbone,
    cal_indices,
)

cal_teacher_r1 = teacher_rank1(
    cal_indices
)


print(
    "CAL trained 160x80 R1:",
    f"{trained_cal['low_rank1']:.3%}",
)

print(
    "CAL trained 256x128 R1:",
    f"{trained_cal['full_rank1']:.3%}",
)

print(
    "CAL teacher R1:",
    f"{cal_teacher_r1:.3%}",
)


print()
print("Re-evaluating held-out identities...")

original_val = evaluate_model(
    original_backbone,
    val_indices,
)

trained_val = evaluate_model(
    trained_backbone,
    val_indices,
)

teacher_val_r1 = teacher_rank1(
    val_indices
)


low_gain = (
    trained_val["low_rank1"]
    - original_val["low_rank1"]
)

full_delta = (
    trained_val["full_rank1"]
    - teacher_val_r1
)


print(
    "Original 160x80 R1:",
    f"{original_val['low_rank1']:.3%}",
)

print(
    "Trained 160x80 R1:",
    f"{trained_val['low_rank1']:.3%}",
)

print(
    "Low gain:",
    f"{low_gain:+.3%}",
)

print(
    "Original full R1:",
    f"{teacher_val_r1:.3%}",
)

print(
    "Trained full R1:",
    f"{trained_val['full_rank1']:.3%}",
)

print(
    "Full delta:",
    f"{full_delta:+.3%}",
)

print(
    "Low/full agreement:",
    f"{trained_val['low_full_identity_agreement']:.3%}",
)


passed = (
    low_gain >= 0.08
    and
    full_delta >= -0.01
    and
    trained_val[
        "rank1_gap"
    ] <= 0.08
    and
    trained_val[
        "low_full_identity_agreement"
    ] >= 0.85
)


if not passed:
    raise RuntimeError(
        "Recovered checkpoint failed the original "
        "development gates."
    )


print()
print("PASS: Preserved checkpoint passes development gates.")


print()
print("Running corrected timing benchmark...")


sample_path = (
    records[
        int(val_indices[0])
    ][0]
)

sample_input = capture_input(
    capture_model,
    capture_backbone,
    [sample_path],
)


timing = {}


for threads in (1, 4):
    torch.set_num_threads(
        threads
    )

    full_times = []
    low_times = []

    with torch.no_grad():

        for _ in range(10):
            trained_backbone(
                sample_input
            )

            low_input = F.interpolate(
                sample_input,
                size=LOW_SIZE,
                mode="bilinear",
                align_corners=False,
            )

            trained_backbone(
                low_input
            )

        for iteration in range(100):

            if iteration % 2 == 0:
                modes = (
                    "full",
                    "low",
                )
            else:
                modes = (
                    "low",
                    "full",
                )

            for mode in modes:

                if mode == "full":
                    started = (
                        time.perf_counter()
                    )

                    trained_backbone(
                        sample_input
                    )

                    full_times.append(
                        time.perf_counter()
                        - started
                    )

                else:
                    started = (
                        time.perf_counter()
                    )

                    low_input = F.interpolate(
                        sample_input,
                        size=LOW_SIZE,
                        mode="bilinear",
                        align_corners=False,
                    )

                    trained_backbone(
                        low_input
                    )

                    low_times.append(
                        time.perf_counter()
                        - started
                    )


    if not full_times or not low_times:
        raise RuntimeError(
            "Timing samples unexpectedly empty."
        )


    full_median = float(
        statistics.median(
            full_times
        )
    )

    low_median = float(
        statistics.median(
            low_times
        )
    )

    timing[str(threads)] = {
        "full_median_seconds":
            full_median,

        "low_160x80_including_resize_median_seconds":
            low_median,

        "low_full_ratio":
            low_median
            / full_median,

        "low_resolution_reduction":
            1.0
            - (
                low_median
                / full_median
            ),

        "samples_per_path":
            100,
    }

    print(
        f"threads={threads} "
        f"full={full_median:.6f}s "
        f"low={low_median:.6f}s "
        f"saving="
        f"{timing[str(threads)]['low_resolution_reduction']:.3%}"
    )


torch.set_num_threads(4)


model_sha = sha256_file(
    MODEL_PATH
)


dataset_manifest = hashlib.sha256()

for path, pid, camid in records:
    dataset_manifest.update(
        path.name.encode(
            "utf-8"
        )
    )

    dataset_manifest.update(
        b"\0"
    )

    dataset_manifest.update(
        str(
            path.stat().st_size
        ).encode(
            "ascii"
        )
    )

    dataset_manifest.update(
        b"\n"
    )


result = {
    "status":
        "passed",

    "experiment":
        "joint multi-resolution OSNet fine-tuning",

    "date":
        "2026-11-22",

    "base_model":
        MODEL_ALIAS,

    "architecture": {
        "low_resolution":
            [160, 80],

        "full_resolution":
            [256, 128],

        "frozen_modules":
            [
                "conv1",
                "conv2",
            ],

        "same_backbone_both_resolutions":
            True,

        "trainable_backbone_parameters":
            1925340,
    },

    "dataset": {
        "name":
            "Market-1501",

        "development_images":
            12936,

        "development_identities":
            751,

        "filename_size_manifest_sha256":
            dataset_manifest.hexdigest(),

        "raw_dataset_committed_to_git":
            False,
    },

    "development_split": {
        "gradient_training": {
            "rule":
                "PID % 5 in {0,1,2}",

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
        },

        "calibration": {
            "rule":
                "PID % 5 == 3",

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
        },

        "held_out_validation": {
            "rule":
                "PID % 5 == 4",

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
        },

        "standard_market_test_used":
            False,
    },

    "training": {
        "epochs":
            8,

        "selected_epoch":
            8,

        "batch_size":
            16,

        "seed":
            checkpoint.get(
                "seed",
                29601,
            ),

        "optimizer":
            "AdamW",

        "backbone_learning_rate":
            5e-5,

        "classifier_learning_rate":
            5e-4,

        "weight_decay":
            5e-4,

        "loss_weights": {
            "low_teacher_cosine":
                1.00,

            "full_teacher_cosine":
                0.25,

            "relational_similarity":
                0.50,

            "cross_resolution_consistency":
                0.25,

            "identity_classification":
                0.50,
        },

        "epoch_history":
            epoch_history,
    },

    "calibration_recovery": {
        **trained_cal,

        "teacher_full_rank1":
            cal_teacher_r1,

        "full_delta_vs_teacher":
            trained_cal[
                "full_rank1"
            ]
            - cal_teacher_r1,
    },

    "held_out_validation": {
        **trained_val,

        "original_low_rank1":
            original_val[
                "low_rank1"
            ],

        "original_full_teacher_rank1":
            teacher_val_r1,

        "low_rank1_gain_vs_original":
            low_gain,

        "full_rank1_delta_vs_original_teacher":
            full_delta,
    },

    "timing":
        timing,

    "checkpoint": {
        "runtime_path":
            str(
                MODEL_PATH
            ),

        "bytes":
            MODEL_PATH.stat().st_size,

        "sha256":
            model_sha,

        "binary_committed_to_git":
            False,
    },

    "recovery_note": (
        "All eight Step 29F-B training epochs completed "
        "and epoch 8 was saved. The original batch failed "
        "after training because of a timing-loop bug that "
        "left one timing list empty. The preserved model "
        "was reloaded and evaluated without retraining."
    ),

    "metric_scope_warning": (
        "The 88% development numbers are identity-disjoint "
        "development metrics from the Market-1501 training "
        "population. They are NOT the final standard "
        "Market-1501 query/gallery benchmark."
    ),

    "standard_market_test_used":
        False,
}


write_json(
    RESULT_PATH,
    result,
)


config = {
    "version":
        1,

    "strategy":
        "joint multi-resolution OSNet",

    "base_model":
        MODEL_ALIAS,

    "low_resolution":
        [160, 80],

    "full_resolution":
        [256, 128],

    "frozen_modules":
        [
            "conv1",
            "conv2",
        ],

    "epochs":
        8,

    "selected_epoch":
        8,

    "batch_size":
        16,

    "optimizer":
        "AdamW",

    "backbone_learning_rate":
        5e-5,

    "classifier_learning_rate":
        5e-4,

    "weight_decay":
        5e-4,

    "checkpoint_sha256":
        model_sha,

    "test_set_used":
        False,
}


write_json(
    CONFIG_PATH,
    config,
)


print()
print("Preserving small runtime JSON experiment records...")


if RUNTIME_HISTORY_DIR.exists():
    shutil.rmtree(
        RUNTIME_HISTORY_DIR
    )


RUNTIME_HISTORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


copied = []


benchmarks = (
    RUNTIME
    / "benchmarks"
)


for source in sorted(
    benchmarks.rglob(
        "*.json"
    )
):
    if source.stat().st_size > 262144:
        continue

    text = source.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if "BEGIN PRIVATE KEY" in text:
        continue

    relative = source.relative_to(
        benchmarks
    )

    safe_name = "__".join(
        relative.parts
    )

    destination = (
        RUNTIME_HISTORY_DIR
        / safe_name
    )

    shutil.copy2(
        source,
        destination,
    )

    copied.append(
        destination.name
    )


(
    RUNTIME_HISTORY_DIR
    / "README.md"
).write_text(
    """# Runtime Experiment History

This directory contains small JSON records copied from development experiments.

Both successful and failed experiments are intentionally retained. Failed
validation is research evidence and is not treated as a broken artifact.

Large NPY/NPZ caches, raw datasets and model binaries remain outside GitHub.
Their relevant dimensions, hashes and experimental roles are documented in
the repository.
""",
    encoding="utf-8",
)


print(
    "Runtime JSON records copied:",
    len(copied),
)


epoch_rows = []


for item in epoch_history:
    cal = item[
        "calibration"
    ]

    epoch_rows.append(
        "| "
        + str(
            item[
                "epoch"
            ]
        )
        + " | "
        + f"{cal['low_rank1']:.3%}"
        + " | "
        + f"{cal['full_rank1']:.3%}"
        + " | "
        + f"{item['calibration_low_gain_vs_baseline']:+.3%}"
        + " | "
        + f"{item['calibration_full_delta_vs_teacher']:+.3%}"
        + " | "
        + f"{cal['low_full_identity_agreement']:.3%}"
        + " | "
        + f"{cal['low_full_embedding_cosine']:.4f}"
        + " |"
    )


timing_lines = []


for threads in sorted(
    timing,
    key=int,
):
    item = timing[
        threads
    ]

    timing_lines.append(
        "- "
        + threads
        + " Torch thread(s): full "
        + f"{item['full_median_seconds']:.6f}s"
        + ", low "
        + f"{item['low_160x80_including_resize_median_seconds']:.6f}s"
        + ", low-resolution saving "
        + f"{item['low_resolution_reduction']:.3%}"
        + "."
    )


UPDATE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


UPDATE_PATH.write_text(
    f"""# RAC-ReID Project Update - 22 November 2026

## Project

RAC-ReID stands for Resource-Adaptive Cascaded Person Re-Identification.

The research question is whether a ReID system can dynamically reduce
query-side computation for easier searches while preserving identification
quality and allocate more computation only when necessary.

Public application:

https://reid.nabidanzum.com

GitHub:

https://github.com/nabid-anzum/rac-reid

## Dataset

Primary dataset: Market-1501.

Development population:

- 12,936 images
- 751 identities

Identity-disjoint development split:

- gradient training: 7,811 images / 449 identities
- calibration: 2,515 images / 144 identities
- held-out development validation: 2,610 images / 158 identities

The standard Market-1501 query/gallery test split has not been used for the
new multi-resolution training or checkpoint selection.

## Established standard baselines

| Experiment | mAP | Rank-1 |
|---|---:|---:|
| OSNet x1.0 | 37.368% | 66.330% |
| OSNet x0.25 | 31.008% | 59.768% |
| Initial proxy adaptive cascade | 2.015% | 14.252% |

## Why the architecture changed

Several adaptive approaches were tested before joint training.

- Handcrafted proxy Top-K pruning failed because valid identities were removed
  before the learned OSNet stage.
- A separate OSNet x0.25 first pass duplicated too much real CPU work.
- A post-hoc x0.25-to-x1 embedding adapter damaged retrieval geometry.
- Cheap image-quality features could not safely route enough queries.
- Frozen conv2/conv3 exits were not discriminative enough for ReID.
- Progressive 224x112 to 256x128 routing had good routing quality but only
  about 2.9% best-case stable CPU saving.

These negative results motivated joint multi-resolution training.

## Training feasibility

CPU profiling showed:

- full OSNet fine-tuning: about 14.50 min/epoch
- freeze conv1+conv2: about 5.84 min/epoch
- freeze conv1+conv2+conv3: about 5.02 min/epoch

The selected configuration freezes conv1 and conv2 while allowing the
remaining 1,925,340 backbone parameters to adapt.

## Joint multi-resolution training

Low-resource resolution: 160x80.

Full resolution: 256x128.

Same OSNet x1.0 backbone is used for both.

Training configuration:

- 8 epochs
- batch size 16
- AdamW
- backbone LR 5e-5
- classifier LR 5e-4
- weight decay 5e-4
- selected checkpoint: epoch 8

Objective:

L =
1.00 * low-resolution teacher cosine loss
+ 0.25 * full-resolution teacher cosine loss
+ 0.50 * relational similarity loss
+ 0.25 * cross-resolution consistency loss
+ 0.50 * identity classification loss

## Epoch history

| Epoch | Low R1 | Full R1 | Low gain | Full delta | Agreement | Cosine |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(epoch_rows)}

## Held-out development result

Original 160x80 Rank-1: {original_val['low_rank1']:.3%}

Trained 160x80 Rank-1: {trained_val['low_rank1']:.3%}

Low-resolution improvement: {low_gain:+.3%}

Original full-resolution Rank-1: {teacher_val_r1:.3%}

Trained full-resolution Rank-1: {trained_val['full_rank1']:.3%}

Full-resolution change: {full_delta:+.3%}

Low/full identity agreement:
{trained_val['low_full_identity_agreement']:.3%}

Low/full embedding cosine:
{trained_val['low_full_embedding_cosine']:.4f}

IMPORTANT:

These are held-out development metrics using identities from the Market-1501
training population. They are not the final standard Market-1501 benchmark.

## Corrected inference timing

{chr(10).join(timing_lines)}

## Why Step 29F-B appeared to fail

The model training itself did not fail.

All eight epochs completed and epoch 8 was saved.

The command then reached a separate timing-report function. A Python loop bug
left one timing array empty, causing:

statistics.StatisticsError: no median for empty data

The checkpoint was preserved. It was reloaded and validated without retraining.

## Checkpoint

Runtime checkpoint:

{MODEL_PATH}

SHA-256:

{model_sha}

The binary model is not committed to GitHub. Its exact hash, architecture,
training configuration and results are stored here.

## What GitHub now preserves

- source code
- evaluation and recovery code
- experiment configurations
- epoch-by-epoch training history
- successful results
- failed experiment records
- methodology decisions
- dataset split information
- checkpoint SHA-256
- reproducibility protocol
- teacher-facing project update

Raw Market-1501 images, large embedding caches and temporary model caches are
not committed.

## Next step

Freeze this trained backbone.

Then train a small confidence controller using only production-observable
160x80 retrieval statistics.

Only if that passes the predetermined development gates will the complete
adaptive architecture be frozen and evaluated on the untouched standard
Market-1501 query/gallery test.
""",
    encoding="utf-8",
)


HISTORY_DOC.write_text(
    """# RAC-ReID Experiment History

This file records the important architecture decisions and negative results.

| Step | Experiment | Main result | Decision |
|---|---|---|---|
| 27 | OSNet x1.0 baseline | 37.368% mAP, 66.330% Rank-1 | Valid full learned baseline |
| 28C | Handcrafted proxy adaptive cascade | 2.015% mAP, 14.252% Rank-1, 100% heavy invocation | Rejected: proxy shortlist recall bottleneck |
| 29B | OSNet x0.25 baseline | 31.008% mAP, 59.768% Rank-1 | Valid lightweight baseline |
| 29C3 | Learned correctness router | Initially promising but later found methodologically invalid | Superseded because routing features had ground-truth-assisted filtering |
| 29C4 | Production-observable correctness router | 3.448% held-out coverage, 84.444% precision, compute negative | Rejected |
| 29C5 | x0.25/x1 agreement router | 98.146% precision, 61.992% exits | Router worked, but independent CNN cascade had too little real CPU benefit |
| 29C6 | Teacher-distilled embedding adapter | Agreement and Rank-1 worsened | Rejected |
| 29C7 | Image-quality pre-router | 0.690% useful coverage | Rejected |
| 29D-B | Frozen shared-backbone early exit | conv3 exit 0.651%, 17.893% early-head Rank-1 | Rejected |
| 29E-A | Multi-resolution audit | 224x112 retained strong retrieval | Investigated further |
| 29E-B | 224-to-256 progressive router | 97.366% precision, 85.824% exits, -0.421 pp Rank-1 | Routing worked but CPU economics failed |
| 29E-C | Stable timing audit | Best projected saving only +2.929% | Progressive pretrained branch closed |
| 29F-A2 | Training feasibility | Full 14.50 min/epoch; freeze conv1+conv2 5.84 min/epoch | Joint training practical |
| 29F-B | Joint 160x80/256x128 OSNet training | Large held-out low-resolution recovery with preserved full-resolution quality | Current direction |

Negative experiments are intentionally retained. Gates were not lowered after
failed experiments merely to obtain a passing result.

The standard Market-1501 test query/gallery split remains reserved for the
final frozen-architecture evaluation.
""",
    encoding="utf-8",
)


PROTOCOL_PATH.write_text(
    f"""# Multi-Resolution Training Protocol

## Model

Base model:

{MODEL_ALIAS}

Spatial budgets:

- low: 160x80
- full: 256x128

Frozen modules:

- conv1
- conv2

Trainable backbone parameters:

1,925,340

## Development data

Market-1501 training population:

- 12,936 images
- 751 identities

Split:

- PID % 5 in {{0,1,2}}: gradient training
- PID % 5 == 3: checkpoint selection
- PID % 5 == 4: held-out development validation

The standard Market-1501 query/gallery test split is not used.

## Optimization

- epochs: 8
- batch size: 16
- optimizer: AdamW
- backbone learning rate: 5e-5
- classifier learning rate: 5e-4
- weight decay: 5e-4
- seed: {checkpoint.get('seed', 29601)}

Loss:

L =
1.00 * low teacher cosine
+ 0.25 * full teacher cosine
+ 0.50 * relational similarity
+ 0.25 * cross-resolution consistency
+ 0.50 * identity classification

## Selected checkpoint

Epoch: 8

SHA-256:

{model_sha}

## Scientific rule

Development identities may be used to train, calibrate and validate the
architecture according to the fixed identity-disjoint split.

The standard Market-1501 query/gallery test split must remain untouched until
the architecture and adaptive controller are frozen.
""",
    encoding="utf-8",
)


readme_text = README.read_text(
    encoding="utf-8"
)


marker = (
    "## Current Research Update - 22 November 2026"
)


if marker not in readme_text:
    readme_text += """

## Current Research Update - 22 November 2026

The current research phase uses a jointly trained multi-resolution OSNet
backbone at 160x80 and 256x128.

The latest training completed 8 epochs and selected epoch 8.

Detailed teacher-facing update:

- `docs/PROJECT_UPDATE_2026-11-22.md`

Complete experiment history:

- `docs/EXPERIMENT_HISTORY.md`

Training protocol:

- `docs/MULTIRESOLUTION_TRAINING_PROTOCOL.md`

Machine-readable result:

- `results/market1501_multiresolution_training.json`

The standard Market-1501 query/gallery test remains untouched for the final
frozen-architecture evaluation.
"""


README.write_text(
    readme_text,
    encoding="utf-8",
)


methodology_text = (
    METHODOLOGY.read_text(
        encoding="utf-8"
    )
)


method_marker = (
    "## Research Traceability - 22 November 2026"
)


if method_marker not in methodology_text:
    methodology_text += """

## Research Traceability - 22 November 2026

The repository now retains both successful and unsuccessful development
experiments.

Negative results are preserved because they justify architecture changes and
prevent post-hoc threshold manipulation.

The current direction is joint cross-resolution OSNet training at 160x80 and
256x128 with conv1 and conv2 frozen.

See:

- `docs/PROJECT_UPDATE_2026-11-22.md`
- `docs/EXPERIMENT_HISTORY.md`
- `docs/MULTIRESOLUTION_TRAINING_PROTOCOL.md`
- `results/market1501_multiresolution_training.json`
- `results/runtime_history/`

Large datasets, pretrained-model caches, learned checkpoint binaries and
temporary NPY/NPZ feature caches remain outside GitHub, while the repository
stores their scientific role, configuration and hashes.
"""


METHODOLOGY.write_text(
    methodology_text,
    encoding="utf-8",
)


RECOVERY_SCRIPT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


shutil.copy2(
    Path(__file__),
    RECOVERY_SCRIPT,
)


print()
print("=================================================")
print(" RECOVERY / DOCUMENTATION COMPLETE")
print("=================================================")

print(
    "Epochs preserved:",
    len(epoch_history),
)

print(
    "Selected epoch:",
    checkpoint["epoch"],
)

print(
    "Original 160x80 R1:",
    f"{original_val['low_rank1']:.3%}",
)

print(
    "Trained 160x80 R1:",
    f"{trained_val['low_rank1']:.3%}",
)

print(
    "Low gain:",
    f"{low_gain:+.3%}",
)

print(
    "Original full R1:",
    f"{teacher_val_r1:.3%}",
)

print(
    "Trained full R1:",
    f"{trained_val['full_rank1']:.3%}",
)

print(
    "Full delta:",
    f"{full_delta:+.3%}",
)

print(
    "Agreement:",
    f"{trained_val['low_full_identity_agreement']:.3%}",
)

print(
    "Checkpoint SHA:",
    model_sha,
)

print(
    "Market standard test used:",
    False,
)
