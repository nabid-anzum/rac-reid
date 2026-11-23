#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT = Path("/root/rac-reid")
RUNTIME = PROJECT / ".runtime"

REFERENCE = (
    PROJECT
    / "scripts/evaluate_cuhk03_zero_shot.py"
)

REFERENCE_SHA = (
    "7e689881e569204477ea0f444833420a"
    "e065c9baa406408bb57a064682e2bbc3"
)

MODEL = (
    RUNTIME
    / "models/cuhk03-supervised/"
    "osnet_x1_0_cuhk03_multires.pt"
)

MODEL_SHA = (
    "f879574a3cf21f9b106f51fbd2bef692"
    "e6044761a3bb17280c5da3a37fd1a399"
)

TRAINING_RESULT = (
    PROJECT
    / "results/cuhk03_supervised_training_result.json"
)

ZERO_RESULT = (
    PROJECT
    / "results/cuhk03_zero_shot_epoch8_final.json"
)

ZERO_RESULT_SHA = (
    "9cf940063b73af21b1b67d84b7ed590b"
    "a378f6ff5c89105ffabee806259a34d7"
)

EXP = (
    RUNTIME
    / "experiments/cuhk03-supervised-benchmark"
)

STARTED = (
    EXP
    / "BENCHMARK_STARTED.json"
)

COMPLETE = (
    EXP
    / "BENCHMARK_COMPLETE.json"
)

RESULT = (
    EXP
    / "cuhk03_supervised_benchmark_final.json"
)

DET_METRICS = (
    EXP
    / "detected_supervised_metrics.json"
)

LAB_METRICS = (
    EXP
    / "labeled_supervised_metrics.json"
)

DATASET = (
    PROJECT
    / "data/datasets/cuhk03/cuhk03-np"
)

VARIANT_ORDER = (
    "detected",
    "labeled",
)


def utc_now():
    return (
        datetime.now(
            timezone.utc
        )
        .replace(
            microsecond=0
        )
        .isoformat()
    )


def sha256_file(
    path: Path,
):
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


def load_reference_module():
    if sha256_file(
        REFERENCE
    ) != REFERENCE_SHA:
        raise RuntimeError(
            "Experiment-2 evaluator changed."
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "rac_cuhk03_reference_eval",
            REFERENCE,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not import reference evaluator."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


class _CaptureBenchmarkInput(RuntimeError):
    pass


class BenchmarkCheckpointEncoder:
    """
    Benchmark-only OSNet checkpoint adapter.

    This intentionally does NOT modify or weaken
    MultiResolutionOSNetReIDEncoder, whose production
    contract is the original epoch-8 checkpoint.

    Preprocessing/resizing mirrors the RAC-ReID
    multi-resolution training/evaluation path:
      source preprocessing -> 256x128 full tensor
      -> optional bilinear 160x80 resize
      -> frozen checkpoint backbone
      -> L2-normalized 512-D embedding.
    """

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

    def __init__(
        self,
        checkpoint_path,
        expected_sha256,
        expected_epoch,
    ):
        import time

        from reid import ReIDModel

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.expected_sha256 = (
            expected_sha256
        )

        self.expected_epoch = int(
            expected_epoch
        )

        if not self.checkpoint_path.is_file():
            raise RuntimeError(
                "Benchmark checkpoint missing: "
                f"{self.checkpoint_path}"
            )

        actual_sha = sha256_file(
            self.checkpoint_path
        )

        if actual_sha != self.expected_sha256:
            raise RuntimeError(
                "Benchmark checkpoint SHA mismatch."
            )

        torch.set_num_threads(
            4
        )

        started = time.perf_counter()

        self.capture_model = (
            ReIDModel.from_pretrained(
                self.BASE_MODEL,
                device="cpu",
            )
        )

        self.capture_backbone = (
            self.capture_model._backbone
        )

        self.trained_model = (
            ReIDModel.from_pretrained(
                self.BASE_MODEL,
                device="cpu",
            )
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        actual_epoch = int(
            checkpoint.get(
                "epoch",
                -1,
            )
        )

        if actual_epoch != self.expected_epoch:
            raise RuntimeError(
                "Unexpected benchmark checkpoint "
                f"epoch {actual_epoch}; expected "
                f"{self.expected_epoch}."
            )

        selected_epoch = checkpoint.get(
            "selected_epoch"
        )

        if (
            selected_epoch is not None
            and int(selected_epoch)
            != self.expected_epoch
        ):
            raise RuntimeError(
                "Checkpoint selected_epoch does "
                "not match expected benchmark epoch."
            )

        state = checkpoint.get(
            "backbone_state_dict"
        )

        if not isinstance(
            state,
            dict,
        ):
            raise RuntimeError(
                "Checkpoint is missing "
                "backbone_state_dict."
            )

        self.trained_model._backbone.load_state_dict(
            state,
            strict=True,
        )

        self.trained_model._backbone.eval()
        self.capture_backbone.eval()

        self.model_load_seconds = (
            time.perf_counter()
            - started
        )

        self.checkpoint_sha256 = (
            actual_sha
        )

    def _capture_preprocessed(
        self,
        image_path,
    ):
        path = Path(
            image_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        captured = {}

        def hook(
            module,
            inputs,
        ):
            captured["tensor"] = (
                inputs[0]
                .detach()
                .cpu()
                .clone()
            )

            raise _CaptureBenchmarkInput()

        handle = (
            self.capture_backbone
            .register_forward_pre_hook(
                hook
            )
        )

        try:
            try:
                self.capture_model.extract_embeddings_from_paths(
                    [
                        str(path),
                    ]
                )

            except _CaptureBenchmarkInput:
                pass

        finally:
            handle.remove()

        tensor = captured.get(
            "tensor"
        )

        if tensor is None:
            raise RuntimeError(
                "Could not capture benchmark "
                "preprocessing tensor."
            )

        if tensor.ndim != 4:
            raise RuntimeError(
                "Unexpected preprocessing tensor "
                f"shape: {tuple(tensor.shape)}"
            )

        return tensor.contiguous()

    def embed_path_timed_mode(
        self,
        image_path,
        mode,
    ):
        import time

        import torch.nn.functional as F

        normalized_mode = (
            str(mode)
            .strip()
            .lower()
        )

        if normalized_mode == "efficient":
            size = self.LOW_SIZE

        elif normalized_mode == "full":
            size = self.FULL_SIZE

        else:
            raise ValueError(
                "Benchmark mode must be "
                "'efficient' or 'full'."
            )

        tensor = (
            self._capture_preprocessed(
                image_path
            )
        )

        if tuple(
            tensor.shape[-2:]
        ) != tuple(size):

            tensor = F.interpolate(
                tensor,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        started = (
            time.perf_counter()
        )

        with torch.no_grad():

            raw = (
                self.trained_model
                ._backbone(
                    tensor
                )
            )

        inference_seconds = (
            time.perf_counter()
            - started
        )

        array = (
            raw
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
            .reshape(-1)
        )

        if array.shape != (
            512,
        ):
            raise RuntimeError(
                "Unexpected benchmark embedding "
                f"shape: {array.shape}"
            )

        norm = float(
            np.linalg.norm(
                array
            )
        )

        if (
            not np.isfinite(norm)
            or norm <= 0.0
        ):
            raise RuntimeError(
                "Invalid benchmark embedding norm."
            )

        embedding = (
            array / norm
        ).astype(
            np.float32
        )

        return (
            embedding,
            inference_seconds,
            0.0,
        )


def main():

    EXP.mkdir(
        parents=True,
        exist_ok=True,
    )

    if COMPLETE.exists():
        print(
            "Experiment-3 benchmark "
            "already complete."
        )

        print(
            COMPLETE.read_text(
                encoding="utf-8"
            )
        )

        return 0

    if sha256_file(
        MODEL
    ) != MODEL_SHA:
        raise RuntimeError(
            "Experiment-3 checkpoint changed."
        )

    if sha256_file(
        ZERO_RESULT
    ) != ZERO_RESULT_SHA:
        raise RuntimeError(
            "Experiment-2 result changed."
        )

    training = json.loads(
        TRAINING_RESULT.read_text(
            encoding="utf-8"
        )
    )

    if training[
        "selected_epoch"
    ] != 3:
        raise RuntimeError(
            "Selected epoch changed."
        )

    if training[
        "checkpoint_sha256"
    ] != MODEL_SHA:
        raise RuntimeError(
            "Training result/checkpoint mismatch."
        )

    if training[
        "official_cuhk03_query_gallery_used"
    ] is not False:
        raise RuntimeError(
            "Training record indicates "
            "official test contamination."
        )

    reference = (
        load_reference_module()
    )

    # Reuse Experiment-2 dataset parsing,
    # caching, split protocol and metric code.
    #
    # The production MultiResolution encoder
    # is intentionally restricted to the
    # original epoch-8 checkpoint, so the
    # Experiment-3 checkpoint is loaded through
    # the benchmark-only adapter defined above.
    reference.MODEL = MODEL
    reference.EXPECTED_MODEL_SHA = (
        MODEL_SHA
    )
    reference.EXP = EXP
    reference.DATASET = DATASET

    encoder = (
        BenchmarkCheckpointEncoder(
            checkpoint_path=MODEL,
            expected_sha256=MODEL_SHA,
            expected_epoch=3,
        )
    )

    if torch.get_num_threads() != 4:
        raise RuntimeError(
            "Expected four Torch threads."
        )

    # Compatibility smoke test uses
    # CUHK03 TRAINING data only.
    if not STARTED.exists():

        train_paths = (
            reference.list_images(
                DATASET
                / "detected"
                / "bounding_box_train"
            )
        )

        smoke = train_paths[0]

        low = reference.embed_one(
            encoder,
            smoke,
            "efficient",
        )

        full = reference.embed_one(
            encoder,
            smoke,
            "full",
        )

        print(
            "PRE-BENCHMARK TRAINING SMOKE:",
            smoke.name,
        )

        print(
            "  efficient norm:",
            f"{np.linalg.norm(low):.6f}",
        )

        print(
            "  full norm:",
            f"{np.linalg.norm(full):.6f}",
        )

        write_json(
            STARTED,
            {
                "status":
                    "BENCHMARK_STARTED",

                "started_utc":
                    utc_now(),

                "experiment":
                    3,

                "checkpoint_sha256":
                    MODEL_SHA,

                "selected_epoch":
                    3,

                "primary_variant":
                    "detected",

                "secondary_variant":
                    "labeled",

                "training_during_benchmark":
                    False,

                "tuning_during_benchmark":
                    False,

                "post_benchmark_tuning_allowed":
                    False,

                "evaluation_implementation":
                    (
                        "Experiment-2 split/metric logic reused; "
                        "benchmark-only epoch-3 checkpoint "
                        "adapter mirrors the same RAC-ReID "
                        "preprocessing/resizing path"
                    ),

                "reference_evaluator_sha256":
                    REFERENCE_SHA,
            },
        )

        print()
        print(
            "========================================"
        )

        print(
            " EXPERIMENT-3 BENCHMARK STARTED"
        )

        print(
            " DETECTED = PRIMARY"
        )

        print(
            " LABELED = SECONDARY"
        )

        print(
            " NO TRAINING OR TUNING"
        )

        print(
            "========================================"
        )

    results = {}

    metric_files = {
        "detected":
            DET_METRICS,

        "labeled":
            LAB_METRICS,
    }

    for variant in VARIANT_ORDER:

        metric_path = (
            metric_files[
                variant
            ]
        )

        if metric_path.exists():

            print(
                f"Using existing completed "
                f"{variant} metric record."
            )

            results[
                variant
            ] = json.loads(
                metric_path.read_text(
                    encoding="utf-8"
                )
            )

            continue

        print()
        print(
            "========================================"
        )

        print(
            f" {variant.upper()} "
            f"{'(PRIMARY)' if variant == 'detected' else '(SECONDARY)'}"
        )

        print(
            "========================================"
        )

        result = (
            reference.evaluate_variant(
                encoder,
                variant,
            )
        )

        write_json(
            metric_path,
            result,
        )

        results[
            variant
        ] = result

        print()
        print(
            f"{variant.upper()} COMPLETE"
        )

        print(
            " Efficient R1:",
            f"{result['efficient_160x80']['rank1']:.3f}%",
        )

        print(
            " Full R1:",
            f"{result['full_256x128']['rank1']:.3f}%",
        )

        print(
            " R1 retention:",
            f"{result['comparison']['rank1_retention_percent']:.3f}%",
        )

    zero = json.loads(
        ZERO_RESULT.read_text(
            encoding="utf-8"
        )
    )

    comparisons = {}

    for variant in VARIANT_ORDER:

        trained = results[
            variant
        ]

        previous = zero[
            variant
        ]

        comparisons[
            variant
        ] = {
            "efficient_map_gain_pp":
                (
                    trained[
                        "efficient_160x80"
                    ][
                        "map"
                    ]
                    -
                    previous[
                        "efficient_160x80"
                    ][
                        "map"
                    ]
                ),

            "efficient_rank1_gain_pp":
                (
                    trained[
                        "efficient_160x80"
                    ][
                        "rank1"
                    ]
                    -
                    previous[
                        "efficient_160x80"
                    ][
                        "rank1"
                    ]
                ),

            "full_map_gain_pp":
                (
                    trained[
                        "full_256x128"
                    ][
                        "map"
                    ]
                    -
                    previous[
                        "full_256x128"
                    ][
                        "map"
                    ]
                ),

            "full_rank1_gain_pp":
                (
                    trained[
                        "full_256x128"
                    ][
                        "rank1"
                    ]
                    -
                    previous[
                        "full_256x128"
                    ][
                        "rank1"
                    ]
                ),

            "zero_shot_efficient_rank1":
                previous[
                    "efficient_160x80"
                ][
                    "rank1"
                ],

            "zero_shot_full_rank1":
                previous[
                    "full_256x128"
                ][
                    "rank1"
                ],

            "supervised_efficient_rank1":
                trained[
                    "efficient_160x80"
                ][
                    "rank1"
                ],

            "supervised_full_rank1":
                trained[
                    "full_256x128"
                ][
                    "rank1"
                ],
        }

    final = {
        "status":
            "POST_TRAINING_BENCHMARK_FINAL",

        "experiment":
            (
                "Experiment 3 - CUHK03-NP "
                "supervised multi-resolution "
                "post-training benchmark"
            ),

        "completed_utc":
            utc_now(),

        "checkpoint_sha256":
            MODEL_SHA,

        "selected_epoch":
            3,

        "training_during_benchmark":
            False,

        "tuning_during_benchmark":
            False,

        "post_benchmark_tuning_allowed":
            False,

        "benchmark_role":
            (
                "standardized post-training "
                "benchmark; CUHK03 query/gallery "
                "was previously observed in "
                "Experiment 2"
            ),

        "evaluation_implementation": {
            "reused_experiment_2_split_and_metrics":
                True,

            "benchmark_checkpoint_adapter":
                (
                    "Experiment-3-specific loader; "
                    "production epoch-8 loader unchanged"
                ),

            "reference_script":
                (
                    "scripts/"
                    "evaluate_cuhk03_zero_shot.py"
                ),

            "reference_sha256":
                REFERENCE_SHA,

            "distance":
                (
                    "cosine similarity on "
                    "L2-normalized embeddings"
                ),

            "same_pid_same_camera_is_junk":
                True,

            "reranking":
                False,
        },

        "primary_variant":
            "detected",

        "secondary_variant":
            "labeled",

        "detected":
            results[
                "detected"
            ],

        "labeled":
            results[
                "labeled"
            ],

        "comparison_to_experiment_2_zero_shot":
            comparisons,
    }

    write_json(
        RESULT,
        final,
    )

    result_sha = (
        sha256_file(
            RESULT
        )
    )

    write_json(
        COMPLETE,
        {
            "status":
                "BENCHMARK_COMPLETE",

            "completed_utc":
                final[
                    "completed_utc"
                ],

            "checkpoint_sha256":
                MODEL_SHA,

            "result_path":
                str(
                    RESULT
                ),

            "result_sha256":
                result_sha,

            "primary_variant":
                "detected",

            "secondary_variant":
                "labeled",

            "post_benchmark_tuning_allowed":
                False,
        },
    )

    print()
    print(
        "========================================"
    )

    print(
        " EXPERIMENT-3 CUHK03 FINAL"
    )

    print(
        "========================================"
    )

    for variant in VARIANT_ORDER:

        data = results[
            variant
        ]

        comp = comparisons[
            variant
        ]

        print()
        print(
            variant.upper()
            + (
                " (PRIMARY)"
                if variant == "detected"
                else " (SECONDARY)"
            )
        )

        print(
            "  Efficient 160x80"
        )

        print(
            f"    mAP:    "
            f"{data['efficient_160x80']['map']:.3f}%"
        )

        print(
            f"    Rank-1: "
            f"{data['efficient_160x80']['rank1']:.3f}%"
        )

        print(
            "  Full 256x128"
        )

        print(
            f"    mAP:    "
            f"{data['full_256x128']['map']:.3f}%"
        )

        print(
            f"    Rank-1: "
            f"{data['full_256x128']['rank1']:.3f}%"
        )

        print(
            "  Efficient Rank-1 retention:"
        )

        print(
            f"    "
            f"{data['comparison']['rank1_retention_percent']:.3f}%"
        )

        print(
            "  Gain vs Experiment 2"
        )

        print(
            f"    Efficient R1: "
            f"{comp['efficient_rank1_gain_pp']:+.3f} pp"
        )

        print(
            f"    Full R1:      "
            f"{comp['full_rank1_gain_pp']:+.3f} pp"
        )

    print()
    print(
        "Result:",
        RESULT,
    )

    print(
        "SHA256:",
        result_sha,
    )

    print()
    print(
        "POST-BENCHMARK TUNING:"
    )

    print(
        "  NOT ALLOWED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
