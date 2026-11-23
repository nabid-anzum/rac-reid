#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rac_reid.learned_reid import (
    MultiResolutionOSNetReIDEncoder,
)


PROJECT = Path("/root/rac-reid")
RUNTIME = PROJECT / ".runtime"

EXP = (
    RUNTIME
    / "experiments/cuhk03-zero-shot"
)

DATASET = (
    PROJECT
    / "data/datasets/cuhk03/cuhk03-np"
)

MODEL = (
    RUNTIME
    / "models/multires/"
    "osnet_x1_0_multires_160x80.pt"
)

MARKET_RESULT = (
    PROJECT
    / "results/"
    "market1501_multires_epoch8_final.json"
)

PROTOCOL = (
    EXP
    / "EXPERIMENT_2_PROTOCOL.json"
)

STARTED = (
    EXP
    / "ZERO_SHOT_EVALUATION_STARTED.json"
)

COMPLETE = (
    EXP
    / "ZERO_SHOT_EVALUATION_COMPLETE.json"
)

RESULT = (
    EXP
    / "cuhk03_zero_shot_epoch8_final.json"
)


EXPECTED_HEAD = (
    "f4f49db012c6c12004c0c6495fc778728aa134b9"
)

EXPECTED_MODEL_SHA = (
    "d0570ff0fec0905b75bef1b90ec5147b"
    "96ac7b51d8ba9f30de5fa3f0c0757dec"
)

EXPECTED_MARKET_SHA = (
    "a7b14a330d5dd6691635d32199aa72ace"
    "3c1389d5f00e88b78a05665da3bc8f1"
)

EXPECTED = {
    "detected": {
        "query_count": 1400,
        "gallery_count": 5332,
        "query_manifest":
            (
                "9e60de74c816d0999464294e5ef32fa22"
                "264b8b1593efa5406ed8658dabec7d3"
            ),
        "gallery_manifest":
            (
                "f61fc82da3f78452b4df191975d7ff661"
                "3b08aec73f6df1be610ce90c1425d51"
            ),
    },

    "labeled": {
        "query_count": 1400,
        "gallery_count": 5328,
        "query_manifest":
            (
                "21ab4673fc58fc3146c7bd2a8dae9dbe"
                "0f895e282ddac524e753b45ea97799a7"
            ),
        "gallery_manifest":
            (
                "4557120c6b09b58709dc9daa0809d60e"
                "ed9f7462fafc1e25734997babfdd250e"
            ),
    },
}

VARIANT_ORDER = (
    "detected",
    "labeled",
)

EMBED_DIM = 512

PATTERN = re.compile(
    r"^(-?\d+)_c(\d+)"
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def list_images(
    directory: Path,
) -> list[Path]:
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    return sorted(
        [
            p for p in directory.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in extensions
            )
        ],
        key=lambda p: p.name,
    )


def manifest(
    paths: list[Path],
) -> str:
    h = hashlib.sha256()

    for p in paths:
        h.update(
            (
                f"{p.name} "
                f"{p.stat().st_size}\n"
            ).encode("utf-8")
        )

    return h.hexdigest()


def parse_identity(
    path: Path,
) -> tuple[int, int]:
    m = PATTERN.match(
        path.name
    )

    if m is None:
        raise RuntimeError(
            "Unexpected CUHK03 filename: "
            + path.name
        )

    return (
        int(m.group(1)),
        int(m.group(2)),
    )


def atomic_json(
    path: Path,
    payload,
) -> None:
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


def make_encoder():
    return MultiResolutionOSNetReIDEncoder(
        cache_dir=(
            RUNTIME
            / "reid-cache"
        ),
        checkpoint_path=MODEL,
        expected_sha256=(
            EXPECTED_MODEL_SHA
        ),
        base_model_name=(
            "osnet_x1_0_msmt17_combineall"
        ),
        query_size=(160, 80),
        gallery_size=(256, 128),
        torch_threads=4,
    )


def embed_one(
    encoder,
    path: Path,
    mode: str,
) -> np.ndarray:
    value = (
        encoder
        .embed_path_timed_mode(
            path,
            mode=mode,
        )
    )

    if isinstance(
        value,
        (tuple, list),
    ):
        embedding = value[0]
    else:
        embedding = value

    if torch.is_tensor(
        embedding
    ):
        embedding = (
            embedding.detach()
            .cpu()
            .numpy()
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    if embedding.shape != (
        EMBED_DIM,
    ):
        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embedding.shape}"
        )

    norm = float(
        np.linalg.norm(
            embedding
        )
    )

    if not np.isfinite(norm):
        raise RuntimeError(
            "Non-finite embedding."
        )

    if abs(norm - 1.0) > 1e-3:
        raise RuntimeError(
            "Embedding is not normalized: "
            f"{norm}"
        )

    return embedding


def cache_path(
    variant: str,
    name: str,
) -> Path:
    return (
        EXP
        / f"{variant}_{name}.npy"
    )


def progress_path(
    variant: str,
    name: str,
) -> Path:
    return (
        EXP
        / f"{variant}_{name}.progress.json"
    )


def extract_collection(
    encoder,
    *,
    variant: str,
    name: str,
    paths: list[Path],
    mode: str,
) -> tuple[np.ndarray, dict]:
    out = cache_path(
        variant,
        name,
    )

    progress = progress_path(
        variant,
        name,
    )

    total = len(paths)

    if out.exists():

        if not progress.exists():
            raise RuntimeError(
                f"{out.name} exists without "
                f"{progress.name}"
            )

        state = json.loads(
            progress.read_text(
                encoding="utf-8"
            )
        )

        assert state["variant"] == variant
        assert state["mode"] == mode
        assert state["total"] == total

        matrix = np.load(
            out,
            mmap_mode="r+",
        )

        if matrix.shape != (
            total,
            EMBED_DIM,
        ):
            raise RuntimeError(
                "Cached matrix shape mismatch: "
                f"{matrix.shape}"
            )

        completed = int(
            state["completed"]
        )

        prior_wall = float(
            state.get(
                "wall_seconds",
                0.0,
            )
        )

        print(
            f"RESUME {variant}/{name}: "
            f"{completed}/{total}",
            flush=True,
        )

    else:

        matrix = (
            np.lib.format
            .open_memmap(
                out,
                mode="w+",
                dtype=np.float32,
                shape=(
                    total,
                    EMBED_DIM,
                ),
            )
        )

        completed = 0
        prior_wall = 0.0

        atomic_json(
            progress,
            {
                "variant": variant,
                "name": name,
                "mode": mode,
                "total": total,
                "completed": 0,
                "wall_seconds": 0.0,
                "checkpoint_sha256":
                    EXPECTED_MODEL_SHA,
            },
        )

    started = time.perf_counter()

    for index in range(
        completed,
        total,
    ):

        matrix[index] = embed_one(
            encoder,
            paths[index],
            mode,
        )

        done = index + 1

        if (
            done % 100 == 0
            or done == total
        ):

            matrix.flush()

            current_wall = (
                prior_wall
                + time.perf_counter()
                - started
            )

            atomic_json(
                progress,
                {
                    "variant": variant,
                    "name": name,
                    "mode": mode,
                    "total": total,
                    "completed": done,
                    "wall_seconds":
                        current_wall,
                    "checkpoint_sha256":
                        EXPECTED_MODEL_SHA,
                },
            )

        if (
            done % 250 == 0
            or done == total
        ):
            print(
                f"{variant}/{name}: "
                f"{done}/{total}",
                flush=True,
            )

    matrix.flush()

    state = json.loads(
        progress.read_text(
            encoding="utf-8"
        )
    )

    return (
        np.load(
            out,
            mmap_mode="r",
        ),
        state,
    )


def evaluate(
    *,
    variant: str,
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_paths: list[Path],
    gallery_paths: list[Path],
) -> dict:

    q_pid = np.empty(
        len(query_paths),
        dtype=np.int32,
    )

    q_cam = np.empty(
        len(query_paths),
        dtype=np.int16,
    )

    for i, path in enumerate(
        query_paths
    ):
        (
            q_pid[i],
            q_cam[i],
        ) = parse_identity(path)

    g_pid = np.empty(
        len(gallery_paths),
        dtype=np.int32,
    )

    g_cam = np.empty(
        len(gallery_paths),
        dtype=np.int16,
    )

    for i, path in enumerate(
        gallery_paths
    ):
        (
            g_pid[i],
            g_cam[i],
        ) = parse_identity(path)

    print(
        f"{variant}: computing "
        "query-gallery similarity matrix...",
        flush=True,
    )

    metric_started = (
        time.perf_counter()
    )

    scores = np.asarray(
        query_embeddings,
        dtype=np.float32,
    ) @ np.asarray(
        gallery_embeddings,
        dtype=np.float32,
    ).T

    print(
        f"{variant}: similarity matrix "
        f"{scores.shape} ready.",
        flush=True,
    )

    ap_sum = 0.0
    inp_sum = 0.0

    r1 = 0
    r5 = 0
    r10 = 0

    valid = 0

    for index in range(
        len(query_paths)
    ):

        order = np.argsort(
            -scores[index],
            kind="stable",
        )

        ordered_pid = (
            g_pid[order]
        )

        ordered_cam = (
            g_cam[order]
        )

        # CUHK03-NP protocol:
        # same-person / same-camera true
        # matches are junk.
        junk = (
            (ordered_pid == -1)
            |
            (
                (ordered_pid == q_pid[index])
                &
                (ordered_cam == q_cam[index])
            )
        )

        keep_pid = (
            ordered_pid[~junk]
        )

        matches = (
            keep_pid
            == q_pid[index]
        )

        positives = np.flatnonzero(
            matches
        )

        if positives.size == 0:
            continue

        valid += 1

        first = int(
            positives[0]
        )

        r1 += int(
            first < 1
        )

        r5 += int(
            first < 5
        )

        r10 += int(
            first < 10
        )

        precision = (
            np.arange(
                1,
                positives.size + 1,
                dtype=np.float64,
            )
            /
            (
                positives.astype(
                    np.float64
                )
                + 1.0
            )
        )

        ap_sum += float(
            precision.mean()
        )

        inp_sum += float(
            positives.size
            /
            (
                float(
                    positives[-1]
                )
                + 1.0
            )
        )

        if (
            (index + 1) % 250 == 0
            or
            index + 1
            == len(query_paths)
        ):
            print(
                f"{variant}: scored "
                f"{index + 1}/"
                f"{len(query_paths)} queries",
                flush=True,
            )

    wall = (
        time.perf_counter()
        - metric_started
    )

    if valid != len(
        query_paths
    ):
        raise RuntimeError(
            f"{variant}: valid queries "
            f"{valid}/"
            f"{len(query_paths)}"
        )

    return {
        "map":
            ap_sum
            / valid
            * 100.0,

        "rank1":
            r1
            / valid
            * 100.0,

        "rank5":
            r5
            / valid
            * 100.0,

        "rank10":
            r10
            / valid
            * 100.0,

        "minp":
            inp_sum
            / valid
            * 100.0,

        "scored_queries":
            valid,

        "metric_wall_seconds":
            wall,
    }


def evaluate_variant(
    encoder,
    variant: str,
) -> dict:

    expected = EXPECTED[
        variant
    ]

    root = (
        DATASET
        / variant
    )

    train_paths = list_images(
        root
        / "bounding_box_train"
    )

    query_paths = list_images(
        root
        / "query"
    )

    gallery_paths = list_images(
        root
        / "bounding_box_test"
    )

    if len(query_paths) != expected[
        "query_count"
    ]:
        raise RuntimeError(
            f"{variant}: query count mismatch"
        )

    if len(gallery_paths) != expected[
        "gallery_count"
    ]:
        raise RuntimeError(
            f"{variant}: gallery count mismatch"
        )

    if manifest(
        query_paths
    ) != expected[
        "query_manifest"
    ]:
        raise RuntimeError(
            f"{variant}: query manifest mismatch"
        )

    if manifest(
        gallery_paths
    ) != expected[
        "gallery_manifest"
    ]:
        raise RuntimeError(
            f"{variant}: gallery manifest mismatch"
        )

    print()
    print(
        "========================================"
    )

    print(
        f" {variant.upper()} VARIANT"
    )

    print(
        "========================================"
    )

    print(
        "Query:",
        len(query_paths),
    )

    print(
        "Gallery:",
        len(gallery_paths),
    )

    # This uses the TRAIN split only.
    # It occurs before any CUHK03 test image
    # is embedded and is only an encoder sanity check.
    if not STARTED.exists():

        smoke = train_paths[0]

        low = embed_one(
            encoder,
            smoke,
            "efficient",
        )

        full = embed_one(
            encoder,
            smoke,
            "full",
        )

        print(
            f"{variant} training smoke:",
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

    print()
    print(
        f"{variant}: "
        "full 256x128 gallery embeddings"
    )

    gallery_embeddings, gallery_state = (
        extract_collection(
            encoder,
            variant=variant,
            name="gallery_full_256x128",
            paths=gallery_paths,
            mode="full",
        )
    )

    print()
    print(
        f"{variant}: "
        "efficient 160x80 query embeddings"
    )

    query_low, low_state = (
        extract_collection(
            encoder,
            variant=variant,
            name="query_efficient_160x80",
            paths=query_paths,
            mode="efficient",
        )
    )

    print()
    print(
        f"{variant}: "
        "full 256x128 query embeddings"
    )

    query_full, full_state = (
        extract_collection(
            encoder,
            variant=variant,
            name="query_full_256x128",
            paths=query_paths,
            mode="full",
        )
    )

    efficient = evaluate(
        variant=(
            f"{variant}/efficient-160x80"
        ),
        query_embeddings=query_low,
        gallery_embeddings=(
            gallery_embeddings
        ),
        query_paths=query_paths,
        gallery_paths=gallery_paths,
    )

    full = evaluate(
        variant=(
            f"{variant}/full-256x128"
        ),
        query_embeddings=query_full,
        gallery_embeddings=(
            gallery_embeddings
        ),
        query_paths=query_paths,
        gallery_paths=gallery_paths,
    )

    comparison = {
        "map_delta_pp_efficient_minus_full":
            (
                efficient["map"]
                - full["map"]
            ),

        "rank1_delta_pp_efficient_minus_full":
            (
                efficient["rank1"]
                - full["rank1"]
            ),

        "rank5_delta_pp_efficient_minus_full":
            (
                efficient["rank5"]
                - full["rank5"]
            ),

        "rank10_delta_pp_efficient_minus_full":
            (
                efficient["rank10"]
                - full["rank10"]
            ),

        "minp_delta_pp_efficient_minus_full":
            (
                efficient["minp"]
                - full["minp"]
            ),

        "rank1_retention_percent":
            (
                efficient["rank1"]
                / full["rank1"]
                * 100.0
                if full["rank1"] > 0
                else None
            ),

        "map_retention_percent":
            (
                efficient["map"]
                / full["map"]
                * 100.0
                if full["map"] > 0
                else None
            ),
    }

    return {
        "query_images":
            len(query_paths),

        "gallery_images":
            len(gallery_paths),

        "test_identities":
            len({
                parse_identity(p)[0]
                for p in query_paths
            }),

        "query_manifest_sha256":
            expected[
                "query_manifest"
            ],

        "gallery_manifest_sha256":
            expected[
                "gallery_manifest"
            ],

        "gallery_embedding_wall_seconds":
            float(
                gallery_state[
                    "wall_seconds"
                ]
            ),

        "efficient_query_embedding_wall_seconds":
            float(
                low_state[
                    "wall_seconds"
                ]
            ),

        "full_query_embedding_wall_seconds":
            float(
                full_state[
                    "wall_seconds"
                ]
            ),

        "efficient_160x80":
            efficient,

        "full_256x128":
            full,

        "comparison":
            comparison,
    }


def main() -> int:

    EXP.mkdir(
        parents=True,
        exist_ok=True,
    )

    if COMPLETE.exists():

        print(
            "Experiment 2 is already complete."
        )

        print(
            COMPLETE.read_text(
                encoding="utf-8"
            )
        )

        return 0

    head = (
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

    if head != EXPECTED_HEAD:
        raise RuntimeError(
            "Frozen source commit changed."
        )

    if (
        sha256_file(MODEL)
        != EXPECTED_MODEL_SHA
    ):
        raise RuntimeError(
            "Frozen checkpoint changed."
        )

    if (
        sha256_file(
            MARKET_RESULT
        )
        != EXPECTED_MARKET_SHA
    ):
        raise RuntimeError(
            "Market final result changed."
        )

    protocol = json.loads(
        PROTOCOL.read_text(
            encoding="utf-8"
        )
    )

    if protocol[
        "training_on_cuhk03"
    ] is not False:
        raise RuntimeError(
            "Protocol permits CUHK03 training."
        )

    if protocol[
        "tuning_on_cuhk03"
    ] is not False:
        raise RuntimeError(
            "Protocol permits CUHK03 tuning."
        )

    if protocol[
        "primary_dataset_variant"
    ] != "detected":
        raise RuntimeError(
            "Primary variant changed."
        )

    encoder = make_encoder()

    if torch.get_num_threads() != 4:
        raise RuntimeError(
            "Expected four Torch threads."
        )

    # Smoke both variants on TRAIN images only
    # before the zero-shot TEST marker is created.
    if not STARTED.exists():

        for variant in VARIANT_ORDER:

            train_paths = list_images(
                DATASET
                / variant
                / "bounding_box_train"
            )

            smoke = train_paths[0]

            low = embed_one(
                encoder,
                smoke,
                "efficient",
            )

            full = embed_one(
                encoder,
                smoke,
                "full",
            )

            print()
            print(
                f"PRE-TEST {variant} "
                f"training smoke: "
                f"{smoke.name}"
            )

            print(
                "  efficient norm:",
                f"{np.linalg.norm(low):.6f}",
            )

            print(
                "  full norm:",
                f"{np.linalg.norm(full):.6f}",
            )

        atomic_json(
            STARTED,
            {
                "status":
                    "ZERO_SHOT_EVALUATION_STARTED",

                "started_utc":
                    utc_now(),

                "frozen_source_commit":
                    EXPECTED_HEAD,

                "frozen_checkpoint_sha256":
                    EXPECTED_MODEL_SHA,

                "training_on_cuhk03":
                    False,

                "tuning_on_cuhk03":
                    False,

                "primary_variant":
                    "detected",

                "secondary_variant":
                    "labeled",

                "policy":
                    (
                        "Report both variants. "
                        "Detected remains primary "
                        "regardless of result."
                    ),
            },
        )

        print()
        print(
            "========================================"
        )

        print(
            " CUHK03 ZERO-SHOT TEST STARTED"
        )

        print(
            " DETECTED = PRIMARY"
        )

        print(
            " LABELED = SECONDARY"
        )

        print(
            " NO CUHK03 TRAINING OR TUNING"
        )

        print(
            "========================================"
        )

    else:

        started = json.loads(
            STARTED.read_text(
                encoding="utf-8"
            )
        )

        if started[
            "frozen_checkpoint_sha256"
        ] != EXPECTED_MODEL_SHA:
            raise RuntimeError(
                "STARTED marker checkpoint mismatch."
            )

        print(
            "RESUMING SAME CUHK03 "
            "ZERO-SHOT EVALUATION."
        )

    results = {}

    # Important:
    # preregistered primary is evaluated first.
    for variant in VARIANT_ORDER:

        variant_result_path = (
            EXP
            / f"{variant}_zero_shot_metrics.json"
        )

        if variant_result_path.exists():

            print()
            print(
                f"Using completed {variant} "
                "metric record."
            )

            results[
                variant
            ] = json.loads(
                variant_result_path
                .read_text(
                    encoding="utf-8"
                )
            )

            continue

        variant_result = (
            evaluate_variant(
                encoder,
                variant,
            )
        )

        atomic_json(
            variant_result_path,
            variant_result,
        )

        results[
            variant
        ] = variant_result

        print()
        print(
            f"{variant.upper()} COMPLETE"
        )

        print(
            " Efficient R1:",
            f"{variant_result['efficient_160x80']['rank1']:.3f}%",
        )

        print(
            " Full R1:",
            f"{variant_result['full_256x128']['rank1']:.3f}%",
        )

        print(
            " R1 retention:",
            f"{variant_result['comparison']['rank1_retention_percent']:.3f}%",
        )

    market = json.loads(
        MARKET_RESULT.read_text(
            encoding="utf-8"
        )
    )

    final = {
        "experiment":
            (
                "Experiment 2 - frozen Market-trained "
                "RAC-ReID zero-shot CUHK03-NP"
            ),

        "status":
            "ZERO_SHOT_FINAL",

        "completed_utc":
            utc_now(),

        "frozen_source_commit":
            EXPECTED_HEAD,

        "frozen_checkpoint_sha256":
            EXPECTED_MODEL_SHA,

        "training_dataset":
            "Market-1501 development population",

        "evaluation_dataset":
            "CUHK03-NP",

        "training_on_cuhk03":
            False,

        "tuning_on_cuhk03":
            False,

        "model":
            "osnet_x1_0_multires_epoch8",

        "architecture":
            "OSNet x1.0",

        "device":
            "cpu",

        "torch_threads":
            torch.get_num_threads(),

        "embedding_dim":
            EMBED_DIM,

        "query_modes": {
            "efficient":
                [160, 80],

            "full":
                [256, 128],
        },

        "gallery_resolution":
            [256, 128],

        "evaluation_protocol": {
            "protocol":
                "CUHK03-NP new protocol",

            "primary_variant":
                "detected",

            "secondary_variant":
                "labeled",

            "same_pid_same_camera_is_junk":
                True,

            "pid_minus_1_is_junk":
                True,

            "reranking":
                False,

            "distance":
                (
                    "cosine similarity on "
                    "L2-normalized embeddings"
                ),

            "cosine_euclidean_rank_equivalence":
                True,
        },

        "detected":
            results["detected"],

        "labeled":
            results["labeled"],

        "market1501_frozen_reference": {
            "efficient_160x80": {
                "map":
                    market[
                        "efficient_160x80"
                    ][
                        "map"
                    ],

                "rank1":
                    market[
                        "efficient_160x80"
                    ][
                        "rank1"
                    ],
            },

            "full_256x128": {
                "map":
                    market[
                        "full_256x128"
                    ][
                        "map"
                    ],

                "rank1":
                    market[
                        "full_256x128"
                    ][
                        "rank1"
                    ],
            },
        },

        "scientific_policy": {
            "detected_primary_was_preregistered":
                True,

            "both_variants_reported":
                True,

            "cuhk03_used_for_training":
                False,

            "cuhk03_used_for_model_selection":
                False,

            "cuhk03_used_for_threshold_tuning":
                False,

            "market_experiment_modified":
                False,
        },
    }

    atomic_json(
        RESULT,
        final,
    )

    result_sha = sha256_file(
        RESULT
    )

    atomic_json(
        COMPLETE,
        {
            "status":
                "ZERO_SHOT_EVALUATION_COMPLETE",

            "completed_utc":
                final["completed_utc"],

            "result_path":
                str(RESULT),

            "result_sha256":
                result_sha,

            "primary_variant":
                "detected",

            "secondary_variant":
                "labeled",

            "training_on_cuhk03":
                False,

            "tuning_on_cuhk03":
                False,
        },
    )

    print()
    print(
        "========================================"
    )

    print(
        " CUHK03-NP ZERO-SHOT FINAL RESULTS"
    )

    print(
        "========================================"
    )

    for variant in VARIANT_ORDER:

        data = results[
            variant
        ]

        e = data[
            "efficient_160x80"
        ]

        f = data[
            "full_256x128"
        ]

        c = data[
            "comparison"
        ]

        print()
        print(
            f"{variant.upper()}"
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
            f"    mAP:    {e['map']:.3f}%"
        )

        print(
            f"    Rank-1: {e['rank1']:.3f}%"
        )

        print(
            f"    Rank-5: {e['rank5']:.3f}%"
        )

        print(
            f"    Rank-10:{e['rank10']:.3f}%"
        )

        print(
            f"    mINP:   {e['minp']:.3f}%"
        )

        print(
            "  Full 256x128"
        )

        print(
            f"    mAP:    {f['map']:.3f}%"
        )

        print(
            f"    Rank-1: {f['rank1']:.3f}%"
        )

        print(
            f"    Rank-5: {f['rank5']:.3f}%"
        )

        print(
            f"    Rank-10:{f['rank10']:.3f}%"
        )

        print(
            f"    mINP:   {f['minp']:.3f}%"
        )

        print(
            "  Efficient - Full"
        )

        print(
            "    mAP delta: "
            f"{c['map_delta_pp_efficient_minus_full']:+.3f} pp"
        )

        print(
            "    R1 delta:  "
            f"{c['rank1_delta_pp_efficient_minus_full']:+.3f} pp"
        )

        print(
            "    R1 retention: "
            f"{c['rank1_retention_percent']:.3f}%"
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
        "Experiment 2 complete."
    )

    print(
        "No CUHK03 training or tuning "
        "was performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
