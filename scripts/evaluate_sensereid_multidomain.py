#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reid import ReIDModel


PROJECT = Path("/root/rac-reid")
RUNTIME = PROJECT / ".runtime"

BASE_MODEL = (
    "osnet_x1_0_msmt17_combineall"
)

MARKET_MODEL = (
    RUNTIME
    / "models/multires/"
    "osnet_x1_0_multires_160x80.pt"
)

MARKET_SHA = (
    "d0570ff0fec0905b75bef1b90ec5147b"
    "96ac7b51d8ba9f30de5fa3f0c0757dec"
)

CUHK_MODEL = (
    RUNTIME
    / "models/cuhk03-supervised/"
    "osnet_x1_0_cuhk03_multires.pt"
)

CUHK_SHA = (
    "f879574a3cf21f9b106f51fbd2bef692"
    "e6044761a3bb17280c5da3a37fd1a399"
)

SENSE_ARCHIVE = (
    PROJECT
    / "data/datasets/sensereid/"
    "SenseReID.zip"
)

SENSE_ARCHIVE_SHA = (
    "10f02fda87794853c174f2de9bff7343"
    "edcd12421483273329780e7150cfaa57"
)

QUERY_DIR = (
    PROJECT
    / "data/datasets/sensereid/"
    "SenseReID/test_probe"
)

GALLERY_DIR = (
    PROJECT
    / "data/datasets/sensereid/"
    "SenseReID/test_gallery"
)

QUERY_MANIFEST = (
    "044427adcb3b5925a005db4668b968e0"
    "dd219f01f89655d9612fbfa033994f46"
)

GALLERY_MANIFEST = (
    "4e8392d697ff8527d9a187d71320e453"
    "52c6063d557170f564c5c57c66efde93"
)

TARGET_EXP = (
    RUNTIME
    / "experiments/"
    "multidomain-market-cuhk-sensereid/"
    "sensereid-final"
)

STARTED = (
    TARGET_EXP
    / "TARGET_EVALUATION_STARTED.json"
)

COMPLETE = (
    TARGET_EXP
    / "TARGET_EVALUATION_COMPLETE.json"
)

FINAL = (
    TARGET_EXP
    / "sensereid_multidomain_final.json"
)

MARKET_METRICS = (
    TARGET_EXP
    / "market_single_source_metrics.json"
)

CUHK_METRICS = (
    TARGET_EXP
    / "cuhk_single_source_metrics.json"
)

JOINT_METRICS = (
    TARGET_EXP
    / "multidomain_metrics.json"
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


class CaptureInput(RuntimeError):
    pass


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
    path,
):
    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as handle:
        for block in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def write_json(
    path,
    payload,
):
    path = Path(path)

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


def parse_split(
    root,
    expected_images,
    expected_ids,
    expected_manifest,
):
    paths = sorted(
        root.glob(
            "*.jpg"
        ),
        key=lambda p: p.name,
    )

    if len(paths) != expected_images:
        raise RuntimeError(
            f"{root.name}: image count changed."
        )

    manifest = filename_size_manifest(
        paths
    )

    if manifest != expected_manifest:
        raise RuntimeError(
            f"{root.name}: manifest changed."
        )

    pids = []
    camids = []

    for path in paths:
        parts = path.stem.split(
            "_"
        )

        if len(parts) != 2:
            raise RuntimeError(
                f"Unexpected SenseReID filename "
                f"{path.name}"
            )

        pids.append(
            int(
                parts[0]
            )
        )

        camids.append(
            int(
                parts[1]
            )
        )

    pids = np.asarray(
        pids,
        dtype=np.int64,
    )

    camids = np.asarray(
        camids,
        dtype=np.int64,
    )

    if (
        len(
            set(
                pids.tolist()
            )
        )
        != expected_ids
    ):
        raise RuntimeError(
            f"{root.name}: identity count changed."
        )

    return (
        paths,
        pids,
        camids,
        manifest,
    )


def make_capture_model():
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

    return (
        capture_model,
        capture_backbone,
    )


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
        captured[
            "tensor"
        ] = (
            inputs[0]
            .detach()
            .cpu()
            .clone()
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

    tensor = captured.get(
        "tensor"
    )

    if tensor is None:
        raise RuntimeError(
            "Could not capture preprocessing tensor."
        )

    return tensor.contiguous()


def load_checkpoint_backbone(
    *,
    checkpoint_path,
    expected_sha,
    expected_epoch,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if sha256_file(
        checkpoint_path
    ) != expected_sha:
        raise RuntimeError(
            "Checkpoint SHA mismatch."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    actual_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )

    if actual_epoch != int(
        expected_epoch
    ):
        raise RuntimeError(
            f"Checkpoint epoch mismatch: "
            f"actual={actual_epoch}, "
            f"expected={expected_epoch}"
        )

    state = checkpoint.get(
        "backbone_state_dict"
    )

    if not isinstance(
        state,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint missing backbone_state_dict."
        )

    model = (
        ReIDModel.from_pretrained(
            BASE_MODEL,
            device="cpu",
        )
    )

    backbone = (
        model._backbone
    )

    backbone.load_state_dict(
        state,
        strict=True,
    )

    backbone.eval()

    return (
        model,
        backbone,
        checkpoint,
    )


def embed_paths(
    *,
    capture_model,
    capture_backbone,
    backbone,
    paths,
    mode,
    label,
):
    outputs = []

    if mode == "efficient":
        size = LOW_SIZE

    elif mode == "full":
        size = FULL_SIZE

    else:
        raise ValueError(
            mode
        )

    with torch.no_grad():

        for offset in range(
            0,
            len(paths),
            32,
        ):
            batch_paths = (
                paths[
                    offset:
                    offset + 32
                ]
            )

            tensor = capture_input(
                capture_model,
                capture_backbone,
                batch_paths,
            )

            if (
                tuple(
                    tensor.shape[
                        -2:
                    ]
                )
                != tuple(
                    size
                )
            ):
                tensor = F.interpolate(
                    tensor,
                    size=size,
                    mode="bilinear",
                    align_corners=False,
                )

            raw = backbone(
                tensor
            )

            norm = F.normalize(
                raw,
                p=2,
                dim=1,
            )

            outputs.append(
                norm
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            done = min(
                offset + 32,
                len(paths),
            )

            if (
                done % 256 == 0
                or done == len(paths)
            ):
                print(
                    f"{label}: "
                    f"{done}/{len(paths)}",
                    flush=True,
                )

    array = np.concatenate(
        outputs,
        axis=0,
    )

    if (
        array.ndim != 2
        or array.shape[1]
        != EMBED_DIM
    ):
        raise RuntimeError(
            f"Unexpected embedding matrix "
            f"{array.shape}"
        )

    return array


def evaluate_rank(
    *,
    query,
    gallery,
    query_pids,
    query_camids,
    gallery_pids,
    gallery_camids,
):
    similarity = (
        query
        @ gallery.T
    )

    max_rank = 10

    all_cmc = []
    all_ap = []
    all_inp = []

    valid_queries = 0

    for index in range(
        len(
            query_pids
        )
    ):
        scores = similarity[
            index
        ]

        order = np.argsort(
            -scores,
            kind="mergesort",
        )

        same_pid = (
            gallery_pids
            ==
            query_pids[
                index
            ]
        )

        same_cam = (
            gallery_camids
            ==
            query_camids[
                index
            ]
        )

        remove = (
            same_pid
            &
            same_cam
        )

        order = order[
            ~remove[
                order
            ]
        ]

        matches = (
            gallery_pids[
                order
            ]
            ==
            query_pids[
                index
            ]
        ).astype(
            np.int32
        )

        num_rel = int(
            matches.sum()
        )

        if num_rel == 0:
            continue

        valid_queries += 1

        cumulative = (
            matches.cumsum()
        )

        cmc = cumulative.copy()

        cmc[
            cmc > 1
        ] = 1

        all_cmc.append(
            cmc[
                :
                max_rank
            ]
        )

        precision = (
            cumulative
            /
            (
                np.arange(
                    len(
                        matches
                    )
                )
                + 1.0
            )
        )

        ap = float(
            (
                precision
                * matches
            ).sum()
            / num_rel
        )

        positive_positions = (
            np.flatnonzero(
                matches
            )
        )

        last_position = int(
            positive_positions[
                -1
            ]
        )

        inp = float(
            num_rel
            /
            (
                last_position
                + 1
            )
        )

        all_ap.append(
            ap
        )

        all_inp.append(
            inp
        )

    if valid_queries == 0:
        raise RuntimeError(
            "No valid SenseReID queries."
        )

    cmc = np.mean(
        np.stack(
            all_cmc,
            axis=0,
        ),
        axis=0,
    )

    return {
        "query_images":
            int(
                len(
                    query_pids
                )
            ),

        "gallery_images":
            int(
                len(
                    gallery_pids
                )
            ),

        "scored_queries":
            int(
                valid_queries
            ),

        "map":
            float(
                np.mean(
                    all_ap
                )
                * 100.0
            ),

        "rank1":
            float(
                cmc[
                    0
                ]
                * 100.0
            ),

        "rank5":
            float(
                cmc[
                    4
                ]
                * 100.0
            ),

        "rank10":
            float(
                cmc[
                    9
                ]
                * 100.0
            ),

        "minp":
            float(
                np.mean(
                    all_inp
                )
                * 100.0
            ),

        "same_pid_same_camera_removed":
            True,

        "reranking":
            False,
    }


def evaluate_model(
    *,
    model_name,
    checkpoint_path,
    expected_sha,
    expected_epoch,
    query_paths,
    query_pids,
    query_camids,
    gallery_paths,
    gallery_pids,
    gallery_camids,
    capture_model,
    capture_backbone,
):
    print()
    print(
        "========================================"
    )

    print(
        model_name
    )

    print(
        "========================================"
    )

    model, backbone, checkpoint = (
        load_checkpoint_backbone(
            checkpoint_path=checkpoint_path,
            expected_sha=expected_sha,
            expected_epoch=expected_epoch,
        )
    )

    gallery_full = embed_paths(
        capture_model=capture_model,
        capture_backbone=capture_backbone,
        backbone=backbone,
        paths=gallery_paths,
        mode="full",
        label=(
            model_name
            + "/gallery-full"
        ),
    )

    query_low = embed_paths(
        capture_model=capture_model,
        capture_backbone=capture_backbone,
        backbone=backbone,
        paths=query_paths,
        mode="efficient",
        label=(
            model_name
            + "/query-efficient"
        ),
    )

    query_full = embed_paths(
        capture_model=capture_model,
        capture_backbone=capture_backbone,
        backbone=backbone,
        paths=query_paths,
        mode="full",
        label=(
            model_name
            + "/query-full"
        ),
    )

    efficient = evaluate_rank(
        query=query_low,
        gallery=gallery_full,
        query_pids=query_pids,
        query_camids=query_camids,
        gallery_pids=gallery_pids,
        gallery_camids=gallery_camids,
    )

    full = evaluate_rank(
        query=query_full,
        gallery=gallery_full,
        query_pids=query_pids,
        query_camids=query_camids,
        gallery_pids=gallery_pids,
        gallery_camids=gallery_camids,
    )

    comparison = {
        "rank1_delta_pp_efficient_minus_full":
            (
                efficient[
                    "rank1"
                ]
                -
                full[
                    "rank1"
                ]
            ),

        "map_delta_pp_efficient_minus_full":
            (
                efficient[
                    "map"
                ]
                -
                full[
                    "map"
                ]
            ),

        "rank1_retention_percent":
            (
                efficient[
                    "rank1"
                ]
                /
                full[
                    "rank1"
                ]
                * 100.0
                if full[
                    "rank1"
                ] > 0.0
                else 0.0
            ),
    }

    result = {
        "model":
            model_name,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "checkpoint_sha256":
            expected_sha,

        "checkpoint_epoch":
            int(
                expected_epoch
            ),

        "efficient_160x80":
            efficient,

        "full_256x128":
            full,

        "comparison":
            comparison,
    }

    del backbone
    del model

    gc.collect()

    return result


def delta(
    joint,
    baseline,
):
    return {
        "efficient_map_pp":
            (
                joint[
                    "efficient_160x80"
                ][
                    "map"
                ]
                -
                baseline[
                    "efficient_160x80"
                ][
                    "map"
                ]
            ),

        "efficient_rank1_pp":
            (
                joint[
                    "efficient_160x80"
                ][
                    "rank1"
                ]
                -
                baseline[
                    "efficient_160x80"
                ][
                    "rank1"
                ]
            ),

        "full_map_pp":
            (
                joint[
                    "full_256x128"
                ][
                    "map"
                ]
                -
                baseline[
                    "full_256x128"
                ][
                    "map"
                ]
            ),

        "full_rank1_pp":
            (
                joint[
                    "full_256x128"
                ][
                    "rank1"
                ]
                -
                baseline[
                    "full_256x128"
                ][
                    "rank1"
                ]
            ),
    }


def self_test():
    # Synthetic metric test.
    #
    # The original four-gallery-item version
    # left only three candidates after
    # same-PID/same-camera removal, while the
    # evaluator reports Rank-5 and Rank-10.
    #
    # This replacement changes ONLY synthetic
    # self-test data. Production/benchmark
    # metric logic remains untouched.

    query = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    gallery = np.asarray(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.00, 1.00],
            [0.01, 0.99],
            [0.80, 0.60],
            [0.75, 0.66],
            [0.70, 0.714],
            [0.60, 0.80],
            [-0.20, 0.98],
            [0.98, -0.20],
            [-0.80, 0.60],
            [0.60, -0.80],
        ],
        dtype=np.float32,
    )

    query = (
        query
        /
        np.linalg.norm(
            query,
            axis=1,
            keepdims=True,
        )
    )

    gallery = (
        gallery
        /
        np.linalg.norm(
            gallery,
            axis=1,
            keepdims=True,
        )
    )

    result = evaluate_rank(
        query=query,
        gallery=gallery,
        query_pids=np.asarray(
            [
                1,
                2,
            ],
            dtype=np.int64,
        ),
        query_camids=np.asarray(
            [
                0,
                0,
            ],
            dtype=np.int64,
        ),
        gallery_pids=np.asarray(
            [
                1,
                1,
                2,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            dtype=np.int64,
        ),
        gallery_camids=np.asarray(
            [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            dtype=np.int64,
        ),
    )

    assert result["scored_queries"] == 2
    assert result["rank1"] == 100.0
    assert result["rank5"] == 100.0
    assert result["rank10"] == 100.0

    print("SELF TEST PASS")

    print(
        "  scored_queries:",
        result["scored_queries"],
    )

    print(
        "  rank1:",
        f"{result['rank1']:.3f}%",
    )

    print(
        "  rank5:",
        f"{result['rank5']:.3f}%",
    )

    print(
        "  rank10:",
        f"{result['rank10']:.3f}%",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--joint-model",
        type=Path,
    )

    parser.add_argument(
        "--joint-sha",
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if (
        args.joint_model is None
        or not args.joint_sha
    ):
        raise SystemExit(
            "--joint-model and --joint-sha "
            "are required."
        )

    torch.set_num_threads(
        4
    )

    TARGET_EXP.mkdir(
        parents=True,
        exist_ok=True,
    )

    if COMPLETE.exists():

        print(
            "SenseReID target evaluation "
            "already complete."
        )

        print(
            FINAL.read_text(
                encoding="utf-8"
            )
        )

        return 0

    if sha256_file(
        SENSE_ARCHIVE
    ) != SENSE_ARCHIVE_SHA:
        raise RuntimeError(
            "SenseReID archive changed."
        )

    (
        query_paths,
        query_pids,
        query_camids,
        query_manifest,
    ) = parse_split(
        QUERY_DIR,
        1040,
        521,
        QUERY_MANIFEST,
    )

    (
        gallery_paths,
        gallery_pids,
        gallery_camids,
        gallery_manifest,
    ) = parse_split(
        GALLERY_DIR,
        3388,
        1718,
        GALLERY_MANIFEST,
    )

    missing = (
        set(
            query_pids.tolist()
        )
        -
        set(
            gallery_pids.tolist()
        )
    )

    if missing:
        raise RuntimeError(
            "SenseReID query identities "
            "missing from gallery."
        )

    if sha256_file(
        MARKET_MODEL
    ) != MARKET_SHA:
        raise RuntimeError(
            "Market checkpoint changed."
        )

    if sha256_file(
        CUHK_MODEL
    ) != CUHK_SHA:
        raise RuntimeError(
            "CUHK checkpoint changed."
        )

    if sha256_file(
        args.joint_model
    ) != args.joint_sha:
        raise RuntimeError(
            "Joint checkpoint changed."
        )

    joint_checkpoint = torch.load(
        args.joint_model,
        map_location="cpu",
        weights_only=False,
    )

    joint_epoch = int(
        joint_checkpoint.get(
            "selected_epoch",
            joint_checkpoint.get(
                "epoch",
                -1,
            ),
        )
    )

    if joint_epoch < 1:
        raise RuntimeError(
            "Invalid joint selected epoch."
        )

    model_specs = [
        {
            "key":
                "market_single_source",

            "name":
                "Experiment 1 Market-trained",

            "path":
                MARKET_MODEL,

            "sha":
                MARKET_SHA,

            "epoch":
                8,

            "metric_file":
                MARKET_METRICS,
        },

        {
            "key":
                "cuhk_single_source",

            "name":
                "Experiment 3 CUHK-trained",

            "path":
                CUHK_MODEL,

            "sha":
                CUHK_SHA,

            "epoch":
                3,

            "metric_file":
                CUHK_METRICS,
        },

        {
            "key":
                "multidomain_exp4",

            "name":
                "Experiment 4 Market+CUHK",

            "path":
                args.joint_model,

            "sha":
                args.joint_sha,

            "epoch":
                joint_epoch,

            "metric_file":
                JOINT_METRICS,
        },
    ]

    # All checkpoint validation and target
    # filename/manifests are complete here.
    # No SenseReID image has yet been passed
    # through a model in this process.

    if not STARTED.exists():
        write_json(
            STARTED,
            {
                "status":
                    "TARGET_EVALUATION_STARTED",

                "started_utc":
                    utc_now(),

                "dataset":
                    "SenseReID",

                "evaluation_run":
                    1,

                "query_images":
                    1040,

                "query_identities":
                    521,

                "gallery_images":
                    3388,

                "gallery_identities":
                    1718,

                "query_manifest_sha256":
                    query_manifest,

                "gallery_manifest_sha256":
                    gallery_manifest,

                "models":
                    [
                        {
                            "key":
                                spec[
                                    "key"
                                ],

                            "checkpoint_sha256":
                                spec[
                                    "sha"
                                ],

                            "epoch":
                                spec[
                                    "epoch"
                                ],
                        }
                        for spec in model_specs
                    ],

                "target_used_for_training":
                    False,

                "target_used_for_selection":
                    False,

                "post_target_tuning_allowed":
                    False,

                "resume_after_infrastructure_interruption_allowed":
                    True,
            },
        )

    print(
        "========================================"
    )

    print(
        " SENSEREID TARGET EVALUATION"
    )

    print(
        " SINGLE FROZEN RUN"
    )

    print(
        " NO TRAINING / NO TUNING"
    )

    print(
        "========================================"
    )

    capture_model, capture_backbone = (
        make_capture_model()
    )

    results = {}

    for spec in model_specs:

        metric_file = spec[
            "metric_file"
        ]

        if metric_file.exists():
            print(
                "Using existing metric file:",
                metric_file.name,
            )

            result = json.loads(
                metric_file.read_text(
                    encoding="utf-8"
                )
            )

        else:
            result = evaluate_model(
                model_name=spec[
                    "name"
                ],
                checkpoint_path=spec[
                    "path"
                ],
                expected_sha=spec[
                    "sha"
                ],
                expected_epoch=spec[
                    "epoch"
                ],
                query_paths=query_paths,
                query_pids=query_pids,
                query_camids=query_camids,
                gallery_paths=gallery_paths,
                gallery_pids=gallery_pids,
                gallery_camids=gallery_camids,
                capture_model=capture_model,
                capture_backbone=capture_backbone,
            )

            write_json(
                metric_file,
                result,
            )

        results[
            spec[
                "key"
            ]
        ] = result

        print()
        print(
            spec[
                "name"
            ]
        )

        print(
            "  Efficient mAP:",
            f"{result['efficient_160x80']['map']:.3f}%"
        )

        print(
            "  Efficient R1:",
            f"{result['efficient_160x80']['rank1']:.3f}%"
        )

        print(
            "  Full mAP:",
            f"{result['full_256x128']['map']:.3f}%"
        )

        print(
            "  Full R1:",
            f"{result['full_256x128']['rank1']:.3f}%"
        )

        print(
            "  Efficient R1 retention:",
            f"{result['comparison']['rank1_retention_percent']:.3f}%"
        )

    joint = results[
        "multidomain_exp4"
    ]

    market = results[
        "market_single_source"
    ]

    cuhk = results[
        "cuhk_single_source"
    ]

    final = {
        "status":
            "SENSEREID_FINAL",

        "experiment":
            4,

        "dataset":
            "SenseReID",

        "completed_utc":
            utc_now(),

        "evaluation_run":
            1,

        "query_images":
            1040,

        "query_identities":
            521,

        "gallery_images":
            3388,

        "gallery_identities":
            1718,

        "archive_sha256":
            SENSE_ARCHIVE_SHA,

        "query_manifest_sha256":
            QUERY_MANIFEST,

        "gallery_manifest_sha256":
            GALLERY_MANIFEST,

        "published_metadata_discrepancy_documented":
            True,

        "target_used_for_training":
            False,

        "target_used_for_selection":
            False,

        "post_target_tuning_allowed":
            False,

        "standard_protocol": {
            "gallery_resolution":
                [
                    256,
                    128,
                ],

            "efficient_query_resolution":
                [
                    160,
                    80,
                ],

            "full_query_resolution":
                [
                    256,
                    128,
                ],

            "similarity":
                (
                    "cosine on L2-normalized "
                    "embeddings"
                ),

            "same_pid_same_camera_removed":
                True,

            "reranking":
                False,
        },

        "models":
            results,

        "comparison": {
            "joint_vs_market_single_source":
                delta(
                    joint,
                    market,
                ),

            "joint_vs_cuhk_single_source":
                delta(
                    joint,
                    cuhk,
                ),
        },
    }

    write_json(
        FINAL,
        final,
    )

    result_sha = (
        sha256_file(
            FINAL
        )
    )

    write_json(
        COMPLETE,
        {
            "status":
                "TARGET_EVALUATION_COMPLETE",

            "completed_utc":
                final[
                    "completed_utc"
                ],

            "evaluation_run":
                1,

            "result_path":
                str(
                    FINAL
                ),

            "result_sha256":
                result_sha,

            "joint_checkpoint_sha256":
                args.joint_sha,

            "post_target_tuning_allowed":
                False,
        },
    )

    print()
    print(
        "========================================"
    )

    print(
        " SENSEREID FINAL COMPARISON"
    )

    print(
        "========================================"
    )

    for key in (
        "market_single_source",
        "cuhk_single_source",
        "multidomain_exp4",
    ):
        result = results[
            key
        ]

        print()
        print(
            result[
                "model"
            ]
        )

        print(
            " Efficient:",
            f"mAP={result['efficient_160x80']['map']:.3f}% "
            f"R1={result['efficient_160x80']['rank1']:.3f}%"
        )

        print(
            " Full:",
            f"mAP={result['full_256x128']['map']:.3f}% "
            f"R1={result['full_256x128']['rank1']:.3f}%"
        )

        print(
            " Retention:",
            f"{result['comparison']['rank1_retention_percent']:.3f}%"
        )

    print()
    print(
        "Joint vs Market Efficient R1:",
        f"{final['comparison']['joint_vs_market_single_source']['efficient_rank1_pp']:+.3f} pp"
    )

    print(
        "Joint vs CUHK Efficient R1:",
        f"{final['comparison']['joint_vs_cuhk_single_source']['efficient_rank1_pp']:+.3f} pp"
    )

    print(
        "Final result SHA256:",
        result_sha,
    )

    print(
        "POST-TARGET TUNING: NOT ALLOWED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
