from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time

from pathlib import Path

import numpy as np

from reid import load_market1501

from rac_reid.image_signals import (
    ImageSignalExtractor,
)

from rac_reid.learned_reid import (
    OSNetReIDEncoder,
)

from rac_reid.resource_controller import (
    QuerySignals,
    ResourceController,
)

from rac_reid.retrieval import (
    BudgetedVisualEncoder,
)


MODEL_ALIAS = (
    "osnet_x1_0_msmt17_combineall"
)

EXPECTED_QUERY = 3368
EXPECTED_GALLERY = 19732


def atomic_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
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


def dataset_signature(
    query,
    gallery,
) -> str:
    digest = (
        hashlib.sha256()
    )

    for split_name, split in (
        (
            "query",
            query,
        ),
        (
            "gallery",
            gallery,
        ),
    ):
        digest.update(
            split_name.encode(
                "utf-8"
            )
        )

        for (
            path,
            pid,
            camid,
        ) in zip(
            split.image_paths,
            split.pids,
            split.camids,
        ):
            digest.update(
                str(
                    Path(path).name
                ).encode(
                    "utf-8"
                )
            )

            digest.update(
                f"|{int(pid)}|{int(camid)}\n"
                .encode(
                    "utf-8"
                )
            )

    return (
        digest.hexdigest()
    )


def cache_is_valid(
    meta_path: Path,
    data_path: Path,
    signature: str,
    count: int,
) -> bool:
    if (
        not meta_path.is_file()
        or not data_path.is_file()
    ):
        return False

    try:
        meta = json.loads(
            meta_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return False

    return (
        meta.get(
            "dataset_signature"
        )
        == signature
        and int(
            meta.get(
                "count",
                -1,
            )
        )
        == count
        and bool(
            meta.get(
                "complete",
                False,
            )
        )
    )


def build_proxy_gallery(
    gallery_paths,
    encoder,
    cache_dir: Path,
    signature: str,
):
    final_path = (
        cache_dir
        / "proxy_gallery.npy"
    )

    final_meta = (
        cache_dir
        / "proxy_gallery.json"
    )

    partial_path = (
        cache_dir
        / "proxy_gallery.partial.npy"
    )

    progress_path = (
        cache_dir
        / "proxy_gallery.progress.json"
    )

    count = len(
        gallery_paths
    )

    if cache_is_valid(
        final_meta,
        final_path,
        signature,
        count,
    ):
        array = np.load(
            final_path,
            mmap_mode="r",
        )

        print(
            "Proxy gallery cache: REUSED",
            flush=True,
        )

        print(
            "Proxy descriptor shape:",
            array.shape,
            flush=True,
        )

        return (
            array,
            True,
            0.0,
        )

    print(
        "Building full-budget proxy gallery cache...",
        flush=True,
    )

    started = (
        time.perf_counter()
    )

    resume = 0
    mmap = None

    if (
        partial_path.is_file()
        and progress_path.is_file()
    ):
        try:
            progress = json.loads(
                progress_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                progress.get(
                    "dataset_signature"
                )
                == signature
                and int(
                    progress.get(
                        "count",
                        -1,
                    )
                )
                == count
            ):
                resume = int(
                    progress[
                        "completed"
                    ]
                )

                mmap = np.load(
                    partial_path,
                    mmap_mode="r+",
                )

                print(
                    "Resuming proxy cache at:",
                    resume,
                    flush=True,
                )

        except Exception:
            mmap = None
            resume = 0

    if mmap is None:
        partial_path.unlink(
            missing_ok=True
        )

        progress_path.unlink(
            missing_ok=True
        )

        first = (
            encoder.encode(
                gallery_paths[0],
                retention=1.0,
            )
            .descriptor
        )

        descriptor_dim = int(
            first.shape[0]
        )

        mmap = (
            np.lib.format
            .open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    count,
                    descriptor_dim,
                ),
            )
        )

        mmap[
            0
        ] = np.asarray(
            first,
            dtype=np.float32,
        )

        resume = 1

        atomic_json(
            progress_path,
            {
                "dataset_signature":
                    signature,
                "count":
                    count,
                "completed":
                    resume,
                "descriptor_dim":
                    descriptor_dim,
            },
        )

    for index in range(
        resume,
        count,
    ):
        descriptor = (
            encoder.encode(
                gallery_paths[
                    index
                ],
                retention=1.0,
            )
            .descriptor
        )

        mmap[
            index
        ] = np.asarray(
            descriptor,
            dtype=np.float32,
        )

        completed = (
            index + 1
        )

        if (
            completed % 250 == 0
            or completed == count
        ):
            mmap.flush()

            atomic_json(
                progress_path,
                {
                    "dataset_signature":
                        signature,
                    "count":
                        count,
                    "completed":
                        completed,
                    "descriptor_dim":
                        int(
                            mmap.shape[1]
                        ),
                },
            )

            print(
                "Proxy gallery:",
                completed,
                "/",
                count,
                flush=True,
            )

    shape = tuple(
        mmap.shape
    )

    mmap.flush()
    del mmap

    os.replace(
        partial_path,
        final_path,
    )

    progress_path.unlink(
        missing_ok=True
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    atomic_json(
        final_meta,
        {
            "dataset_signature":
                signature,
            "count":
                count,
            "shape":
                list(
                    shape
                ),
            "complete":
                True,
            "build_seconds":
                elapsed,
        },
    )

    array = np.load(
        final_path,
        mmap_mode="r",
    )

    print(
        "PASS: Proxy gallery cache complete.",
        flush=True,
    )

    return (
        array,
        False,
        elapsed,
    )


def build_osnet_gallery(
    gallery_paths,
    learned_encoder,
    cache_dir: Path,
    signature: str,
):
    final_path = (
        cache_dir
        / "osnet_gallery.npy"
    )

    final_meta = (
        cache_dir
        / "osnet_gallery.json"
    )

    partial_path = (
        cache_dir
        / "osnet_gallery.partial.npy"
    )

    progress_path = (
        cache_dir
        / "osnet_gallery.progress.json"
    )

    count = len(
        gallery_paths
    )

    if cache_is_valid(
        final_meta,
        final_path,
        signature,
        count,
    ):
        array = np.load(
            final_path,
            mmap_mode="r",
        )

        print(
            "OSNet gallery cache: REUSED",
            flush=True,
        )

        print(
            "OSNet gallery shape:",
            array.shape,
            flush=True,
        )

        return (
            array,
            True,
            0.0,
        )

    print(
        "Building offline OSNet gallery embedding cache...",
        flush=True,
    )

    print(
        "This gallery cost is offline precomputation and is "
        "not counted as per-query adaptive compute.",
        flush=True,
    )

    started = (
        time.perf_counter()
    )

    resume = 0
    mmap = None

    if (
        partial_path.is_file()
        and progress_path.is_file()
    ):
        try:
            progress = json.loads(
                progress_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                progress.get(
                    "dataset_signature"
                )
                == signature
                and int(
                    progress.get(
                        "count",
                        -1,
                    )
                )
                == count
            ):
                resume = int(
                    progress[
                        "completed"
                    ]
                )

                mmap = np.load(
                    partial_path,
                    mmap_mode="r+",
                )

                print(
                    "Resuming OSNet gallery cache at:",
                    resume,
                    flush=True,
                )

        except Exception:
            mmap = None
            resume = 0

    if mmap is None:
        partial_path.unlink(
            missing_ok=True
        )

        progress_path.unlink(
            missing_ok=True
        )

        first = (
            learned_encoder
            .embed_path(
                gallery_paths[
                    0
                ]
            )
        )

        embedding_dim = int(
            first.shape[0]
        )

        mmap = (
            np.lib.format
            .open_memmap(
                partial_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    count,
                    embedding_dim,
                ),
            )
        )

        mmap[
            0
        ] = first

        resume = 1

        atomic_json(
            progress_path,
            {
                "dataset_signature":
                    signature,
                "count":
                    count,
                "completed":
                    resume,
                "embedding_dim":
                    embedding_dim,
                "model":
                    MODEL_ALIAS,
            },
        )

    for index in range(
        resume,
        count,
    ):
        mmap[
            index
        ] = (
            learned_encoder
            .embed_path(
                gallery_paths[
                    index
                ]
            )
        )

        completed = (
            index + 1
        )

        if (
            completed % 250 == 0
            or completed == count
        ):
            mmap.flush()

            atomic_json(
                progress_path,
                {
                    "dataset_signature":
                        signature,
                    "count":
                        count,
                    "completed":
                        completed,
                    "embedding_dim":
                        int(
                            mmap.shape[1]
                        ),
                    "model":
                        MODEL_ALIAS,
                },
            )

            print(
                "OSNet gallery:",
                completed,
                "/",
                count,
                flush=True,
            )

    shape = tuple(
        mmap.shape
    )

    mmap.flush()
    del mmap

    os.replace(
        partial_path,
        final_path,
    )

    progress_path.unlink(
        missing_ok=True
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    atomic_json(
        final_meta,
        {
            "dataset_signature":
                signature,
            "count":
                count,
            "shape":
                list(
                    shape
                ),
            "model":
                MODEL_ALIAS,
            "complete":
                True,
            "build_seconds":
                elapsed,
        },
    )

    array = np.load(
        final_path,
        mmap_mode="r",
    )

    print(
        "PASS: OSNet gallery cache complete.",
        flush=True,
    )

    return (
        array,
        False,
        elapsed,
    )


def proxy_scores(
    gallery_descriptors,
    query_descriptor,
):
    query = np.asarray(
        query_descriptor,
        dtype=np.float32,
    )

    difference = (
        gallery_descriptors
        - query[
            None,
            :
        ]
    )

    rms = np.sqrt(
        np.mean(
            difference
            * difference,
            axis=1,
        )
    )

    scores = np.exp(
        -6.0
        * rms
    )

    return scores.astype(
        np.float32,
        copy=False,
    )


def evaluate_ranking(
    order,
    query_pid,
    query_camid,
    gallery_pids,
    gallery_camids,
    junk_pids,
):
    ranked_pids = (
        gallery_pids[
            order
        ]
    )

    ranked_camids = (
        gallery_camids[
            order
        ]
    )

    junk = np.zeros(
        ranked_pids.shape,
        dtype=bool,
    )

    for junk_pid in junk_pids:
        junk |= (
            ranked_pids
            == int(
                junk_pid
            )
        )

    junk |= (
        (
            ranked_pids
            == query_pid
        )
        & (
            ranked_camids
            == query_camid
        )
    )

    kept_pids = (
        ranked_pids[
            ~junk
        ]
    )

    matches = (
        kept_pids
        == query_pid
    )

    hit_positions = (
        np.flatnonzero(
            matches
        )
    )

    if hit_positions.size == 0:
        return None

    hit_numbers = np.arange(
        1,
        hit_positions.size
        + 1,
        dtype=np.float64,
    )

    precisions = (
        hit_numbers
        / (
            hit_positions
            .astype(
                np.float64
            )
            + 1.0
        )
    )

    average_precision = float(
        np.mean(
            precisions
        )
    )

    inp = float(
        hit_positions.size
        / (
            int(
                hit_positions[
                    -1
                ]
            )
            + 1
        )
    )

    first_hit = int(
        hit_positions[
            0
        ]
    )

    return {
        "ap":
            average_precision,
        "inp":
            inp,
        "rank1":
            int(
                first_hit < 1
            ),
        "rank5":
            int(
                first_hit < 5
            ),
        "rank10":
            int(
                first_hit < 10
            ),
    }


def default_stats():
    return {
        "processed_queries":
            0,
        "valid_queries":
            0,
        "ap_sum":
            0.0,
        "inp_sum":
            0.0,
        "rank1_hits":
            0,
        "rank5_hits":
            0,
        "rank10_hits":
            0,
        "osnet_invocations":
            0,
        "early_exits":
            0,
        "proxy_tokens_sum":
            0.0,
        "proxy_saving_sum":
            0.0,
        "stage_count_sum":
            0.0,
        "proxy_query_seconds_sum":
            0.0,
        "osnet_query_seconds_sum":
            0.0,
        "invoked_candidate_positive_hits":
            0,
        "route_counts": {},
        "initial_difficulty_counts": {},
        "stage_exit_counts": {},
    }


def increment(
    mapping,
    key,
) -> None:
    mapping[
        str(
            key
        )
    ] = (
        int(
            mapping.get(
                str(
                    key
                ),
                0,
            )
        )
        + 1
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
        "--baseline",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--runtime-output",
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

    benchmark_started = (
        time.perf_counter()
    )

    dataset = (
        args.dataset
        .resolve()
    )

    baseline_path = (
        args.baseline
        .resolve()
    )

    policy_path = (
        args.policy
        .resolve()
    )

    cache_dir = (
        args.cache_dir
        .resolve()
    )

    runtime_output = (
        args.runtime_output
        .resolve()
    )

    output = (
        args.output
        .resolve()
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = json.loads(
        baseline_path.read_text(
            encoding="utf-8"
        )
    )

    controller = (
        ResourceController
        .from_json(
            policy_path
        )
    )

    image_extractor = (
        ImageSignalExtractor()
    )

    encoder = (
        BudgetedVisualEncoder()
    )

    learned_encoder = (
        OSNetReIDEncoder(
            cache_dir=(
                Path(
                    os.environ[
                        "REID_CACHE_DIR"
                    ]
                )
            ),
            model_name=(
                MODEL_ALIAS
            ),
            torch_threads=(
                args.threads
            ),
        )
    )

    query, gallery = (
        load_market1501(
            str(
                dataset
            )
        )
    )

    query_paths = [
        Path(
            path
        )
        for path
        in query.image_paths
    ]

    gallery_paths = [
        Path(
            path
        )
        for path
        in gallery.image_paths
    ]

    if len(
        query_paths
    ) != EXPECTED_QUERY:
        raise SystemExit(
            "ERROR: Unexpected query count: "
            f"{len(query_paths)}"
        )

    if len(
        gallery_paths
    ) != EXPECTED_GALLERY:
        raise SystemExit(
            "ERROR: Unexpected gallery count: "
            f"{len(gallery_paths)}"
        )

    query_pids = np.asarray(
        query.pids,
        dtype=np.int64,
    )

    query_camids = np.asarray(
        query.camids,
        dtype=np.int64,
    )

    gallery_pids = np.asarray(
        gallery.pids,
        dtype=np.int64,
    )

    gallery_camids = np.asarray(
        gallery.camids,
        dtype=np.int64,
    )

    junk_pids = set(
        int(
            value
        )
        for value
        in gallery.gallery_junk_pids
    )

    signature = (
        dataset_signature(
            query,
            gallery,
        )
    )

    policy_sha = (
        sha256_file(
            policy_path
        )
    )

    print(
        "Dataset:",
        dataset,
        flush=True,
    )

    print(
        "Queries:",
        len(
            query_paths
        ),
        flush=True,
    )

    print(
        "Gallery:",
        len(
            gallery_paths
        ),
        flush=True,
    )

    print(
        "Junk PIDs:",
        sorted(
            junk_pids
        ),
        flush=True,
    )

    print(
        "Dataset signature:",
        signature,
        flush=True,
    )

    print(
        "Policy SHA256:",
        policy_sha,
        flush=True,
    )

    print(
        "Learned model:",
        MODEL_ALIAS,
        flush=True,
    )

    print(
        "Important: Market-1501 already contains "
        "pedestrian crops.",
        flush=True,
    )

    print(
        "YOLO is excluded from this retrieval-layer "
        "benchmark for comparability with the OSNet baseline.",
        flush=True,
    )

    print(
        "Detector-derived controller inputs are neutralized: "
        "confidence=1, foreground=1, occlusion=0, crowding=0.",
        flush=True,
    )

    print(
        "No thresholds are tuned on the Market-1501 test set.",
        flush=True,
    )

    if runtime_output.is_file():
        try:
            saved = json.loads(
                runtime_output.read_text(
                    encoding="utf-8"
                )
            )

            if (
                saved.get(
                    "dataset_signature"
                )
                == signature
                and saved.get(
                    "policy_sha256"
                )
                == policy_sha
                and saved.get(
                    "model"
                )[
                    "alias"
                ]
                == MODEL_ALIAS
                and int(
                    saved[
                        "dataset"
                    ][
                        "scored_queries"
                    ]
                )
                == EXPECTED_QUERY
            ):
                output.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    runtime_output,
                    output,
                )

                print(
                    "PASS: Existing completed runtime result "
                    "reused; no evaluation rerun required.",
                    flush=True,
                )

                return

        except Exception:
            pass

    (
        proxy_gallery,
        proxy_reused,
        proxy_cache_seconds,
    ) = build_proxy_gallery(
        gallery_paths,
        encoder,
        cache_dir,
        signature,
    )

    (
        osnet_gallery,
        osnet_reused,
        osnet_cache_seconds,
    ) = build_osnet_gallery(
        gallery_paths,
        learned_encoder,
        cache_dir,
        signature,
    )

    if (
        proxy_gallery.shape[
            0
        ]
        != EXPECTED_GALLERY
    ):
        raise SystemExit(
            "ERROR: Proxy cache gallery count mismatch."
        )

    if (
        osnet_gallery.shape[
            0
        ]
        != EXPECTED_GALLERY
    ):
        raise SystemExit(
            "ERROR: OSNet cache gallery count mismatch."
        )

    checkpoint_path = (
        cache_dir
        / "query_progress.json"
    )

    stats = (
        default_stats()
    )

    start_query = 0

    if checkpoint_path.is_file():
        try:
            checkpoint = json.loads(
                checkpoint_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                checkpoint.get(
                    "dataset_signature"
                )
                == signature
                and checkpoint.get(
                    "policy_sha256"
                )
                == policy_sha
                and checkpoint.get(
                    "model"
                )
                == MODEL_ALIAS
            ):
                saved_stats = (
                    checkpoint[
                        "stats"
                    ]
                )

                stats.update(
                    saved_stats
                )

                start_query = int(
                    stats[
                        "processed_queries"
                    ]
                )

                print(
                    "Resuming query evaluation at:",
                    start_query,
                    flush=True,
                )

        except Exception:
            stats = (
                default_stats()
            )

            start_query = 0

    evaluation_started = (
        time.perf_counter()
    )

    for query_index in range(
        start_query,
        EXPECTED_QUERY,
    ):
        query_path = (
            query_paths[
                query_index
            ]
        )

        query_pid = int(
            query_pids[
                query_index
            ]
        )

        query_camid = int(
            query_camids[
                query_index
            ]
        )

        query_started = (
            time.perf_counter()
        )

        image_signals = (
            image_extractor
            .extract(
                query_path
            )
        )

        signals = QuerySignals(
            detector_confidence=1.0,
            foreground_ratio=1.0,
            blur=(
                image_signals.blur
            ),
            visual_entropy=(
                image_signals
                .visual_entropy
            ),
            occlusion=0.0,
            crowding=0.0,
        )

        plan = (
            controller
            .initial_plan(
                signals
            )
        )

        increment(
            stats[
                "initial_difficulty_counts"
            ],
            plan.difficulty.value,
        )

        prepared = (
            encoder.prepare(
                query_path
            )
        )

        progressive = (
            encoder
            .start_progressive(
                prepared
            )
        )

        cumulative_tokens = 0
        stage_count = 0
        final_order = None
        final_action = None

        decision = None

        for stage_number in range(
            1,
            5,
        ):
            encoded = (
                encoder
                .encode_progressive(
                    progressive,
                    plan.token_retention,
                )
            )

            cumulative_tokens += (
                encoded
                .new_tokens_processed
            )

            stage_count += 1

            scores = (
                proxy_scores(
                    proxy_gallery,
                    encoded.descriptor,
                )
            )

            order = np.argsort(
                -scores,
                kind="stable",
            )

            top_count = min(
                plan.top_k,
                len(
                    gallery_paths
                ),
            )

            top_scores = (
                scores[
                    order[
                        :top_count
                    ]
                ]
            )

            decision = (
                controller
                .route_after_retrieval(
                    top_scores.tolist(),
                    plan,
                )
            )

            final_order = order
            final_action = (
                decision.action
            )

            if (
                decision.action
                == "refine"
            ):
                if (
                    decision.next_plan
                    is None
                ):
                    raise RuntimeError(
                        "Refinement without next plan."
                    )

                plan = (
                    decision.next_plan
                )

                continue

            break

        if (
            final_order
            is None
            or decision
            is None
        ):
            raise RuntimeError(
                "Adaptive retrieval produced no ranking."
            )

        proxy_query_seconds = (
            time.perf_counter()
            - query_started
        )

        stats[
            "proxy_query_seconds_sum"
        ] += float(
            proxy_query_seconds
        )

        stats[
            "stage_count_sum"
        ] += float(
            stage_count
        )

        stats[
            "proxy_tokens_sum"
        ] += float(
            cumulative_tokens
        )

        total_tokens = int(
            prepared.total_tokens
        )

        proxy_saving = (
            1.0
            - (
                cumulative_tokens
                / total_tokens
            )
        )

        stats[
            "proxy_saving_sum"
        ] += float(
            proxy_saving
        )

        increment(
            stats[
                "route_counts"
            ],
            final_action,
        )

        increment(
            stats[
                "stage_exit_counts"
            ],
            stage_count,
        )

        if (
            final_action
            == "early_exit"
        ):
            stats[
                "early_exits"
            ] += 1

        if (
            final_action
            == "invoke_vllm"
        ):
            stats[
                "osnet_invocations"
            ] += 1

            candidate_count = min(
                plan.top_k,
                EXPECTED_GALLERY,
            )

            candidate_indices = (
                final_order[
                    :candidate_count
                ]
            )

            candidate_valid_positive = (
                (
                    gallery_pids[
                        candidate_indices
                    ]
                    == query_pid
                )
                & (
                    gallery_camids[
                        candidate_indices
                    ]
                    != query_camid
                )
            )

            for junk_pid in junk_pids:
                candidate_valid_positive &= (
                    gallery_pids[
                        candidate_indices
                    ]
                    != junk_pid
                )

            if np.any(
                candidate_valid_positive
            ):
                stats[
                    "invoked_candidate_positive_hits"
                ] += 1

            (
                query_embedding,
                inference_seconds,
                _,
            ) = (
                learned_encoder
                .embed_path_timed(
                    query_path
                )
            )

            stats[
                "osnet_query_seconds_sum"
            ] += float(
                inference_seconds
            )

            candidate_embeddings = np.asarray(
                osnet_gallery[
                    candidate_indices
                ],
                dtype=np.float32,
            )

            cosine = (
                candidate_embeddings
                @ query_embedding
            )

            learned_order = np.argsort(
                -cosine,
                kind="stable",
            )

            reranked = (
                candidate_indices[
                    learned_order
                ]
            )

            final_order = np.concatenate(
                (
                    reranked,
                    final_order[
                        candidate_count:
                    ],
                )
            )

        metrics = (
            evaluate_ranking(
                final_order,
                query_pid,
                query_camid,
                gallery_pids,
                gallery_camids,
                junk_pids,
            )
        )

        stats[
            "processed_queries"
        ] = (
            query_index
            + 1
        )

        if metrics is not None:
            stats[
                "valid_queries"
            ] += 1

            stats[
                "ap_sum"
            ] += (
                metrics[
                    "ap"
                ]
            )

            stats[
                "inp_sum"
            ] += (
                metrics[
                    "inp"
                ]
            )

            stats[
                "rank1_hits"
            ] += (
                metrics[
                    "rank1"
                ]
            )

            stats[
                "rank5_hits"
            ] += (
                metrics[
                    "rank5"
                ]
            )

            stats[
                "rank10_hits"
            ] += (
                metrics[
                    "rank10"
                ]
            )

        completed = (
            query_index
            + 1
        )

        if (
            completed % 25 == 0
            or completed
            == EXPECTED_QUERY
        ):
            atomic_json(
                checkpoint_path,
                {
                    "dataset_signature":
                        signature,
                    "policy_sha256":
                        policy_sha,
                    "model":
                        MODEL_ALIAS,
                    "stats":
                        stats,
                },
            )

        if (
            completed % 50 == 0
            or completed
            == EXPECTED_QUERY
        ):
            valid = max(
                1,
                int(
                    stats[
                        "valid_queries"
                    ]
                ),
            )

            print(
                "Queries:",
                completed,
                "/",
                EXPECTED_QUERY,
                "| mAP:",
                f"{100.0 * stats['ap_sum'] / valid:.3f}%",
                "| Rank-1:",
                (
                    f"{100.0 * stats['rank1_hits'] / valid:.3f}%"
                ),
                "| OSNet:",
                (
                    f"{100.0 * stats['osnet_invocations'] / completed:.1f}%"
                ),
                "| patch saving:",
                (
                    f"{100.0 * stats['proxy_saving_sum'] / completed:.1f}%"
                ),
                flush=True,
            )

    evaluation_seconds = (
        time.perf_counter()
        - evaluation_started
    )

    valid = int(
        stats[
            "valid_queries"
        ]
    )

    processed = int(
        stats[
            "processed_queries"
        ]
    )

    if valid != EXPECTED_QUERY:
        raise SystemExit(
            "ERROR: Expected all 3368 queries "
            f"to be valid, got {valid}."
        )

    if processed != EXPECTED_QUERY:
        raise SystemExit(
            "ERROR: Not all queries were processed."
        )

    metrics_percent = {
        "map":
            100.0
            * stats[
                "ap_sum"
            ]
            / valid,
        "rank1":
            100.0
            * stats[
                "rank1_hits"
            ]
            / valid,
        "rank5":
            100.0
            * stats[
                "rank5_hits"
            ]
            / valid,
        "rank10":
            100.0
            * stats[
                "rank10_hits"
            ]
            / valid,
        "minp":
            100.0
            * stats[
                "inp_sum"
            ]
            / valid,
    }

    invocation_rate = (
        stats[
            "osnet_invocations"
        ]
        / processed
    )

    early_exit_rate = (
        stats[
            "early_exits"
        ]
        / processed
    )

    average_proxy_patches = (
        stats[
            "proxy_tokens_sum"
        ]
        / processed
    )

    proxy_patch_saving = (
        stats[
            "proxy_saving_sum"
        ]
        / processed
    )

    average_stages = (
        stats[
            "stage_count_sum"
        ]
        / processed
    )

    average_proxy_seconds = (
        stats[
            "proxy_query_seconds_sum"
        ]
        / processed
    )

    if (
        stats[
            "osnet_invocations"
        ]
        > 0
    ):
        average_osnet_seconds = (
            stats[
                "osnet_query_seconds_sum"
            ]
            / stats[
                "osnet_invocations"
            ]
        )

        candidate_recall = (
            stats[
                "invoked_candidate_positive_hits"
            ]
            / stats[
                "osnet_invocations"
            ]
        )

    else:
        average_osnet_seconds = 0.0
        candidate_recall = None

    query_side_osnet_forward_reduction = (
        1.0
        - invocation_rate
    )

    baseline_metrics = (
        baseline[
            "metrics_percent"
        ]
    )

    delta = {
        name:
            float(
                metrics_percent[
                    name
                ]
                - float(
                    baseline_metrics[
                        name
                    ]
                )
            )
        for name in (
            "map",
            "rank1",
            "rank5",
            "rank10",
            "minp",
        )
    }

    benchmark_seconds = (
        time.perf_counter()
        - benchmark_started
    )

    result = {
        "benchmark":
            (
                "Market-1501 RAC-ReID "
                "retrieval-layer adaptive ablation"
            ),
        "dataset_signature":
            signature,
        "policy_sha256":
            policy_sha,
        "dataset": {
            "name":
                "Market-1501-v15.09.15",
            "query_images":
                EXPECTED_QUERY,
            "gallery_images":
                EXPECTED_GALLERY,
            "scored_queries":
                valid,
            "gallery_junk_pids":
                sorted(
                    junk_pids
                ),
        },
        "model": {
            "alias":
                MODEL_ALIAS,
            "learned_architecture":
                "OSNet x1.0",
            "learned_rerank_similarity":
                "cosine on normalized embeddings",
            "proxy_encoder":
                (
                    "handcrafted identity-aware "
                    "visual-patch descriptor"
                ),
        },
        "scope": {
            "type":
                "retrieval-layer ablation",
            "yolo_included":
                False,
            "reason":
                (
                    "Market-1501 provides person crops; "
                    "YOLO is excluded to match the "
                    "full-OSNet retrieval baseline scope."
                ),
            "detector_evidence":
                {
                    "detector_confidence":
                        1.0,
                    "foreground_ratio":
                        1.0,
                    "occlusion":
                        0.0,
                    "crowding":
                        0.0,
                },
            "threshold_tuning_on_test_set":
                False,
        },
        "protocol": {
            "same_identity_same_camera":
                "ignored",
            "gallery_junk_pids":
                sorted(
                    junk_pids
                ),
            "pid_zero":
                "retained as distractor if present",
            "map":
                (
                    "mean precision at each valid "
                    "positive rank"
                ),
            "minp":
                (
                    "valid positives divided by rank "
                    "of last positive"
                ),
        },
        "metrics_percent":
            metrics_percent,
        "baseline_full_osnet_percent":
            {
                key:
                    float(
                        baseline_metrics[
                            key
                        ]
                    )
                for key in (
                    "map",
                    "rank1",
                    "rank5",
                    "rank10",
                    "minp",
                )
            },
        "delta_vs_full_osnet_pp":
            delta,
        "efficiency": {
            "osnet_invocation_rate":
                invocation_rate,
            "query_side_osnet_forward_reduction":
                query_side_osnet_forward_reduction,
            "early_exit_rate":
                early_exit_rate,
            "average_proxy_patches_processed":
                average_proxy_patches,
            "proxy_patch_saving":
                proxy_patch_saving,
            "average_adaptive_stages":
                average_stages,
            "average_proxy_query_seconds":
                average_proxy_seconds,
            "average_osnet_query_inference_seconds_when_invoked":
                average_osnet_seconds,
            "hard_candidate_positive_recall_when_invoked":
                candidate_recall,
        },
        "routing": {
            "final_routes":
                stats[
                    "route_counts"
                ],
            "initial_difficulty":
                stats[
                    "initial_difficulty_counts"
                ],
            "stage_exit_counts":
                stats[
                    "stage_exit_counts"
                ],
        },
        "runtime": {
            "device":
                "cpu",
            "threads":
                args.threads,
            "proxy_gallery_cache_reused":
                proxy_reused,
            "osnet_gallery_cache_reused":
                osnet_reused,
            "proxy_gallery_cache_build_seconds":
                proxy_cache_seconds,
            "osnet_gallery_cache_build_seconds":
                osnet_cache_seconds,
            "query_evaluation_wall_seconds":
                evaluation_seconds,
            "total_step_wall_seconds":
                benchmark_seconds,
            "python":
                platform.python_version(),
            "machine":
                platform.machine(),
        },
        "scientific_notes": [
            (
                "The patch budget is a handcrafted "
                "visual-patch proxy and is not a "
                "transformer token count."
            ),
            (
                "OSNet gallery embeddings are offline "
                "precomputation, matching the deployed "
                "gallery workflow."
            ),
            (
                "Only queries routed to the learned stage "
                "receive an OSNet query forward pass."
            ),
            (
                "When OSNet is invoked, it reranks only "
                "the adaptive proxy candidate set; the "
                "remaining gallery retains proxy order."
            ),
            (
                "This fixed policy was not tuned on the "
                "Market-1501 test queries."
            ),
            (
                "Because Market-1501 contains pedestrian "
                "crops, detector-derived difficulty "
                "evidence is neutralized in this ablation."
            ),
            (
                "Accuracy loss is reported even if it "
                "shows the current adaptive policy is not "
                "yet competitive."
            ),
        ],
    }

    atomic_json(
        runtime_output,
        result,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        runtime_output,
        output,
    )

    checkpoint_path.unlink(
        missing_ok=True
    )

    print()
    print(
        "========================================",
        flush=True,
    )

    print(
        " RAC-REID ADAPTIVE RESULTS",
        flush=True,
    )

    print(
        "========================================",
        flush=True,
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
            f"{metrics_percent[key]:.3f}% "
            f"(delta {delta[key]:+.3f} pp)",
            flush=True,
        )

    print(
        "OSNet invocation:",
        f"{invocation_rate:.1%}",
        flush=True,
    )

    print(
        "Query-side OSNet forward reduction:",
        (
            f"{query_side_osnet_forward_reduction:.1%}"
        ),
        flush=True,
    )

    print(
        "Early exit:",
        f"{early_exit_rate:.1%}",
        flush=True,
    )

    print(
        "Average proxy patches:",
        f"{average_proxy_patches:.2f}",
        flush=True,
    )

    print(
        "Proxy patch saving:",
        f"{proxy_patch_saving:.1%}",
        flush=True,
    )

    print(
        "Average stages:",
        f"{average_stages:.3f}",
        flush=True,
    )

    print(
        "Result:",
        output,
        flush=True,
    )

    print(
        "PASS: Full adaptive Market-1501 "
        "ablation finished.",
        flush=True,
    )


if __name__ == "__main__":
    main()
