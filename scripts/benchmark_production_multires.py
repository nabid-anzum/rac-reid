#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PROJECT = Path("/root/rac-reid")
WEB_APP = PROJECT / "scripts/web_app.py"

MODEL = (
    PROJECT
    / ".runtime/models/multires/"
    "osnet_x1_0_multires_160x80.pt"
)

EXPECTED_MODEL_SHA = (
    "d0570ff0fec0905b75bef1b90ec5147b"
    "96ac7b51d8ba9f30de5fa3f0c0757dec"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_web():
    name = "rac_29he2_benchmark"

    spec = importlib.util.spec_from_file_location(
        name,
        WEB_APP,
    )

    assert spec
    assert spec.loader

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def percentile(values: list[float], q: float) -> float:
    values = sorted(float(v) for v in values)

    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return values[lo]

    weight = pos - lo

    return (
        values[lo] * (1.0 - weight)
        + values[hi] * weight
    )


def stats(values: list[float]) -> dict[str, float | int]:
    values = [float(v) for v in values]

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "stdev": statistics.stdev(values)
        if len(values) > 1
        else 0.0,
        "min": min(values),
        "max": max(values),
    }


def saving(low: float, full: float) -> float:
    return (1.0 - low / full) * 100.0


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")

    if path.exists():
        for line in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()

    return platform.processor() or "unknown"


def cleanup(prepared) -> None:
    try:
        crop = Path(prepared.crop_path)
        parent = crop.parent

        root = PROJECT / ".runtime/query-work"

        if parent == root or root in parent.parents:
            shutil.rmtree(
                parent,
                ignore_errors=True,
            )
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    assert args.runs >= 10
    assert args.warmup >= 1

    assert sha256(MODEL) == EXPECTED_MODEL_SHA

    module = load_web()

    # Construct application first because AppState currently
    # applies its own CPU-thread setting.
    state = module.AppState()

    entries = state.gallery.entries()

    if len(entries) < 2:
        raise RuntimeError(
            "At least two gallery identities required."
        )

    timing_entry = entries[0]
    query = Path(timing_entry["original"])

    if not query.is_file():
        raise RuntimeError("Timing input missing.")

    qtext = str(query)

    if (
        "Market-1501-v15.09.15/query" in qtext
        or
        "Market-1501-v15.09.15/bounding_box_test"
        in qtext
    ):
        raise RuntimeError(
            "Refusing official Market-1501 test data."
        )

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    def run_once(
        mode: str,
        measured_index=None,
        pair_index=None,
        order_position=None,
    ) -> dict[str, Any]:

        started = time.perf_counter()

        events: dict[str, float] = {}

        def progress(
            code,
            title,
            detail,
            percent,
        ):
            events[str(code)] = (
                time.perf_counter() - started
            )

        prepared, result, learned, timings = (
            state.search_multires(
                query,
                mode=mode,
                progress_callback=progress,
            )
        )

        wall = time.perf_counter() - started

        try:
            assert (
                result.get("pipeline_kind")
                == "trained_multires"
            )

            required = (
                "locating_person",
                "person_located",
                "reid_preprocessing_complete",
                "query_embedding",
                "gallery_embedding_search",
            )

            for code in required:
                if code not in events:
                    raise RuntimeError(
                        f"Missing event: {code}"
                    )

            detector = (
                events["person_located"]
                - events["locating_person"]
            )

            crop = (
                events["reid_preprocessing_complete"]
                - events["person_located"]
            )

            observed_learned = (
                events["gallery_embedding_search"]
                - events["query_embedding"]
            )

            osnet = float(
                learned.inference_seconds
            )

            learned_seconds = float(
                timings["learned_seconds"]
            )

            gallery_overhead = max(
                0.0,
                learned_seconds - osnet,
            )

            top_identity = (
                learned.candidates[0].label
                if learned.candidates
                else ""
            )

            return {
                "measured_index": measured_index,
                "pair_index": pair_index,
                "order_position": order_position,
                "mode": mode,
                "query_resolution":
                    "160x80"
                    if mode == "efficient"
                    else "256x128",
                "detector_seconds": detector,
                "crop_seconds": crop,
                "preprocess_seconds": float(
                    timings["preprocess_seconds"]
                ),
                "osnet_inference_seconds": osnet,
                "learned_stage_seconds":
                    learned_seconds,
                "observed_embedding_to_gallery_seconds":
                    observed_learned,
                "gallery_ranking_overhead_seconds":
                    gallery_overhead,
                "total_backend_seconds": float(
                    timings["total_seconds"]
                ),
                "wall_seconds": wall,
                "top_identity": top_identity,
                "top1_cosine": float(
                    learned.top1
                ),
                "top1_top2_margin": float(
                    learned.margin
                ),
                "gallery_count": len(
                    learned.candidates
                ),
            }

        finally:
            cleanup(prepared)

    print("Timing input:", query.name)
    print(
        "Timing identity:",
        timing_entry["label"],
    )
    print("Gallery identities:", len(entries))

    print()
    print("Warm-up...")

    for i in range(args.warmup):
        order = (
            ("efficient", "full")
            if i % 2 == 0
            else ("full", "efficient")
        )

        for mode in order:
            run_once(mode)

    # The application establishes its CPU defaults when
    # constructing/loading the model. Override only after
    # warm-up so the measured runs really use four threads.
    torch.set_num_threads(4)

    actual_threads = torch.get_num_threads()

    print("Warm-up complete.")
    print(
        "Measured Torch threads:",
        actual_threads,
    )

    if actual_threads != 4:
        raise RuntimeError(
            f"Expected 4 Torch threads, got "
            f"{actual_threads}"
        )

    print()
    print("Measured alternating runs...")

    rows: list[dict[str, Any]] = []
    measured_index = 0

    for pair in range(args.runs):
        order = (
            ("efficient", "full")
            if pair % 2 == 0
            else ("full", "efficient")
        )

        for position, mode in enumerate(
            order,
            start=1,
        ):
            measured_index += 1

            row = run_once(
                mode,
                measured_index,
                pair + 1,
                position,
            )

            rows.append(row)

            print(
                f"{measured_index:02d}/"
                f"{args.runs * 2:02d} "
                f"{mode:9s} "
                f"osnet="
                f"{row['osnet_inference_seconds']:.6f}s "
                f"total="
                f"{row['total_backend_seconds']:.6f}s"
            )

    metric_names = (
        "detector_seconds",
        "crop_seconds",
        "preprocess_seconds",
        "osnet_inference_seconds",
        "learned_stage_seconds",
        "observed_embedding_to_gallery_seconds",
        "gallery_ranking_overhead_seconds",
        "total_backend_seconds",
        "wall_seconds",
    )

    summary = {}

    for mode in ("efficient", "full"):
        subset = [
            row
            for row in rows
            if row["mode"] == mode
        ]

        summary[mode] = {
            "resolution":
                "160x80"
                if mode == "efficient"
                else "256x128",
            "runs": len(subset),
            "metrics": {
                name: stats(
                    [
                        row[name]
                        for row in subset
                    ]
                )
                for name in metric_names
            },
            "top_identity_counts":
                dict(
                    Counter(
                        row["top_identity"]
                        for row in subset
                    )
                ),
            "top1_cosine":
                stats(
                    [
                        row["top1_cosine"]
                        for row in subset
                    ]
                ),
            "top1_top2_margin":
                stats(
                    [
                        row["top1_top2_margin"]
                        for row in subset
                    ]
                ),
        }

    e = summary["efficient"]["metrics"]
    f = summary["full"]["metrics"]

    savings = {
        "osnet_inference_median_percent":
            saving(
                e["osnet_inference_seconds"]["median"],
                f["osnet_inference_seconds"]["median"],
            ),
        "learned_stage_median_percent":
            saving(
                e["learned_stage_seconds"]["median"],
                f["learned_stage_seconds"]["median"],
            ),
        "total_backend_median_percent":
            saving(
                e["total_backend_seconds"]["median"],
                f["total_backend_seconds"]["median"],
            ),
        "wall_median_percent":
            saving(
                e["wall_seconds"]["median"],
                f["wall_seconds"]["median"],
            ),
    }

    source_commit = (
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

    result = {
        "benchmark_name":
            (
                "RAC-ReID controlled "
                "production multi-resolution benchmark"
            ),
        "date": "2026-11-23",
        "source_git_commit": source_commit,
        "checkpoint_sha256":
            EXPECTED_MODEL_SHA,
        "model":
            "osnet_x1_0_multires_epoch8",
        "device": "cpu",
        "torch_threads": actual_threads,
        "cpu": cpu_model(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "warmup_runs_per_mode":
            args.warmup,
        "measured_runs_per_mode":
            args.runs,
        "total_measured_searches":
            len(rows),
        "order":
            (
                "paired alternating: "
                "efficient/full then full/efficient"
            ),
        "timing_input": {
            "source":
                (
                    "existing enrolled gallery "
                    "original; timing-only input"
                ),
            "basename": query.name,
            "identity_label":
                timing_entry["label"],
            "accuracy_evaluation":
                False,
        },
        "gallery_identity_count":
            len(entries),
        "standard_market1501_test_used":
            False,
        "trained_model_official_test_used":
            False,
        "modes": summary,
        "median_savings_percent":
            savings,
        "methodology_notes": [
            (
                "Five warm-up runs per mode "
                "are excluded."
            ),
            (
                "Thirty measured runs per mode "
                "are used."
            ),
            (
                "Efficient/full ordering alternates "
                "within pairs."
            ),
            (
                "The same input is used for "
                "both resolutions."
            ),
            (
                "YOLO detection and crop are "
                "repeated for each search."
            ),
            (
                "OSNet inference time is reported "
                "separately."
            ),
            (
                "Timing input is an enrolled image "
                "and is not accuracy evidence."
            ),
            (
                "Official trained-model Market-1501 "
                "test data is not used."
            ),
        ],
    }

    args.json_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.csv_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.json_output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with args.csv_output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(
                rows[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "========================================"
    )
    print(
        " PRODUCTION MULTI-RES BENCHMARK RESULTS"
    )
    print(
        "========================================"
    )

    for mode in ("efficient", "full"):
        m = summary[mode]["metrics"]

        print()
        print(
            mode.upper(),
            summary[mode]["resolution"],
        )

        print(
            "  OSNet median:",
            f"{m['osnet_inference_seconds']['median']:.6f}s",
        )
        print(
            "  OSNet mean:  ",
            f"{m['osnet_inference_seconds']['mean']:.6f}s",
        )
        print(
            "  OSNet p95:   ",
            f"{m['osnet_inference_seconds']['p95']:.6f}s",
        )
        print(
            "  Total median:",
            f"{m['total_backend_seconds']['median']:.6f}s",
        )
        print(
            "  Total p95:   ",
            f"{m['total_backend_seconds']['p95']:.6f}s",
        )

    print()
    print(
        "Median OSNet saving:",
        f"{savings['osnet_inference_median_percent']:+.3f}%",
    )
    print(
        "Median learned-stage saving:",
        f"{savings['learned_stage_median_percent']:+.3f}%",
    )
    print(
        "Median total-backend saving:",
        f"{savings['total_backend_median_percent']:+.3f}%",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
