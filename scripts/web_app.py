from __future__ import annotations

import threading

import time

import argparse
import cgi
import html
import json
import secrets
import shutil
import uuid
from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
)
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rac_reid import (
    AdaptiveReIDPipeline,
    BudgetedVisualEncoder,
    GalleryIndex,
    ImageSignalExtractor,
    NoPersonDetectedError,
    OSNetReIDEncoder,
    PersonQueryPreprocessor,
    ResourceController,
    YOLOPersonDetector,
)


from rac_reid.learned_reid import MultiResolutionOSNetReIDEncoder

ROOT = Path(
    __file__
).resolve().parents[1]

RUNTIME = (
    ROOT
    / ".runtime"
)

GALLERY_ROOT = (
    ROOT
    / "data"
    / "gallery"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "yolo26n.pt"
)

REID_CACHE = (
    RUNTIME
    / "reid-cache"
)

MULTIRES_CHECKPOINT = (
    RUNTIME
    / "models"
    / "multires"
    / "osnet_x1_0_multires_160x80.pt"
)

MULTIRES_CHECKPOINT_SHA = (
    "d0570ff0fec0905b75bef1b90ec5147b96ac7b51d8ba9f30de5fa3f0c0757dec"
)

ADMIN_TOKEN_FILE = (
    RUNTIME
    / "admin.token"
)

MAX_UPLOAD = (
    12
    * 1024
    * 1024
)

ALLOWED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


class SearchJobStore:
    """Thread-safe in-memory progress state for live search jobs."""

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_jobs: int = 100,
    ) -> None:
        self.ttl_seconds = float(
            ttl_seconds
        )

        self.max_jobs = int(
            max_jobs
        )

        self._lock = (
            threading.Lock()
        )

        self._jobs = {}

    def _cleanup_locked(
        self,
    ) -> None:
        now = time.time()

        expired = [
            job_id
            for (
                job_id,
                job,
            )
            in self._jobs.items()
            if (
                now
                - job["created_epoch"]
                > self.ttl_seconds
            )
        ]

        for job_id in expired:
            self._jobs.pop(
                job_id,
                None,
            )

        if (
            len(self._jobs)
            <= self.max_jobs
        ):
            return

        ordered = sorted(
            self._jobs.items(),
            key=lambda item:
                item[1][
                    "created_epoch"
                ],
        )

        remove_count = (
            len(self._jobs)
            - self.max_jobs
        )

        for (
            job_id,
            _,
        ) in ordered[
            :remove_count
        ]:
            self._jobs.pop(
                job_id,
                None,
            )

    @staticmethod
    def _event(
        job,
        code: str,
        title: str,
        detail: str,
        progress: int,
    ):
        elapsed = (
            time.perf_counter()
            - job[
                "started_monotonic"
            ]
        )

        return {
            "sequence":
                len(
                    job[
                        "events"
                    ]
                )
                + 1,
            "code":
                str(code),
            "title":
                str(title),
            "detail":
                str(detail),
            "progress":
                int(
                    max(
                        0,
                        min(
                            100,
                            progress,
                        ),
                    )
                ),
            "elapsed_seconds":
                float(
                    elapsed
                ),
        }

    def create(
        self,
    ) -> str:
        with self._lock:
            self._cleanup_locked()

            job_id = (
                uuid.uuid4()
                .hex
            )

            job = {
                "id":
                    job_id,
                "status":
                    "running",
                "progress":
                    2,
                "created_epoch":
                    time.time(),
                "started_monotonic":
                    time.perf_counter(),
                "events":
                    [],
                "current":
                    None,
                "error":
                    None,
                "result":
                    None,
            }

            event = (
                self._event(
                    job,
                    "upload_received",
                    "Upload received",
                    (
                        "The server accepted the image and "
                        "created a RAC-ReID search job."
                    ),
                    2,
                )
            )

            job[
                "events"
            ].append(
                event
            )

            job[
                "current"
            ] = event

            self._jobs[
                job_id
            ] = job

            return job_id

    def update(
        self,
        job_id: str,
        code: str,
        title: str,
        detail: str,
        progress: int,
    ) -> None:
        with self._lock:
            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return

            if (
                job[
                    "status"
                ]
                != "running"
            ):
                return

            progress = max(
                int(
                    job[
                        "progress"
                    ]
                ),
                int(
                    progress
                ),
            )

            event = (
                self._event(
                    job,
                    code,
                    title,
                    detail,
                    progress,
                )
            )

            job[
                "progress"
            ] = progress

            job[
                "current"
            ] = event

            job[
                "events"
            ].append(
                event
            )

            if (
                len(
                    job[
                        "events"
                    ]
                )
                > 250
            ):
                job[
                    "events"
                ] = (
                    job[
                        "events"
                    ][
                        -250:
                    ]
                )

    def finish(
        self,
        job_id: str,
        result,
    ) -> None:
        with self._lock:
            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return

            event = (
                self._event(
                    job,
                    "complete",
                    "Search complete",
                    (
                        "RAC-ReID finished the query and "
                        "generated the final ranking."
                    ),
                    100,
                )
            )

            job[
                "events"
            ].append(
                event
            )

            job[
                "current"
            ] = event

            job[
                "progress"
            ] = 100

            job[
                "status"
            ] = "done"

            job[
                "result"
            ] = result

    def fail(
        self,
        job_id: str,
        error: str,
    ) -> None:
        with self._lock:
            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return

            event = (
                self._event(
                    job,
                    "failed",
                    "Search failed",
                    str(
                        error
                    ),
                    int(
                        job[
                            "progress"
                        ]
                    ),
                )
            )

            job[
                "events"
            ].append(
                event
            )

            job[
                "current"
            ] = event

            job[
                "status"
            ] = "error"

            job[
                "error"
            ] = str(
                error
            )

    def snapshot(
        self,
        job_id: str,
    ):
        with self._lock:
            self._cleanup_locked()

            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return None

            elapsed = (
                time.perf_counter()
                - job[
                    "started_monotonic"
                ]
            )

            return {
                "id":
                    job[
                        "id"
                    ],
                "status":
                    job[
                        "status"
                    ],
                "progress":
                    job[
                        "progress"
                    ],
                "elapsed_seconds":
                    float(
                        elapsed
                    ),
                "current":
                    (
                        dict(
                            job[
                                "current"
                            ]
                        )
                        if job[
                            "current"
                        ]
                        is not None
                        else None
                    ),
                "events":
                    [
                        dict(
                            event
                        )
                        for event
                        in job[
                            "events"
                        ]
                    ],
                "error":
                    job[
                        "error"
                    ],
            }

    def result(
        self,
        job_id: str,
    ):
        with self._lock:
            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return None

            return job[
                "result"
            ]


class GalleryStore:
    def __init__(
        self,
        root: Path,
        preprocessor: PersonQueryPreprocessor,
        learned_encoder: OSNetReIDEncoder,
    ) -> None:
        self.root = root
        self.preprocessor = (
            preprocessor
        )
        self.learned_encoder = (
            learned_encoder
        )

        self.originals = (
            root
            / "originals"
        )

        self.crops = (
            root
            / "crops"
        )

        self.embeddings = (
            root
            / "embeddings"
        )

        self.metadata = (
            root
            / "metadata.json"
        )

        self.originals.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.crops.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embeddings.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._cached_index = None

    def entries(
        self,
    ) -> list[
        dict[str, Any]
    ]:
        if not self.metadata.is_file():
            return []

        data = json.loads(
            self.metadata.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            list,
        ):
            raise RuntimeError(
                "Gallery metadata is invalid."
            )

        return data

    def _write(
        self,
        entries: list[
            dict[str, Any]
        ],
    ) -> None:
        temp = (
            self.metadata
            .with_suffix(
                ".json.tmp"
            )
        )

        temp.write_text(
            json.dumps(
                entries,
                indent=2,
            ),
            encoding="utf-8",
        )

        temp.replace(
            self.metadata
        )

    def add(
        self,
        label: str,
        uploaded: Path,
    ) -> dict[
        str,
        Any
    ]:
        label = label.strip()

        if not label:
            raise ValueError(
                "Gallery label is required."
            )

        if len(label) > 80:
            raise ValueError(
                "Gallery label is too long."
            )

        entries = self.entries()

        if any(
            item["label"].casefold()
            == label.casefold()
            for item in entries
        ):
            raise ValueError(
                "Gallery labels must be unique."
            )

        identifier = (
            uuid.uuid4().hex
        )

        workspace = (
            RUNTIME
            / "gallery-work"
            / identifier
        )

        prepared = (
            self.preprocessor.prepare(
                uploaded,
                workspace,
            )
        )

        suffix = (
            uploaded.suffix.lower()
        )

        if suffix not in ALLOWED_SUFFIXES:
            suffix = ".jpg"

        original_target = (
            self.originals
            / f"{identifier}{suffix}"
        )

        crop_target = (
            self.crops
            / f"{identifier}.jpg"
        )

        embedding_target = (
            self.embeddings
            / f"{identifier}.npy"
        )

        try:
            shutil.copy2(
                uploaded,
                original_target,
            )

            shutil.copy2(
                prepared.crop_path,
                crop_target,
            )

            self.learned_encoder.save_embedding(
                crop_target,
                embedding_target,
            )

        except Exception:
            original_target.unlink(
                missing_ok=True
            )

            crop_target.unlink(
                missing_ok=True
            )

            embedding_target.unlink(
                missing_ok=True
            )

            raise

        entry = {
            "id": identifier,
            "label": label,
            "original": str(
                original_target
            ),
            "crop": str(
                crop_target
            ),
            "embedding": str(
                embedding_target
            ),
            "confidence": (
                prepared
                .primary_confidence
            ),
            "people": (
                prepared
                .person_count
            ),
        }

        entries.append(
            entry
        )

        self._write(
            entries
        )

        self._cached_index = None

        return entry

    def index(
        self,
        encoder: BudgetedVisualEncoder,
    ) -> GalleryIndex:
        entries = self.entries()

        if len(entries) < 2:
            raise ValueError(
                "Add at least two gallery identities before searching."
            )

        if self._cached_index is None:
            mapping = {
                item["label"]:
                    Path(
                        item["crop"]
                    )
                for item in entries
            }

            self._cached_index = (
                GalleryIndex.build(
                    mapping,
                    encoder,
                )
            )

        return self._cached_index

    def learned_rerank(
        self,
        query_crop: Path,
        cheap_candidates,
    ):
        entries = self.entries()

        by_label = {
            item["label"]:
                item
            for item in entries
        }

        labels = [
            candidate.label
            for candidate
            in cheap_candidates
        ]

        cheap_scores = {
            candidate.label:
                candidate.score
            for candidate
            in cheap_candidates
        }

        learned = {}

        changed = False

        for label in labels:
            entry = by_label.get(
                label
            )

            if entry is None:
                continue

            embedding_value = (
                entry.get(
                    "embedding"
                )
            )

            if embedding_value:
                embedding_path = Path(
                    embedding_value
                )
            else:
                embedding_path = (
                    self.embeddings
                    / f"{entry['id']}.npy"
                )

            if not embedding_path.is_file():
                self.learned_encoder.save_embedding(
                    Path(
                        entry["crop"]
                    ),
                    embedding_path,
                )

                entry[
                    "embedding"
                ] = str(
                    embedding_path
                )

                changed = True

            learned[
                label
            ] = (
                self.learned_encoder
                .load_embedding(
                    embedding_path
                )
            )

        if changed:
            self._write(
                entries
            )

        if len(learned) < 2:
            raise RuntimeError(
                "At least two learned gallery "
                "embeddings are required."
            )

        return (
            self.learned_encoder
            .rerank_precomputed(
                query_path=query_crop,
                gallery_embeddings=learned,
                cheap_scores=cheap_scores,
            )
        )


    def learned_search_all(
        self,
        query_crop: Path,
        mode: str = "efficient",
    ):
        """
        Search every learned gallery embedding.

        Gallery embeddings are precomputed once
        at 256x128. The query uses either 160x80
        efficient mode or explicit 256x128 full
        mode.
        """
        entries = self.entries()

        if len(entries) < 2:
            raise ValueError(
                "Add at least two gallery "
                "identities before searching."
            )

        learned = {}

        changed = False

        for entry in entries:
            label = str(
                entry["label"]
            )

            embedding_value = (
                entry.get(
                    "embedding"
                )
            )

            if embedding_value:
                embedding_path = Path(
                    embedding_value
                )

            else:
                embedding_path = (
                    self.embeddings
                    / f"{entry['id']}.npy"
                )

            if not embedding_path.is_file():
                self.learned_encoder.save_embedding(
                    Path(
                        entry["crop"]
                    ),
                    embedding_path,
                )

                entry[
                    "embedding"
                ] = str(
                    embedding_path
                )

                changed = True

            learned[
                label
            ] = (
                self.learned_encoder
                .load_embedding(
                    embedding_path
                )
            )

        if changed:
            self._write(
                entries
            )

        cheap_scores = {
            label: 0.0
            for label
            in learned
        }

        return (
            self.learned_encoder
            .rerank_precomputed_mode(
                query_path=query_crop,
                gallery_embeddings=(
                    learned
                ),
                cheap_scores=(
                    cheap_scores
                ),
                mode=mode,
            )
        )


class AppState:
    def __init__(
        self,
    ) -> None:
        self.controller = (
            ResourceController.from_json(
                ROOT
                / "configs"
                / "resource_policy.json"
            )
        )

        self.image_extractor = (
            ImageSignalExtractor()
        )

        self.encoder = (
            BudgetedVisualEncoder()
        )

        self.detector = (
            YOLOPersonDetector(
                MODEL_PATH,
                torch_threads=2,
            )
        )

        self.preprocessor = (
            PersonQueryPreprocessor(
                detector=self.detector,
                controller=self.controller,
                image_extractor=(
                    self.image_extractor
                ),
            )
        )

        self.learned_encoder = (
            MultiResolutionOSNetReIDEncoder(
                cache_dir=(
                    REID_CACHE
                ),
                checkpoint_path=(
                    MULTIRES_CHECKPOINT
                ),
                expected_sha256=(
                    MULTIRES_CHECKPOINT_SHA
                ),
                query_size=(
                    160,
                    80,
                ),
                gallery_size=(
                    256,
                    128,
                ),
                torch_threads=4,
            )
        )

        self.gallery = (
            GalleryStore(
                GALLERY_ROOT,
                self.preprocessor,
                self.learned_encoder,
            )
        )

        self.jobs = (
            SearchJobStore()
        )

        # YOLO / Torch state is shared by the web process.
        # Serialize expensive searches rather than running
        # multiple inference pipelines against the same
        # model state concurrently.
        self._search_lock = (
            threading.Lock()
        )

    def search(
        self,
        uploaded: Path,
        progress_callback=None,
    ):
        total_started = (
            time.perf_counter()
        )

        identifier = (
            uuid.uuid4().hex
        )

        workspace = (
            RUNTIME
            / "query-work"
            / identifier
        )

        preprocess_started = (
            time.perf_counter()
        )

        prepared = (
            self.preprocessor.prepare(
                uploaded,
                workspace,
                progress_callback=(
                    progress_callback
                ),
            )
        )

        preprocess_seconds = (
            time.perf_counter()
            - preprocess_started
        )

        if progress_callback is not None:
            progress_callback(
                "gallery_index",
                "Loading gallery search index",
                (
                    "Loading the cached proxy gallery index "
                    "or building it once if required."
                ),
                44,
            )

        index_started = (
            time.perf_counter()
        )

        index = (
            self.gallery.index(
                self.encoder
            )
        )

        gallery_index_seconds = (
            time.perf_counter()
            - index_started
        )

        if progress_callback is not None:
            progress_callback(
                "gallery_index_ready",
                "Gallery index ready",
                (
                    f"Proxy gallery contains "
                    f"{len(index.records)} identity record(s)."
                ),
                45,
            )

        pipeline = (
            AdaptiveReIDPipeline(
                controller=(
                    self.controller
                ),
                image_extractor=(
                    self.image_extractor
                ),
                encoder=(
                    self.encoder
                ),
                gallery=index,
            )
        )

        adaptive_started = (
            time.perf_counter()
        )

        result = (
            pipeline.run(
                prepared.crop_path,
                prepared.evidence,
                progress_callback=(
                    progress_callback
                ),
            )
        )

        adaptive_seconds = (
            time.perf_counter()
            - adaptive_started
        )

        learned_result = None
        learned_seconds = 0.0

        if (
            result.final_action
            == "invoke_vllm"
        ):
            if progress_callback is not None:
                progress_callback(
                    "osnet_embedding",
                    "Running learned OSNet embedding",
                    (
                        "The proxy cascade remained ambiguous. "
                        "Computing one learned 512-D query "
                        "embedding."
                    ),
                    89,
                )

            learned_started = (
                time.perf_counter()
            )

            learned_result = (
                self.gallery
                .learned_rerank(
                    prepared.crop_path,
                    result.candidates,
                )
            )

            learned_seconds = (
                time.perf_counter()
                - learned_started
            )

            if progress_callback is not None:
                progress_callback(
                    "learned_matching",
                    "Comparing learned identity embeddings",
                    (
                        f"OSNet reranking complete; "
                        f"top cosine="
                        f"{learned_result.top1:.4f}, "
                        f"margin="
                        f"{learned_result.margin:.4f}."
                    ),
                    95,
                )

        else:
            if progress_callback is not None:
                progress_callback(
                    "osnet_skipped",
                    "OSNet skipped",
                    (
                        "The adaptive proxy cascade returned "
                        "without requiring learned reranking."
                    ),
                    94,
                )

        if progress_callback is not None:
            progress_callback(
                "final_ranking",
                "Finalizing result",
                (
                    "Selecting the final ranked identity "
                    "list and collecting timing metrics."
                ),
                98,
            )

        total_seconds = (
            time.perf_counter()
            - total_started
        )

        timings = {
            "preprocess_seconds":
                preprocess_seconds,
            "gallery_index_seconds":
                gallery_index_seconds,
            "adaptive_seconds":
                adaptive_seconds,
            "learned_seconds":
                learned_seconds,
            "total_seconds":
                total_seconds,
        }

        return (
            prepared,
            result,
            learned_result,
            timings,
        )

    def search_multires(
        self,
        uploaded: Path,
        mode: str = "efficient",
        progress_callback=None,
    ):
        """
        Direct trained-OSNet gallery search.

        This is the backend path that the next
        frontend milestone will expose.
        """
        resolution = (
            self.learned_encoder
            .resolution_for_mode(
                mode
            )
        )

        total_started = (
            time.perf_counter()
        )

        identifier = (
            uuid.uuid4().hex
        )

        workspace = (
            RUNTIME
            / "query-work"
            / identifier
        )

        preprocess_started = (
            time.perf_counter()
        )

        prepared = (
            self.preprocessor.prepare(
                uploaded,
                workspace,
                progress_callback=(
                    progress_callback
                ),
            )
        )

        preprocess_seconds = (
            time.perf_counter()
            - preprocess_started
        )

        if progress_callback is not None:
            progress_callback(
                "multires_query",
                (
                    "Efficient query embedding"
                    if mode == "efficient"
                    else
                    "Full-quality query embedding"
                ),
                (
                    f"Running the trained OSNet "
                    f"checkpoint at "
                    f"{resolution[0]}x"
                    f"{resolution[1]} for the "
                    f"query crop."
                ),
                58,
            )

        learned_started = (
            time.perf_counter()
        )

        learned_result = (
            self.gallery
            .learned_search_all(
                prepared.crop_path,
                mode=mode,
            )
        )

        learned_seconds = (
            time.perf_counter()
            - learned_started
        )

        if progress_callback is not None:
            progress_callback(
                "multires_gallery_match",
                "Searching learned gallery",
                (
                    f"Compared one 512-D "
                    f"{resolution[0]}x"
                    f"{resolution[1]} query "
                    f"embedding against all "
                    f"stored 256x128 gallery "
                    f"embeddings. "
                    f"Top cosine="
                    f"{learned_result.top1:.4f}; "
                    f"margin="
                    f"{learned_result.margin:.4f}."
                ),
                94,
            )

            progress_callback(
                "final_ranking",
                "Finalizing learned ranking",
                (
                    "Returning the complete "
                    "OSNet gallery ranking."
                ),
                98,
            )

        total_seconds = (
            time.perf_counter()
            - total_started
        )

        timings = {
            "preprocess_seconds":
                preprocess_seconds,

            "learned_seconds":
                learned_seconds,

            "total_seconds":
                total_seconds,

            "query_height":
                resolution[0],

            "query_width":
                resolution[1],

            "gallery_height":
                256,

            "gallery_width":
                128,

            "mode":
                mode,
        }

        return (
            prepared,
            learned_result,
            timings,
        )

    def start_search(
        self,
        uploaded: Path,
    ) -> str:
        job_id = (
            self.jobs.create()
        )

        worker = (
            threading.Thread(
                target=(
                    self._run_search_job
                ),
                args=(
                    job_id,
                    uploaded,
                ),
                name=(
                    "rac-search-"
                    + job_id[:8]
                ),
                daemon=True,
            )
        )

        worker.start()

        return job_id

    def _run_search_job(
        self,
        job_id: str,
        uploaded: Path,
    ) -> None:
        def report(
            code: str,
            title: str,
            detail: str,
            progress: int,
        ) -> None:
            self.jobs.update(
                job_id,
                code,
                title,
                detail,
                progress,
            )

        acquired = (
            self._search_lock
            .acquire(
                blocking=False
            )
        )

        if not acquired:
            report(
                "queue_wait",
                "Waiting for search worker",
                (
                    "Another RAC-ReID query is running. "
                    "This job is queued to protect shared "
                    "CPU model state."
                ),
                4,
            )

            self._search_lock.acquire()

        try:
            report(
                "job_started",
                "Search worker started",
                (
                    "Beginning the RAC-ReID backend "
                    "execution pipeline."
                ),
                6,
            )

            result = (
                self.search(
                    uploaded,
                    progress_callback=(
                        report
                    ),
                )
            )

            self.jobs.finish(
                job_id,
                result,
            )

        except Exception as exc:
            self.jobs.fail(
                job_id,
                str(
                    exc
                ),
            )

        finally:
            self._search_lock.release()



class ReIDHandler(
    BaseHTTPRequestHandler
):
    state: AppState

    server_version = (
        "RAC-ReID/0.6"
    )

    def do_GET(
        self,
    ) -> None:
        if self.path.startswith(
            "/api/search/"
        ):
            job_id = (
                self.path[
                    len(
                        "/api/search/"
                    ):
                ]
                .split(
                    "?",
                    1,
                )[0]
                .strip()
            )

            snapshot = (
                self.state
                .jobs
                .snapshot(
                    job_id
                )
            )

            if snapshot is None:
                self._json(
                    404,
                    {
                        "error":
                            "Search job not found."
                    },
                )
                return

            if (
                snapshot[
                    "status"
                ]
                == "done"
            ):
                stored = (
                    self.state
                    .jobs
                    .result(
                        job_id
                    )
                )

                if stored is not None:
                    (
                        prepared,
                        result,
                        learned_result,
                        timings,
                    ) = stored

                    snapshot[
                        "result_html"
                    ] = (
                        self._result_html(
                            prepared,
                            result,
                            learned_result,
                            timings,
                            progress_events=(
                                snapshot[
                                    "events"
                                ]
                            ),
                        )
                    )

            self._json(
                200,
                snapshot,
            )

            return

        if self.path == "/health":
            payload = {
                "status": "ok",
                "gallery_count": len(
                    self.state
                    .gallery
                    .entries()
                ),
                "model": (
                    MODEL_PATH.name
                ),
                "learned_model": (
                    self.state
                    .learned_encoder
                    .model_name
                ),
                "learned_model_loaded": (
                    self.state
                    .learned_encoder
                    .loaded
                ),
                "device": "cpu",
            }

            self._json(
                200,
                payload,
            )

            return

        if self.path == "/":
            self._html(
                200,
                self._page(),
            )

            return

        self._html(
            404,
            self._page(
                message="Not found."
            ),
        )

    def do_POST(
        self,
    ) -> None:
        if self.path == "/search/start":
            try:
                form = (
                    self._form()
                )

                upload = (
                    self._save_image(
                        form
                    )
                )

                job_id = (
                    self.state
                    .start_search(
                        upload
                    )
                )

                self._json(
                    202,
                    {
                        "job_id":
                            job_id,
                        "status":
                            "running",
                        "poll_url":
                            (
                                "/api/search/"
                                + job_id
                            ),
                    },
                )

            except (
                ValueError,
                NoPersonDetectedError,
            ) as exc:
                self._json(
                    400,
                    {
                        "error":
                            str(
                                exc
                            )
                    },
                )

            except Exception as exc:
                self._json(
                    500,
                    {
                        "error":
                            (
                                "Unable to start search: "
                                + str(
                                    exc
                                )
                            )
                    },
                )

            return

        try:
            if self.path == "/gallery":
                form = (
                    self._form()
                )

                self._require_admin(
                    form
                )

                label = (
                    form.getfirst(
                        "label",
                        "",
                    )
                )

                upload = (
                    self._save_image(
                        form
                    )
                )

                entry = (
                    self.state
                    .gallery
                    .add(
                        label,
                        upload,
                    )
                )

                self._html(
                    200,
                    self._page(
                        message=(
                            "Added gallery identity "
                            f"{entry['label']}."
                        )
                    ),
                )

                return

            if self.path == "/search":
                form = (
                    self._form()
                )

                upload = (
                    self._save_image(
                        form
                    )
                )

                (
                    prepared,
                    result,
                    learned_result,
                    timings,
                ) = (
                    self.state.search(
                        upload
                    )
                )

                self._html(
                    200,
                    self._page(
                        result_html=(
                            self._result_html(
                                prepared,
                                result,
                                learned_result,
                                timings,
                            )
                        )
                    ),
                )

                return

            self._html(
                404,
                self._page(
                    message="Not found."
                ),
            )

        except PermissionError as exc:
            self._html(
                403,
                self._page(
                    message=str(exc)
                ),
            )

        except (
            ValueError,
            NoPersonDetectedError,
        ) as exc:
            self._html(
                400,
                self._page(
                    message=str(exc)
                ),
            )

        except Exception as exc:
            self._html(
                500,
                self._page(
                    message=(
                        "Internal error: "
                        f"{exc}"
                    )
                ),
            )

    def _require_admin(
        self,
        form,
    ) -> None:
        if not ADMIN_TOKEN_FILE.is_file():
            raise PermissionError(
                "Gallery administration "
                "is unavailable."
            )

        expected = (
            ADMIN_TOKEN_FILE
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

        provided = (
            form.getfirst(
                "admin_token",
                "",
            )
            .strip()
        )

        if not expected or not secrets.compare_digest(
            provided,
            expected,
        ):
            raise PermissionError(
                "Invalid gallery admin token."
            )

    def _form(
        self,
    ):
        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Invalid Content-Length."
            ) from exc

        if length <= 0:
            raise ValueError(
                "Empty request."
            )

        if length > MAX_UPLOAD:
            raise ValueError(
                "Upload exceeds 12 MB limit."
            )

        content_type = (
            self.headers.get(
                "Content-Type",
                "",
            )
        )

        if not content_type.startswith(
            "multipart/form-data"
        ):
            raise ValueError(
                "Expected multipart upload."
            )

        environment = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": (
                content_type
            ),
            "CONTENT_LENGTH": str(
                length
            ),
        }

        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environment,
            keep_blank_values=True,
        )

    def _save_image(
        self,
        form,
    ) -> Path:
        if "image" not in form:
            raise ValueError(
                "Image file is required."
            )

        item = form[
            "image"
        ]

        if isinstance(
            item,
            list,
        ):
            item = item[0]

        filename = getattr(
            item,
            "filename",
            None,
        )

        if not filename:
            raise ValueError(
                "Image filename is missing."
            )

        suffix = (
            Path(filename)
            .suffix
            .lower()
        )

        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(
                "Use JPG, JPEG, PNG, or WEBP."
            )

        data = (
            item.file.read(
                MAX_UPLOAD + 1
            )
        )

        if len(data) > MAX_UPLOAD:
            raise ValueError(
                "Upload exceeds 12 MB limit."
            )

        uploads = (
            RUNTIME
            / "uploads"
        )

        uploads.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            uploads
            / (
                uuid.uuid4().hex
                + suffix
            )
        )

        path.write_bytes(
            data
        )

        try:
            with Image.open(
                path
            ) as image:
                image.verify()
        except Exception as exc:
            path.unlink(
                missing_ok=True
            )

            raise ValueError(
                "Uploaded file is not a valid image."
            ) from exc

        return path

    def _page(
        self,
        message: str = "",
        result_html: str = "",
    ) -> str:
        entries = (
            self.state
            .gallery
            .entries()
        )

        gallery_items = (
            "".join(
                "<li>"
                + "<span class=\"gallery-avatar\">"
                + html.escape(
                    item["label"][:1].upper()
                )
                + "</span>"
                + "<div><strong>"
                + html.escape(
                    item["label"]
                )
                + "</strong>"
                + "<small>Detector confidence "
                + f"{item['confidence']:.3f}"
                + "</small></div>"
                + "</li>"
                for item in entries
            )
            or """
    <li class="empty-gallery">
      No gallery identities yet.
    </li>
    """
        )

        message_html = ""

        if message:
            message_html = (
                '<div class="notice">'
                + html.escape(
                    message
                )
                + "</div>"
            )

        baseline_html = (
            "<p>Verified baseline unavailable.</p>"
        )

        baseline_path = (
            ROOT
            / "results"
            / "market1501_osnet_crossdomain.json"
        )

        if baseline_path.is_file():
            try:
                baseline = json.loads(
                    baseline_path.read_text(
                        encoding="utf-8"
                    )
                )

                metrics = (
                    baseline[
                        "metrics_percent"
                    ]
                )

                wall = float(
                    baseline[
                        "runtime"
                    ][
                        "evaluation_wall_seconds"
                    ]
                )

                baseline_html = f"""
    <div class="benchmark-metrics">
      <div>
        <span>mAP</span>
        <strong>{float(metrics["map"]):.3f}%</strong>
      </div>
      <div>
        <span>Rank-1</span>
        <strong>{float(metrics["rank1"]):.3f}%</strong>
      </div>
      <div>
        <span>Rank-5</span>
        <strong>{float(metrics["rank5"]):.3f}%</strong>
      </div>
      <div>
        <span>Rank-10</span>
        <strong>{float(metrics["rank10"]):.3f}%</strong>
      </div>
      <div>
        <span>Queries</span>
        <strong>3,368</strong>
      </div>
      <div>
        <span>CPU benchmark</span>
        <strong>{wall / 60.0:.1f} min</strong>
      </div>
    </div>
    """

            except Exception:
                baseline_html = (
                    "<p>Baseline could not be rendered.</p>"
                )

        adaptive_html = """
    <div class="pending-benchmark">
      <span class="pending-dot"></span>
      <div>
        <strong>Adaptive benchmark pending</strong>
        <small>
          Step 28C will populate the baseline-vs-adaptive
          accuracy and efficiency comparison.
        </small>
      </div>
    </div>
    """

        adaptive_path = (
            ROOT
            / "results"
            / "market1501_rac_adaptive.json"
        )

        if adaptive_path.is_file():
            try:
                adaptive = json.loads(
                    adaptive_path.read_text(
                        encoding="utf-8"
                    )
                )

                metrics = (
                    adaptive[
                        "metrics_percent"
                    ]
                )

                efficiency = (
                    adaptive[
                        "efficiency"
                    ]
                )

                adaptive_html = f"""
    <div class="adaptive-result">
      <div>
        <span>Adaptive mAP</span>
        <strong>{float(metrics["map"]):.3f}%</strong>
      </div>
      <div>
        <span>Adaptive Rank-1</span>
        <strong>{float(metrics["rank1"]):.3f}%</strong>
      </div>
      <div>
        <span>OSNet invocation</span>
        <strong>
          {float(efficiency["osnet_invocation_rate"]):.1%}
        </strong>
      </div>
      <div>
        <span>Proxy patch saving</span>
        <strong>
          {float(efficiency["proxy_patch_saving"]):.1%}
        </strong>
      </div>
    </div>
    """
            except Exception:
                pass

        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <meta
      name="viewport"
      content="width=device-width,initial-scale=1"
    >
    <title>RAC-ReID Research Console</title>

    <style>
    :root {{
      --bg: #070b14;
      --bg2: #0c1220;
      --panel: rgba(17, 25, 40, .88);
      --panel2: rgba(23, 33, 51, .88);
      --text: #f4f7fb;
      --muted: #93a4bd;
      --line: rgba(148, 163, 184, .15);
      --blue: #5794ff;
      --blue2: #8bb7ff;
      --green: #43d79e;
      --amber: #ffc766;
      --red: #ff6d7a;
      --purple: #ae8bff;
      --shadow: 0 24px 80px rgba(0,0,0,.25);
    }}

    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(
          circle at 15% -10%,
          rgba(69, 121, 255, .20),
          transparent 34%
        ),
        radial-gradient(
          circle at 95% 0%,
          rgba(142, 92, 246, .12),
          transparent 32%
        ),
        linear-gradient(
          180deg,
          #070b14,
          #0b101c 55%,
          #070b14
        );
      font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
      line-height: 1.5;
    }}

    a {{
      color: inherit;
    }}

    .shell {{
      width: min(1180px, calc(100% - 28px));
      margin: 0 auto;
      padding: 22px 0 70px;
    }}

    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 10px 2px 22px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 11px;
    }}

    .logo {{
      display: grid;
      place-items: center;
      width: 38px;
      height: 38px;
      border-radius: 12px;
      background:
        linear-gradient(
          135deg,
          #3377ff,
          #896bff
        );
      font-weight: 900;
      box-shadow:
        0 8px 30px rgba(69,121,255,.30);
    }}

    .brand strong {{
      display: block;
    }}

    .brand span {{
      display: block;
      color: var(--muted);
      font-size: .76rem;
    }}

    .node-status {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 11px;
      border: 1px solid rgba(67, 215, 158, .22);
      border-radius: 999px;
      background: rgba(67,215,158,.07);
      color: #8ff0c7;
      font-size: .78rem;
    }}

    .node-dot {{
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--green);
      box-shadow:
        0 0 14px var(--green);
    }}

    .hero {{
      display: grid;
      grid-template-columns:
        minmax(0, 1.4fr)
        minmax(280px, .6fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }}

    .hero-main,
    .hero-side,
    .card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
      backdrop-filter: blur(16px);
      box-shadow: var(--shadow);
    }}

    .hero-main {{
      padding: 32px;
    }}

    .hero-side {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 24px;
      background:
        linear-gradient(
          150deg,
          rgba(87,148,255,.12),
          rgba(174,139,255,.06)
        ),
        var(--panel);
    }}

    .eyebrow {{
      color: #8cb7ff;
      text-transform: uppercase;
      letter-spacing: .12em;
      font-size: .71rem;
      font-weight: 800;
    }}

    h1 {{
      margin: 8px 0 12px;
      font-size: clamp(2.4rem, 7vw, 4.6rem);
      line-height: .98;
      letter-spacing: -.055em;
    }}

    h2 {{
      margin: 0 0 10px;
      letter-spacing: -.025em;
    }}

    h3 {{
      margin-top: 28px;
    }}

    .hero p {{
      max-width: 710px;
      margin-bottom: 0;
      color: var(--muted);
      font-size: 1.03rem;
    }}

    .stack-badge {{
      display: inline-block;
      margin: 4px 4px 4px 0;
      padding: 5px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: #cdd8e7;
      font-size: .72rem;
    }}

    .card {{
      padding: 24px;
      margin: 18px 0;
    }}

    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 16px;
    }}

    .card-header p {{
      margin: 5px 0 0;
      color: var(--muted);
    }}

    .pipeline-map {{
      display: grid;
      grid-template-columns:
        repeat(7, minmax(105px, 1fr));
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}

    .pipeline-step {{
      min-width: 105px;
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(255,255,255,.025);
    }}

    .pipeline-step span {{
      display: block;
      color: var(--blue2);
      font-size: .67rem;
      font-weight: 800;
      letter-spacing: .08em;
    }}

    .pipeline-step strong {{
      display: block;
      margin: 5px 0 3px;
      font-size: .84rem;
    }}

    .pipeline-step small {{
      display: block;
      color: var(--muted);
      font-size: .69rem;
    }}

    .science-note {{
      margin-top: 14px;
      padding: 13px 14px;
      border: 1px solid rgba(87,148,255,.22);
      border-radius: 12px;
      background: rgba(87,148,255,.07);
      color: #c9d9f4;
      font-size: .86rem;
    }}

    .benchmark-metrics,
    .adaptive-result,
    .metric-grid,
    .saving-box {{
      display: grid;
      grid-template-columns:
        repeat(auto-fit, minmax(125px, 1fr));
      gap: 9px;
    }}

    .benchmark-metrics > div,
    .adaptive-result > div,
    .metric-grid > div,
    .saving-box > div {{
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: rgba(255,255,255,.025);
    }}

    .benchmark-metrics span,
    .adaptive-result span,
    .metric-grid span,
    .saving-box span {{
      display: block;
      color: var(--muted);
      font-size: .72rem;
    }}

    .benchmark-metrics strong,
    .adaptive-result strong,
    .metric-grid strong,
    .saving-box strong {{
      display: block;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }}

    .pending-benchmark {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
      padding: 12px;
      border-radius: 11px;
      background: rgba(255,199,102,.06);
      border: 1px solid rgba(255,199,102,.16);
    }}

    .pending-benchmark small {{
      display: block;
      color: var(--muted);
    }}

    .pending-dot {{
      flex: 0 0 auto;
      width: 8px;
      height: 8px;
      margin-top: 7px;
      border-radius: 50%;
      background: var(--amber);
    }}

    .workspace {{
      display: grid;
      grid-template-columns:
        minmax(0, 1.25fr)
        minmax(300px, .75fr);
      gap: 18px;
    }}

    input,
    button {{
      font: inherit;
    }}

    input[type="text"],
    input[type="password"] {{
      width: 100%;
      padding: 11px 12px;
      margin: 6px 0 13px;
      color: var(--text);
      background: rgba(255,255,255,.035);
      border: 1px solid var(--line);
      border-radius: 10px;
      outline: none;
    }}

    input[type="file"] {{
      width: 100%;
      margin: 10px 0 14px;
      color: var(--muted);
    }}

    button {{
      padding: 11px 16px;
      border: 0;
      border-radius: 10px;
      color: white;
      background:
        linear-gradient(
          135deg,
          #3e7eff,
          #7667f7
        );
      font-weight: 750;
      cursor: pointer;
      box-shadow:
        0 8px 28px rgba(62,126,255,.20);
    }}

    button:hover {{
      filter: brightness(1.06);
    }}

    button:disabled {{
      opacity: .55;
      cursor: wait;
    }}

    .search-preview {{
      display: none;
      margin: 12px 0;
      border-radius: 13px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #05080e;
    }}

    .search-preview.visible {{
      display: block;
    }}

    .search-preview img {{
      display: block;
      width: 100%;
      max-height: 290px;
      object-fit: contain;
    }}

    .live-console {{
      display: none;
      margin-top: 16px;
      border: 1px solid rgba(87,148,255,.28);
      border-radius: 16px;
      overflow: hidden;
      background: #080d17;
    }}

    .live-console.visible {{
      display: block;
    }}

    .console-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 15px;
      border-bottom: 1px solid var(--line);
    }}

    .run-state {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: #9bc0ff;
      font-size: .78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .08em;
    }}

    .run-spinner {{
      width: 9px;
      height: 9px;
      border: 2px solid rgba(87,148,255,.28);
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin .75s linear infinite;
    }}

    .run-state.done {{
      color: var(--green);
    }}

    .run-state.done .run-spinner {{
      border: 0;
      background: var(--green);
      animation: none;
    }}

    .run-state.error {{
      color: var(--red);
    }}

    .run-state.error .run-spinner {{
      border: 0;
      background: var(--red);
      animation: none;
    }}

    .console-elapsed {{
      color: var(--muted);
      font-size: .78rem;
    }}

    .progress-wrap {{
      padding: 15px;
    }}

    .progress-meta {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
    }}

    .progress-meta strong {{
      font-size: .9rem;
    }}

    .progress-meta span {{
      color: var(--blue2);
      font-size: .82rem;
      font-weight: 700;
    }}

    .progress-track {{
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
      overflow: hidden;
    }}

    .progress-bar {{
      height: 100%;
      width: 0;
      border-radius: inherit;
      background:
        linear-gradient(
          90deg,
          #3578ff,
          #7f69ff,
          #43d79e
        );
      transition: width .3s ease;
    }}

    .current-operation {{
      margin-top: 13px;
      padding: 12px;
      border-radius: 11px;
      background: rgba(87,148,255,.06);
    }}

    .current-operation strong {{
      display: block;
    }}

    .current-operation small {{
      display: block;
      margin-top: 3px;
      color: var(--muted);
    }}

    .event-log {{
      max-height: 430px;
      overflow-y: auto;
      border-top: 1px solid var(--line);
    }}

    .event {{
      display: grid;
      grid-template-columns: 24px 1fr auto;
      gap: 9px;
      align-items: start;
      padding: 11px 15px;
      border-bottom: 1px solid rgba(148,163,184,.08);
    }}

    .event:last-child {{
      border-bottom: 0;
    }}

    .event-icon {{
      display: grid;
      place-items: center;
      width: 20px;
      height: 20px;
      margin-top: 1px;
      border-radius: 50%;
      background: rgba(67,215,158,.11);
      color: var(--green);
      font-size: .67rem;
    }}

    .event.running .event-icon {{
      background: rgba(87,148,255,.12);
      color: var(--blue);
    }}

    .event.error .event-icon {{
      background: rgba(255,109,122,.12);
      color: var(--red);
    }}

    .event strong {{
      display: block;
      font-size: .82rem;
    }}

    .event small {{
      display: block;
      color: var(--muted);
      margin-top: 2px;
      font-size: .72rem;
    }}

    .event-time {{
      color: var(--muted);
      font-family: ui-monospace, monospace;
      font-size: .68rem;
    }}

    .notice {{
      margin: 12px 0;
      padding: 12px 14px;
      border: 1px solid rgba(255,199,102,.20);
      border-radius: 11px;
      background: rgba(255,199,102,.07);
    }}

    .small {{
      color: var(--muted);
      font-size: .84rem;
    }}

    .gallery-list {{
      list-style: none;
      margin: 12px 0 0;
      padding: 0;
    }}

    .gallery-list li {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 9px 0;
      border-bottom: 1px solid rgba(148,163,184,.08);
    }}

    .gallery-list li:last-child {{
      border-bottom: 0;
    }}

    .gallery-avatar {{
      display: grid;
      place-items: center;
      width: 34px;
      height: 34px;
      border-radius: 10px;
      background: rgba(87,148,255,.10);
      color: var(--blue2);
      font-weight: 800;
    }}

    .gallery-list small {{
      display: block;
      color: var(--muted);
    }}

    .result-card {{
      border-color: rgba(67,215,158,.20);
    }}

    .result-hero {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      flex-wrap: wrap;
    }}

    .final-match {{
      min-width: 220px;
      padding: 15px;
      border-radius: 12px;
      background: rgba(67,215,158,.08);
      border: 1px solid rgba(67,215,158,.20);
    }}

    .final-match span,
    .final-match small {{
      display: block;
      color: var(--muted);
    }}

    .final-match strong {{
      display: block;
      margin: 4px 0;
      font-size: 1.18rem;
    }}

    .signal {{
      margin: 12px 0;
    }}

    .signal-head {{
      display: flex;
      justify-content: space-between;
    }}

    .signal small {{
      display: block;
      margin-top: 3px;
      color: var(--muted);
    }}

    .meter,
    .budget-bar {{
      height: 8px;
      margin-top: 5px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(255,255,255,.08);
    }}

    .meter span,
    .budget-bar span {{
      display: block;
      height: 100%;
      background:
        linear-gradient(
          90deg,
          var(--blue),
          var(--purple)
        );
    }}

    .thresholds {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: .75rem;
    }}

    .stage {{
      margin: 12px 0;
      padding: 15px;
      border: 1px solid var(--line);
      border-radius: 13px;
      background: rgba(255,255,255,.018);
    }}

    .stage-title,
    .budget-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }}

    .stage-number {{
      display: inline-grid;
      width: 27px;
      height: 27px;
      margin-right: 6px;
      place-items: center;
      border-radius: 50%;
      background: var(--blue);
      color: white;
    }}

    .pill {{
      padding: 4px 8px;
      border-radius: 999px;
      color: var(--blue2);
      background: rgba(87,148,255,.09);
      font-size: .71rem;
      font-weight: 800;
    }}

    .budget-head {{
      margin-top: 12px;
      color: var(--muted);
      font-size: .8rem;
    }}

    .reason {{
      margin-bottom: 0;
      color: var(--muted);
    }}

    .learned {{
      padding: 15px;
      border-radius: 12px;
    }}

    .learned.skipped {{
      border: 1px solid rgba(67,215,158,.20);
      background: rgba(67,215,158,.06);
    }}

    .learned.invoked {{
      border: 1px solid rgba(87,148,255,.20);
      background: rgba(87,148,255,.06);
    }}

    .learned-title {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}

    .status-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--green);
    }}

    .invoked .status-dot {{
      background: var(--blue);
    }}

    .timing-flow {{
      display: flex;
      align-items: stretch;
      flex-wrap: wrap;
      gap: 7px;
    }}

    .timing-flow > div:not(.arrow) {{
      flex: 1 1 135px;
      padding: 11px;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}

    .timing-flow span {{
      display: block;
      color: var(--muted);
      font-size: .71rem;
    }}

    .arrow {{
      display: grid;
      place-items: center;
      color: var(--muted);
    }}

    .total-time {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-top: 9px;
      padding: 13px;
      border-radius: 10px;
      background: rgba(255,255,255,.05);
    }}

    .history-timeline {{
      position: relative;
      margin: 15px 0;
    }}

    .history-event {{
      display: grid;
      grid-template-columns: 28px 1fr;
      gap: 10px;
      padding-bottom: 14px;
    }}

    .history-marker {{
      display: grid;
      place-items: center;
      width: 23px;
      height: 23px;
      border-radius: 50%;
      background: rgba(67,215,158,.11);
      color: var(--green);
      font-size: .7rem;
    }}

    .history-title {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}

    .history-title span {{
      color: var(--muted);
      font-family: ui-monospace, monospace;
      font-size: .71rem;
    }}

    .history-event p {{
      margin: 2px 0 0;
      color: var(--muted);
      font-size: .78rem;
    }}

    .table-scroll {{
      overflow-x: auto;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
    }}

    th,
    td {{
      padding: 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}

    th {{
      color: var(--muted);
      font-size: .73rem;
    }}

    #live-result {{
      margin-top: 18px;
    }}

    footer {{
      padding: 24px 3px;
      color: var(--muted);
      font-size: .76rem;
    }}

    @keyframes spin {{
      to {{
        transform: rotate(360deg);
      }}
    }}

    @media (max-width: 900px) {{
      .hero,
      .workspace {{
        grid-template-columns: 1fr;
      }}

      .pipeline-map {{
        grid-template-columns:
          repeat(7, 125px);
      }}
    }}

    @media (max-width: 580px) {{
      .shell {{
        width: min(100% - 18px, 1180px);
      }}

      .hero-main,
      .hero-side,
      .card {{
        padding: 17px;
        border-radius: 15px;
      }}

      .topbar {{
        align-items: flex-start;
      }}

      .node-status {{
        font-size: .67rem;
      }}

      .arrow {{
        display: none;
      }}
    }}
    </style>

    <script>
    let activeJobId = null;
    let pollTimer = null;

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function previewSearchImage(input) {{
      const box =
        document.getElementById(
          "search-preview"
        );

      const image =
        document.getElementById(
          "search-preview-image"
        );

      if (!box || !image) {{
        return;
      }}

      if (
        !input.files
        || !input.files[0]
      ) {{
        box.classList.remove(
          "visible"
        );
        return;
      }}

      image.src =
        URL.createObjectURL(
          input.files[0]
        );

      box.classList.add(
        "visible"
      );
    }}

    function resetLiveConsole() {{
      const consoleBox =
        document.getElementById(
          "live-console"
        );

      consoleBox.classList.add(
        "visible"
      );

      document.getElementById(
        "run-state"
      ).className =
        "run-state";

      document.getElementById(
        "run-state-text"
      ).textContent =
        "Running";

      document.getElementById(
        "progress-bar"
      ).style.width =
        "2%";

      document.getElementById(
        "progress-value"
      ).textContent =
        "2%";

      document.getElementById(
        "current-title"
      ).textContent =
        "Uploading query";

      document.getElementById(
        "current-detail"
      ).textContent =
        "Starting RAC-ReID search job...";

      document.getElementById(
        "event-log"
      ).innerHTML =
        "";

      document.getElementById(
        "live-result"
      ).innerHTML =
        "";
    }}

    function renderEvents(data) {{
      const events =
        data.events || [];

      const log =
        document.getElementById(
          "event-log"
        );

      log.innerHTML =
        events.map(
          (event, index) => {{
            const isLast =
              index ===
              events.length - 1;

            let state =
              "done";

            let icon =
              "✓";

            if (
              data.status === "running"
              && isLast
            ) {{
              state =
                "running";

              icon =
                "●";
            }}

            if (
              data.status === "error"
              && isLast
            ) {{
              state =
                "error";

              icon =
                "!";
            }}

            return `
    <div class="event ${{state}}">
      <div class="event-icon">
        ${{icon}}
      </div>

      <div>
        <strong>
          ${{escapeHtml(event.title)}}
        </strong>

        <small>
          ${{escapeHtml(event.detail)}}
        </small>
      </div>

      <div class="event-time">
        ${{Number(event.elapsed_seconds).toFixed(3)}}s
      </div>
    </div>`;
          }}
        ).join("");

      log.scrollTop =
        log.scrollHeight;
    }}

    function renderProgress(data) {{
      const progress =
        Math.max(
          0,
          Math.min(
            100,
            Number(
              data.progress || 0
            )
          )
        );

      document.getElementById(
        "progress-bar"
      ).style.width =
        `${{progress}}%`;

      document.getElementById(
        "progress-value"
      ).textContent =
        `${{progress}}%`;

      document.getElementById(
        "console-elapsed"
      ).textContent =
        `${{
          Number(
            data.elapsed_seconds || 0
          ).toFixed(2)
        }} s`;

      if (data.current) {{
        document.getElementById(
          "current-title"
        ).textContent =
          data.current.title;

        document.getElementById(
          "current-detail"
        ).textContent =
          data.current.detail;
      }}

      renderEvents(
        data
      );
    }}

    function finishButton() {{
      const button =
        document.getElementById(
          "search-button"
        );

      if (button) {{
        button.disabled = false;
        button.textContent =
          "Run RAC-ReID search";
      }}
    }}

    async function pollSearch() {{
      if (!activeJobId) {{
        return;
      }}

      try {{
        const response =
          await fetch(
            "/api/search/"
            + encodeURIComponent(
              activeJobId
            ),
            {{
              cache: "no-store"
            }}
          );

        const data =
          await response.json();

        if (!response.ok) {{
          throw new Error(
            data.error
            || "Unable to read search progress."
          );
        }}

        renderProgress(
          data
        );

        if (
          data.status === "done"
        ) {{
          const state =
            document.getElementById(
              "run-state"
            );

          state.className =
            "run-state done";

          document.getElementById(
            "run-state-text"
          ).textContent =
            "Completed";

          if (data.result_html) {{
            document.getElementById(
              "live-result"
            ).innerHTML =
              data.result_html;
          }}

          finishButton();

          document.getElementById(
            "live-result"
          ).scrollIntoView({{
            behavior: "smooth",
            block: "start"
          }});

          return;
        }}

        if (
          data.status === "error"
        ) {{
          const state =
            document.getElementById(
              "run-state"
            );

          state.className =
            "run-state error";

          document.getElementById(
            "run-state-text"
          ).textContent =
            "Failed";

          finishButton();

          return;
        }}

        pollTimer =
          setTimeout(
            pollSearch,
            350
          );

      }} catch (error) {{
        const state =
          document.getElementById(
            "run-state"
          );

        state.className =
          "run-state error";

        document.getElementById(
          "run-state-text"
        ).textContent =
          "Connection error";

        document.getElementById(
          "current-title"
        ).textContent =
          "Unable to read progress";

        document.getElementById(
          "current-detail"
        ).textContent =
          String(error);

        finishButton();
      }}
    }}

    async function startLiveSearch(form) {{
      if (pollTimer) {{
        clearTimeout(
          pollTimer
        );
      }}

      resetLiveConsole();

      const button =
        document.getElementById(
          "search-button"
        );

      button.disabled = true;
      button.textContent =
        "Search running...";

      try {{
        const response =
          await fetch(
            "/search/start",
            {{
              method: "POST",
              body: new FormData(
                form
              )
            }}
          );

        const data =
          await response.json();

        if (!response.ok) {{
          throw new Error(
            data.error
            || "Unable to start search."
          );
        }}

        activeJobId =
          data.job_id;

        document.getElementById(
          "job-id"
        ).textContent =
          activeJobId.slice(
            0,
            12
          );

        pollSearch();

      }} catch (error) {{
        const state =
          document.getElementById(
            "run-state"
          );

        state.className =
          "run-state error";

        document.getElementById(
          "run-state-text"
        ).textContent =
          "Failed";

        document.getElementById(
          "current-title"
        ).textContent =
          "Search could not start";

        document.getElementById(
          "current-detail"
        ).textContent =
          String(error);

        finishButton();
      }}

      return false;
    }}
    </script>

    </head>

    <body>

    <div class="shell">

      <header class="topbar">
        <div class="brand">
          <div class="logo">R</div>
          <div>
            <strong>RAC-ReID</strong>
            <span>Research execution console</span>
          </div>
        </div>

        <div class="node-status">
          <span class="node-dot"></span>
          CPU RESEARCH NODE ONLINE
        </div>
      </header>

      <section class="hero">

        <div class="hero-main">
          <div class="eyebrow">
            Resource-Adaptive Person Re-Identification
          </div>

          <h1>
            See every decision.
          </h1>

          <p>
            Upload a person image and watch RAC-ReID locate,
            analyze, prune, search, refine, rerank and decide
            in real time.
          </p>
        </div>

        <div class="hero-side">
          <div>
            <div class="eyebrow">
              Runtime stack
            </div>

            <div style="margin-top:12px">
              <span class="stack-badge">YOLO26n</span>
              <span class="stack-badge">Adaptive proxy</span>
              <span class="stack-badge">OSNet x1.0</span>
              <span class="stack-badge">CPU-only</span>
            </div>
          </div>

          <p class="small">
            Semantic VLLM is not connected yet.
            The current 30/55/80% resource values are
            visual-patch proxy budgets.
          </p>
        </div>

      </section>

      {message_html}

      <section class="card">

        <div class="card-header">
          <div>
            <div class="eyebrow">
              Backend architecture
            </div>
            <h2>What RAC-ReID actually does</h2>
          </div>
        </div>

        <div class="pipeline-map">

          <div class="pipeline-step">
            <span>01</span>
            <strong>Validate</strong>
            <small>Check uploaded image</small>
          </div>

          <div class="pipeline-step">
            <span>02</span>
            <strong>Locate</strong>
            <small>YOLO person detection</small>
          </div>

          <div class="pipeline-step">
            <span>03</span>
            <strong>Analyze</strong>
            <small>Quality + difficulty signals</small>
          </div>

          <div class="pipeline-step">
            <span>04</span>
            <strong>Prune</strong>
            <small>Identity-aware patch selection</small>
          </div>

          <div class="pipeline-step">
            <span>05</span>
            <strong>Search</strong>
            <small>Gallery candidate retrieval</small>
          </div>

          <div class="pipeline-step">
            <span>06</span>
            <strong>Refine</strong>
            <small>Restore patches if uncertain</small>
          </div>

          <div class="pipeline-step">
            <span>07</span>
            <strong>Rerank</strong>
            <small>Conditional OSNet</small>
          </div>

        </div>

        <div class="science-note">
          <strong>Research terminology:</strong>
          pruning currently refers to handcrafted
          visual-patch proxy selection.
          It is deliberately not presented as learned
          transformer-token pruning.
        </div>

      </section>

      <section class="card">

        <div class="card-header">
          <div>
            <div class="eyebrow">
              Verified baseline
            </div>
            <h2>Market-1501 full OSNet</h2>
            <p>
              3,368 standard queries against 19,732 gallery images.
            </p>
          </div>
        </div>

        {baseline_html}

        {adaptive_html}

      </section>

      <div class="workspace">

        <main>

          <section class="card">

            <div class="card-header">
              <div>
                <div class="eyebrow">
                  Live query
                </div>
                <h2>Run RAC-ReID</h2>
                <p>
                  The dashboard below reflects events emitted by
                  the backend, not a simulated animation.
                </p>
              </div>
            </div>

            <form
              method="post"
              action="/search"
              enctype="multipart/form-data"
              onsubmit="startLiveSearch(this); return false;"
            >
              <input
                type="file"
                name="image"
                accept="image/*"
                onchange="previewSearchImage(this)"
                required
              >

              <div
                id="search-preview"
                class="search-preview"
              >
                <img
                  id="search-preview-image"
                  alt="Selected query image preview"
                >
              </div>

              <button
                id="search-button"
                type="submit"
              >
                Run RAC-ReID search
              </button>
            </form>

            <div
              id="live-console"
              class="live-console"
              aria-live="polite"
            >

              <div class="console-head">

                <div
                  id="run-state"
                  class="run-state"
                >
                  <span class="run-spinner"></span>
                  <span id="run-state-text">
                    Running
                  </span>
                </div>

                <div class="console-elapsed">
                  Job
                  <span id="job-id">—</span>
                  ·
                  <span id="console-elapsed">0.00 s</span>
                </div>

              </div>

              <div class="progress-wrap">

                <div class="progress-meta">
                  <strong id="current-title">
                    Starting
                  </strong>

                  <span id="progress-value">
                    0%
                  </span>
                </div>

                <div class="progress-track">
                  <div
                    id="progress-bar"
                    class="progress-bar"
                  ></div>
                </div>

                <div class="current-operation">
                  <strong>
                    Current backend operation
                  </strong>
                  <small id="current-detail">
                    Waiting for server...
                  </small>
                </div>

              </div>

              <div
                id="event-log"
                class="event-log"
              ></div>

            </div>

          </section>

          <div id="live-result">
            {result_html}
          </div>

        </main>

        <aside>

          <section class="card">

            <div class="eyebrow">
              Search gallery
            </div>

            <h2>
              Identities ({len(entries)})
            </h2>

            <ul class="gallery-list">
              {gallery_items}
            </ul>

            <p class="small">
              A minimum of two identities is required
              for comparative retrieval.
            </p>

          </section>

          <section class="card">

            <div class="eyebrow">
              Admin
            </div>

            <h2>Add identity</h2>

            <p class="small">
              OSNet embeddings are precomputed once at gallery
              insertion time and reused during searches.
            </p>

            <form
              method="post"
              action="/gallery"
              enctype="multipart/form-data"
            >

              <label>Identity label</label>

              <input
                type="text"
                name="label"
                maxlength="80"
                required
              >

              <label>Admin token</label>

              <input
                type="password"
                name="admin_token"
                autocomplete="off"
                required
              >

              <input
                type="file"
                name="image"
                accept="image/*"
                required
              >

              <button type="submit">
                Add identity
              </button>

            </form>

          </section>

        </aside>

      </div>

      <footer>
        RAC-ReID · MSc research prototype ·
        live backend instrumentation ·
        CPU-only research deployment
      </footer>

    </div>

    </body>
    </html>"""

    def _result_html(
        self,
        prepared,
        result,
        learned_result,
        timings,
        progress_events=None,
    ) -> str:
        difficulty_score = (
            self.state
            .controller
            .difficulty_score(
                result.query_signals
            )
        )

        signals = (
            result.query_signals
        )

        signal_items = [
            (
                "Detector confidence",
                signals.detector_confidence,
                "Higher means the detector is more confident.",
            ),
            (
                "Foreground ratio",
                signals.foreground_ratio,
                "Detected-person occupancy inside the source scene.",
            ),
            (
                "Blur",
                signals.blur,
                "Higher means the query is estimated to be blurrier.",
            ),
            (
                "Visual entropy",
                signals.visual_entropy,
                "Image-complexity proxy used by the controller.",
            ),
            (
                "Occlusion proxy",
                signals.occlusion,
                "Heuristic proxy; not ground-truth occlusion.",
            ),
            (
                "Crowding proxy",
                signals.crowding,
                "Heuristic estimate from nearby detections.",
            ),
        ]

        signal_html = ""

        for (
            name,
            value,
            note,
        ) in signal_items:
            value = float(
                value
            )

            signal_html += f"""
    <div class="signal">
      <div class="signal-head">
        <span>{html.escape(name)}</span>
        <strong>{value:.3f}</strong>
      </div>
      <div class="meter">
        <span style="width:{value:.1%}"></span>
      </div>
      <small>{html.escape(note)}</small>
    </div>
    """

        stage_html = ""

        for stage in result.stages:
            if (
                stage.action
                == "invoke_vllm"
            ):
                display_action = (
                    "escalate_to_osnet"
                )
            else:
                display_action = (
                    stage.action
                )

            unused_ratio = (
                1.0
                - (
                    stage.retained_tokens
                    / stage.total_tokens
                )
            )

            stage_html += f"""
    <div class="stage">
      <div class="stage-title">
        <div>
          <span class="stage-number">
            {stage.stage_number}
          </span>
          <strong>
            {html.escape(stage.difficulty.title())} stage
          </strong>
        </div>
        <span class="pill">
          {html.escape(display_action)}
        </span>
      </div>

      <div class="budget-head">
        <span>
          Visual-patch proxy budget:
          <strong>{stage.token_retention:.0%}</strong>
        </span>
        <span>
          {stage.retained_tokens}/{stage.total_tokens} patches
        </span>
      </div>

      <div class="budget-bar">
        <span style="width:{stage.token_retention:.1%}"></span>
      </div>

      <div class="metric-grid">
        <div>
          <span>New patches</span>
          <strong>{stage.new_tokens_processed}</strong>
        </div>
        <div>
          <span>Cumulative</span>
          <strong>{stage.cumulative_tokens_processed}</strong>
        </div>
        <div>
          <span>Top-K</span>
          <strong>{stage.top_k}</strong>
        </div>
        <div>
          <span>Top-1</span>
          <strong>{stage.top1_score:.3f}</strong>
        </div>
        <div>
          <span>Margin</span>
          <strong>{stage.margin:.3f}</strong>
        </div>
        <div>
          <span>Stage time</span>
          <strong>{stage.elapsed_seconds:.4f} s</strong>
        </div>
        <div>
          <span>Unused budget</span>
          <strong>{unused_ratio:.1%}</strong>
        </div>
      </div>

      <p class="reason">
        <strong>Backend decision:</strong>
        {html.escape(stage.reason)}
      </p>
    </div>
    """

        cheap_rows = ""

        for (
            rank,
            candidate,
        ) in enumerate(
            result.candidates,
            start=1,
        ):
            cheap_rows += (
                "<tr>"
                f"<td>{rank}</td>"
                "<td>"
                + html.escape(
                    candidate.label
                )
                + "</td>"
                f"<td>{candidate.score:.4f}</td>"
                "</tr>"
            )

        if learned_result is None:
            learned_html = """
    <div class="learned skipped">
      <div class="learned-title">
        <span class="status-dot"></span>
        <strong>OSNet skipped</strong>
      </div>
      <p>
        The adaptive proxy path returned before learned
        person-ReID reranking was necessary.
      </p>
      <p class="small">
        No OSNet query embedding was computed.
      </p>
    </div>
    """

            if result.candidates:
                final_label = (
                    result.candidates[0]
                    .label
                )

                score_text = (
                    "Proxy similarity "
                    f"{result.candidates[0].score:.4f}"
                )

            else:
                final_label = "No candidate"
                score_text = "No score"

        else:
            learned_rows = ""

            for (
                rank,
                candidate,
            ) in enumerate(
                learned_result.candidates,
                start=1,
            ):
                learned_rows += (
                    "<tr>"
                    f"<td>{rank}</td>"
                    "<td>"
                    + html.escape(
                        candidate.label
                    )
                    + "</td>"
                    f"<td>{candidate.cosine_similarity:.4f}</td>"
                    f"<td>{candidate.cheap_score:.4f}</td>"
                    "</tr>"
                )

            final_label = (
                learned_result
                .candidates[0]
                .label
            )

            score_text = (
                "OSNet cosine "
                f"{learned_result.top1:.4f}"
            )

            learned_html = f"""
    <div class="learned invoked">
      <div class="learned-title">
        <span class="status-dot"></span>
        <strong>OSNet invoked</strong>
      </div>

      <p>
        Proxy retrieval remained ambiguous, so RAC-ReID
        escalated the selected candidates to learned OSNet
        person-ReID matching.
      </p>

      <div class="metric-grid">
        <div>
          <span>Model</span>
          <strong>{html.escape(learned_result.model_name)}</strong>
        </div>
        <div>
          <span>Embedding</span>
          <strong>{learned_result.embedding_dim}-D</strong>
        </div>
        <div>
          <span>Inference</span>
          <strong>{learned_result.inference_seconds:.4f} s</strong>
        </div>
        <div>
          <span>Cold load</span>
          <strong>{learned_result.model_load_seconds:.4f} s</strong>
        </div>
        <div>
          <span>Top-1 cosine</span>
          <strong>{learned_result.top1:.4f}</strong>
        </div>
        <div>
          <span>Margin</span>
          <strong>{learned_result.margin:.4f}</strong>
        </div>
      </div>

      <div class="table-scroll">
        <table>
          <tr>
            <th>Rank</th>
            <th>Identity</th>
            <th>OSNet cosine</th>
            <th>Proxy score</th>
          </tr>
          {learned_rows}
        </table>
      </div>
    </div>
    """

        timeline_html = ""

        if progress_events:
            for event in progress_events:
                timeline_html += f"""
    <div class="history-event">
      <div class="history-marker">✓</div>
      <div>
        <div class="history-title">
          {html.escape(str(event.get("title", "")))}
          <span>
            {float(event.get("elapsed_seconds", 0.0)):.3f}s
          </span>
        </div>
        <p>
          {html.escape(str(event.get("detail", "")))}
        </p>
      </div>
    </div>
    """

        if timeline_html:
            timeline_section = f"""
    <h3>Full backend execution timeline</h3>

    <p class="small">
      This is the actual sequence emitted by the server while
      this query was running.
    </p>

    <div class="history-timeline">
      {timeline_html}
    </div>
    """
        else:
            timeline_section = ""

        preprocess_seconds = float(
            timings[
                "preprocess_seconds"
            ]
        )

        index_seconds = float(
            timings[
                "gallery_index_seconds"
            ]
        )

        adaptive_seconds = float(
            timings[
                "adaptive_seconds"
            ]
        )

        learned_seconds = float(
            timings[
                "learned_seconds"
            ]
        )

        total_seconds = float(
            timings[
                "total_seconds"
            ]
        )

        if learned_result is None:
            learned_time = "skipped"
        else:
            learned_time = (
                f"{learned_seconds:.3f} s"
            )

        if (
            result.final_action
            == "invoke_vllm"
        ):
            final_route = (
                "escalate_to_osnet"
            )
        else:
            final_route = (
                result.final_action
            )

        return f"""
    <div class="card result-card">

      <div class="result-hero">
        <div>
          <div class="eyebrow">
            Actual backend execution trace
          </div>
          <h2>Search complete</h2>
          <p>
            These values were measured during this exact query.
          </p>
        </div>

        <div class="final-match">
          <span>Final top match</span>
          <strong>{html.escape(final_label)}</strong>
          <small>{html.escape(score_text)}</small>
        </div>
      </div>

      <div class="science-note">
        <strong>Scientific boundary:</strong>
        30% / 55% / 80% represent handcrafted
        <strong>visual-patch proxy budgets</strong>,
        not transformer/VLM token counts.
      </div>

      {timeline_section}

      <h3>Detection and localization</h3>

      <div class="metric-grid">
        <div>
          <span>People detected</span>
          <strong>{prepared.person_count}</strong>
        </div>
        <div>
          <span>Detector confidence</span>
          <strong>{prepared.primary_confidence:.3f}</strong>
        </div>
        <div>
          <span>Bounding box</span>
          <strong>
            {prepared.bbox[0]},
            {prepared.bbox[1]},
            {prepared.bbox[2]},
            {prepared.bbox[3]}
          </strong>
        </div>
        <div>
          <span>Initial difficulty</span>
          <strong>{html.escape(prepared.initial_difficulty)}</strong>
        </div>
        <div>
          <span>Difficulty score</span>
          <strong>{difficulty_score:.3f}</strong>
        </div>
        <div>
          <span>Crop margin</span>
          <strong>{prepared.crop_margin:.0%}</strong>
        </div>
      </div>

      <h3>Difficulty signals</h3>
      <div class="signals">
        {signal_html}
      </div>

      <div class="thresholds">
        <span>Easy ≤ 0.35</span>
        <span>Medium ≤ 0.65</span>
        <span>Hard &gt; 0.65</span>
      </div>

      <h3>Pruning, searching and progressive refinement</h3>

      {stage_html}

      <div class="saving-box">
        <div>
          <span>Unique proxy patches</span>
          <strong>{result.total_query_tokens_processed}</strong>
        </div>
        <div>
          <span>Proxy patch saving</span>
          <strong>{result.token_saving_ratio:.1%}</strong>
        </div>
        <div>
          <span>Final route</span>
          <strong>{html.escape(final_route)}</strong>
        </div>
      </div>

      <h3>Conditional learned ReID</h3>
      {learned_html}

      <h3>Backend timing</h3>

      <div class="timing-flow">
        <div>
          <span>YOLO + preprocessing</span>
          <strong>{preprocess_seconds:.3f} s</strong>
        </div>
        <div class="arrow">→</div>
        <div>
          <span>Gallery index</span>
          <strong>{index_seconds:.3f} s</strong>
        </div>
        <div class="arrow">→</div>
        <div>
          <span>Adaptive retrieval</span>
          <strong>{adaptive_seconds:.3f} s</strong>
        </div>
        <div class="arrow">→</div>
        <div>
          <span>OSNet</span>
          <strong>{learned_time}</strong>
        </div>
      </div>

      <div class="total-time">
        <span>Total backend search time</span>
        <strong>{total_seconds:.3f} s</strong>
      </div>

      <h3>Proxy candidate ranking</h3>

      <div class="table-scroll">
        <table>
          <tr>
            <th>Rank</th>
            <th>Identity</th>
            <th>Proxy score</th>
          </tr>
          {cheap_rows}
        </table>
      </div>

    </div>
    """

    def _html(
        self,
        status: int,
        body: str,
    ) -> None:
        encoded = body.encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(encoded)),
        )

        self.end_headers()

        self.wfile.write(
            encoded
        )

    def _json(
        self,
        status: int,
        payload: dict[
            str,
            Any
        ],
    ) -> None:
        encoded = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(encoded)),
        )

        self.end_headers()

        self.wfile.write(
            encoded
        )


def main() -> None:
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8877,
    )

    parser.add_argument(
        "--check",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    state = AppState()

    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "gallery_count": len(
                        state
                        .gallery
                        .entries()
                    ),
                    "model": (
                        MODEL_PATH.name
                    ),
                    "learned_model": (
                        state
                        .learned_encoder
                        .model_name
                    ),
                    "learned_model_loaded": (
                        state
                        .learned_encoder
                        .loaded
                    ),
                    "device": "cpu",
                },
                indent=2,
            )
        )

        return

    ReIDHandler.state = state

    server = HTTPServer(
        (
            args.host,
            args.port,
        ),
        ReIDHandler,
    )

    print(
        f"RAC-ReID listening on "
        f"http://{args.host}:{args.port}"
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
