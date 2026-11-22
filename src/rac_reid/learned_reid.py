from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LearnedCandidate:
    label: str
    cosine_similarity: float
    cheap_score: float


@dataclass(frozen=True)
class LearnedReIDResult:
    model_name: str
    embedding_dim: int
    inference_seconds: float
    model_load_seconds: float
    candidates: tuple[
        LearnedCandidate,
        ...
    ]

    @property
    def top1(
        self,
    ) -> float:
        if not self.candidates:
            return 0.0

        return (
            self.candidates[
                0
            ].cosine_similarity
        )

    @property
    def margin(
        self,
    ) -> float:
        if len(
            self.candidates
        ) < 2:
            return self.top1

        return (
            self.candidates[
                0
            ].cosine_similarity
            - self.candidates[
                1
            ].cosine_similarity
        )


class OSNetReIDEncoder:
    DEFAULT_MODEL = (
        "osnet_x1_0_msmt17_combineall"
    )

    def __init__(
        self,
        cache_dir: str | Path,
        model_name: str = DEFAULT_MODEL,
        torch_threads: int = 2,
    ) -> None:
        self.cache_dir = Path(
            cache_dir
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model_name = (
            model_name
        )

        self.torch_threads = max(
            1,
            int(
                torch_threads
            ),
        )

        self._model = None

        self._lock = (
            threading.Lock()
        )

        self._last_load_seconds = (
            0.0
        )

        self._configure_environment()

    @property
    def loaded(
        self,
    ) -> bool:
        return self._model is not None

    @property
    def last_load_seconds(
        self,
    ) -> float:
        return (
            self._last_load_seconds
        )

    def _configure_environment(
        self,
    ) -> None:
        hf_home = (
            self.cache_dir
            / "huggingface"
        )

        hf_hub = (
            hf_home
            / "hub"
        )

        hf_home.mkdir(
            parents=True,
            exist_ok=True,
        )

        hf_hub.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.environ[
            "REID_CACHE_DIR"
        ] = str(
            self.cache_dir
        )

        os.environ[
            "HF_HOME"
        ] = str(
            hf_home
        )

        os.environ[
            "HF_HUB_CACHE"
        ] = str(
            hf_hub
        )

        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = ""

    def _ensure_model(
        self,
    ):
        if self._model is not None:
            return (
                self._model,
                0.0,
            )

        with self._lock:
            if self._model is not None:
                return (
                    self._model,
                    0.0,
                )

            import torch
            from reid import (
                ReIDModel,
            )

            torch.set_num_threads(
                self.torch_threads
            )

            started = (
                time.perf_counter()
            )

            model = (
                ReIDModel
                .from_pretrained(
                    self.model_name,
                    device="cpu",
                )
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            self._model = model

            self._last_load_seconds = (
                elapsed
            )

            return (
                model,
                elapsed,
            )

    @staticmethod
    def normalize(
        vector: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if array.size < 1:
            raise ValueError(
                "Embedding is empty."
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise ValueError(
                "Embedding contains "
                "non-finite values."
            )

        norm = float(
            np.linalg.norm(
                array
            )
        )

        if norm <= 0.0:
            raise ValueError(
                "Embedding has zero norm."
            )

        return (
            array / norm
        ).astype(
            np.float32
        )

    @classmethod
    def cosine_similarity(
        cls,
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        a = cls.normalize(
            left
        )

        b = cls.normalize(
            right
        )

        if a.shape != b.shape:
            raise ValueError(
                "Embedding dimensions "
                "do not match."
            )

        return float(
            np.clip(
                np.dot(
                    a,
                    b,
                ),
                -1.0,
                1.0,
            )
        )

    def embed_path_timed(
        self,
        image_path: str | Path,
    ) -> tuple[
        np.ndarray,
        float,
        float,
    ]:
        path = Path(
            image_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: "
                f"{path}"
            )

        model, load_seconds = (
            self._ensure_model()
        )

        started = (
            time.perf_counter()
        )

        raw = (
            model
            .extract_embeddings_from_paths(
                [
                    str(path),
                ]
            )
        )

        inference_seconds = (
            time.perf_counter()
            - started
        )

        if hasattr(
            raw,
            "detach",
        ):
            raw = (
                raw
                .detach()
                .cpu()
                .numpy()
            )

        array = np.asarray(
            raw,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise RuntimeError(
                "OSNet returned an "
                "unexpected embedding shape."
            )

        if array.shape[0] != 1:
            raise RuntimeError(
                "Expected exactly one "
                "OSNet embedding."
            )

        embedding = (
            self.normalize(
                array[
                    0
                ]
            )
        )

        return (
            embedding,
            inference_seconds,
            load_seconds,
        )

    def embed_path(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        (
            embedding,
            _,
            _,
        ) = (
            self.embed_path_timed(
                image_path
            )
        )

        return embedding

    def save_embedding(
        self,
        image_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        embedding = (
            self.embed_path(
                image_path
            )
        )

        np.save(
            output,
            embedding,
            allow_pickle=False,
        )

        return output

    @staticmethod
    def load_embedding(
        path: str | Path,
    ) -> np.ndarray:
        array = np.load(
            Path(
                path
            ),
            allow_pickle=False,
        )

        return (
            OSNetReIDEncoder
            .normalize(
                array
            )
        )

    @classmethod
    def rank_from_embeddings(
        cls,
        query_embedding: np.ndarray,
        gallery_embeddings: Mapping[
            str,
            np.ndarray,
        ],
        cheap_scores: Mapping[
            str,
            float,
        ],
        model_name: str,
        inference_seconds: float = 0.0,
        model_load_seconds: float = 0.0,
    ) -> LearnedReIDResult:
        query = cls.normalize(
            query_embedding
        )

        candidates: list[
            LearnedCandidate
        ] = []

        for (
            label,
            embedding,
        ) in gallery_embeddings.items():
            score = (
                cls.cosine_similarity(
                    query,
                    embedding,
                )
            )

            candidates.append(
                LearnedCandidate(
                    label=label,
                    cosine_similarity=(
                        score
                    ),
                    cheap_score=float(
                        cheap_scores.get(
                            label,
                            0.0,
                        )
                    ),
                )
            )

        candidates.sort(
            key=lambda item: (
                item.cosine_similarity,
                item.cheap_score,
                item.label,
            ),
            reverse=True,
        )

        return LearnedReIDResult(
            model_name=(
                model_name
            ),
            embedding_dim=int(
                query.shape[
                    0
                ]
            ),
            inference_seconds=float(
                inference_seconds
            ),
            model_load_seconds=float(
                model_load_seconds
            ),
            candidates=tuple(
                candidates
            ),
        )

    def rerank_precomputed(
        self,
        query_path: str | Path,
        gallery_embeddings: Mapping[
            str,
            np.ndarray,
        ],
        cheap_scores: Mapping[
            str,
            float,
        ],
    ) -> LearnedReIDResult:
        (
            query_embedding,
            inference_seconds,
            load_seconds,
        ) = (
            self.embed_path_timed(
                query_path
            )
        )

        return (
            self.rank_from_embeddings(
                query_embedding=(
                    query_embedding
                ),
                gallery_embeddings=(
                    gallery_embeddings
                ),
                cheap_scores=(
                    cheap_scores
                ),
                model_name=(
                    self.model_name
                ),
                inference_seconds=(
                    inference_seconds
                ),
                model_load_seconds=(
                    load_seconds
                ),
            )
        )



class MultiResolutionOSNetReIDEncoder(
    OSNetReIDEncoder
):
    """
    OSNet x1.0 wrapper for the jointly trained
    RAC-ReID 160x80 / 256x128 checkpoint.

    Gallery embeddings are generated at 256x128.

    Query embeddings default to 160x80 for the
    efficient path. 256x128 remains available as
    an explicit full-quality mode.
    """

    VARIANT_NAME = (
        "osnet_x1_0_multires_epoch8"
    )

    _VALID_MODES = {
        "efficient": (160, 80),
        "full": (256, 128),
    }

    class _CaptureStop(
        RuntimeError
    ):
        pass

    def __init__(
        self,
        cache_dir: str | Path,
        checkpoint_path: str | Path,
        expected_sha256: str,
        base_model_name: str = (
            OSNetReIDEncoder.DEFAULT_MODEL
        ),
        query_size: tuple[int, int] = (
            160,
            80,
        ),
        gallery_size: tuple[int, int] = (
            256,
            128,
        ),
        torch_threads: int = 4,
    ) -> None:
        super().__init__(
            cache_dir=cache_dir,
            model_name=base_model_name,
            torch_threads=torch_threads,
        )

        self.base_model_name = (
            base_model_name
        )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.expected_sha256 = (
            expected_sha256.lower()
        )

        self.query_size = (
            int(query_size[0]),
            int(query_size[1]),
        )

        self.gallery_size = (
            int(gallery_size[0]),
            int(gallery_size[1]),
        )

        self.model_name = (
            self.VARIANT_NAME
        )

        self._checkpoint_sha256 = None

        self._inference_lock = (
            threading.Lock()
        )

        os.environ[
            "HF_HUB_OFFLINE"
        ] = "1"

        os.environ[
            "TRANSFORMERS_OFFLINE"
        ] = "1"

    @property
    def checkpoint_sha256(
        self,
    ) -> str | None:
        return self._checkpoint_sha256

    @property
    def query_resolution(
        self,
    ) -> tuple[int, int]:
        return self.query_size

    @property
    def gallery_resolution(
        self,
    ) -> tuple[int, int]:
        return self.gallery_size

    @classmethod
    def resolution_for_mode(
        cls,
        mode: str,
    ) -> tuple[int, int]:
        normalized = (
            str(mode)
            .strip()
            .lower()
        )

        if normalized not in (
            cls._VALID_MODES
        ):
            raise ValueError(
                "Search mode must be "
                "'efficient' or 'full'."
            )

        return cls._VALID_MODES[
            normalized
        ]

    def _verify_checkpoint(
        self,
    ) -> str:
        import hashlib

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "Multi-resolution checkpoint "
                f"not found: "
                f"{self.checkpoint_path}"
            )

        digest = hashlib.sha256()

        with self.checkpoint_path.open(
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

        actual = digest.hexdigest()

        if (
            actual
            != self.expected_sha256
        ):
            raise RuntimeError(
                "Multi-resolution checkpoint "
                "SHA-256 mismatch."
            )

        return actual

    def _ensure_model(
        self,
    ):
        if self._model is not None:
            return (
                self._model,
                0.0,
            )

        with self._lock:
            if self._model is not None:
                return (
                    self._model,
                    0.0,
                )

            import torch

            from reid import (
                ReIDModel,
            )

            torch.set_num_threads(
                self.torch_threads
            )

            actual_sha = (
                self._verify_checkpoint()
            )

            started = (
                time.perf_counter()
            )

            model = (
                ReIDModel
                .from_pretrained(
                    self.base_model_name,
                    device="cpu",
                )
            )

            checkpoint = torch.load(
                self.checkpoint_path,
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
                    "Expected epoch-8 "
                    "multi-resolution checkpoint."
                )

            state = checkpoint.get(
                "backbone_state_dict"
            )

            if not isinstance(
                state,
                dict,
            ):
                raise RuntimeError(
                    "Checkpoint does not contain "
                    "backbone_state_dict."
                )

            model._backbone.load_state_dict(
                state,
                strict=True,
            )

            model._backbone.eval()

            elapsed = (
                time.perf_counter()
                - started
            )

            self._model = model

            self._checkpoint_sha256 = (
                actual_sha
            )

            self._last_load_seconds = (
                elapsed
            )

            return (
                model,
                elapsed,
            )

    def _capture_preprocessed(
        self,
        model,
        path: Path,
    ):
        import torch

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

            raise self._CaptureStop()

        handle = (
            model
            ._backbone
            .register_forward_pre_hook(
                hook
            )
        )

        try:
            try:
                model.extract_embeddings_from_paths(
                    [
                        str(path),
                    ]
                )

            except self._CaptureStop:
                pass

        finally:
            handle.remove()

        array = captured.get(
            "array"
        )

        if array is None:
            raise RuntimeError(
                "Unable to capture OSNet "
                "preprocessing tensor."
            )

        return torch.from_numpy(
            np.array(
                array,
                dtype=np.float32,
                copy=True,
            )
        ).contiguous()

    def embed_path_timed_at_size(
        self,
        image_path: str | Path,
        size: tuple[int, int],
    ) -> tuple[
        np.ndarray,
        float,
        float,
    ]:
        path = Path(
            image_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        height = int(
            size[0]
        )

        width = int(
            size[1]
        )

        if (
            height < 1
            or width < 1
        ):
            raise ValueError(
                "Resolution must contain "
                "positive dimensions."
            )

        model, load_seconds = (
            self._ensure_model()
        )

        import torch
        import torch.nn.functional as F

        started = (
            time.perf_counter()
        )

        with self._inference_lock:
            tensor = (
                self._capture_preprocessed(
                    model,
                    path,
                )
            )

            target = (
                height,
                width,
            )

            if (
                tuple(
                    tensor.shape[-2:]
                )
                != target
            ):
                tensor = F.interpolate(
                    tensor,
                    size=target,
                    mode="bilinear",
                    align_corners=False,
                )

            with torch.inference_mode():
                raw = (
                    model
                    ._backbone(
                        tensor
                    )
                )

        inference_seconds = (
            time.perf_counter()
            - started
        )

        if hasattr(
            raw,
            "detach",
        ):
            raw = (
                raw
                .detach()
                .cpu()
                .numpy()
            )

        array = np.asarray(
            raw,
            dtype=np.float32,
        )

        if (
            array.ndim != 2
            or array.shape[0] != 1
        ):
            raise RuntimeError(
                "Multi-resolution OSNet "
                "returned an unexpected "
                "embedding shape."
            )

        embedding = self.normalize(
            array[0]
        )

        return (
            embedding,
            inference_seconds,
            load_seconds,
        )

    def embed_path_timed_mode(
        self,
        image_path: str | Path,
        mode: str = "efficient",
    ) -> tuple[
        np.ndarray,
        float,
        float,
    ]:
        return self.embed_path_timed_at_size(
            image_path,
            self.resolution_for_mode(
                mode
            ),
        )

    def embed_path_timed(
        self,
        image_path: str | Path,
    ) -> tuple[
        np.ndarray,
        float,
        float,
    ]:
        return self.embed_path_timed_at_size(
            image_path,
            self.query_size,
        )

    def save_embedding(
        self,
        image_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Persist a gallery embedding.

        Gallery embeddings deliberately use
        the full 256x128 representation.
        """
        output = Path(
            output_path
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            embedding,
            _,
            _,
        ) = (
            self.embed_path_timed_at_size(
                image_path,
                self.gallery_size,
            )
        )

        np.save(
            output,
            embedding,
            allow_pickle=False,
        )

        return output

    def rerank_precomputed_mode(
        self,
        query_path: str | Path,
        gallery_embeddings: Mapping[
            str,
            np.ndarray,
        ],
        cheap_scores: Mapping[
            str,
            float,
        ],
        mode: str = "efficient",
    ) -> LearnedReIDResult:
        (
            query_embedding,
            inference_seconds,
            load_seconds,
        ) = (
            self.embed_path_timed_mode(
                query_path,
                mode=mode,
            )
        )

        return self.rank_from_embeddings(
            query_embedding=(
                query_embedding
            ),
            gallery_embeddings=(
                gallery_embeddings
            ),
            cheap_scores=(
                cheap_scores
            ),
            model_name=(
                self.model_name
                + ":"
                + str(mode)
            ),
            inference_seconds=(
                inference_seconds
            ),
            model_load_seconds=(
                load_seconds
            ),
        )

    def rerank_precomputed(
        self,
        query_path: str | Path,
        gallery_embeddings: Mapping[
            str,
            np.ndarray,
        ],
        cheap_scores: Mapping[
            str,
            float,
        ],
    ) -> LearnedReIDResult:
        return self.rerank_precomputed_mode(
            query_path=query_path,
            gallery_embeddings=(
                gallery_embeddings
            ),
            cheap_scores=(
                cheap_scores
            ),
            mode="efficient",
        )
