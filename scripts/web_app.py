from __future__ import annotations

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
            OSNetReIDEncoder(
                cache_dir=(
                    REID_CACHE
                ),
                torch_threads=2,
            )
        )

        self.gallery = (
            GalleryStore(
                GALLERY_ROOT,
                self.preprocessor,
                self.learned_encoder,
            )
        )

    def search(
        self,
        uploaded: Path,
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
            )
        )

        preprocess_seconds = (
            time.perf_counter()
            - preprocess_started
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
                + html.escape(
                    item["label"]
                )
                + " <span class=\"small\">"
                + "detector confidence "
                + f"{item['confidence']:.3f}"
                + "</span></li>"
                for item in entries
            )
            or "<li>No gallery identities yet.</li>"
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
    <div class="metric-grid">
      <div>
        <span>mAP</span>
        <strong>
          {float(metrics["map"]):.3f}%
        </strong>
      </div>

      <div>
        <span>Rank-1</span>
        <strong>
          {float(metrics["rank1"]):.3f}%
        </strong>
      </div>

      <div>
        <span>Rank-5</span>
        <strong>
          {float(metrics["rank5"]):.3f}%
        </strong>
      </div>

      <div>
        <span>Rank-10</span>
        <strong>
          {float(metrics["rank10"]):.3f}%
        </strong>
      </div>

      <div>
        <span>Queries</span>
        <strong>3,368</strong>
      </div>

      <div>
        <span>Full evaluation</span>
        <strong>{wall / 60.0:.1f} min</strong>
      </div>
    </div>
    """

            except Exception:
                baseline_html = (
                    "<p>Baseline file could not be rendered.</p>"
                )

        adaptive_html = """
    <div class="pending">
      <strong>Adaptive benchmark:</strong>
      pending Step 28C.
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
      <h3>Adaptive RAC-ReID benchmark</h3>

      <div class="metric-grid">
        <div>
          <span>mAP</span>
          <strong>
            {float(metrics["map"]):.3f}%
          </strong>
        </div>

        <div>
          <span>Rank-1</span>
          <strong>
            {float(metrics["rank1"]):.3f}%
          </strong>
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
    </div>
    """

            except Exception:
                adaptive_html = """
    <div class="pending">
      Adaptive result exists but could not be rendered.
    </div>
    """

        return f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <meta
      name="viewport"
      content="width=device-width,initial-scale=1"
    >
    <title>RAC-ReID Research Demo</title>

    <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #e4e7ec;
      --accent: #2357d8;
      --accent-soft: #eef3ff;
      --green: #16825d;
      --green-soft: #eaf8f2;
      --amber-soft: #fff7df;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
      line-height: 1.5;
    }}

    .container {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 34px 20px 70px;
    }}

    .hero {{
      padding: 20px 0 8px;
    }}

    .eyebrow {{
      color: var(--accent);
      font-size: .75rem;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 7px 0 10px;
      font-size: clamp(2.2rem, 6vw, 3.8rem);
      letter-spacing: -.05em;
    }}

    h2 {{
      margin-top: 0;
    }}

    h3 {{
      margin-top: 28px;
    }}

    .hero p {{
      max-width: 760px;
      color: var(--muted);
      font-size: 1.04rem;
    }}

    .card {{
      margin: 18px 0;
      padding: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow:
        0 1px 2px rgba(16, 24, 40, .04);
    }}

    .two-column {{
      display: grid;
      grid-template-columns:
        repeat(auto-fit, minmax(300px, 1fr));
      gap: 18px;
    }}

    .architecture {{
      display: grid;
      grid-template-columns:
        repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
    }}

    .architecture div {{
      padding: 15px;
      background: #fafbfc;
      border: 1px solid var(--line);
      border-radius: 12px;
    }}

    .architecture span {{
      display: block;
      color: var(--accent);
      font-size: .75rem;
      font-weight: 800;
    }}

    .architecture strong {{
      display: block;
      margin: 4px 0;
    }}

    .architecture small {{
      color: var(--muted);
    }}

    input,
    button {{
      font: inherit;
    }}

    input[type="text"],
    input[type="password"],
    input[type="file"] {{
      width: 100%;
      margin: 6px 0 14px;
    }}

    input[type="text"],
    input[type="password"] {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 9px;
    }}

    button {{
      padding: 11px 16px;
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }}

    button:disabled {{
      opacity: .6;
      cursor: wait;
    }}

    .notice,
    .pending {{
      padding: 12px 14px;
      background: var(--amber-soft);
      border-radius: 10px;
    }}

    .small {{
      color: var(--muted);
      font-size: .9rem;
    }}

    .science-note {{
      padding: 14px;
      margin: 16px 0;
      background: var(--accent-soft);
      border: 1px solid #cbd8ff;
      border-radius: 11px;
    }}

    .metric-grid,
    .saving-box {{
      display: grid;
      grid-template-columns:
        repeat(auto-fit, minmax(130px, 1fr));
      gap: 9px;
    }}

    .metric-grid > div,
    .saving-box > div {{
      padding: 12px;
      background: #fafbfc;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}

    .metric-grid span,
    .saving-box span {{
      display: block;
      color: var(--muted);
      font-size: .77rem;
    }}

    .metric-grid strong,
    .saving-box strong {{
      display: block;
      margin-top: 3px;
      overflow-wrap: anywhere;
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
      background: var(--green-soft);
      border: 1px solid #b8e5d4;
      border-radius: 12px;
    }}

    .final-match span,
    .final-match small {{
      display: block;
      color: var(--muted);
    }}

    .final-match strong {{
      display: block;
      font-size: 1.25rem;
      margin: 4px 0;
    }}

    .signal {{
      margin: 13px 0;
    }}

    .signal-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }}

    .signal small {{
      display: block;
      color: var(--muted);
      margin-top: 4px;
    }}

    .meter,
    .budget-bar {{
      height: 9px;
      margin-top: 6px;
      overflow: hidden;
      background: #e9edf3;
      border-radius: 999px;
    }}

    .meter span,
    .budget-bar span {{
      display: block;
      height: 100%;
      background: var(--accent);
    }}

    .thresholds {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      margin-top: 14px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: .78rem;
    }}

    .stage {{
      margin: 12px 0;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 13px;
    }}

    .stage-title,
    .budget-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}

    .stage-number {{
      display: inline-grid;
      width: 28px;
      height: 28px;
      margin-right: 8px;
      place-items: center;
      border-radius: 50%;
      background: var(--accent);
      color: white;
    }}

    .pill {{
      padding: 5px 9px;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      font-size: .75rem;
      font-weight: 800;
    }}

    .budget-head {{
      margin-top: 14px;
      color: var(--muted);
      font-size: .84rem;
    }}

    .reason {{
      margin-bottom: 0;
      color: var(--muted);
    }}

    .learned {{
      padding: 17px;
      border-radius: 12px;
    }}

    .learned.skipped {{
      background: var(--green-soft);
      border: 1px solid #b8e5d4;
    }}

    .learned.invoked {{
      background: var(--accent-soft);
      border: 1px solid #cbd8ff;
    }}

    .learned-title {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .status-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
    }}

    .invoked .status-dot {{
      background: var(--accent);
    }}

    .timing-flow {{
      display: flex;
      flex-wrap: wrap;
      align-items: stretch;
      gap: 8px;
    }}

    .timing-flow > div:not(.arrow) {{
      flex: 1 1 145px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}

    .timing-flow span {{
      display: block;
      color: var(--muted);
      font-size: .77rem;
    }}

    .arrow {{
      display: grid;
      place-items: center;
      color: var(--muted);
    }}

    .total-time {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 10px;
      padding: 14px;
      background: #17202a;
      color: white;
      border-radius: 10px;
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
      padding: 9px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }}

    th {{
      color: var(--muted);
      font-size: .78rem;
    }}

    .processing {{
      display: none;
      margin-top: 14px;
      padding: 14px;
      background: var(--accent-soft);
      border: 1px solid #cbd8ff;
      border-radius: 11px;
    }}

    .processing.visible {{
      display: block;
    }}

    .processing-steps {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 9px;
    }}

    .processing-steps span {{
      padding: 4px 8px;
      background: white;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: .76rem;
    }}

    .loader {{
      display: inline-block;
      width: 12px;
      height: 12px;
      margin-right: 7px;
      border: 2px solid #cbd8ff;
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .8s linear infinite;
    }}

    @keyframes spin {{
      to {{
        transform: rotate(360deg);
      }}
    }}

    footer {{
      margin-top: 25px;
      color: var(--muted);
      font-size: .8rem;
    }}

    @media (max-width: 640px) {{
      .container {{
        padding: 20px 12px 50px;
      }}

      .card {{
        padding: 18px;
      }}

      .arrow {{
        display: none;
      }}
    }}
    </style>

    <script>
    function showProcessing(form) {{
      const panel =
        document.getElementById(
          "processing-panel"
        );

      if (panel) {{
        panel.classList.add(
          "visible"
        );
      }}

      const button =
        form.querySelector(
          'button[type="submit"]'
        );

      if (button) {{
        button.disabled = true;
        button.textContent =
          "Processing query...";
      }}

      return true;
    }}
    </script>

    </head>

    <body>

    <div class="container">

      <section class="hero">
        <div class="eyebrow">
          MSc research prototype
        </div>

        <h1>RAC-ReID</h1>

        <p>
          Resource-Adaptive Cascaded Person Re-Identification.
          The system starts with a query-specific compute budget
          and restores additional resources only when the current
          retrieval result remains uncertain.
        </p>
      </section>

      {message_html}

      <div class="card">

        <h2>How the backend processes a query</h2>

        <div class="architecture">

          <div>
            <span>01</span>
            <strong>Detect</strong>
            <small>YOLO finds the person.</small>
          </div>

          <div>
            <span>02</span>
            <strong>Measure</strong>
            <small>Six signals estimate difficulty.</small>
          </div>

          <div>
            <span>03</span>
            <strong>Allocate</strong>
            <small>Assign 30%, 55%, or 80% proxy patches.</small>
          </div>

          <div>
            <span>04</span>
            <strong>Decide</strong>
            <small>Exit early or restore additional patches.</small>
          </div>

          <div>
            <span>05</span>
            <strong>Escalate</strong>
            <small>Ambiguous hard cases use OSNet reranking.</small>
          </div>

        </div>

        <div class="science-note">
          <strong>Important research distinction:</strong>
          the current resource budget refers to handcrafted
          visual-patch proxy processing.
          It is not transformer/VLM token pruning yet.
        </div>

      </div>

      <div class="card">

        <div class="eyebrow">
          Verified full-compute baseline
        </div>

        <h2>Market-1501 OSNet baseline</h2>

        <p class="small">
          Standard cross-domain evaluation:
          3,368 queries against 19,732 gallery images.
        </p>

        {baseline_html}

        {adaptive_html}

      </div>

      <div class="two-column">

        <div class="card">

          <h2>1. Add gallery identity</h2>

          <p class="small">
            Admin-only operation.
            OSNet gallery embeddings are computed once when an
            identity is added and reused during future searches.
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
              required
              maxlength="80"
            >

            <label>Admin token</label>

            <input
              type="password"
              name="admin_token"
              autocomplete="off"
              required
            >

            <label>Image</label>

            <input
              type="file"
              name="image"
              accept="image/*"
              required
            >

            <button type="submit">
              Add to gallery
            </button>

          </form>

        </div>

        <div class="card">

          <h2>2. Search person</h2>

          <p class="small">
            The completed result will expose the real backend
            difficulty signals, resource stages, routing reasons,
            timings, patch savings, and OSNet decision.
          </p>

          <form
            method="post"
            action="/search"
            enctype="multipart/form-data"
            onsubmit="return showProcessing(this)"
          >

            <input
              type="file"
              name="image"
              accept="image/*"
              required
            >

            <button type="submit">
              Run RAC-ReID search
            </button>

          </form>

          <div
            id="processing-panel"
            class="processing"
          >
            <div>
              <span class="loader"></span>
              <strong>Backend processing...</strong>
            </div>

            <div class="processing-steps">
              <span>YOLO</span>
              <span>difficulty</span>
              <span>proxy patches</span>
              <span>confidence routing</span>
              <span>conditional OSNet</span>
            </div>

            <p class="small">
              This is the expected pipeline indicator.
              After completion, the result reports the actual
              path and measured values used by the backend.
            </p>
          </div>

        </div>

      </div>

      <div class="card">

        <h2>Gallery ({len(entries)})</h2>

        <ul>
          {gallery_items}
        </ul>

        <p class="small">
          At least two gallery identities are required to perform
          comparative retrieval.
        </p>

      </div>

      {result_html}

      <footer>
        RAC-ReID research prototype · CPU deployment ·
        verified Market-1501 baseline ·
        semantic VLLM stage not yet connected
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
                "Heuristic proxy; this is not ground-truth occlusion.",
            ),
            (
                "Crowding proxy",
                signals.crowding,
                "Heuristic estimate derived from nearby detections.",
            ),
        ]

        signal_html = ""

        for (
            name,
            value,
            note,
        ) in signal_items:
            value = float(value)

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
            if stage.action == "invoke_vllm":
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
          <span>Unused proxy budget</span>
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
        person-ReID reranking was needed.
      </p>

      <p class="small">
        No OSNet query embedding was computed for this request.
      </p>
    </div>
    """

            if result.candidates:
                final_label = (
                    result.candidates[0]
                    .label
                )

                final_score = (
                    result.candidates[0]
                    .score
                )

                score_text = (
                    f"Proxy similarity {final_score:.4f}"
                )
            else:
                final_label = (
                    "No candidate"
                )

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
        The hard adaptive proxy stage remained ambiguous,
        so the candidate set was escalated to the learned
        OSNet person-ReID model.
      </p>

      <div class="metric-grid">
        <div>
          <span>Model</span>
          <strong>
            {html.escape(learned_result.model_name)}
          </strong>
        </div>

        <div>
          <span>Embedding</span>
          <strong>{learned_result.embedding_dim}-D</strong>
        </div>

        <div>
          <span>Query inference</span>
          <strong>
            {learned_result.inference_seconds:.4f} s
          </strong>
        </div>

        <div>
          <span>Cold model load</span>
          <strong>
            {learned_result.model_load_seconds:.4f} s
          </strong>
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

      <p class="small">
        Cosine similarity is a ranking score here,
        not a calibrated probability.
      </p>

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

        if result.final_action == "invoke_vllm":
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
            Every value below comes from the backend execution
            of this uploaded query.
          </p>
        </div>

        <div class="final-match">
          <span>Final top match</span>
          <strong>
            {html.escape(final_label)}
          </strong>
          <small>{html.escape(score_text)}</small>
        </div>
      </div>

      <div class="science-note">
        <strong>Scientific boundary:</strong>
        30% / 55% / 80% are
        <strong>visual-patch proxy budgets</strong>
        from the current handcrafted CPU encoder.
        They are not transformer or VLM token counts.
      </div>

      <h3>1. Detection and adaptive crop</h3>

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
          <strong>
            {html.escape(prepared.initial_difficulty)}
          </strong>
        </div>

        <div>
          <span>Difficulty score</span>
          <strong>{difficulty_score:.3f}</strong>
        </div>

        <div>
          <span>Crop margin</span>
          <strong>{prepared.crop_margin:.0%}</strong>
        </div>

        <div>
          <span>Detection + crop time</span>
          <strong>{preprocess_seconds:.3f} s</strong>
        </div>
      </div>

      <h3>2. Difficulty signals</h3>

      <p class="small">
        These are the six values passed into the resource
        controller for this query.
      </p>

      <div class="signals">
        {signal_html}
      </div>

      <div class="thresholds">
        <span>Easy ≤ 0.35</span>
        <span>Medium ≤ 0.65</span>
        <span>Hard &gt; 0.65</span>
      </div>

      <h3>3. Progressive resource allocation</h3>

      <p class="small">
        Previously processed proxy patches are reused.
        A refinement stage processes only newly restored patches.
      </p>

      {stage_html}

      <div class="saving-box">
        <div>
          <span>Unique proxy patches processed</span>
          <strong>
            {result.total_query_tokens_processed}
          </strong>
        </div>

        <div>
          <span>Proxy patch saving vs full query</span>
          <strong>
            {result.token_saving_ratio:.1%}
          </strong>
        </div>

        <div>
          <span>Final route</span>
          <strong>
            {html.escape(final_route)}
          </strong>
        </div>
      </div>

      <h3>4. Conditional learned ReID</h3>

      {learned_html}

      <h3>5. Backend timing</h3>

      <div class="timing-flow">
        <div>
          <span>YOLO + crop</span>
          <strong>{preprocess_seconds:.3f} s</strong>
        </div>

        <div class="arrow">→</div>

        <div>
          <span>Proxy gallery cache/build</span>
          <strong>{index_seconds:.3f} s</strong>
        </div>

        <div class="arrow">→</div>

        <div>
          <span>Adaptive proxy</span>
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

      <p class="small">
        Gallery OSNet embeddings are precomputed when gallery
        identities are added. The proxy gallery index is cached
        after its first construction.
      </p>

      <h3>6. Proxy ranking before learned reranking</h3>

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
