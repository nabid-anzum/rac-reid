from __future__ import annotations

import argparse
import cgi
import html
import json
import shutil
import uuid
from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
)
from pathlib import Path
from typing import Any

from PIL import Image

from rac_reid import (
    AdaptiveReIDPipeline,
    BudgetedVisualEncoder,
    GalleryIndex,
    ImageSignalExtractor,
    NoPersonDetectedError,
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
    ) -> None:
        self.root = root
        self.preprocessor = (
            preprocessor
        )

        self.originals = (
            root
            / "originals"
        )

        self.crops = (
            root
            / "crops"
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

        shutil.copy2(
            uploaded,
            original_target,
        )

        shutil.copy2(
            prepared.crop_path,
            crop_target,
        )

        entry = {
            "id": identifier,
            "label": label,
            "original": str(
                original_target
            ),
            "crop": str(
                crop_target
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

        self.gallery = (
            GalleryStore(
                GALLERY_ROOT,
                self.preprocessor,
            )
        )

    def search(
        self,
        uploaded: Path,
    ):
        identifier = (
            uuid.uuid4().hex
        )

        workspace = (
            RUNTIME
            / "query-work"
            / identifier
        )

        prepared = (
            self.preprocessor.prepare(
                uploaded,
                workspace,
            )
        )

        index = (
            self.gallery.index(
                self.encoder
            )
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

        result = (
            pipeline.run(
                prepared.crop_path,
                prepared.evidence,
            )
        )

        return (
            prepared,
            result,
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

                prepared, result = (
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
                + " - detector confidence "
                + f"{item['confidence']:.3f}"
                + "</li>"
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

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAC-ReID</title>
<style>
body {{
  font-family: system-ui, sans-serif;
  max-width: 920px;
  margin: 40px auto;
  padding: 0 20px;
  line-height: 1.5;
  background: #fafafa;
  color: #181818;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 12px;
  padding: 20px;
  margin: 18px 0;
}}
input, button {{
  padding: 10px;
  margin: 5px 0;
}}
button {{
  cursor: pointer;
}}
.notice {{
  background: #f0f0f0;
  border-radius: 8px;
  padding: 12px;
  margin: 12px 0;
}}
table {{
  width: 100%;
  border-collapse: collapse;
}}
th, td {{
  text-align: left;
  border-bottom: 1px solid #ddd;
  padding: 8px;
}}
.small {{
  color: #666;
  font-size: 0.92rem;
}}
</style>
</head>
<body>
<h1>RAC-ReID</h1>
<p>
Resource-Adaptive Cascaded Person Re-Identification research demo.
</p>
<p class="small">
CPU-only detector and adaptive token pipeline.
The semantic VLLM reranking stage is not connected yet.
</p>
{message_html}

<div class="card">
<h2>1. Add gallery identity</h2>
<form method="post" action="/gallery" enctype="multipart/form-data">
<label>Identity label</label><br>
<input name="label" required maxlength="80"><br>
<label>Image</label><br>
<input type="file" name="image" accept="image/*" required><br>
<button type="submit">Add to gallery</button>
</form>
</div>

<div class="card">
<h2>2. Search person</h2>
<form method="post" action="/search" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required><br>
<button type="submit">Run RAC-ReID search</button>
</form>
</div>

<div class="card">
<h2>Gallery ({len(entries)})</h2>
<ul>{gallery_items}</ul>
</div>

{result_html}

</body>
</html>"""

    def _result_html(
        self,
        prepared,
        result,
    ) -> str:
        candidate_rows = ""

        for rank, candidate in enumerate(
            result.candidates,
            start=1,
        ):
            candidate_rows += (
                "<tr>"
                f"<td>{rank}</td>"
                "<td>"
                + html.escape(
                    candidate.label
                )
                + "</td>"
                f"<td>{candidate.score:.3f}</td>"
                "</tr>"
            )

        stage_rows = ""

        for stage in result.stages:
            stage_rows += (
                "<tr>"
                f"<td>{stage.stage_number}</td>"
                f"<td>{html.escape(stage.difficulty)}</td>"
                f"<td>{stage.token_retention:.0%}</td>"
                f"<td>{stage.new_tokens_processed}</td>"
                f"<td>{stage.cumulative_tokens_processed}</td>"
                f"<td>{stage.top1_score:.3f}</td>"
                f"<td>{stage.margin:.3f}</td>"
                f"<td>{html.escape(stage.action)}</td>"
                "</tr>"
            )

        semantic_note = ""

        if (
            result.final_action
            == "invoke_vllm"
        ):
            semantic_note = (
                '<p class="small">'
                "Visual retrieval reached the semantic-routing threshold. "
                "The future VLLM stage would rerank these candidates."
                "</p>"
            )

        return f"""
<div class="card">
<h2>Search result</h2>

<p>
Detected people: {prepared.person_count}<br>
Primary detector confidence: {prepared.primary_confidence:.3f}<br>
Initial difficulty: {html.escape(prepared.initial_difficulty)}<br>
Adaptive crop margin: {prepared.crop_margin:.0%}<br>
Unique tokens processed: {result.total_query_tokens_processed}<br>
Token saving vs full query: {result.token_saving_ratio:.1%}<br>
Final action: {html.escape(result.final_action)}
</p>

{semantic_note}

<h3>Adaptive stages</h3>
<table>
<tr>
<th>Stage</th>
<th>Class</th>
<th>Budget</th>
<th>New tokens</th>
<th>Cumulative</th>
<th>Top-1</th>
<th>Margin</th>
<th>Action</th>
</tr>
{stage_rows}
</table>

<h3>Ranked candidates</h3>
<table>
<tr>
<th>Rank</th>
<th>Identity</th>
<th>Score</th>
</tr>
{candidate_rows}
</table>
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
