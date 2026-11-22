from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

from rac_reid.person_detector import (
    PersonQueryPreprocessor,
)


WEB = Path(
    "/root/rac-reid/scripts/web_app.py"
)


def load_web():
    name = "rac_production_path_test"

    spec = (
        importlib.util
        .spec_from_file_location(
            name,
            WEB,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Cannot import web app."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


class ProductionReIDPathTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(
        cls,
    ):
        cls.web = load_web()

    def test_prepare_reid_excludes_proxy_logic(
        self,
    ):
        source = inspect.getsource(
            PersonQueryPreprocessor
            .prepare_reid
        )

        self.assertNotIn(
            "image_extractor",
            source,
        )

        self.assertNotIn(
            "controller.initial_plan",
            source,
        )

        self.assertNotIn(
            "QuerySignals",
            source,
        )

        self.assertIn(
            "crop_margin: float = 0.10",
            source,
        )

    def test_live_search_uses_prepare_reid(
        self,
    ):
        source = inspect.getsource(
            self.web
            .AppState
            .search_multires
        )

        self.assertIn(
            "prepare_reid",
            source,
        )

    def test_gallery_uses_prepare_reid(
        self,
    ):
        source = inspect.getsource(
            self.web
            .GalleryStore
            .add
        )

        self.assertIn(
            "prepare_reid",
            source,
        )

    def test_public_pipeline_is_current(
        self,
    ):
        source = inspect.getsource(
            self.web
            .ReIDHandler
            ._page
        )

        for marker in (
            "<strong>Validate</strong>",
            "<strong>Locate</strong>",
            "<strong>Crop</strong>",
            "<strong>Resize</strong>",
            "<strong>Embed</strong>",
            "<strong>Search</strong>",
            "<strong>Rank</strong>",
        ):
            self.assertIn(
                marker,
                source,
            )

        for obsolete in (
            "Research terminology:",
            "30/55/80%",
            "<strong>Analyze</strong>",
            "<strong>Prune</strong>",
            "<strong>Refine</strong>",
            "<strong>Rerank</strong>",
        ):
            self.assertNotIn(
                obsolete,
                source,
            )

    def test_not_labelled_prototype(
        self,
    ):
        source = inspect.getsource(
            self.web
            .ReIDHandler
            ._page
        )

        self.assertIn(
            "MSc research implementation",
            source,
        )

        self.assertNotIn(
            "MSc research prototype",
            source,
        )


if __name__ == "__main__":
    unittest.main()
