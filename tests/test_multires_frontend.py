from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path


WEB = Path(
    "/root/rac-reid/scripts/web_app.py"
)


def load_web():
    name = "rac_multires_frontend_tests"

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


class MultiResolutionFrontendTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(
        cls,
    ):
        cls.module = load_web()

    def test_default_mode_is_efficient(
        self,
    ):
        signature = inspect.signature(
            self.module
            .AppState
            .start_search
        )

        self.assertEqual(
            signature
            .parameters["mode"]
            .default,
            "efficient",
        )

    def test_primary_worker_uses_multires(
        self,
    ):
        source = inspect.getsource(
            self.module
            .AppState
            ._run_search_job
        )

        self.assertIn(
            "search_multires",
            source,
        )

        self.assertNotIn(
            "self.search(",
            source,
        )

    def test_post_reads_mode(
        self,
    ):
        source = inspect.getsource(
            self.module
            .ReIDHandler
            .do_POST
        )

        self.assertIn(
            '"mode"',
            source,
        )

        self.assertIn(
            "mode=mode",
            source,
        )

    def test_live_form_has_modes(
        self,
    ):
        source = inspect.getsource(
            self.module
            .ReIDHandler
            ._page
        )

        self.assertIn(
            'name="mode"',
            source,
        )

        self.assertIn(
            "Efficient - 160x80",
            source,
        )

        self.assertIn(
            "Full quality - 256x128",
            source,
        )

    def test_async_formdata_preserved(
        self,
    ):
        source = inspect.getsource(
            self.module
            .ReIDHandler
            ._page
        )

        self.assertIn(
            '"/search/start"',
            source,
        )

        self.assertIn(
            "new FormData(",
            source,
        )

    def test_multires_renderer(
        self,
    ):
        source = inspect.getsource(
            self.module
            .ReIDHandler
            ._result_html
        )

        self.assertIn(
            "trained_multires",
            source,
        )

        self.assertIn(
            "Identity ranking",
            source,
        )

        self.assertTrue(
            hasattr(
                self.module
                .ReIDHandler,
                "_legacy_result_html",
            )
        )


if __name__ == "__main__":
    unittest.main()
