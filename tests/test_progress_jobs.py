from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


WEB = Path(
    "/root/rac-reid/scripts/web_app.py"
)


def load_web():
    name = (
        "rac_web_progress_tests"
    )

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
            "Unable to import web app."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


class SearchJobStoreTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(
        cls,
    ):
        cls.module = (
            load_web()
        )

    def test_progress_is_monotonic(
        self,
    ):
        store = (
            self.module
            .SearchJobStore()
        )

        job = (
            store.create()
        )

        store.update(
            job,
            "a",
            "A",
            "first",
            40,
        )

        store.update(
            job,
            "b",
            "B",
            "second",
            20,
        )

        snapshot = (
            store.snapshot(
                job
            )
        )

        self.assertIsNotNone(
            snapshot
        )

        self.assertEqual(
            snapshot[
                "progress"
            ],
            40,
        )

    def test_finished_job_reaches_100(
        self,
    ):
        store = (
            self.module
            .SearchJobStore()
        )

        job = (
            store.create()
        )

        store.finish(
            job,
            (
                "result",
            ),
        )

        snapshot = (
            store.snapshot(
                job
            )
        )

        self.assertEqual(
            snapshot[
                "status"
            ],
            "done",
        )

        self.assertEqual(
            snapshot[
                "progress"
            ],
            100,
        )

        self.assertEqual(
            snapshot[
                "events"
            ][-1][
                "code"
            ],
            "complete",
        )

    def test_result_is_not_exposed_in_snapshot(
        self,
    ):
        store = (
            self.module
            .SearchJobStore()
        )

        job = (
            store.create()
        )

        expected = (
            object(),
        )

        store.finish(
            job,
            expected,
        )

        snapshot = (
            store.snapshot(
                job
            )
        )

        self.assertNotIn(
            "result",
            snapshot,
        )

        self.assertIs(
            store.result(
                job
            )[0],
            expected[0],
        )


if __name__ == "__main__":
    unittest.main()
