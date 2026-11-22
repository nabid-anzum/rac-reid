from __future__ import annotations

import unittest
from pathlib import Path

from rac_reid import (
    DifficultyLevel,
    QuerySignals,
    ResourceController,
)


ROOT = Path(__file__).resolve().parents[1]

CONTROLLER = ResourceController.from_json(
    ROOT / "configs" / "resource_policy.json"
)


class ResourceControllerTests(unittest.TestCase):
    def test_easy_query_gets_small_budget(self) -> None:
        plan = CONTROLLER.initial_plan(
            QuerySignals(
                0.98,
                0.90,
                0.05,
                0.15,
                0.05,
                0.05,
            )
        )

        self.assertEqual(
            plan.difficulty,
            DifficultyLevel.EASY,
        )
        self.assertEqual(plan.top_k, 5)
        self.assertAlmostEqual(plan.token_retention, 0.30)
        self.assertFalse(plan.vllm_eligible)

    def test_hard_query_gets_large_budget(self) -> None:
        plan = CONTROLLER.initial_plan(
            QuerySignals(
                0.55,
                0.35,
                0.80,
                0.85,
                0.90,
                0.75,
            )
        )

        self.assertEqual(
            plan.difficulty,
            DifficultyLevel.HARD,
        )
        self.assertEqual(plan.top_k, 20)
        self.assertAlmostEqual(plan.token_retention, 0.80)
        self.assertTrue(plan.vllm_eligible)

    def test_confident_result_exits_early(self) -> None:
        plan = CONTROLLER.plan_for(
            DifficultyLevel.EASY
        )

        decision = CONTROLLER.route_after_retrieval(
            [0.94, 0.69, 0.60],
            plan,
        )

        self.assertEqual(
            decision.action,
            "early_exit",
        )
        self.assertIsNone(decision.next_plan)

    def test_ambiguous_easy_result_refines(self) -> None:
        plan = CONTROLLER.plan_for(
            DifficultyLevel.EASY
        )

        decision = CONTROLLER.route_after_retrieval(
            [0.82, 0.78, 0.75],
            plan,
        )

        self.assertEqual(decision.action, "refine")
        self.assertIsNotNone(decision.next_plan)
        self.assertEqual(
            decision.next_plan.difficulty,
            DifficultyLevel.MEDIUM,
        )

    def test_ambiguous_medium_result_refines(self) -> None:
        plan = CONTROLLER.plan_for(
            DifficultyLevel.MEDIUM
        )

        decision = CONTROLLER.route_after_retrieval(
            [0.82, 0.79, 0.77],
            plan,
        )

        self.assertEqual(decision.action, "refine")
        self.assertEqual(
            decision.next_plan.difficulty,
            DifficultyLevel.HARD,
        )

    def test_ambiguous_hard_result_invokes_vllm(self) -> None:
        plan = CONTROLLER.plan_for(
            DifficultyLevel.HARD
        )

        decision = CONTROLLER.route_after_retrieval(
            [0.83, 0.81, 0.79],
            plan,
        )

        self.assertEqual(
            decision.action,
            "invoke_vllm",
        )

    def test_invalid_signal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QuerySignals(
                1.2,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
