from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_json(path: Path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def analyze(
    path: Path,
    low_full_ratio: float,
):
    data = np.load(path)

    low = (
        np.asarray(
            data["low_correct"]
        )
        .astype(bool)
    )

    full = (
        np.asarray(
            data["full_correct"]
        )
        .astype(bool)
    )

    agreement = (
        np.asarray(
            data["raw_agreement"]
        )
        .astype(bool)
    )

    both_correct = low & full

    low_only = (
        low
        & ~full
    )

    full_rescue = (
        ~low
        & full
    )

    both_wrong = (
        ~low
        & ~full
    )

    n = len(low)

    low_rank1 = float(
        np.mean(low)
    )

    full_rank1 = float(
        np.mean(full)
    )

    oracle_rank1 = float(
        np.mean(
            low | full
        )
    )

    full_rescue_rate = float(
        np.mean(
            full_rescue
        )
    )

    low_only_rate = float(
        np.mean(
            low_only
        )
    )

    low_wrong_rate = float(
        np.mean(
            ~low
        )
    )

    if np.sum(~low) > 0:
        rescue_given_low_wrong = float(
            np.mean(
                full[
                    ~low
                ]
            )
        )
    else:
        rescue_given_low_wrong = 0.0

    perfect_refine_rate = (
        full_rescue_rate
    )

    perfect_exit_rate = (
        1.0
        - perfect_refine_rate
    )

    perfect_progressive_cost = (
        low_full_ratio
        + perfect_refine_rate
    )

    perfect_progressive_saving = (
        1.0
        - perfect_progressive_cost
    )

    break_even_exit_rate = (
        low_full_ratio
    )

    five_percent_exit_rate = (
        low_full_ratio
        + 0.05
    )

    return {
        "queries":
            int(n),

        "low_rank1":
            low_rank1,

        "full_rank1":
            full_rank1,

        "low_minus_full_rank1":
            low_rank1
            - full_rank1,

        "raw_low_full_identity_agreement":
            float(
                np.mean(
                    agreement
                )
            ),

        "outcome_categories": {
            "both_correct": {
                "count":
                    int(
                        np.sum(
                            both_correct
                        )
                    ),

                "rate":
                    float(
                        np.mean(
                            both_correct
                        )
                    ),
            },

            "low_correct_full_wrong": {
                "count":
                    int(
                        np.sum(
                            low_only
                        )
                    ),

                "rate":
                    low_only_rate,
            },

            "low_wrong_full_correct_rescue": {
                "count":
                    int(
                        np.sum(
                            full_rescue
                        )
                    ),

                "rate":
                    full_rescue_rate,
            },

            "both_wrong": {
                "count":
                    int(
                        np.sum(
                            both_wrong
                        )
                    ),

                "rate":
                    float(
                        np.mean(
                            both_wrong
                        )
                    ),
            },
        },

        "low_wrong_rate":
            low_wrong_rate,

        "full_rescue_given_low_wrong":
            rescue_given_low_wrong,

        "oracle_adaptive_rank1":
            oracle_rank1,

        "oracle_gain_vs_low":
            oracle_rank1
            - low_rank1,

        "oracle_gain_vs_full":
            oracle_rank1
            - full_rank1,

        "compute": {
            "low_full_ratio":
                low_full_ratio,

            "fixed_low_saving":
                1.0
                - low_full_ratio,

            "break_even_required_exit_rate_ignoring_router":
                break_even_exit_rate,

            "five_percent_saving_required_exit_rate_ignoring_router":
                five_percent_exit_rate,

            "oracle_refine_only_actual_rescues_rate":
                perfect_refine_rate,

            "oracle_exit_rate":
                perfect_exit_rate,

            "oracle_progressive_saving_ignoring_router":
                perfect_progressive_saving,
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--router-result",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    router = json.loads(
        args.router_result.read_text(
            encoding="utf-8"
        )
    )

    low_full_ratio = float(
        router[
            "measured_compute"
        ][
            "low_full_ratio"
        ]
    )

    result = {
        "status":
            "completed",

        "experiment":
            "low-resolution refinement headroom audit",

        "purpose":
            (
                "Measure whether full-resolution inference "
                "actually rescues errors made by the 160x80 "
                "query path, before training another router."
            ),

        "production_changed":
            False,

        "standard_market_test_used":
            False,

        "calibration":
            analyze(
                args.cache
                / "calibration_router_features.npz",
                low_full_ratio,
            ),
    }

    validation_path = (
        args.cache
        / "validation_router_features.npz"
    )

    if validation_path.is_file():
        result[
            "heldout_validation"
        ] = analyze(
            validation_path,
            low_full_ratio,
        )
    else:
        result[
            "heldout_validation"
        ] = None

    cal = result[
        "calibration"
    ]

    rescue_rate = (
        cal[
            "outcome_categories"
        ][
            "low_wrong_full_correct_rescue"
        ][
            "rate"
        ]
    )

    low_only_rate = (
        cal[
            "outcome_categories"
        ][
            "low_correct_full_wrong"
        ][
            "rate"
        ]
    )

    if (
        rescue_rate
        > 0.0
        and cal[
            "compute"
        ][
            "oracle_progressive_saving_ignoring_router"
        ]
        >= 0.05
    ):
        recommendation = (
            "A refinement-benefit router remains scientifically "
            "worth testing. Train it to predict whether full "
            "resolution is likely to rescue a low-resolution "
            "failure, rather than merely predicting whether the "
            "low-resolution result is correct."
        )

    elif (
        cal[
            "low_rank1"
        ]
        >= (
            cal[
                "full_rank1"
            ]
            - 0.005
        )
    ):
        recommendation = (
            "The fixed 160x80 model is already a strong efficiency "
            "baseline. Full-resolution refinement has limited "
            "development headroom and should not be forced into "
            "production merely to preserve an adaptive design."
        )

    else:
        recommendation = (
            "Full-resolution refinement has insufficient economic "
            "headroom under the current two-resolution design. "
            "Revisit the adaptive architecture before deployment."
        )

    result[
        "interpretation"
    ] = {
        "calibration_full_rescue_rate":
            rescue_rate,

        "calibration_low_only_advantage_rate":
            low_only_rate,

        "recommendation":
            recommendation,
    }

    write_json(
        args.output,
        result,
    )

    print(
        "=" * 72
    )

    print(
        "LOW-RESOLUTION REFINEMENT HEADROOM"
    )

    print(
        "=" * 72
    )

    print(
        "Calibration queries:",
        cal["queries"],
    )

    print(
        "160x80 Rank-1:",
        f"{cal['low_rank1']:.3%}",
    )

    print(
        "256x128 Rank-1:",
        f"{cal['full_rank1']:.3%}",
    )

    print(
        "160x80 minus full:",
        f"{cal['low_minus_full_rank1']:+.3%}",
    )

    categories = (
        cal[
            "outcome_categories"
        ]
    )

    print()
    print(
        "Both correct:",
        f"{categories['both_correct']['rate']:.3%}",
        f"({categories['both_correct']['count']})",
    )

    print(
        "Low correct / full wrong:",
        f"{categories['low_correct_full_wrong']['rate']:.3%}",
        f"({categories['low_correct_full_wrong']['count']})",
    )

    print(
        "LOW WRONG / FULL RESCUES:",
        f"{categories['low_wrong_full_correct_rescue']['rate']:.3%}",
        f"({categories['low_wrong_full_correct_rescue']['count']})",
    )

    print(
        "Both wrong:",
        f"{categories['both_wrong']['rate']:.3%}",
        f"({categories['both_wrong']['count']})",
    )

    print()
    print(
        "When low is wrong, full rescues:",
        f"{cal['full_rescue_given_low_wrong']:.3%}",
    )

    print(
        "Oracle adaptive Rank-1:",
        f"{cal['oracle_adaptive_rank1']:.3%}",
    )

    print(
        "Oracle gain vs low:",
        f"{cal['oracle_gain_vs_low']:+.3%}",
    )

    print(
        "Oracle gain vs full:",
        f"{cal['oracle_gain_vs_full']:+.3%}",
    )

    compute = cal[
        "compute"
    ]

    print()
    print(
        "Fixed 160x80 CPU saving:",
        f"{compute['fixed_low_saving']:.3%}",
    )

    print(
        "Required exit rate for break-even:",
        f"{compute['break_even_required_exit_rate_ignoring_router']:.3%}",
    )

    print(
        "Required exit rate for 5% saving:",
        f"{compute['five_percent_saving_required_exit_rate_ignoring_router']:.3%}",
    )

    print(
        "Perfect-router refine rate:",
        f"{compute['oracle_refine_only_actual_rescues_rate']:.3%}",
    )

    print(
        "Perfect-router projected saving:",
        f"{compute['oracle_progressive_saving_ignoring_router']:.3%}",
    )

    print()
    print(
        "Recommendation:"
    )

    print(
        result[
            "interpretation"
        ][
            "recommendation"
        ]
    )

    print()
    print(
        "Standard Market test used:",
        False,
    )


if __name__ == "__main__":
    main()
