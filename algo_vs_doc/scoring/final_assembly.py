"""Apply the shared cohort enrolment funnel."""

from __future__ import annotations

from typing import Callable, List, Tuple

import pandas as pd

#: Exclusions defining the outcome-eligible primary target population.
OUTCOME_ELIGIBILITY_STEPS: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
    ("not_yet_discharged", lambda df: df["icu_discharged_at"].isna()),
    ("died_on_icu", lambda df: df["died_on_icu"] == True),  # noqa: E712
    ("palliative_care", lambda df: df["palliative_care"] == True),  # noqa: E712
    ("do_not_readmit_policy", lambda df: df["do_not_readmit"] == True),  # noqa: E712
    (
        "transferred_to_other_hospital",
        lambda df: df["discharge_destination"] == "external_transfer",
    ),
    ("discharged_home", lambda df: df["discharge_destination"] == "home"),
    ("icu_los_lt_12h", lambda df: df["icu_los_lt_12h"] == True),  # noqa: E712
]

#: Exclusions needed for a paired physician-versus-model comparison.
PREDICTION_COMPLETENESS_STEPS: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
    ("missing_doc_prediction", lambda df: df["doc_predicted_at"].isna()),
    (
        "missing_algo_prediction",
        lambda df: df["algo_predicted_at_doc_prediction"].isna(),
    ),
]

#: Complete ordered enrolment funnel.
FUNNEL_STEPS = OUTCOME_ELIGIBILITY_STEPS + PREDICTION_COMPLETENESS_STEPS


def apply_enrolment_funnel(
    algo_doc: pd.DataFrame,
    *,
    steps: List[Tuple[str, Callable[[pd.DataFrame], pd.Series]]] = FUNNEL_STEPS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exclusion steps in order and return remaining rows with step counts.

    The report contains ``step``, ``excluded``, and ``remaining`` columns, preceded by the total
    admission count.
    """
    remaining = algo_doc
    report_rows = [
        {"step": "total_icu_admissions", "excluded": 0, "remaining": len(remaining)}
    ]
    for name, predicate in steps:
        excluded_mask = predicate(remaining)
        excluded_count = int(excluded_mask.sum())
        remaining = remaining[~excluded_mask]
        report_rows.append(
            {"step": name, "excluded": excluded_count, "remaining": len(remaining)}
        )
    return remaining, pd.DataFrame(report_rows)
