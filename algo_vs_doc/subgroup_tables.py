"""Presentation of descriptive subgroup outcomes for supplementary reporting."""

import pandas as pd

from algo_vs_doc.specialties import SPECIALTY_LABELS
from algo_vs_doc.word_tables import WordTable


SECTIONS = {
    "admitting_specialty": "Admitting specialty",
    "admission_type": "Admission type",
    "icu_los_group": "Index ICU length of stay",
}
LABELS = {
    **SPECIALTY_LABELS,
    "medical": "Medical",
    "elective_surgery": "Elective surgery",
    "emergency_surgery": "Emergency surgery",
    "other_or_unknown": "Unknown",
    "unknown": "Unknown",
    "<24 hours": "<24 hours",
    "24-<72 hours": "24-72 hours",
    "72-<168 hours": "72-168 hours",
    ">=168 hours": "\u2265168 hours",
}
ORDER = {
    "admitting_specialty": list(SPECIALTY_LABELS) + ["unknown"],
    "admission_type": [
        "medical",
        "elective_surgery",
        "emergency_surgery",
        "other_or_unknown",
    ],
    "icu_los_group": [
        "<24 hours",
        "24-<72 hours",
        "72-<168 hours",
        ">=168 hours",
        "unknown",
    ],
}


def subgroup_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Format existing subgroup counts without recalculating outcomes or categories."""
    selected = counts.loc[counts["variable"].isin(SECTIONS)].copy()
    if selected.empty:
        raise ValueError(
            "no specialty, admission-type, or length-of-stay counts available"
        )
    keys = ["cohort", "variable", "level"]
    if selected.duplicated(keys).any():
        raise ValueError("duplicate subgroup counts")
    if selected[["n", "events_n"]].isna().any().any():
        raise ValueError("subgroup counts must be observed")
    if ((selected.events_n < 0) | (selected.n < selected.events_n)).any():
        raise ValueError("event counts must lie between zero and discharge counts")
    cohorts = list(selected.cohort.unique())
    cohorts = [c for c in ("Combined", "VUmc", "LUMC") if c in cohorts] + sorted(
        set(cohorts) - {"Combined", "VUmc", "LUMC"}
    )
    indexed = selected.set_index(keys)
    rows = []
    for variable, section in SECTIONS.items():
        levels = selected.loc[selected.variable.eq(variable), "level"].unique()
        ordered = [level for level in ORDER[variable] if level in levels]
        ordered += sorted(set(levels) - set(ordered))
        for level in ordered:
            row = {"Characteristic": section, "Subgroup": LABELS.get(level, level)}
            for cohort in cohorts:
                key = (cohort, variable, level)
                if key in indexed.index:
                    values = indexed.loc[key]
                    n, events = int(values["n"]), int(values["events_n"])
                    row[cohort] = (
                        f"{events}/{n} ({100 * events / n:.1f})" if n else "0/0 (NA)"
                    )
                else:
                    # An absent level is empty only if this variable was summarized.
                    available = (
                        (selected.cohort == cohort) & (selected.variable == variable)
                    ).any()
                    row[cohort] = "0/0 (NA)" if available else "Not available"
            rows.append(row)
    return pd.DataFrame(rows)


def subgroup_word_table(summary: pd.DataFrame) -> WordTable:
    """Create the supplementary table with explicit outcome and denominator notes."""
    return WordTable(
        "eTable 4. Seven-day outcomes by clinical subgroup",
        summary,
        notes=(
            "Values are events/discharges (%). Events are ICU readmission or death within "
            "seven days; a discharge with both components counts once. The denominator is "
            "the number of enrolled ICU discharges in the subgroup, not unique patients.",
            "Length of stay refers to the index ICU stay. The displayed ranges 24-72 "
            "and 72-168 hours include the lower boundary and exclude the upper boundary; "
            "the outer categories are <24 and \u2265168 hours.",
            "Unknown characteristics remain explicit. Unknown indicates no positive admission-type flag.",
            "NA indicates an empty subgroup. Not available indicates an absent subgroup "
            "summary. Percentages are within subgroups and are not percentages of all events.",
            "These additional descriptive summaries do not test subgroup differences or "
            "estimate subgroup predictive performance. Discharge destinations and "
            "readmission origins are reported separately in eTable 3.",
        ),
        group_column=["Characteristic", "Subgroup"],
        merge_group_cells=True,
    )
