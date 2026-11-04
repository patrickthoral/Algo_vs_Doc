import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from algo_vs_doc import exploratory_contrasts


def test_joint_contrasts_and_notebook_exports(tmp_path):
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "patient_cluster_id": np.repeat(np.arange(40), 2),
            "outcome": np.tile([False, False, True, True], 20),
            "doc_prediction": rng.uniform(0.05, 0.8, 80),
            "doc_prediction_cat": np.tile(["low", "medium", "high", "medium"], 20),
            "doc_prediction_cat_score": np.tile([0, 1, 2, 1], 20),
            "algo_prediction_at_doc_prediction": rng.uniform(0.05, 0.8, 80),
        }
    )
    rows, samples = exploratory_contrasts.supplementary_contrasts(
        frame,
        n_bootstraps=3,
        n_splits=2,
    )
    assert rows.groupby("metric").size().to_dict() == {"auroc": 3, "auprc": 10}
    assert (
        samples.groupby("contrast_id").replicate.apply(list).map(tuple).nunique() == 1
    )
    pair = rows.loc[
        rows.agent_1.eq("Combined (Category)")
        & rows.agent_2.eq("Combined (Percentage)")
    ]
    assert len(pair) == 2
    secondary = rows.family.eq("secondary")
    assert secondary.sum() == 2
    assert rows.loc[secondary, "p_holm"].notna().all()
    assert rows.loc[~secondary, "p_holm"].isna().all()
    root = Path(__file__).resolve().parents[3]
    notebook = json.loads(
        (root / "notebooks/4-analysis.ipynb").read_text(encoding="utf-8")
    )
    source = next(
        "".join(c.get("source", []))
        for c in notebook["cells"]
        if "exploratory_contrasts.supplementary_contrasts("
        in "".join(c.get("source", []))
    )
    existing = rows.iloc[:2].copy()
    existing["contrast_id"] = [20, 21]
    existing["family"] = ["primary", "exploratory"]
    existing_samples = samples.loc[samples.contrast_id.isin([0, 1])].copy()
    existing_samples["contrast_id"] = existing_samples["contrast_id"].replace(
        {0: 20, 1: 21}
    )
    (tmp_path / "Combined").mkdir()
    (tmp_path / "plot_data" / "Combined").mkdir(parents=True)
    context = dict(
        os=os,
        pd=pd,
        cohorts=SimpleNamespace(COHORT_ORDER=("Combined",)),
        analysis_cohorts={"Combined": frame},
        analysis_states={
            "Combined": {
                "auroc_contrasts": existing,
                "auroc_bootstraps": existing_samples,
            }
        },
        n_bootstraps=3,
        random_seed=42,
        outputs_folder=str(tmp_path),
        plot_data_folder=str(tmp_path / "plot_data"),
    )
    exec(compile(source, "exploratory notebook cell", "exec"), context)
    result = context["auroc_contrasts"]
    pd.testing.assert_frame_equal(
        result.loc[result.contrast_id.isin([20, 21])].reset_index(drop=True),
        existing.reset_index(drop=True),
    )
    assert len(result) == 5
    assert len(pd.read_csv(tmp_path / "Combined/auprc_contrasts.csv")) == 10
