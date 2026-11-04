import numpy as np
import pandas as pd

from algo_vs_doc import clustered_inference


def _frame():
    rng = np.random.default_rng(42)
    size = 120
    census = rng.integers(5, 20, size=size)
    availability_probability = 1 / (1 + np.exp(-(-0.5 + 0.08 * census)))
    return pd.DataFrame(
        {
            "hospital": np.where(np.arange(size) % 2, "LUMC", "VUmc"),
            "patient_id": np.arange(size) % 80,
            "icu_discharged_at": pd.Timestamp("2024-01-01")
            + pd.to_timedelta((np.arange(size) // 2) % 30, unit="D"),
            "doc_predicted_at": pd.Series(
                np.where(
                    rng.random(size) < availability_probability,
                    pd.Timestamp("2024-01-01"),
                    pd.NaT,
                )
            ),
            "n_census_at_discharge": census,
            "n_external_transfers_same_day": rng.integers(0, 3, size=size),
            "n_elective_surgery_admissions_same_day": rng.integers(0, 6, size=size),
            "discharge_after_18h": rng.random(size) < 0.2,
            "discharge_is_weekend": rng.random(size) < 0.3,
            "age_at_admission": rng.normal(65, 12, size=size),
            "sofa_discharge": rng.integers(0, 10, size=size),
            "is_medical": rng.random(size) < 0.5,
            "emergency_surgery": rng.random(size) < 0.2,
            "elective_surgery": rng.random(size) < 0.3,
        }
    )


def test_prediction_availability_model_uses_two_cluster_dimensions():
    result = clustered_inference.prediction_availability_model(
        _frame(), prediction_column="doc_predicted_at"
    ).set_index("variable")

    assert "n_census_at_discharge" in result.index
    assert "admission_type_medical" in result.index
    assert "admission_type_other_or_unknown" not in result.index
    assert result.loc["n_census_at_discharge", "odds_ratio"] > 0
    assert "p" not in result.columns
    assert "p_holm" not in result.columns
    assert result["patient_clusters"].iloc[0] == 80
    assert result["hospital_date_clusters"].iloc[0] == 60
