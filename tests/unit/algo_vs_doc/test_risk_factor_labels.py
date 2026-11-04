import pandas as pd

from algo_vs_doc import risk_factors


def test_model_feature_labels_harmonize_site_specific_variants():
    values = pd.Series(
        [
            "heart_rate__mean__last_24h",
            "heart_rate__mean__last_24h__padded",
        ]
    )

    assert risk_factors.model_feature_labels(values).tolist() == [
        "Heart rate",
        "Heart rate",
    ]


def test_physician_factor_display_labels_harmonize_hospital_wording():
    values = pd.Series(["Cardiaal belaste voorgeschiedenis", "Comorbiditeit: Cardiaal"])

    assert risk_factors.physician_factor_display_labels(values).tolist() == [
        "Cardiovascular comorbidity",
        "Cardiovascular comorbidity",
    ]
