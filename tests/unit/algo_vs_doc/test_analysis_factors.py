import pandas as pd
import pytest

from algo_vs_doc import analysis


def test_construct_shap_data_frame_accepts_non_contiguous_series_indices():
    shap = pd.Series(["{'age': 0.1}", "{'age': -0.2}"], index=[4, 9])
    patients = pd.Series([101, 102], index=[4, 9])
    discharges = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]), index=[4, 9])

    result = analysis.construct_shap_data_frame(shap, patients, discharges)

    assert result["patient_id"].tolist() == [101, 102]
    assert result["feature_name"].tolist() == ["age", "age"]


def _predictions():
    return pd.DataFrame(
        [
            {
                "patient_id": 1,
                "icu_discharged_at": pd.Timestamp("2024-01-02"),
                "hospital": "LUMC",
                "doc_factors": "doc_factors:10||doc_factors_2:10",
                "doc_other_factors": None,
            },
            {
                "patient_id": 2,
                "icu_discharged_at": pd.Timestamp("2024-01-03"),
                "hospital": "VUmc",
                "doc_factors": "doc_factors:30",
                "doc_other_factors": "pancreatitis, maagperforatie",
            },
        ]
    )


def test_selected_and_free_text_factors_land_in_one_frame_keyed_to_their_stay():
    factors = analysis.construct_doc_factors_data_frame(_predictions())
    assert len(factors) == 4
    by_patient = factors.groupby("patient_id").size()
    assert by_patient[1] == 2 and by_patient[2] == 2
    assert factors["factor_text"].dropna().tolist() == ["pancreatitis, maagperforatie"]


def test_free_text_with_a_comma_is_not_split_into_two_factors():
    factors = analysis.construct_doc_factors_data_frame(_predictions())
    assert (factors["factor_text"] == "pancreatitis, maagperforatie").sum() == 1


def test_both_kinds_of_factor_get_a_category():
    factors = analysis.construct_doc_factors_data_frame(_predictions())
    categorized = analysis.get_new_doc_features(factors)
    assert set(categorized.loc[categorized["patient_id"] == 1, "new_factors"]) == {
        "Logistic",
        "Circulatory",
    }
    assert set(categorized.loc[categorized["patient_id"] == 2, "new_factors"]) == {
        "Circulatory",
        "Abdominal",
    }


def test_two_factors_in_one_category_count_once_per_stay():
    predictions = (
        _predictions()
        .iloc[[1]]
        .assign(doc_factors="doc_factors:29||doc_factors:30", doc_other_factors=None)
    )  # both Circulatory
    factors = analysis.construct_doc_factors_data_frame(predictions)
    assert len(analysis.get_new_doc_features(factors)) == 1


def test_unrecognised_free_text_raises_like_any_other_gap_in_the_mapping():
    # a factor cited by a physician but counted in no category would understate agreement
    predictions = _predictions().iloc[[1]].assign(doc_other_factors="iets nieuws")
    factors = analysis.construct_doc_factors_data_frame(predictions)
    with pytest.raises(KeyError, match="not covered by the mapping"):
        analysis.get_new_doc_features(factors)


def test_an_unmapped_option_id_still_raises():
    predictions = _predictions().iloc[[0]].assign(doc_factors="doc_factors:999")
    factors = analysis.construct_doc_factors_data_frame(predictions)
    with pytest.raises(KeyError):
        analysis.get_new_doc_features(factors)


def test_a_stay_with_no_factors_at_all_contributes_no_rows():
    predictions = (
        _predictions().iloc[[0]].assign(doc_factors=None, doc_other_factors=None)
    )
    assert len(analysis.construct_doc_factors_data_frame(predictions)) == 0


def test_labels_make_the_descriptive_table_readable():
    # Descriptive tables need readable labels instead of factor tokens.
    factors = analysis.add_factor_labels(
        analysis.construct_doc_factors_data_frame(_predictions())
    )
    labels = factors["factor_label"].tolist()
    assert "Circulatie: bv. bloeddruk, ritme, ischemie, hartfalen, ..." in labels
    assert "Logistiek: zorgzwaarte voor afdeling" in labels
    assert "pancreatitis, maagperforatie" in labels  # free text is its own label


def test_labelling_does_not_disturb_the_ids_the_analysis_keys_on():
    factors = analysis.construct_doc_factors_data_frame(_predictions())
    labelled = analysis.add_factor_labels(factors)
    assert labelled["factor_id"].tolist() == factors["factor_id"].tolist()
    assert "factor_label" not in factors.columns  # the input is not modified
