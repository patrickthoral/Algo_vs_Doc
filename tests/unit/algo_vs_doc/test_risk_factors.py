import pandas as pd
import pytest

from algo_vs_doc import risk_factors


def _factors(rows):
    """Rows of ``(hospital, field, option_id)``."""
    return pd.DataFrame(
        [{"hospital": h, "factor_field": f, "factor_id": i} for h, f, i in rows]
    )


def test_every_mapped_category_is_in_the_canonical_vocabulary():
    used = set(risk_factors.MODEL_FEATURE_CATEGORIES.values())
    used |= set(risk_factors.PHYSICIAN_FREE_TEXT_CATEGORIES.values())
    for fields in risk_factors.PHYSICIAN_FACTOR_CATEGORIES.values():
        for mapping in fields.values():
            used |= set(mapping.values())
    assert used <= set(risk_factors.CATEGORIES), used - set(risk_factors.CATEGORIES)


def test_liver_chemistry_and_abdominal_factors_share_one_category():
    # the model's liver chemistry must be able to match a physician citing liver insufficiency
    assert (
        risk_factors.MODEL_FEATURE_CATEGORIES["bilirubin_total__mean__last_24h"]
        == "Abdominal"
    )
    assert (
        risk_factors.PHYSICIAN_FACTOR_CATEGORIES["LUMC"]["doc_factors_2"][15]
        == "Abdominal"
    )
    assert (
        risk_factors.PHYSICIAN_FACTOR_CATEGORIES["VUmc"]["doc_factors"][41]
        == "Abdominal"
    )
    assert "Liver" not in risk_factors.CATEGORIES


def test_comparable_categories_exclude_only_the_physician_only_ones():
    comparable = risk_factors.comparable_categories()
    assert "Logistic" not in comparable and "Medication" not in comparable
    assert "Abdominal" in comparable


def test_every_comparable_category_really_exists_on_both_sides():
    # the guarantee the coverage metric depends on
    model = set(risk_factors.MODEL_FEATURE_CATEGORIES.values())
    physician = set()
    for fields in risk_factors.PHYSICIAN_FACTOR_CATEGORIES.values():
        for mapping in fields.values():
            physician |= set(mapping.values())
    for category in risk_factors.comparable_categories():
        assert category in model, f"{category} unreachable from model features"
        assert category in physician, f"{category} unreachable from physician factors"


def test_the_same_option_id_means_different_things_per_field_and_hospital():
    """The reason the field is part of the key. LUMC split one question across two form
    sections whose option ids restart, so id 10 is logistic in one and circulatory in the
    other - a distinction that was lost once before by reading the id alone."""
    result = risk_factors.categorize_physician_factors(
        _factors(
            [
                ("LUMC", "doc_factors", 10),
                ("LUMC", "doc_factors_2", 10),
                ("VUmc", "doc_factors", 10),
            ]
        )
    )
    assert result["new_factors"].tolist() == ["Logistic", "Circulatory", "General"]


def test_an_unmapped_option_id_fails_loudly_naming_its_field():
    with pytest.raises(KeyError, match="not covered by the mapping"):
        risk_factors.categorize_physician_factors(
            _factors([("LUMC", "doc_factors", 99)])
        )


def test_an_unknown_hospital_or_field_fails_loudly():
    with pytest.raises(KeyError, match="no physician factor list"):
        risk_factors.categorize_physician_factors(
            _factors([("LUMC", "doc_factors_9", 1)])
        )
    with pytest.raises(KeyError, match="no physician factor list"):
        risk_factors.categorize_physician_factors(
            _factors([("Elsewhere", "doc_factors", 1)])
        )


def test_categorize_physician_factors_does_not_modify_the_input():
    data = _factors([("VUmc", "doc_factors", 30)])
    before = data.copy()
    risk_factors.categorize_physician_factors(data)
    pd.testing.assert_frame_equal(data, before)


def test_free_text_with_a_comma_stays_one_factor():
    # the bug that motivated keying on ids: comma-splitting turned this single factor into two
    # fragments, neither of which mapped to anything
    categories = risk_factors.categorize_free_text(
        pd.Series(["pancreatitis, maagperforatie"])
    )
    assert categories.tolist() == ["Abdominal"]


def test_unrecognised_free_text_fails_rather_than_going_uncounted():
    # the trial is closed, so this vocabulary is closed too - an answer it does not cover means
    # the mapping was never finished, and the factor would silently be counted in no category
    with pytest.raises(KeyError, match="not covered by the mapping"):
        risk_factors.categorize_free_text(pd.Series(["iets nieuws"]))


def test_free_text_is_matched_regardless_of_case_and_surrounding_whitespace():
    # a difference in typing is not a difference in meaning, and would otherwise raise
    categories = risk_factors.categorize_free_text(pd.Series(["  COVID  ", "covid"]))
    assert categories.tolist() == ["Respiratory", "Respiratory"]


def test_a_blank_free_text_answer_is_left_uncategorized_rather_than_raising():
    # no free-text answer given is legitimately missing, not a gap in the mapping
    assert risk_factors.categorize_free_text(pd.Series([None])).isna().all()


def test_categorize_model_features_maps_and_reports_unmapped_features():
    shap_data = pd.DataFrame([{"feature_name": "lactate__last__overall"}])
    assert risk_factors.categorize_model_features(shap_data)[
        "new_feature_name"
    ].tolist() == ["Circulatory"]
    with pytest.raises(KeyError, match="not covered by the mapping"):
        risk_factors.categorize_model_features(
            pd.DataFrame([{"feature_name": "brand_new_feature"}])
        )


def test_categorize_model_features_covers_both_trial_model_vocabularies():
    expected = {
        "body_mass_index__mean__last_24h": "General",
        "cough_stimulus__mode__last_24h__cough_reflex_normal": "Respiratory",
        "cough_stimulus__mode__last_24h__cough_reflex_reasonable": "Respiratory",
        "creatine_kinase_mb_mass__is_measured__last_24h": "Circulatory",
        "length_of_stay_hours": "General",
        "o2_flow__change_since_first__last_24h": "Respiratory",
        "o2_flow__mean__last_24h": "Respiratory",
        "pco2_arterial__mean__last_24h": "Respiratory",
        "respiratory_rate_measured__last__overall": "Respiratory",
        "respiratory_rate_measured__mean__last_24h__padded": "Respiratory",
        "sex__female": "General",
        "sex__male": "General",
        "temperature__change_since_first__last_24h": "Infection",
        "time_since_start": "General",
        "tracheobronchial_toilet_quantity__total_dose__last_24h": "Respiratory",
        "tracheobronchial_toilet_quantity__total_value__last_24h": "Respiratory",
    }
    frame = pd.DataFrame({"feature_name": list(expected)})

    result = risk_factors.categorize_model_features(frame)

    assert result.set_index("feature_name")["new_feature_name"].to_dict() == expected


def test_categorize_model_features_does_not_modify_the_input():
    shap_data = pd.DataFrame([{"feature_name": "age"}])
    before = shap_data.copy()
    risk_factors.categorize_model_features(shap_data)
    pd.testing.assert_frame_equal(shap_data, before)


def test_the_unselected_option_record_still_refers_to_real_options():
    # the record must not drift from the mapping it annotates
    for hospital, field, option_id in risk_factors.UNSELECTED_FACTOR_OPTIONS:
        assert option_id in risk_factors.PHYSICIAN_FACTOR_CATEGORIES[hospital][field]


def test_an_id_that_is_not_a_number_at_all_fails_rather_than_reading_as_no_factor():
    # numeric coercion would turn this into NA, which downstream is indistinguishable from a
    # physician having selected nothing
    with pytest.raises(KeyError, match="not covered by the mapping"):
        risk_factors.categorize_physician_factors(
            _factors([("LUMC", "doc_factors", "n/a")])
        )


def test_a_blank_id_is_left_uncategorized_rather_than_raising():
    # a physician who selected no factor is legitimately missing, not an ETL defect
    result = risk_factors.categorize_physician_factors(
        _factors([("LUMC", "doc_factors", None)])
    )
    assert result["new_factors"].isna().all()


def test_explode_factor_tokens_carries_the_source_row_back():
    frame = pd.DataFrame(
        [
            {"hospital": "LUMC", "doc_factors": "doc_factors:5||doc_factors_2:15"},
            {"hospital": "VUmc", "doc_factors": None},
        ],
        index=["stay_a", "stay_b"],
    )
    exploded = risk_factors.explode_factor_tokens(frame)
    assert exploded["source_index"].tolist() == ["stay_a", "stay_a"]
    assert exploded["factor_field"].tolist() == ["doc_factors", "doc_factors_2"]
    assert exploded["factor_id"].tolist() == ["5", "15"]


def test_every_option_has_exactly_one_label():
    # the descriptions live both here and as the comments that make the category mapping
    # reviewable; this keeps an option from being added to one and forgotten in the other
    def keys(table):
        return {
            (h, f, i)
            for h, fields in table.items()
            for f, ids in fields.items()
            for i in ids
        }

    assert keys(risk_factors.PHYSICIAN_FACTOR_LABELS) == keys(
        risk_factors.PHYSICIAN_FACTOR_CATEGORIES
    )


def test_labels_are_never_used_as_a_lookup_key():
    # the original bug: matching on descriptions, several of which contain commas
    assert risk_factors.factor_labels(
        _factors([("LUMC", "doc_factors_2", 15)])
    ).tolist() == ["Abdominaal: bv. leverinsufficientie"]
    with pytest.raises(KeyError, match="physician factor label"):
        risk_factors.factor_labels(_factors([("LUMC", "doc_factors", 99)]))


def test_the_free_text_vocabulary_carries_no_mangled_text():
    # "Hematologie patient" (with a diaeresis) was mapped in its broken form for a long time,
    # because the export was read with the wrong codepage; these are the characters that appear
    # when that happens
    mojibake = set("\u2030\u0160\u008d\u008f\u0090\u009d\u02dc\u0161\u017e\u0178\u0192")
    for text in risk_factors.PHYSICIAN_FREE_TEXT_CATEGORIES:
        assert not (set(text) & mojibake), text
    for fields in risk_factors.PHYSICIAN_FACTOR_LABELS.values():
        for labels in fields.values():
            for text in labels.values():
                assert not (set(text) & mojibake), text
