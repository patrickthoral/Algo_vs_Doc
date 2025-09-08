import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from itertools import combinations

hospital = "Pacmed Medical Center"


def combine_predictions(data, prediction1, prediction2, outcome, combined_prediction):
    data[combined_prediction] = np.nan
    mask = np.isnan(data.loc[:, [prediction1, prediction2]]).sum(axis=1) == 0
    data.loc[data[prediction1] == 0, prediction1] = 0.000001
    data.loc[data[prediction1] == 1, prediction1] = 0.999999
    log1 = np.log(data.loc[mask, prediction1] / (1 - data.loc[mask, prediction1]))
    log2 = np.log(data.loc[mask, prediction2] / (1 - data.loc[mask, prediction2]))
    X = pd.DataFrame({"log1": log1, "log2": log2})
    y = data.loc[mask, outcome]
    clf = LogisticRegression(random_state=42).fit(X, y)
    print(clf.coef_)
    data.loc[mask, combined_prediction] = clf.predict_proba(X)[:, 1]
    return data


def bootstrap_auc(n, y, prob_y, curve="roc"):
    """
    Bootstrap the predictions by sampling with replacement to calculate 
    the 95% confidence interval of either ROC AUC (curve="roc") or 
    PRC AUC (curve="prc")

    Returns:
    - confidence_lower, confidence_upper
    """
    
    # start with the observed score
    if curve == "roc":
        x_axis, y_axis, thresholds = roc_curve(y, prob_y)
    elif curve == "prc":
        y_axis, x_axis, thresholds = precision_recall_curve(y, prob_y)
    observed = auc(x_axis, y_axis)

    # init bootstrapping
    y.index = range(0, len(y))
    prob_y.index = range(0, len(y))
    n_bootstraps = n
    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, prob_y.shape[0], prob_y.shape[0])
        if len(np.unique(y[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        if curve == "roc":
            x_axis, y_axis, thresholds = roc_curve(y[indices], prob_y[indices])
        elif curve == "prc":
            y_axis, x_axis, thresholds = precision_recall_curve(
                y[indices], prob_y[indices]
            )
        score = auc(x_axis, y_axis)
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return observed, confidence_lower, confidence_upper


def calculate_score(curve, y_true, y_pred):
    if curve == "roc":
        x_axis, y_axis, thresholds = roc_curve(y_true, y_pred)

    elif curve == "prc":
        y_axis, x_axis, thresholds = precision_recall_curve(
            y_true, y_pred
        )
    else:
        raise ValueError(f"Invalid curve/metric specified. Must be either 'roc' or 'prc'")

    model_score = auc(x_axis, y_axis)
    return model_score

def bootstrap_AUC_difference(n_bootstraps, y_true, y_pred_1, y_pred_2, curve="roc"):
    """
    Calculates the AUC difference between two models using bootstrapping

    :param n_bootstraps: number of resamples (bootstraps)
    :param y_true: dataframe of outcome labels
    :param y_pred_1: dataframe of predicted probabilities by model 1
    :param y_pred_2: dataframe of predicted probabilities of model 2
    :param curve: one of ["roc", "prc"]
    """
    y_true.index = range(0, len(y_true))
    y_pred_1.index = range(0, len(y_true))
    y_pred_2.index = range(0, len(y_true))

    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)

    # calculate observed difference (i.e. without all data before resampling)
    model1_score = calculate_score(curve, y_true, y_pred_1)
    model2_score = calculate_score(curve, y_true, y_pred_2)

    observed_difference = model2_score - model1_score

    bootstrapped_differences = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, y_pred_1.shape[0], y_pred_2.shape[0])
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be calculated: reject the sample
            continue

        model1_score = calculate_score(curve, y_true[indices], y_pred_1[indices])
        model2_score = calculate_score(curve, y_true[indices], y_pred_2[indices])
        bootstrapped_differences.append(model2_score - model1_score)

    bootstrapped_differences = np.array(bootstrapped_differences)
    bootstrapped_differences.sort()

    # Computing the lower and upper bound of the 95% confidence interval
    confidence_lower = bootstrapped_differences[int(0.025 * len(bootstrapped_differences))]
    confidence_upper = bootstrapped_differences[int(0.975 * len(bootstrapped_differences))]

    # p value
    # one-sided testing:
    # p_value = min(np.mean(bootstrapped_differences < 0), np.mean(bootstrapped_differences > 0))
    # two-sided testing:
    p_value = 2 * min(np.mean(bootstrapped_differences < 0), np.mean(bootstrapped_differences > 0))

    return observed_difference, confidence_lower, confidence_upper, p_value, bootstrapped_differences, curve


def auc_differences_table(predictions, outcome, prediction_columns, curve="roc"):
    """
    Creates a table comparing the difference in ROC or PRC AUC of models using bootstrapping
    """

    rows = list(prediction_columns)[0:-1]
    columns = list(prediction_columns)[1:]

    table = pd.DataFrame(
        index=rows,
        columns=columns
    )

    data = {}

    comparisons = list(combinations(prediction_columns, 2))
    for comparison in comparisons:
        c1 = comparison[0]
        c2 = comparison[1]
        valid_predictions = ~(
                    (pd.isnull(predictions[prediction_columns[c1]])) |
                    (pd.isnull(predictions[prediction_columns[c2]]))
            )
        y_pred_1 = predictions.loc[
            valid_predictions, prediction_columns[c1]
        ]

        y_pred_2 = predictions.loc[
            valid_predictions, prediction_columns[c2]
        ]

        y_true = predictions.loc[
            valid_predictions, outcome
        ]

        n_bootstraps = 1000
        print(f"Bootstrapping ({n_bootstraps} samples): {c1} vs. {c2}")
        observed_difference, confidence_lower, confidence_upper, p_value, bootstrapped_differences, curve = (
            bootstrap_AUC_difference(
                n_bootstraps, y_true, y_pred_1, y_pred_2, curve=curve))

        # create table and store data
        table.loc[c1, c2] = f"{observed_difference:#.2g} [{confidence_lower:#.2g}, {confidence_upper:#.2g}]\np = {p_value:#.2g}"
        if c1 not in data.keys():
            # create empty dictionary
            data[c1] = {}

        data[c1][c2] = [
            observed_difference,
            confidence_lower,
            confidence_upper,
            p_value,
            bootstrapped_differences,
            curve
        ]

    return table, data


def calibration_loss(y_true, y_prob, bin_size=100):
    y_prob = y_prob.sort_values()
    index = y_prob.index
    y_true = y_true.iloc[index]
    loss = 0.0
    for i in np.arange(0, len(y_prob) - bin_size):
        avg_prob = y_true.iloc[i: i + bin_size].sum() / bin_size
        mean_pred = y_prob.iloc[i: i + bin_size].sum() / bin_size
        loss += np.abs(mean_pred - avg_prob)
    loss /= len(y_prob) - bin_size
    return loss


def performance_table(
        predictions, 
        outcome, 
        prediction_columns,
        curve="roc"):
    """
    Creates a table containing both discriminatory performance using the AUC metric (ROC or PRC) and
    calibration performance using calibration intercept, slope, and loss.
    Uses bootstrapping to estimate 95% confidence intervals.

    Returns:
    - table containing performance metrics
    """
    
    table = pd.DataFrame()
    for c in prediction_columns:
        p = predictions.loc[
            ~pd.isnull(predictions[prediction_columns[c]]), prediction_columns[c]
        ]
        p.loc[p == 1] = 0.9999
        p.loc[p == 0] = 0.0001
        y = predictions.loc[~pd.isnull(predictions[prediction_columns[c]]), outcome]
        # discrimination
        observed, confidence_lower, confidence_upper = bootstrap_auc(1000, y, p, curve=curve)
        # calibration
        logit = np.log(p / (1 - p))
        df = pd.DataFrame(np.transpose([y, logit]), columns=["y", "logit"])
        mod_slope = smf.glm("y~logit", data=df, family=sm.families.Binomial()).fit()
        mod_interc = smf.glm(
            "y~1", data=df, offset=logit, family=sm.families.Binomial()
        ).fit()
        # form table
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [{
                        "n": len(predictions.loc[~pd.isnull(predictions[prediction_columns[c]])]),
                        f"au{curve}_with_confidence_interval".lower(): 
                            f"{observed:#.2g} ["
                            + ", ".join(f"{i:#.2g}" for i in (confidence_lower, confidence_upper))
                            + "]",
                        f"au{curve}": observed,
                        f"au{curve}_confidence_interval_lower": confidence_lower,
                        f"au{curve}_confidence_interval_upper": confidence_upper,
                        "calibration_intercept": 
                            f"{mod_interc.params.iloc[0]:#.2g} ["
                            + ", ".join(f"{i:#.2g}" for i in np.array(mod_interc.conf_int(alpha=0.05))[0, :])
                            + "]",
                        "calibration_slope": 
                            f"{mod_slope.params.iloc[1]:#.2g} ["
                            + ", ".join(f"{i:#.2g}" for i in np.array(mod_slope.conf_int(alpha=0.05))[1, :])
                            + "]",
                        "calibration_loss": f"{calibration_loss(y, p):#.2g}",
                    }]
                )
            ]
        )
    table.index = prediction_columns.keys()
    return table


def create_max_time_difference_df(
        predictions_all_time_points, outcome, prediction_columns,
        curve="roc"
):
    AUC_list = (
            [f"AU{curve.upper()} {c}" for c in list(prediction_columns.keys())]
            + [f"LB {c}" for c in list(prediction_columns.keys())]
            + [f"UB {c}" for c in list(prediction_columns.keys())]
    )
    time_dif_data = pd.DataFrame(
        dict({"max_time_difference": range(4, 25)}, **{AUC: np.nan for AUC in AUC_list})
    )

    for t in time_dif_data.max_time_difference:
        predictions = predictions_all_time_points.copy()

        # remove predictions that are outside window of allowed (max) time difference
        predictions.loc[predictions["true_h_doc"] > t, ["doc_prediction"]] = np.nan
        predictions.loc[predictions["true_h_doc"] > t, ["doc_prediction_cat"]] = np.nan
        predictions.loc[
            predictions["true_h_algo_at_doc_prediction"] > t, ["algo_at_doc_prediction"]
        ] = np.nan

        # bootstrap AUC for each prediction method (e.g. physician, model, ...)
        for c in prediction_columns:
            p = predictions.loc[
                ~pd.isnull(predictions[prediction_columns[c]]), prediction_columns[c]
            ]
            y = predictions.loc[~pd.isnull(predictions[prediction_columns[c]]), outcome]
            observed, confidence_lower, confidence_upper = bootstrap_auc(1000, y, p, curve=curve)
            time_dif_data.loc[
                time_dif_data.max_time_difference == t,
                [f"AU{curve.upper()} {c}", f"LB {c}", f"UB {c}"],
            ] = [observed, confidence_lower, confidence_upper]
    return time_dif_data


def shap_to_dataframe(shap_dictionary, pat_deid, discharge_time):
    shap_dictionary = shap_dictionary.replace("[{", "")
    shap_dictionary = shap_dictionary.replace("}]", "")
    shap_dictionary = shap_dictionary.split("}, {")
    list_shap_dictionaries = [
        eval("{" + dictionary + "}") for dictionary in shap_dictionary
    ]
    shap_data = pd.DataFrame.from_records(list_shap_dictionaries)
    shap_data["patient_id"] = pat_deid
    shap_data["discharge_time"] = discharge_time
    return shap_data


def shap_to_dataframe(shap_dictionary, pat_deid, discharge_time):
    shap_data = pd.DataFrame.from_records(
        {
            "feature_name": list(eval(shap_dictionary).keys()),
            "shap_value": list(eval(shap_dictionary).values()),
        }
    )
    shap_data["patient_id"] = pat_deid
    shap_data["discharge_time"] = discharge_time
    return shap_data


def construct_shap_data_frame(
        shap_variable, patient_id_variable, discharge_time_variable
):
    shap_data =  pd.DataFrame()
    for i in range(0, len(shap_variable)):
        shap = shap_variable[i]
        patient_id = patient_id_variable[i]
        discharge_time = discharge_time_variable[i]
        if not pd.isnull(shap):
                shap_data = pd.concat(
                    [
                        shap_data,
                        shap_to_dataframe(shap, patient_id, discharge_time)
                    ]
                )
    return shap_data


def construct_doc_factors_data_frame(
        factors_variable,
        other_factor_variable,
        patient_id_variable,
        discharge_time_variable,
):
    doc_factors_data = pd.DataFrame()
    for i in range(0, len(factors_variable)):
        main_factors = factors_variable[i]
        other_factors = other_factor_variable[i]
        patient_id = patient_id_variable[i]
        discharge_time = discharge_time_variable[i]

        factors = []

        if main_factors is not None:
            factors += main_factors.split("||")

        if other_factors is not None:
            factors += other_factors.split("||")

        for factor in factors:
            doc_factors_data = pd.concat(
                [
                    doc_factors_data,
                    pd.DataFrame(
                        [{
                            "patient_id": patient_id,
                            "discharge_time": discharge_time,
                            "factors": factor,
                        }]
                    )
                ]
            )
    return doc_factors_data


def get_new_algo_features(shap_data, limit=10, absolute_values=True):
    shap_data = make_new_algo_name_feature(shap_data)
    shap_data["abs_shap_value"] = abs(shap_data.shap_value)
    if absolute_values:
        shap_data = shap_data.sort_values(
            ["patient_id", "discharge_time", "abs_shap_value"], ascending=False
        )
    else:
        shap_data = shap_data.sort_values(
            ["patient_id", "discharge_time", "shap_value"], ascending=False
        )
    shap_data = shap_data.drop_duplicates(
        subset=["patient_id", "discharge_time", "new_feature_name"]
    )
    shap_data = shap_data.groupby(["patient_id", "discharge_time"]).head(limit)
    return shap_data


def get_new_doc_features(doc_data):
    doc_data = make_new_doc_name_feature(doc_data)
    
    doc_data = doc_data.drop_duplicates(
        subset=["patient_id", "discharge_time", "new_factors"]
    )
    return doc_data


def make_new_algo_name_feature(shap_data):
    """
    Groups the individual features into a group of features for comparing the model features
    against the physicians. Combines features from the predictions by the VUmc and LUMC model.
    """
    feature_name_mapping = {
        # General
        "age": "General",
        "length_of_stay_hours": "General",
        "time_since_start": "General",
        "sex__male": "General",
        "sex__female": "General",
        "body_mass_index__mean__last_24h": "General",
        # Respiratory
        "cough_stimulus__mode__last_24h__cough_reflex_normal": "Respiratory",
        "o2_flow__mean__last_24h": "Respiratory",
        "o2_flow__change_since_first__last_24h": "Respiratory",
        "respiratory_rate_measured__mean__last_24h__padded": "Respiratory",
        "respiratory_rate_measured__last__overall": "Respiratory",
        "pco2_arterial__mean__last_24h": "Respiratory",
        "tracheobronchial_toilet_quantity__total_dose__last_24h": "Respiratory",
        "tracheobronchial_toilet_quantity__total_value__last_24h": "Respiratory",
        "o2_saturation__mean__last_24h": "Respiratory",
        "o2_saturation__change_since_first__last_24h": "Respiratory",
        "tidal_volume_per_kg__mean__last_24h__padded": "Respiratory",
        "pao2_over_fio2__last__overall": "Respiratory",
        "fio2__mean__last_24h": "Respiratory",
        "peep__maximum__last_24h": "Respiratory",
        # Circulatory
        "creatine_kinase_mb_mass__last__overall": "Circulatory",
        "heart_rate__mean__last_24h__padded": "Circulatory",
        "heart_rate__mean__last_24h": "Circulatory",
        "arterial_blood_pressure_diastolic__mean__last_24h": "Circulatory",
        "arterial_blood_pressure_systolic__mean__last_24h": "Circulatory",
        "arterial_blood_pressure_mean__change_since_first__last_24h": "Circulatory",
        "lactate__last__overall": "Circulatory",
        "troponin_t__maximum__last_24h": "Circulatory",
        # Neurological
        "glasgow_coma_scale_total__maximum__last_24h": "Neurological",
        "pupil_diameter_max__mean__last_24h": "Neurological",
        "glasgow_coma_scale_total__minimum__last_24h": "Neurological",
        "richmond_agitation_sedation_scale_score__maximum__last_24h": "Neurological",
        # Renal and homeostasis
        "ph_arterial__last__overall": "Renal and homeostasis",
        "glucose__minimum__last_24h": "Renal and homeostasis",
        "sodium__mean__last_24h": "Renal and homeostasis",
        "albumin__mean__last_24h__padded": "Renal and homeostasis",
        "ureum_over_creatinine__mean__last_24h": "Renal and homeostasis",
        "ureum__mean__last_24h": "Renal and homeostasis",
        "fluid_out_urine__total_dose__last_24h": "Renal and homeostasis",
        "creatinine__last__overall": "Renal and homeostasis",
        "phosphate__mean__last_24h": "Renal and homeostasis",
        "chloride__mean__last_24h": "Renal and homeostasis",
        "base_excess__mean__last_24h": "Renal and homeostasis",
        "potassium__mean__last_24h": "Renal and homeostasis",
        "albumin__mean__last_24h": "Renal and homeostasis",
        "fluid_out_urine__total_value__last_24h": "Renal and homeostasis",
        "magnesium__mean__last_24h": "Renal and homeostasis",
        # Hematology
        "thrombocytes__mean__last_24h": "Hematology",
        "hemoglobin__mean__last_24h__padded": "Hematology",
        "prothrombin_time_inr__mean__last_24h__padded": "Hematology",
        "hemoglobin__mean__last_24h": "Hematology",
        "prothrombin_time_inr__mean__last_24h": "Hematology",
        "activated_partial_thromboplastin_time__mean__last_24h": "Hematology",
        # Liver
        "bilirubin_total__mean__last_24h": "Liver",
        "lactate_dehydrogenase__mean__last_24h": "Infection",
        "gamma_glutamyl_transferase__mean__last_24h": "Liver",
        "alanine_transaminase__mean__last_24h": "Liver",
        "aspartate_transaminase__mean__last_24h": "Liver",
        "alkaline_phosphatase__mean__last_24h": "Liver",
        # Infection
        "temperature_skin__mean__last_24h": "Infection",
        "temperature_internal__maximum__last_24h": "Infection",
        "leukocytes__mean__last_24h__padded": "Infection",
        "c_reactive_protein__mean__last_24h": "Infection",
        "temperature__minimum__last_24h": "Infection",
        "leukocytes__mean__last_24h": "Infection"
    }
    shap_data["new_feature_name"] = shap_data.feature_name.apply(
        lambda x: feature_name_mapping[x]
    )
    return shap_data


def make_new_doc_name_feature(doc_data):
    """
    Groups the individual risk factors into a group of risk factors for comparing the physicians risk factors against
    the model features. Combines risk factors documented in  VUmc and LUMC systems.
    """
    mask = ["Hematologie" in factor for factor in doc_data.factors]
    doc_data.loc[mask, "factors"] = "Hematologie pat"
    feature_name_mapping = {
        # VUmc
        # General
        "reden van opname": "General",
        "algemene (ziekte) indruk": "General",
        "leeftijd": "General",
        "eerdere heropname": "General",
        "gewicht-BMI": "General",
        "geslacht": "General",
        "Frailty": "General",
        "Maligniteit": "General",
        # Logistic
        "(verpleegkundige) zorgzwaarte": "Logistic",
        "beleid": "Logistic",
        "Niet-IC beleid ": "Logistic",
        "schizofreen": "Logistic",
        "persoonlijkheids problematiek/borderline": "Logistic",
        "depressief en verward": "Logistic",
        "Psychiatrische voorgeschiedenis": "Logistic",
        "Psychiatrisch zoektebeeld": "Logistic",
        "Psychiatrie": "Logistic",
        "woede-uitbarsting": "Logistic",
        "Isolatiemaatregelen": "Logistic",
        # Respiratory
        "oxygenatie/PaO2": "Respiratory",
        "Pulmonaal belaste voorgeschiedenis": "Respiratory",
        "ventilatie/PaCO2": "Respiratory",
        "st na pneumectomie": "Respiratory",
        "COVID": "Respiratory",
        # Circulatory
        "Bloeddruk": "Circulatory",
        "(recent) hartfalen": "Circulatory",
        "Cardiaal belaste voorgeschiedenis": "Circulatory",
        "(recente) cardiale ischemie": "Circulatory",
        "(recent) vasopressie/inotropie": "Circulatory",
        # Neurological
        "delier": "Neurological",
        "verlaagd bewustzijn/lage EMV score": "Neurological",
        "bewustzijn": "Neurological",
        # Renal and homeostasis
        "nierfalen (bestaand of risico verslechtering)": "Renal and homeostasis",
        # Hematology
        "(recidiverende) bloedingen": "Hematology",
        "Hematologie pat": "Hematology",
        # Infection
        "recidiverende infecties": "Infection",
        # Medication
        "pijnstilling": "Medication",
        # Other
        "pancreatitis, maagperforatie": "Abdominal",
        "nog onvoldoende pijncontrole": "Medication",
        
        # LUMC
        # General
        "Comorbiditeit: Kwetsbaarheid": "General",
        "Comorbiditeit: Overig": "General",
        "Demografisch (leeftijd, geslacht, etniciteit)": "General",
        "Comorbiditeit: Maligniteit": "General",
        "Opname kenmerk: reden van opname": "General",
        "Opname kenmerk: duur van opname of heropname": "General",
        # Logistic
        "Logistiek: zorgzwaarte voor afdeling": "Logistic",
        # Respiratory
        "Respiratie: bv. ventiliatie, zuurstofbehoefte, beademingsduur": "Respiratory",
        "Sputumproductie met uitzuigbehoefte": "Respiratory",
        "Comorbiditeit: Pulmonaal": "Respiratory",
        # Circulatory
        "Circulatie: bv. bloeddruk, ritme, ischemie, hartfalen, vochtbalans, vasopressie/inotropie": "Circulatory",
        "Comorbiditeit: Cardiaal": "Circulatory",
        # Neurological
        "Neurologie: bv. sub maximale EMV, IC verkregen zwakte, risico op verslechtering": "Neurological",
        # Renal and homeostasis
        "Nierfunctie: bv. nierinsufficientie, zuur/base- of electrolytstoornissen": "Renal and homeostasis",
        # Hematology
        "Hematologie: bv. bloedingsrisico": "Hematology",
        # Abdominal
        "Abdominaal: bv. leverinsufficientie": "Abdominal",
        # Infection
        "Infectie: bv. infectie matig onder controle, risico op infectie, isolatiemaatregelen": "Infection"

    }
    doc_data["new_factors"] = doc_data.factors.apply(lambda x: feature_name_mapping[x])
    return doc_data


def get_coverage(doc_feature_list, algo_feature_list):
    D = set(doc_feature_list)
    A = set(algo_feature_list)
    return 100 * (len(D) - len(D - A)) / len(D)


def create_shap_coverage_data(shap_data, doc_factors_data_new):
    cap_data = pd.DataFrame(
        dict(
            {"max_model_features": range(1, 11)},
            **{
                abs_values: np.nan
                for abs_values in ["absolute_values", "no_absolute_values"]
            },
        )
    )

    for limit in range(1, 11):
        for absolute_values in [True, False]:
            shap_data_new = get_new_algo_features(
                shap_data, limit=limit, absolute_values=absolute_values
            )
            # Remove factors with no overlap between algo and doc
            mask_algo = (
                    (shap_data_new.new_feature_name == "Logistic")
                    | (shap_data_new.new_feature_name == "Medication")
                    | (shap_data_new.new_feature_name == "Other")
                    | (shap_data_new.new_feature_name == "Liver")
            )
            shap_data_new = shap_data_new.loc[~mask_algo, :]
            mask_doc = (
                    (doc_factors_data_new.new_factors == "Logistic")
                    | (doc_factors_data_new.new_factors == "Medication")
                    | (doc_factors_data_new.new_factors == "Other")
                    | (doc_factors_data_new.new_factors == "Liver")
            )
            doc_factors_data_new = doc_factors_data_new.loc[~mask_doc, :]
            # Make data frame for comparison overlap factors algo and doc
            doc_factors_grouped = (
                doc_factors_data_new.groupby(["patient_id", "discharge_time"])
                .new_factors.apply(list)
                .reset_index()
            )
            algo_factors_grouped = (
                shap_data_new.groupby(["patient_id", "discharge_time"])
                .new_feature_name.apply(list)
                .reset_index()
            )
            feature_coverage_data = algo_factors_grouped.merge(
                doc_factors_grouped, how="outer", on=["patient_id", "discharge_time"]
            )
            mask = ~pd.isnull(feature_coverage_data.new_feature_name) & ~pd.isnull(
                feature_coverage_data.new_factors
            )
            feature_coverage_data = feature_coverage_data[mask]
            feature_coverage_data["coverage"] = feature_coverage_data.apply(
                lambda x: get_coverage(x.new_factors, x.new_feature_name), axis=1
            )
            if absolute_values:
                cap_data.loc[
                    cap_data.max_model_features == limit, "absolute_values"
                ] = feature_coverage_data.coverage.mean()
            else:
                cap_data.loc[
                    cap_data.max_model_features == limit, "no_absolute_values"
                ] = feature_coverage_data.coverage.mean()
    return cap_data
