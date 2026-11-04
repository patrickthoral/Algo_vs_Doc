import numpy as np
import pandas as pd
import pytest

from algo_vs_doc import calibration, plots


def test_calibration_intercept_and_oe_with_constant_prediction():
    y = np.array([0, 0, 0, 1] * 8)
    p = np.full(len(y), 0.25)
    estimates = calibration.estimates(y, p)
    assert estimates["calibration_in_the_large"] == pytest.approx(0, abs=1e-8)
    assert estimates["observed_expected_ratio"] == 1
    assert np.isnan(estimates["calibration_slope_estimate"])
    result = calibration.bootstrap(
        y, p, clusters=np.repeat(np.arange(16), 2), n_bootstraps=20
    )
    assert result["observed_expected_ratio_valid_bootstraps"] == 20
    assert result["calibration_slope_estimate_failed_bootstraps"] == 20
    assert (
        result["observed_expected_ratio_ci_lower"]
        <= 1
        <= result["observed_expected_ratio_ci_upper"]
    )


def test_endpoint_probabilities_and_degenerate_fits():
    values = calibration.estimates([0, 1], [0, 1])
    assert values["brier_score"] == 0
    assert values["observed_expected_ratio"] == 1
    assert np.isnan(values["calibration_slope_estimate"])
    assert calibration.estimates([0, 0], [0.2, 0.3])["observed_expected_ratio"] == 0


def test_calibration_curve_deciles_and_binary_outcomes(tmp_path):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(8)
    p = np.linspace(0.01, 0.9, 100)
    y = rng.binomial(1, p)
    summary = calibration.deciles(y, p)
    assert len(summary) == 10
    assert summary.n.sum() == 100
    assert len(calibration.deciles(y, np.full(100, 0.2))) == 1
    tables = calibration.plot_tables(
        pd.DataFrame(
            {"outcome": y, "p": p, "patient_cluster_id": np.repeat(np.arange(50), 2)}
        ),
        "outcome",
        {"Model": "p"},
        cluster_column="patient_cluster_id",
        n_bootstraps=20,
        random_seed=42,
    )
    fig = plots.plot_calibration(*tables, output_directory=tmp_path)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert len(ax.collections) == 3
    assert ax.get_aspect() == 1
    histogram = fig.axes[1]
    heights = [patch.get_height() for patch in histogram.patches]
    assert sum(v for v in heights if v > 0) == y.sum()
    assert -sum(v for v in heights if v < 0) == (1 - y).sum()
    assert tables[0].requested_bootstraps.eq(20).all()
    svg_path = tmp_path / "calibration.svg"
    pdf_path = tmp_path / "submission_figures" / "calibration.pdf"
    assert svg_path.exists()
    assert pdf_path.exists()
    first_svg = svg_path.read_bytes()
    plt.close(fig)
    repeated = plots.plot_calibration(*tables, output_directory=tmp_path)
    assert svg_path.read_bytes() == first_svg
    plt.close(repeated)


def test_curve_bootstrap_resamples_whole_patients_and_refits():
    frame = pd.DataFrame(
        {
            "patient_cluster_id": np.repeat(np.arange(20), 2),
            "outcome": [0, 1] * 20,
            "p": np.linspace(0.1, 0.9, 40),
        }
    )
    calls = []

    def refit(sample):
        assert (sample.patient_cluster_id.value_counts() % 2 == 0).all()
        calls.append(len(sample))
        return sample.p

    curve, groups = calibration.curve_intervals(
        frame,
        outcome="outcome",
        prediction="p",
        cluster_column="patient_cluster_id",
        n_bootstraps=10,
        refit=refit,
    )
    assert len(calls) == 10
    assert len(groups) == 10
    assert curve.valid_bootstraps.eq(10).all()
    assert curve[["observed", "lower", "upper"]].notna().all().all()
    assert curve.predicted.iloc[0] == frame.p.min()
    assert curve.predicted.iloc[-1] == frame.p.max()
    finite = curve.dropna(subset=["lower", "upper"])
    assert (finite.lower <= finite.upper).all()


def test_lowess_evaluates_beyond_resample_endpoints():
    p = np.linspace(0.2, 0.8, 100)
    grid = np.linspace(0.1, 0.9, 101)
    fitted = calibration._smooth(p, p, grid)
    np.testing.assert_allclose(fitted, grid, atol=1e-10)


def test_lowess_constant_predictions_do_not_invent_a_slope():
    fitted = calibration._smooth(
        np.array([0, 1]), np.array([0.5, 0.5]), np.array([0.1, 0.5, 0.9])
    )
    np.testing.assert_allclose(fitted, [np.nan, 0.5, np.nan], equal_nan=True)
