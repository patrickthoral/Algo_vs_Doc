import pandas as pd

from algo_vs_doc import bootstrap_storage


def test_bootstrap_storage_writes_csv_with_reproducibility_metadata(tmp_path):
    bootstrap_storage.configure(tmp_path, random_seed=123, n_bootstraps=2)
    bootstrap_storage.persist(
        "Combined: model",
        "auroc",
        [{"value": 0.7}, {"value": 0.8}],
    )

    metadata = pd.read_csv(tmp_path / "run_metadata.csv")
    samples = pd.read_csv(tmp_path / "combined_model__auroc.csv")

    assert metadata.to_dict("records") == [
        {"random_seed": 123, "requested_bootstraps": 2}
    ]
    assert samples["replicate"].tolist() == [0, 1]
    assert samples["random_seed"].tolist() == [123, 123]
