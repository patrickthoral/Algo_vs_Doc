# Analysis results

- `summaries/`: aggregate estimates and descriptive tables used by notebook 5.
- `plot_data/`: calibration summaries and paired contrast bootstrap distributions used by notebook 5.
- `bootstrap/`: individual resampling records.
- `individual_cases/`: case-level inputs used by notebook 5.
- `submission_figures/`: generated submission PDFs.
- `word_figures/`: SVG copies prepared for Word documents.
- `executed_notebooks/`: Papermill execution copies.

Generated files under `results/` remain local and are excluded from Git. Notebooks 2-4 create
analysis results. Notebook 5 also requires the local analysis datasets and case-level inputs; it
writes figures to `figures/` and Word tables to `tables/`. The AUROC and AUPRC comparison matrices
read `plot_data/<cohort>/auroc_contrast_bootstraps.csv` and
`plot_data/<cohort>/auprc_contrast_bootstraps.csv`. Internal `Combined` directories identify the
cohort displayed as Both Hospitals.

`individual_cases/low_risk_event_diagnosis_annotations.csv` contains manually reviewed
annotations. Other case entries can be regenerated, but these annotations cannot be reconstructed
from the analysis notebooks. Preserve the annotation file in an access-controlled, versioned
location before rerunning notebook 5. Do not commit case-level files to this public repository.
