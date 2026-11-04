# Shared bootstrap execution

Notebook 4 initializes a run-scoped coordinator in its settings cell. Each
ordered patient-membership vector, bootstrap count, and seed has one stored
resampling plan. All metrics reuse that plan rather than drawing patients again.
Eligibility filtering remains in the existing analysis functions. Plans with
different ordered patient memberships are kept separate.

The coordinator also caches grouped out-of-fold combination predictions by all
input values, patient groups, input order, cross-validation settings, and seed.
The nested performance, exploratory contrasts, and calibration-curve analyses
therefore reuse identical fits. Changed inputs or settings require new fits.
Copies of prediction arrays are returned so callers cannot modify the cache.

Metric collection and interval summarization remain in separate functions and
notebook cells. Iterating over stored draws is not another resampling run.
LOWESS still needs evaluation for each predictor and replicate when collecting
curve bands; discrimination metrics do not require LOWESS.

The canonical notebook bootstrap count and seed apply to bootstrap analyses in notebook 4. The
coordinator uses numpy.random.default_rng to generate shared resampling plans. This replaces the
previous AUROC RandomState sequence, so bootstrap intervals and p-values can
differ from earlier exports even with the same seed. Fixed-selection analyses
use original-order draws and rank within each resample. Eligibility,
point-estimate rules, and existing interval formulas are unchanged.

Notebook 5 reads exported summaries and bootstrap plot data without rerunning bootstrap inference.
It separately fits the two grouped, out-of-fold combination models to prepare prediction plots.
Run notebook 4 from its settings cell to start a fresh coordinated run. Running
individual metric cells in the same kernel reuses the available plan and fits.
Re-executing the settings cell resets the cache deliberately.

The cache is local to one Python process and analysis run; it is not a persistent
cache of trial data. Only patient draws and prediction arrays are retained.
