# Hodge-SDE — code and fitted-model candidate

Updated 2026-10-16. The training core uses validated exact Hodge projections (`exact_hodge_v1`), and the three parameter-only fitted HCP examples now come from validated exact formal fits. Current main-figure adapters cover Fig2 v12, Fig3 exact_estimator_v2, Fig4 v10, Fig5 v18 and Fig6 v15 with separate hash-locked external inputs. Branch QA and package acceptance are recorded separately in [VALIDATION.md](VALIDATION.md). Subject datasets/tables, observed or predicted time series, clinical records, identifiers, fonts and atlas assets remain outside the package. A public license has not been assigned; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Run a fitted model

From this folder in MATLAB R2022b:

```matlab
trajectory = demo_fitted_model(150);
trajectory = demo_fitted_model(250,false); % deterministic drift
trajectory = demo_fitted_model(450);
test_parameter_export;
```

These are trained exact-model numerical parameters, with reproducible synthetic initial states. The parameter-only simulator needs neither subject data nor the original project loader. The slot order is Sym/Grad/Curl/Harm; the third antisymmetric slot (fourth overall) is exact Harm. [Model contents and additional exports](models/README.md).

## Quick checks

```sh
python -m pip install -r requirements-figures.txt
python -B -m figures.check
python -B -m figures.validation_smoke
python -B -m figures.exact_smoke
```

```matlab
demo_synthetic(false);
test_whiteness_synthetic;
```

[VALIDATION.md](VALIDATION.md) records executed checks. Dependency versions reflect the tested local environment; a fresh installation remains to be tested.

## Current main-figure entrypoints

Set `HODGE_WORK_ROOT` to a separate input/output folder and `HODGE_FONT_FILE` to a lawfully installed Helvetica/Helvetica Neue file. Prepare the external inputs, then run from the package root:

| Figure | Version | Entry |
|---|---|---|
| Fig2 | v12 exact, main core | `python -B -m figures.fig2.render_exact` |
| Fig3 | exact_estimator_v2; frozen corrected truth | `python -B -m figures.fig3.render_exact` |
| Fig4 | v10; exact Harm a–d, original e–g | `python -B -m figures.fig4.render_exact` |
| Fig5 | v18 exact; 843 evaluation pairs | `python -B -m figures.fig5.render_exact` |
| Fig6 | v15 exact_disease_v2, main core | `python -B -m figures.fig6.render_exact` |

Instructions: [Fig2](figures/fig2/README.md), [Fig3](figures/fig3/README_zh.md), [Fig4](figures/fig4/README.md), [all input contracts](figures/INPUTS.md).

Each current entry supports `--check-inputs` without rendering. Historical Fig3 `v13_pipeline` and Fig4 `export_editable`/`render_main` remain legacy workflows, not defaults for current scientific results. Current Fig3 consumes completed exact-estimator outputs without rebuilding truth or statistics; current Fig4 labels a–d Harm and keeps the original e–g experiment distinct from current-model ablation.

Fig2 d and upstream S10 retain the nine-model **Archived workflow timing** call totals. New exact train/test-only timings remain separate source/Methods records; they are not compared as cross-model speed against historical calls that include NBE.

The new ports cover main-figure cores. [Exact input contracts and final S01-S44 numbering](figures/EXACT_INPUTS.md) distinguish fullfit results from historical sensitivity analyses. Existing Fig5 `render_si` and Fig6 `render_si` retain historical branch numbers and inputs; they do not generate the newly numbered complete SI. Fig2/Fig4 export from plotting objects rather than replacing images inside old PDFs. Fig3 links strict synthetic truth, fit caches, selected models and statistics to their provenance. Keep data-containing work folders outside this repository.

The archived validation run instructions and external-data contracts are in [Fig. 5 validation](figures/fig5/validation/README.md) and [Fig. 6 validation](figures/fig6/validation/README.md). Their original/validation labels describe the historical analyses. New training uses `src/hlgsde_exact_hodge.m`: one gauge-fixed graph Laplacian solve and a direct sparse edge-by-triangle least-squares projection, followed by independent numerical acceptance checks. Failed solves raise an error. Old trained models must not be relabelled as exact models; the transition requires separate fits, evaluations and figure inputs.

## Model and evaluation interfaces

- Training: `src/nonlinear_HLGSDE.m`, with separately acquired Hodge and rest-system-id Utilities.
- Parameter export/inference: `src/export_hodge_parameters.m`, `src/simulate_hodge_parameters.m`.
- Evaluation: `src/compute_dynamics_metrics.m`, `score_free_run_by_fold.m`, `audit_segment_status.m`.
- Fair generation: use the Fig2 adapter, including Subspace joint noise and training-residual NMM noise. The legacy `generate_free_run_fold.m` alone is not the seven-model fair-generation entrypoint.
- NBE whiteness: `src/rescore_whiteness_nbe.m` checks and reaggregates external five-fold Q/threshold caches as `mean(Q)/mean(Qthr)`. The original NBE statistic/permutation implementation is external. Archived FigS11 retains its separately documented mean-of-fold-ratios convention.

Historical whole-scan standardization, statistical units and existing analysis settings are preserved. This synchronization exports three already fitted exact models; it does not train models. Keep new exact-model caches separate from archived figure inputs. Fig6 formal computation sources are byte-locked by `figures/fig6/COMPUTATION_LOCK.json`; portable adapters are separate files.

## Provenance and release

[manifest.json](manifest.json) records original-source provenance hashes separately from each current package file hash. Frozen `B_SOURCE_SHA256` / `DYNAMICS_SOURCE_SHA256` values remain provenance keys for compatible historical checkpoints; new run fingerprints also record the actual installed package-module hashes. [Chinese instructions](README_zh.md). Readers obtain HCP, COBRE, MDD and AD data independently under the applicable terms. Before public release, authors must approve code and fitted-parameter redistribution rights, choose a license and test installation with authorized inputs. No public upload has been made.
