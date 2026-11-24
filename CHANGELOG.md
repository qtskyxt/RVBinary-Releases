# Changelog

All notable changes to RVBinary are documented here.

## 0.3.1 — 2026-11-23

- Bundle the supplied Pdist_small.dat in wheel and sdist; default MS17 configs
  need no external table path. Retain optional external overrides and strict
  trial domains, and verify the bundled resource checksum at load.

## 0.3.0 — 2026-11-23

- Add an explicitly restricted MS17 trial model with external period-table
  validation, strict 0.8 <= M1 < 1.2 solar-mass support, no extrapolation,
  normalized period/q/twin/e sampling and bounded Roche validation.
- Select `model.name: ms17` in production serial/MPI pipelines; DM91 and its
  optional mean/sigma settings retain their behavior and scientific digest.
- Record table contents in provenance and cache identity; default to no UMi
  encounter cutoff, with explicit `environment: umi_legacy` for historical use.
- Keep lightweight GUI and the separate website DM91-only. Use a new output
  directory for the new package version; do not mix population models.

## 0.2.2 — 2026-10-18

- Fix production binary and nonbinary mock noise: draw one Gaussian velocity
  perturbation per epoch and reuse it in all observation pairs, preserving
  shared-epoch correlations for stars with three or more epochs.
- Mark the noise algorithm as `epoch-shared/v1` in scientific configuration
  digests. Reject old native shards on append, recovery and PDF creation, and
  reject old generated-workflow manifests on resume.
- Existing mock/PDF/posterior products are not upgraded in place. Regenerate in
  a new output directory; the explicit precomputed Binary2 reader remains a
  legacy-data path and does not correct historical noise.
- Keep orbit generation, reported-error denominators, error-floor convention
  and `consistent`/`legacy` statistic reductions unchanged.

## 0.2.1 — 2026-10-10

- Add reproducible Gaussian RV measurements with configurable noise, observation count and seed, 1σ error bars and observation CSV export.
- Add Chinese/English interface selection while preserving generated results and settings.

## 0.2.0 - 2026-10-10

- Add an optional Streamlit browser interface (`rvbinary gui`) for lightweight,
  reproducible DM91 orbit populations, distributions, single-orbit RV curves,
  full CSV downloads and JSON configuration/provenance downloads.
- Add a framework-independent `rvbinary.mock.generator` API. Orbit count is
  independent of production pipeline `mock.samples` and limited to 10,000.
- Add a bounded, reflected-angle bisection fallback for high-eccentricity
  Kepler solutions that Newton iteration misclassified at floating-point
  precision, without introducing a SciPy dependency.
- Update citation metadata to match the package version.

## 0.1.2 - 2026-09-10

- Allow scheduler scripts to source an explicit Conda initialization file for
  non-interactive PBS and Slurm shells.

## 0.1.1 - 2026-09-08

- Add direct, read-only PDF generation from legacy Binary2
  `mcs_*_combined.h5` mock ensembles without copying the source data.
- Add MPI sample streaming for precomputed unrestricted PDFs and scheduler
  scripts that skip mock generation before serial MCMC.
- Record per-star baked-in error-floor provenance separately from a discrete
  RVBinary error-floor grid.

## 0.1.0 - 2026-09-06

- Add deterministic DM91 orbit simulation with configurable log-period mean
  and uncertainty.
- Add reusable, appendable HDF5 mock stores with external-link indexes,
  recovery, inspection, compaction, and legacy Binary2 readers.
- Add unrestricted and multiple upper-period models without regenerating mocks.
- Add a discrete error-floor grid, evidence weighting, and conditional
  error-floor probabilities.
- Add MCMC binary-fraction inference, convergence diagnostics, CSV/JSON
  summaries, and sensitivity plots.
- Add serial, MPI shard, MPI PDF-reduction, and MPI-enabled HDF5 modes.
- Add Slurm and PBS templates and Python 3.11-3.14 packaging metadata.
