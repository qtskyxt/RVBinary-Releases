# Population model guide

The production Python workflow supports DM91 and the restricted MS17 trial
implementation. The website and lightweight Python GUI currently use DM91 only.

## DM91

Use `name: dm91` (also the default for older unnamed model blocks).
`log10_period_mean` and `log10_period_sigma` configure its log-normal period
population. The MS17 addition does not change DM91 draws or formulas.

## MS17 trial

Use `name: ms17`. As of Python 0.3.1, Pdist_small.dat is bundled in wheel and
sdist, so `period_table` is optional and no Gravity path is required. An explicit
external override remains supported. The bundled trial table was supplied by Tian Qiu; its SHA256 is
490592bec1d0a2c9708053102c38f0d105bb7c8230c23566d41e3a0faca73c50.

The table has 140400 rows and five columns, with total period density in the
last column. Its mass grid 0.80–7.98 solar masses and log-period grid 0.20–7.98
both have spacing 0.02. **Only 0.8 <= M1 < 1.2 solar masses is supported by the
implemented low-mass twin prescription.** The table is trial data and does not
provide a full-domain MS17 calibration. No extrapolation or silent edge clamping
is performed. Invalid or zero-supported probability grids and unsupported stars
are rejected during preflight.

Period sampling follows nearest-logmass/logperiod grid densities integrated over
period cells. The q distribution includes matched low/high power laws and a twin
excess above q=0.95. Eccentricity uses a truncated period-dependent power law;
P<=2 days is circular. Final orbits are checked against primary Roche clearance.
These are corrections to the supplied historical sampler, not bitwise
reproduction of its global NumPy random stream.

`environment: none` is the default. `umi_legacy` adds the historical UMi encounter
maximum-axis limit; it is independent of whether the table is bundled or external.
Neither option expands the trial support.

Both populations retain the per-epoch shared-noise correction introduced in 0.2.2.
The filename/key BETA and Python pairwise_beta naming are retained for API
compatibility; the model addition does not alter statistic reductions.

## Reproducibility and migration

HDF5 source hashes include `ms17-period-table`; scientific model digests include
algorithm identity, table content hash, domains, interpolation and environment.
Bundled and external copies with identical bytes produce identical model
provenance. Changed model/table/environment invalidates native caches; different
native package versions also cannot be appended together. Use a new output
folder when switching models or package versions. Existing outputs are never
converted automatically. Explicit legacy Binary2 input remains a historical
reader and cannot be claimed as newly generated MS17 data.

Start with [the runnable example](../../examples/ms17-trial.yaml) and
[configuration reference](../input-format.md).
