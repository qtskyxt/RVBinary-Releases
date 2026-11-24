# Workflow and Interpretation

## Model selection

Production mock generation supports DM91 (default) and the restricted MS17 trial
model. MS17 defaults to its bundled period table; no external data path is needed.
Its strict mass range is 0.8 <= M1 < 1.2 solar masses; sampled log-periods stay
inside both the table and [0.20, 7.98]. See [model guide](models/README.md).
Changing population, table contents or environment invalidates mock/PDF/workflow
cache identities. Use a new output directory rather than mixing results.

## Stages

1. `rvbinary mock CONFIG` simulates deterministic binary orbits and paired
   non-binary noise realizations. One mock ensemble covers every subsample.
2. `rvbinary pdf CONFIG` selects `TARGETID` members and builds every configured
   period-limit and error-floor PDF.
3. `rvbinary mcmc CONFIG` infers the binary fraction for every PDF and computes
   evidence over binary fraction.
4. `rvbinary run CONFIG` executes or resumes all stages serially.

With `mock.precomputed.format: binary2`, the mock stage becomes a no-op and only
the unrestricted PDF is supported. Scheduler templates run MPI PDF generation
followed by serial `rvbinary mcmc`; they do not regenerate or copy the legacy
mock ensemble.

Random streams are derived from `project.random_seed`, sample index, target ID,
and realization type. Results therefore do not depend on MPI rank count or
partitioning. `run.yaml` records the resolved configuration and `manifest.json`
records artifact digests. A matching rerun resumes without rewriting complete
outputs. Use `--force` to rebuild downstream products under the same mock
configuration. Use a new `project.output` directory when changing mock/model
configuration so the provenance of the existing ensemble remains intact.

## Reusing and Extending Mocks

Generate the full observation parent sample once, then list each scientific
subsample in the YAML. PDF generation selects the requested rows by `TARGETID`.
To add stars without changing existing mock shards:

```bash
mpiexec -n 16 rvbinary mock rvbinary.yaml --extend-stars new_epochs.fits
```

If a job was interrupted after writing shards but before rebuilding indexes:

```bash
rvbinary mock rvbinary.yaml --merge-only
```

This operation creates lightweight HDF5 external-link indexes; it does not copy
all datasets. Use `rvbinary compact INDEX OUTPUT` only when a single physical
file is required, because compaction must copy the full payload once.

## Outputs

The output tree contains reusable mock shards and indexes, one `pdf.npz` and
`posterior.h5` per grid point, `summary.json` beside each posterior, global
`summary.csv`, diagnostic figures, `run.yaml`, and `manifest.json`.

The reported binary-fraction interval is derived from the saved MCMC samples.
Check `converged`, acceptance statistics, and the saved chain before scientific
interpretation. Unless `mcmc.allow_unconverged` is true, unconverged runs retain
their diagnostic products but return a nonzero process status.

For each subsample and period limit, error-floor probabilities normalize the
model evidence across `error_floor.values_kms`. They are conditional on that
finite grid, its configured prior, the RV model, and the histogram support.
They must not be interpreted as a continuous posterior outside the grid.

Comparing binary-fraction posteriors across `period_limits_days` measures model
sensitivity to orbital periods: stars above each upper limit are represented by
the paired non-binary realization during PDF construction.
