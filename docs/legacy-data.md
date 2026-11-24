# Legacy Binary2 Data

RVBinary can inspect and read the historical Binary2 HDF5 organizations used by
`mcs_DM91_table_v4.py`, `mcs_DM91_table_v6.py`, and related PDF scripts. It does
not modify those source files or legacy artifacts.

Inspect a file before use:

```bash
rvbinary inspect /path/to/mcs_table_combined.h5
rvbinary inspect /path/to/mcs_beta_combined.h5
rvbinary inspect /path/to/nberr_combined.h5
```

The inspector recognizes native RVBinary stores and legacy `table_N`, `list_N`,
or `arrays` layouts. It reports sample coverage, dataset shapes, broken links,
duplicates, malformed records, completion, and whether an artifact is usable.
Legacy files are read-only and are not considered resumable mock-generation
artifacts because they lack RVBinary provenance and atomic completion metadata.

## Direct unrestricted analysis

A directory containing exactly these legacy products can drive PDF and MCMC
stages without conversion or copying:

```text
mcs_table_combined.h5
mcs_beta_combined.h5
mcs_nberr_combined.h5
```

Set `mock.precomputed.format: binary2`, point `mock.precomputed.directory` at
that directory, set `mock.samples` to its complete `list_N` count, and leave the
period model unrestricted. `rvbinary pdf` reads the star order from
`table_0/ID` (or `TARGETID`) once, then distributes `list_N/arrays` samples
across MPI ranks. Only fixed-size histogram counts are reduced to rank zero.
The source HDF5 files are never modified.

The historical PDF scripts excluded star statistics outside their fixed
histogram support before density normalization. The Binary2 adapter preserves
that behavior explicitly and stores `binary_out_of_range_count` and
`nonbinary_out_of_range_count` in `pdf.npz`; native RVBinary mock processing
continues to reject out-of-range values.

When those legacy beta and non-binary products already contain a per-star error
floor, use `error_floor_mode: per_star_baked_in`. RVBinary will not apply or
marginalize another error-floor grid. Output PDFs and summaries preserve this
mode explicitly.

To avoid hashing roughly 159 GB before every resumed run, source validation uses
each legacy file's size and nanosecond modification time. Replacing content
while preserving both values can evade this check; use a new output directory
or `--force` after such an unusual in-place replacement.

For new mock ensembles, generate a native mock store from the epoch FITS table. This
adds deterministic seed metadata, model/configuration digests, statistic mode,
source hashes, recoverable shards, and append-new-star support. Existing
`TARGETID` values are preserved; the generated mock sample number is an internal
sample ID and is not a replacement for the stellar identifier.

When validating against historical results, set `statistics.mode: legacy` and
keep the same period distribution, error grid, histogram bins, and random seed.
Do not combine `legacy` and `consistent` artifacts: RVBinary rejects a statistic
mode mismatch before PDF output is written.
