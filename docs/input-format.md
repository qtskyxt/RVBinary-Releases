# Input Data and Configuration

RVBinary reads FITS tables through Astropy and uses `TARGETID` as the public
identifier. Input rows may be in any order; RVBinary groups them stably by ID.

## Epoch Table

`observations.mock_epochs` and optional per-subsample `epochs` files require:

| Column | Meaning | Constraint |
| --- | --- | --- |
| `TARGETID` | Star identifier | Integer representable by signed 64-bit |
| `MJD` | Observation time | Finite days |
| `VRAD` | Measured radial velocity | Finite km/s |
| `VRAD_ERR` | Quoted RV uncertainty | Positive, finite km/s |
| `Mass` | Primary mass | Constant per star, solar masses |
| `meanLOGG` | Surface gravity | Constant per star, cgs dex |

Every analyzed member star must have at least two epochs. Masked or non-finite
values are rejected.

For the MS17 trial model, every mock primary must additionally satisfy
`0.8 <= Mass < 1.2`. Out-of-domain IDs are rejected before science HDF5 writing.

## Member Table

Each `observations.subsamples[].members` file needs one unique `TARGETID` row per
member. For the DESI layout, `loa_mem.fits` is the member table and
`loa_mult.fits` can be supplied as the subsample `epochs` file. When `epochs` is
omitted, observations are selected from `mock_epochs`.

## Configuration

Paths are resolved relative to the YAML file. The main controls are:

```yaml
model:
  name: dm91
  log10_period_mean: 4.8
  log10_period_sigma: 2.3
  include_unrestricted: true
  period_limits_days: [10, 100, 1000, 10000]

mock:
  samples: 10000
  error_floor:
    values_kms: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    prior: uniform
```

`log10_period_mean` and `log10_period_sigma` parameterize the DM91 period
distribution in log10 days. For a finite period limit, binary mock stars above
the limit use their matched non-binary uncertainty realization when the PDF is
built. The original mock table is unchanged, so many limits can reuse it.

### MS17 trial configuration

Replace the DM91 model block with:

```yaml
model:
  name: ms17
  environment: none  # optional: umi_legacy
  include_unrestricted: true
  period_limits_days: [100, 1000]
```

`period_table` is optional: omitted or `null` uses the checksum-verified
`rvbinary/data/Pdist_small.dat` shipped in the installed package. To supply a
replacement, set `period_table: ./my_trial_table.dat`; relative overrides resolve
against the YAML directory and must satisfy all trial grid/domain checks.
The table's broader mass grid is not a claim of full MS17 support. The implemented
mass domain remains `[0.8, 1.2)` solar masses; sampled log periods remain within
both the table and `[0.20, 7.98]`. `log10_period_mean` and `log10_period_sigma`
are DM91-only and are rejected for MS17. `none` does not impose the old UMi
encounter cutoff; `umi_legacy` adds it explicitly. Both models keep primary Roche
clearance and shared-epoch noise. See [model details](models/README.md).

### Shared controls

`error_floor.values_kms` defines a discrete grid. The floor is added in
quadrature to the quoted epoch uncertainty. The current public prior is uniform
over the listed grid points.

To reuse a legacy Binary2 ensemble whose beta products already include each
star's error floor, configure:

```yaml
model:
  include_unrestricted: true
  period_limits_days: []

mock:
  samples: 10000
  error_floor:
    values_kms: [0.0]
  precomputed:
    format: binary2
    directory: /path/to/legacy/mock-directory
    error_floor_mode: per_star_baked_in
```

Precomputed Binary2 inputs cannot be relabelled with `model.name: ms17`.
This mode does not add `0.0` as another error term. The single placeholder
value disables the normal grid while the output metadata records
`per_star_baked_in`. Period limits and additional error-floor grid points are
rejected because the legacy beta-only files cannot reproduce those variations.

`statistics.mode: consistent` is recommended. `legacy` retains the historical
Binary2 statistic behavior for comparisons; artifacts from different modes
cannot be mixed. Run `rvbinary validate CONFIG` before compute jobs. Unknown
configuration keys are errors.
