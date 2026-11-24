# RVBinary

RVBinary generates binary-star radial-velocity mocks and infers binary fractions
from multi-epoch observations through PDF construction and MCMC. It supports
DM91 and a restricted MS17 trial model, with serial and MPI workflows.

This public repository contains versioned release snapshots. Development history
is maintained separately; this snapshot is RVBinary 0.3.1.

## Installation

Requires Python 3.11–3.14. From the repository root, install into your Python
virtual environment:

```bash
git clone https://github.com/qtskyxt/RVBinary-Releases.git
cd RVBinary-Releases
python -m pip install .
```

Alternatively, create a Conda environment:

```bash
conda env create -f environment.yml
conda activate rvbinary
```

For cluster installations, MPI and parallel HDF5 requirements, see the
[HPC guide](docs/hpc.md).

## Quick start

Generate small synthetic inputs and run the complete serial workflow:

```bash
python examples/make_quickstart_data.py
rvbinary validate examples/quickstart.yaml
rvbinary run examples/quickstart.yaml
```

Results are written to `examples/results/quickstart-v1/`, including mock HDF5
files, PDFs, MCMC chains, diagnostic plots and `summary.csv`.

## Use your own observations

Copy the example configuration:

```bash
cp examples/rvbinary.yaml rvbinary.yaml
```

Edit `observations.mock_epochs` and each subsample's `members` path, set
`project.output` and `project.random_seed`, and choose the model, mock sample
count and MCMC settings. Paths are resolved relative to the YAML file. See
[input format and configuration](docs/input-format.md) for the FITS columns
and available options.

Validate and run:

```bash
rvbinary validate rvbinary.yaml
rvbinary run rvbinary.yaml
```

You can also run stages separately with `rvbinary mock`, `rvbinary pdf` and
`rvbinary mcmc`, each followed by the configuration path. Matching runs resume
completed work; use a new output directory when changing the mock/model
configuration. See the [workflow guide](docs/workflow.md) for outputs and reuse.

## Choose a population model

- **DM91 (default):** configurable `model.log10_period_mean` and
  `model.log10_period_sigma`.
- **MS17 trial:** set `model.name: ms17`; the period table is bundled, so no
  external file is required. The supported primary mass is
  `0.8 <= M1 < 1.2 M☉`, with sampled `log10(P/day)` in `[0.20, 7.98]`
  and within the table. Unsupported inputs are rejected without extrapolation.

After generating the quick-start inputs, try MS17 with:

```bash
rvbinary validate examples/ms17-trial.yaml
rvbinary run examples/ms17-trial.yaml
```

See the [model guide](docs/models/README.md) for assumptions and limitations.

## Interactive orbit generation

Open [RVBinary · Binary Orbit Lab](https://qtskyxt.github.io/RVBinary-Web/)
for browser-based orbit generation and noisy RV curves, or run the optional
local interface:

```bash
python -m pip install '.[gui]'
rvbinary gui --port 8501
```

Open <http://127.0.0.1:8501>; stop the server with Ctrl-C. Both interfaces
support Chinese/English and CSV/JSON downloads, and currently use DM91 only.
They are lightweight generators; production PDF/MCMC uses the workflow above.

## Further documentation

- [Models and scientific assumptions](docs/models/README.md)
- [Input data and configuration](docs/input-format.md)
- [Workflow, outputs and interpretation](docs/workflow.md)
- [MPI, HDF5 and scheduler scripts](docs/hpc.md)
- [Using existing Binary2 ensembles](docs/legacy-data.md)
- [Release history and migration notes](CHANGELOG.md)

Citation metadata: [CITATION.cff](CITATION.cff). License:
[BSD 3-Clause](LICENSE). Author: Tian Qiu, Shanghai Jiao Tong University.
