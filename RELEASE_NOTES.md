# RVBinary v0.3.1

First public release snapshot, with installation archives and usage documentation.
The development repository and its history remain separate.

- DM91 mock generation, configurable period distributions, PDF construction and MCMC.
- Restricted MS17 trial: 0.8 <= M1 < 1.2 solar masses, bundled verified period table, no extrapolation.
- Shared per-epoch measurement noise, reusable HDF5 mock storage and serial/MPI workflows.
- Optional local orbit-generation interface and a separate public browser tool.

Python 3.11–3.14 is supported. Install the wheel with `python -m pip install rvbinary-0.3.1-py3-none-any.whl`, or follow README.md to install from source.

Validation: 295 non-MPI tests passed; Ruff checks passed; wheel and source archives passed Twine metadata checks.
MPI execution requires a compatible cluster environment; MPI was not rerun during this release preparation.
Existing scientific outputs are not upgraded automatically. Use a new output directory when changing package versions or population models.

License: BSD 3-Clause. See CITATION.cff for software citation metadata.
