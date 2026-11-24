from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from rvbinary.config import MCMCConfig
from rvbinary.inference.results import POSTERIOR_SCHEMA, write_posterior
from rvbinary.inference.sampler import BinaryFractionProblem, run_binary_fraction_mcmc
from rvbinary.pdf.histogram import PDFModel


def test_unconverged_result_is_saved_before_nonzero_status(tmp_path: Path) -> None:
    pdf = PDFModel(
        edges=np.array([1.0, 2.0, 3.0]),
        binary_density=np.array([0.25, 0.75]),
        nonbinary_density=np.array([0.75, 0.25]),
        binary_counts=np.array([1, 3]),
        nonbinary_counts=np.array([3, 1]),
    )
    problem = BinaryFractionProblem(np.array([1.2, 1.4, 2.2]), pdf)
    config = MCMCConfig(walkers=10, steps=20, burn_in=5)
    result = run_binary_fraction_mcmc(problem, config, seed=42)
    path = tmp_path / "posterior.h5"

    write_posterior(path, result, {"subsample": "FEH0", "error_floor_kms": 0.4})

    assert path.is_file()
    assert result.converged is False
    assert result.exit_status == 1
    assert not list(tmp_path.glob(".*.tmp"))
    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema"] == POSTERIOR_SCHEMA
        assert bool(handle.attrs["completion"])
        assert not bool(handle.attrs["converged"])
        assert json.loads(str(handle.attrs["metadata"]))["subsample"] == "FEH0"
        assert set(("samples", "logl", "frac", "chain", "diagnostics", "summary")) <= set(
            handle
        )
        np.testing.assert_array_equal(handle["chain/values"], result.chain)
        np.testing.assert_array_equal(handle["chain/log_probability"], result.log_probability)
        assert handle["samples"].shape[1] == 1
        assert handle["frac"].shape == (3,)
