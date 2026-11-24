"""Atomic versioned posterior HDF5 output with legacy datasets."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from rvbinary import __version__
from rvbinary.inference.sampler import PosteriorResult

POSTERIOR_SCHEMA = "rvbinary.posterior/v1"


def write_posterior(
    path: Path, result: PosteriorResult, metadata: Mapping[str, object]
) -> None:
    """Atomically save raw chains, diagnostics, summaries, and legacy arrays."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = POSTERIOR_SCHEMA
            handle.attrs["package_version"] = __version__
            handle.attrs["created_at"] = datetime.now(UTC).isoformat()
            handle.attrs["completion"] = False
            handle.attrs["converged"] = result.converged
            handle.attrs["exit_status"] = result.exit_status
            handle.attrs["metadata"] = json.dumps(
                dict(metadata), sort_keys=True, separators=(",", ":"), default=str
            )

            handle.create_dataset("samples", data=result.samples)
            handle.create_dataset("logl", data=result.sample_log_probability)
            handle.create_dataset("frac", data=result.legacy_fraction_summary)

            chain = handle.create_group("chain")
            chain.create_dataset("values", data=result.chain)
            chain.create_dataset("log_probability", data=result.log_probability)
            diagnostics = handle.create_group("diagnostics")
            diagnostics.create_dataset(
                "acceptance_fraction", data=result.acceptance_fraction
            )
            diagnostics.create_dataset(
                "autocorrelation_time",
                data=result.diagnostics.autocorrelation_time,
            )
            diagnostics.create_dataset(
                "effective_sample_size",
                data=result.diagnostics.effective_sample_size,
            )
            diagnostics.attrs["acceptance_mean"] = result.diagnostics.acceptance_mean
            diagnostics.attrs["burn_in"] = result.diagnostics.burn_in
            diagnostics.attrs["thin"] = result.diagnostics.thin
            diagnostics.attrs["reasons"] = json.dumps(result.diagnostics.reasons)

            summary = handle.create_group("summary")
            masses = np.array(sorted(result.credible_intervals), dtype=np.float64)
            summary.create_dataset("credible_mass", data=masses)
            summary.create_dataset(
                "lower",
                data=np.array(
                    [result.credible_intervals[mass][0] for mass in masses],
                    dtype=np.float64,
                ),
            )
            summary.create_dataset(
                "upper",
                data=np.array(
                    [result.credible_intervals[mass][1] for mass in masses],
                    dtype=np.float64,
                ),
            )
            summary.attrs["median"] = result.median
            handle.flush()
            handle.attrs.modify("completion", True)
            handle.flush()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
