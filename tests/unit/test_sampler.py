from __future__ import annotations

import numpy as np
import pytest

from rvbinary.config import MCMCConfig
from rvbinary.inference.diagnostics import diagnose_chain
from rvbinary.inference.sampler import BinaryFractionProblem, run_binary_fraction_mcmc
from rvbinary.pdf.histogram import PDFModel


def simple_problem() -> BinaryFractionProblem:
    pdf = PDFModel(
        edges=np.array([1.0, 2.0, 3.0]),
        binary_density=np.array([0.2, 0.8]),
        nonbinary_density=np.array([0.8, 0.2]),
        binary_counts=np.array([2, 8]),
        nonbinary_counts=np.array([8, 2]),
    )
    return BinaryFractionProblem(np.array([1.2, 1.4, 2.2, 2.5]), pdf)


def test_seeded_sampler_is_reproducible() -> None:
    config = MCMCConfig(walkers=12, steps=80, burn_in=10)

    first = run_binary_fraction_mcmc(simple_problem(), config, seed=81)
    second = run_binary_fraction_mcmc(simple_problem(), config, seed=81)

    np.testing.assert_array_equal(first.chain, second.chain)
    np.testing.assert_array_equal(first.log_probability, second.log_probability)
    np.testing.assert_array_equal(first.samples, second.samples)


def test_short_chain_is_reported_as_unconverged() -> None:
    chain = np.linspace(0.2, 0.8, 40).reshape(10, 4, 1)
    log_probability = np.zeros((10, 4))

    report = diagnose_chain(chain, log_probability, np.full(4, 0.5))

    assert report.converged is False
    assert report.burn_in >= 0
    assert report.thin >= 1
    assert report.reasons


def test_acceptance_fraction_below_point_nine_is_accepted() -> None:
    rng = np.random.default_rng(42)
    chain = rng.normal(0.4, 0.05, size=(5000, 32, 1))
    log_probability = np.zeros((5000, 32))

    report = diagnose_chain(chain, log_probability, np.full(32, 0.85))

    assert report.converged is True
    assert not any("acceptance fraction" in reason for reason in report.reasons)


def test_allow_unconverged_does_not_change_scientific_status() -> None:
    config = MCMCConfig(
        walkers=10, steps=20, burn_in=5, allow_unconverged=True
    )

    result = run_binary_fraction_mcmc(simple_problem(), config, seed=12)

    assert result.converged is False
    assert result.exit_status != 0


def test_sampler_recovers_known_mixture_fraction() -> None:
    pdf = PDFModel(
        edges=np.array([1.0, 2.0, 3.0]),
        binary_density=np.array([0.1, 0.9]),
        nonbinary_density=np.array([0.9, 0.1]),
        binary_counts=np.array([1, 9]),
        nonbinary_counts=np.array([9, 1]),
    )
    observed = np.concatenate((np.full(30, 1.5), np.full(70, 2.5)))
    config = MCMCConfig(walkers=20, steps=500, burn_in=100)

    result = run_binary_fraction_mcmc(
        BinaryFractionProblem(observed, pdf), config, seed=-19
    )

    assert result.median == pytest.approx(0.75, abs=0.12)
