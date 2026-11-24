from __future__ import annotations

import numpy as np
import pytest

from rvbinary.inference.evidence import (
    discrete_credible_interval,
    marginal_log_evidence,
    normalize_discrete_posterior,
)
from rvbinary.inference.likelihood import log_likelihood
from rvbinary.pdf.histogram import PDFModel


def equal_pdf() -> PDFModel:
    return PDFModel(
        edges=np.array([1.0, 2.0, 3.0]),
        binary_density=np.array([0.5, 0.5]),
        nonbinary_density=np.array([0.5, 0.5]),
        binary_counts=np.array([1, 1]),
        nonbinary_counts=np.array([1, 1]),
    )


def test_equal_mixture_models_have_exact_uniform_prior_evidence() -> None:
    observed = np.array([1.2, 2.2, 1.8])
    pdf = equal_pdf()

    result = marginal_log_evidence(observed, pdf, "uniform", quadrature_points=257)

    assert result == pytest.approx(log_likelihood(0.37, observed, pdf))


def test_equal_evidence_recovers_discrete_errorfloor_prior() -> None:
    posterior = normalize_discrete_posterior(
        np.zeros(3), np.array([1.0, 2.0, 1.0])
    )

    np.testing.assert_allclose(posterior, np.array([0.25, 0.5, 0.25]))


def test_discrete_posterior_normalization_is_stable_for_large_logs() -> None:
    posterior = normalize_discrete_posterior(
        np.array([10000.0, 10001.0]), np.ones(2)
    )

    np.testing.assert_allclose(posterior, np.exp([0.0, 1.0]) / np.exp([0.0, 1.0]).sum())


def test_discrete_equal_tailed_credible_interval_sorts_grid() -> None:
    interval = discrete_credible_interval(
        np.array([2.0, 0.0, 1.0]), np.array([0.2, 0.2, 0.6]), mass=0.6
    )

    assert interval == (0.0, 1.0)


def test_evidence_rejects_invalid_prior_density() -> None:
    with pytest.raises(ValueError, match="prior"):
        marginal_log_evidence(
            np.array([1.5]), equal_pdf(), lambda grid: -np.ones_like(grid)
        )
