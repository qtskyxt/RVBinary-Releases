"""Likelihood, evidence, and posterior inference for RVBinary."""

from rvbinary.inference.evidence import (
    discrete_credible_interval,
    marginal_log_evidence,
    normalize_discrete_posterior,
)
from rvbinary.inference.likelihood import log_likelihood
from rvbinary.inference.sampler import (
    BinaryFractionProblem,
    PosteriorResult,
    run_binary_fraction_mcmc,
)

__all__ = [
    "discrete_credible_interval",
    "BinaryFractionProblem",
    "log_likelihood",
    "marginal_log_evidence",
    "normalize_discrete_posterior",
    "PosteriorResult",
    "run_binary_fraction_mcmc",
]
