"""Deterministic emcee sampling for fixed RVBinary PDF models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import emcee
import numpy as np
from numpy.typing import NDArray

from rvbinary.config import MCMCConfig
from rvbinary.inference.diagnostics import ConvergenceReport, diagnose_chain
from rvbinary.inference.likelihood import mixture_log_likelihood, observed_densities
from rvbinary.pdf.histogram import PDFModel


def _highest_density_interval(
    values: NDArray[np.float64], mass: float
) -> tuple[float, float]:
    """Return the narrowest empirical interval containing the requested mass."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    interval_count = int(np.ceil(mass * ordered.size))
    if interval_count < 1 or interval_count >= ordered.size:
        return float(ordered[0]), float(ordered[-1])
    widths = ordered[interval_count:] - ordered[: ordered.size - interval_count]
    start = int(np.argmin(widths))
    return float(ordered[start]), float(ordered[start + interval_count])


def _emcee_seed(seed: int) -> int:
    digest = hashlib.blake2s(str(seed).encode("ascii"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class BinaryFractionProblem:
    """Observed statistics and one fixed binary/non-binary PDF pair."""

    observed: NDArray[np.float64]
    pdf: PDFModel
    _binary_values: NDArray[np.float64] = field(init=False, repr=False)
    _nonbinary_values: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed, dtype=np.float64)
        binary, nonbinary = observed_densities(observed, self.pdf)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "_binary_values", binary)
        object.__setattr__(self, "_nonbinary_values", nonbinary)

    def log_probability(self, parameters: NDArray[np.float64]) -> float:
        """Uniform-prior log probability for emcee."""
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (1,) or not np.isfinite(values[0]):
            return -np.inf
        fraction = float(values[0])
        if fraction < 0.0 or fraction > 1.0:
            return -np.inf
        return mixture_log_likelihood(
            fraction, self._binary_values, self._nonbinary_values
        )


@dataclass(frozen=True)
class PosteriorResult:
    """Raw and post-processed chain products with convergence state."""

    chain: NDArray[np.float64]
    log_probability: NDArray[np.float64]
    samples: NDArray[np.float64]
    sample_log_probability: NDArray[np.float64]
    acceptance_fraction: NDArray[np.float64]
    diagnostics: ConvergenceReport
    median: float
    credible_intervals: dict[float, tuple[float, float]]

    @property
    def converged(self) -> bool:
        return self.diagnostics.converged

    @property
    def exit_status(self) -> int:
        return 0 if self.converged else 1

    @property
    def legacy_fraction_summary(self) -> NDArray[np.float64]:
        mass = min(self.credible_intervals, key=lambda value: abs(value - 0.68))
        lower, upper = self.credible_intervals[mass]
        return np.array([lower, self.median, upper], dtype=np.float64)


def run_binary_fraction_mcmc(
    problem: BinaryFractionProblem,
    config: MCMCConfig,
    seed: int,
    pool: object | None = None,
) -> PosteriorResult:
    """Run a reproducible one-parameter ensemble sampler."""
    if config.burn_in >= config.steps:
        raise ValueError("mcmc burn_in must be smaller than steps")
    if config.walkers < 4:
        raise ValueError("mcmc walkers must be at least 4")
    random_state = np.random.RandomState(_emcee_seed(seed))
    initial = random_state.uniform(0.05, 0.95, size=(config.walkers, 1))
    sampler = emcee.EnsembleSampler(
        config.walkers, 1, problem.log_probability, pool=pool
    )
    sampler.random_state = random_state.get_state()
    sampler.run_mcmc(initial, config.steps, progress=False)

    chain = np.asarray(sampler.get_chain(), dtype=np.float64)
    log_probability = np.asarray(sampler.get_log_prob(), dtype=np.float64)
    acceptance = np.asarray(sampler.acceptance_fraction, dtype=np.float64)
    diagnostics = diagnose_chain(chain, log_probability, acceptance)
    burn_in = min(config.steps - 1, max(config.burn_in, diagnostics.burn_in))
    if burn_in != diagnostics.burn_in:
        ess = (
            (config.steps - burn_in)
            * config.walkers
            / diagnostics.autocorrelation_time
        )
        reasons = list(diagnostics.reasons)
        if np.any(ess < 100.0) and "effective sample size is below 100" not in reasons:
            reasons.append("effective sample size is below 100")
        diagnostics = replace(
            diagnostics,
            converged=not reasons,
            effective_sample_size=np.asarray(ess, dtype=np.float64),
            burn_in=burn_in,
            reasons=tuple(reasons),
        )
    samples = np.asarray(
        sampler.get_chain(discard=burn_in, thin=diagnostics.thin, flat=True),
        dtype=np.float64,
    )
    sample_log_probability = np.asarray(
        sampler.get_log_prob(discard=burn_in, thin=diagnostics.thin, flat=True),
        dtype=np.float64,
    )
    if samples.size == 0:
        raise RuntimeError("MCMC post-processing produced no samples")
    intervals = {
        float(mass): _highest_density_interval(samples[:, 0], float(mass))
        for mass in config.credible_intervals
    }
    return PosteriorResult(
        chain=chain,
        log_probability=log_probability,
        samples=samples,
        sample_log_probability=sample_log_probability,
        acceptance_fraction=acceptance,
        diagnostics=diagnostics,
        median=float(np.median(samples[:, 0])),
        credible_intervals=intervals,
    )
