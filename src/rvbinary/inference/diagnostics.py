"""Convergence diagnostics for one-dimensional ensemble chains."""

from __future__ import annotations

from dataclasses import dataclass

import emcee
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConvergenceReport:
    """Saved chain diagnostics and recommended post-processing parameters."""

    converged: bool
    autocorrelation_time: NDArray[np.float64]
    effective_sample_size: NDArray[np.float64]
    acceptance_mean: float
    burn_in: int
    thin: int
    reasons: tuple[str, ...]


def diagnose_chain(
    chain: NDArray[np.float64],
    log_prob: NDArray[np.float64],
    acceptance: NDArray[np.float64],
) -> ConvergenceReport:
    """Estimate autocorrelation and enforce explicit convergence criteria."""
    values = np.asarray(chain, dtype=np.float64)
    probabilities = np.asarray(log_prob, dtype=np.float64)
    accepted = np.asarray(acceptance, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 1:
        raise ValueError("chain must have shape (steps, walkers, 1)")
    steps, walkers, _ = values.shape
    if probabilities.shape != (steps, walkers):
        raise ValueError("log_prob must match the chain step and walker axes")
    if accepted.shape != (walkers,) or not np.all(np.isfinite(accepted)):
        raise ValueError("acceptance must contain one finite value per walker")
    if np.any((accepted < 0.0) | (accepted > 1.0)):
        raise ValueError("acceptance fractions must be within [0, 1]")
    if not np.all(np.isfinite(values)) or np.any(np.isnan(probabilities)):
        raise ValueError("chain and log probabilities must not contain invalid values")

    reasons: list[str] = []
    try:
        tau = np.asarray(emcee.autocorr.integrated_time(values, quiet=False), dtype=np.float64)
    except emcee.autocorr.AutocorrError as exc:
        tau = np.asarray(exc.tau, dtype=np.float64)
        reasons.append("chain is shorter than 50 autocorrelation times")
    if tau.shape != (1,) or not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        tau = np.full(1, np.inf)
        reasons.append("autocorrelation time is not finite")

    finite_tau = tau[np.isfinite(tau)]
    if finite_tau.size:
        burn_in = min(steps - 1, max(0, int(np.ceil(2.0 * np.max(finite_tau)))))
        thin = max(1, int(np.floor(0.5 * np.min(finite_tau))))
        ess = (steps - burn_in) * walkers / tau
        if steps < 50.0 * float(np.max(finite_tau)) and not any(
            "50 autocorrelation" in reason for reason in reasons
        ):
            reasons.append("chain is shorter than 50 autocorrelation times")
    else:
        burn_in = max(0, steps // 2)
        thin = 1
        ess = np.zeros(1, dtype=np.float64)

    acceptance_mean = float(np.mean(accepted))
    if acceptance_mean < 0.1 or acceptance_mean > 0.9:
        reasons.append("mean acceptance fraction is outside [0.1, 0.9]")
    if np.any(ess < 100.0):
        reasons.append("effective sample size is below 100")
    return ConvergenceReport(
        converged=not reasons,
        autocorrelation_time=tau,
        effective_sample_size=np.asarray(ess, dtype=np.float64),
        acceptance_mean=acceptance_mean,
        burn_in=burn_in,
        thin=thin,
        reasons=tuple(dict.fromkeys(reasons)),
    )
