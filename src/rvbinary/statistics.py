"""Pairwise radial-velocity beta values and star-level reductions."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

StatisticMode = Literal["consistent", "legacy"]


class EpochData(Protocol):
    """Structural input required to calculate pairwise beta values."""

    targetid: NDArray[np.int64]
    vrad_kms: NDArray[np.float64]
    vrad_err_kms: NDArray[np.float64]


def _epoch_arrays(
    data: EpochData,
) -> tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.float64]]:
    targetid = np.asarray(data.targetid)
    vrad_kms = np.asarray(data.vrad_kms, dtype=np.float64)
    vrad_err_kms = np.asarray(data.vrad_err_kms, dtype=np.float64)
    arrays = {
        "targetid": targetid,
        "vrad_kms": vrad_kms,
        "vrad_err_kms": vrad_err_kms,
    }

    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("targetid, vrad_kms, and vrad_err_kms must be one-dimensional")
    if len({values.shape for values in arrays.values()}) != 1:
        raise ValueError("targetid, vrad_kms, and vrad_err_kms must have matching shapes")
    for name, values in arrays.items():
        try:
            finite = np.isfinite(values)
        except TypeError as exc:
            raise ValueError(f"{name} values must be finite") from exc
        if not np.all(finite):
            raise ValueError(f"{name} values must be finite")
    if np.any(vrad_err_kms <= 0):
        raise ValueError("vrad_err_kms values must be positive")

    return targetid, vrad_kms, vrad_err_kms


def pairwise_beta(data: EpochData) -> list[NDArray[np.float64]]:
    """Return every unique epoch-pair beta, grouped by sorted TARGETID."""
    targetid, vrad_kms, vrad_err_kms = _epoch_arrays(data)
    order = np.argsort(targetid, kind="stable")
    sorted_targetid = targetid[order]
    sorted_vrad = vrad_kms[order]
    sorted_vrad_err = vrad_err_kms[order]

    groups: list[NDArray[np.float64]] = []
    targetids, starts, counts = np.unique(
        sorted_targetid, return_index=True, return_counts=True
    )
    for target, start, count in zip(targetids, starts, counts, strict=True):
        if count < 2:
            raise ValueError(f"TARGETID {target} must have at least two epochs")
        first, second = np.triu_indices(count, k=1)
        velocities = sorted_vrad[start : start + count]
        uncertainties = sorted_vrad_err[start : start + count]
        beta = np.abs(velocities[first] - velocities[second]) / np.sqrt(
            np.square(uncertainties[first]) + np.square(uncertainties[second])
        )
        groups.append(np.asarray(beta, dtype=np.float64))
    return groups


def star_statistic(
    pairs: list[NDArray[np.float64]], mode: StatisticMode = "consistent"
) -> NDArray[np.float64]:
    """Reduce each star's pair beta values using an explicit convention."""
    if mode not in ("consistent", "legacy"):
        raise ValueError(f"unsupported statistic mode: {mode}")

    statistics = np.empty(len(pairs), dtype=np.float64)
    for index, pair_group in enumerate(pairs):
        values = np.asarray(pair_group, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"pair group {index} must be one-dimensional and non-empty")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"pair group {index} values must be finite")
        if mode == "consistent":
            statistics[index] = 1.0 + np.sqrt(np.mean(np.square(values)))
        else:
            statistics[index] = np.sqrt(np.mean(np.square(1.0 + values)))
    return statistics
