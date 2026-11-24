"""One-sample orbit simulation and deterministic shared-epoch noise realizations."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rvbinary.io.observations import EpochData
from rvbinary.mock.orbit import radial_velocity, solve_kepler
from rvbinary.models.base import OrbitModel, PrimaryStar
from rvbinary.statistics import StatisticMode, star_statistic

NOISE_MODEL = "epoch-shared/v1"

_EPOCH_COLUMNS = (
    "TARGETID",
    "VRAD",
    "VRAD_ERR",
    "PERIOD",
    "q",
    "ECC",
    "THETA",
    "INCL",
    "OMEGA",
    "DELTAT",
    "STAR_OFFSET",
)


def _readonly(values: NDArray[np.generic]) -> NDArray[np.generic]:
    output = np.array(values, copy=True)
    output.flags.writeable = False
    return output


def _child_rng(
    seed: int,
    sample_index: int,
    targetid: int,
    product_label: str,
    *,
    float_bits: int | None = None,
) -> np.random.Generator:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    target = int(targetid)
    if target < -(2**63) or target >= 2**63:
        raise ValueError("TARGETID must fit in signed int64")
    label = product_label.encode("utf-8")
    hasher = hashlib.blake2b(digest_size=32, person=b"rvbinary-stream")
    seed_bytes = str(seed).encode("ascii")
    hasher.update(struct.pack(">I", len(seed_bytes)))
    hasher.update(seed_bytes)
    hasher.update(struct.pack(">QqI", sample_index, target, len(label)))
    hasher.update(label)
    if float_bits is not None:
        hasher.update(struct.pack(">Q", float_bits))
    entropy = np.frombuffer(hasher.digest(), dtype=">u4").astype(np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _float64_bits(value: float) -> int:
    scalar = np.asarray(value, dtype=np.float64)
    return int(scalar.view(np.uint64).item())


@dataclass(frozen=True)
class PeriodLimitedSample:
    """A non-mutating view recording a period-selection threshold."""

    sample: MockSample
    limit_days: float

    @property
    def orbits_digest(self) -> str:
        return self.sample.orbits_digest


@dataclass(frozen=True)
class MockSample:
    """Immutable unrestricted orbit realization for one Monte Carlo sample."""

    sample_index: int
    arrays: Mapping[str, NDArray[np.generic]]

    def __post_init__(self) -> None:
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        missing = set(_EPOCH_COLUMNS) - set(self.arrays)
        extra = set(self.arrays) - set(_EPOCH_COLUMNS)
        if missing or extra:
            raise ValueError(
                f"mock sample columns mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        arrays = {name: _readonly(np.asarray(self.arrays[name])) for name in _EPOCH_COLUMNS}
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("mock sample arrays must be non-empty and share a row count")
        if arrays["TARGETID"].ndim != 1 or arrays["TARGETID"].dtype != np.dtype(np.int64):
            raise ValueError("TARGETID must be a one-dimensional int64 array")
        if any(values.ndim != 1 for values in arrays.values()):
            raise ValueError("mock sample arrays must be one-dimensional")
        object.__setattr__(self, "arrays", arrays)

    @property
    def targetids(self) -> NDArray[np.int64]:
        return np.unique(self.arrays["TARGETID"])

    @property
    def orbits_digest(self) -> str:
        hasher = hashlib.sha256()
        for name in ("TARGETID", "PERIOD", "q", "ECC", "INCL", "OMEGA"):
            values = np.ascontiguousarray(self.arrays[name])
            encoded_name = name.encode("ascii")
            hasher.update(struct.pack(">I", len(encoded_name)))
            hasher.update(encoded_name)
            hasher.update(values.dtype.str.encode("ascii"))
            hasher.update(values.tobytes())
        return hasher.hexdigest()

    def with_period_limit(self, limit_days: float) -> PeriodLimitedSample:
        if not np.isfinite(limit_days) or limit_days <= 0.0:
            raise ValueError("period limit must be finite and positive")
        return PeriodLimitedSample(self, float(limit_days))


@dataclass(frozen=True)
class BetaSample:
    """Unrestricted binary and nonbinary pair-beta values for one error floor."""

    sample_index: int
    error_floor_kms: float
    statistic_mode: StatisticMode
    targetids: NDArray[np.int64]
    binary_by_star: tuple[NDArray[np.float64], ...]
    nonbinary_by_star: tuple[NDArray[np.float64], ...]
    binary_statistics: NDArray[np.float64]
    nonbinary_statistics: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targetids", _readonly(np.asarray(self.targetids, dtype=np.int64)))
        object.__setattr__(
            self,
            "binary_by_star",
            tuple(
                _readonly(np.asarray(values, dtype=np.float64))
                for values in self.binary_by_star
            ),
        )
        object.__setattr__(
            self,
            "nonbinary_by_star",
            tuple(
                _readonly(np.asarray(values, dtype=np.float64))
                for values in self.nonbinary_by_star
            ),
        )
        object.__setattr__(
            self,
            "binary_statistics",
            _readonly(np.asarray(self.binary_statistics, dtype=np.float64)),
        )
        object.__setattr__(
            self,
            "nonbinary_statistics",
            _readonly(np.asarray(self.nonbinary_statistics, dtype=np.float64)),
        )

    def flattened(self, product: str) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        if product == "binary":
            groups = self.binary_by_star
        elif product == "nonbinary":
            groups = self.nonbinary_by_star
        else:
            raise ValueError(f"unsupported beta product: {product}")
        targetids = np.concatenate(
            [np.repeat(targetid, len(values)) for targetid, values in zip(self.targetids, groups)]
        ).astype(np.int64, copy=False)
        values = np.concatenate(groups).astype(np.float64, copy=False)
        return targetids, values


def _true_anomaly(
    mean_anomaly_rad: float, eccentricity: float, delta_days: NDArray[np.float64], period: float
) -> NDArray[np.float64]:
    mean = mean_anomaly_rad + 2.0 * np.pi * delta_days / period
    eccentric = solve_kepler(mean, eccentricity)
    return np.asarray(
        2.0
        * np.arctan2(
            np.sqrt(1.0 + eccentricity) * np.sin(eccentric / 2.0),
            np.sqrt(1.0 - eccentricity) * np.cos(eccentric / 2.0),
        ),
        dtype=np.float64,
    )


def simulate_sample(
    data: EpochData, model: OrbitModel, seed: int, sample_index: int
) -> MockSample:
    """Draw one unrestricted orbit per exact TARGETID using explicit generators."""
    targetid = np.asarray(data.targetid)
    if targetid.ndim != 1 or targetid.dtype != np.dtype(np.int64) or targetid.size == 0:
        raise ValueError("epoch TARGETID must be a non-empty one-dimensional int64 array")
    input_arrays = (
        data.mjd,
        data.vrad_kms,
        data.vrad_err_kms,
        data.mass_solar,
        data.logg_cgs,
    )
    if any(np.asarray(values).shape != targetid.shape for values in input_arrays):
        raise ValueError("epoch arrays must match TARGETID shape")
    if any(not np.all(np.isfinite(values)) for values in input_arrays):
        raise ValueError("epoch arrays must contain finite values")
    if np.any(data.vrad_err_kms <= 0.0):
        raise ValueError("VRAD_ERR values must be positive")

    order = np.argsort(targetid, kind="stable")
    ids = targetid[order]
    mjd = np.asarray(data.mjd, dtype=np.float64)[order]
    errors = np.asarray(data.vrad_err_kms, dtype=np.float64)[order]
    masses = np.asarray(data.mass_solar, dtype=np.float64)[order]
    logg = np.asarray(data.logg_cgs, dtype=np.float64)[order]
    arrays: dict[str, NDArray[np.generic]] = {
        "TARGETID": ids.astype(np.int64, copy=True),
        "VRAD": np.empty(ids.size, dtype=np.float64),
        "VRAD_ERR": errors.copy(),
        "PERIOD": np.empty(ids.size, dtype=np.float64),
        "q": np.empty(ids.size, dtype=np.float64),
        "ECC": np.empty(ids.size, dtype=np.float64),
        "THETA": np.empty(ids.size, dtype=np.float64),
        "INCL": np.empty(ids.size, dtype=np.float64),
        "OMEGA": np.empty(ids.size, dtype=np.float64),
        "DELTAT": np.empty(ids.size, dtype=np.float64),
        "STAR_OFFSET": np.empty(ids.size, dtype=np.int64),
    }

    unique_ids, starts, counts = np.unique(ids, return_index=True, return_counts=True)
    for star_id, start, count in zip(unique_ids, starts, counts, strict=True):
        rows = slice(start, start + count)
        if np.unique(masses[rows]).size != 1 or np.unique(logg[rows]).size != 1:
            raise ValueError(f"TARGETID {star_id} must have one mass and logg value")
        orbit = model.sample_star(
            PrimaryStar(int(star_id), float(masses[start]), float(logg[start])),
            _child_rng(seed, sample_index, int(star_id), "orbit"),
            sample_index=sample_index,
        )
        delta = mjd[rows] - mjd[start]
        arrays["VRAD"][rows] = radial_velocity(orbit, delta)
        arrays["PERIOD"][rows] = orbit.period_days
        arrays["q"][rows] = orbit.mass_ratio
        arrays["ECC"][rows] = orbit.eccentricity
        arrays["THETA"][rows] = _true_anomaly(
            orbit.mean_anomaly_rad, orbit.eccentricity, delta, orbit.period_days
        )
        arrays["INCL"][rows] = orbit.inclination_rad
        arrays["OMEGA"][rows] = orbit.argument_periastron_rad
        arrays["DELTAT"][rows] = delta
        arrays["STAR_OFFSET"][rows] = np.arange(count, dtype=np.int64)
    return MockSample(sample_index, arrays)


def _pair_beta(
    velocity: NDArray[np.float64],
    errors: NDArray[np.float64],
    floor: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    first, second = np.triu_indices(len(velocity), k=1)
    sigma = np.hypot(errors, floor)
    # One noisy measurement per epoch; every pair reuses that measurement.
    noisy_velocity = velocity + rng.normal(0.0, sigma)
    denominator = np.sqrt(np.square(errors[first]) + np.square(errors[second]))
    return np.asarray(
        np.abs(noisy_velocity[first] - noisy_velocity[second]) / denominator, dtype=np.float64
    )


def derive_beta_realization(
    sample: MockSample,
    error_floor_kms: float,
    seed: int,
    statistic_mode: StatisticMode,
) -> BetaSample:
    """Derive unrestricted pair products from shared noisy epochs."""
    floor = float(error_floor_kms)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("error_floor_kms must be finite and non-negative")
    if statistic_mode not in ("consistent", "legacy"):
        raise ValueError(f"unsupported statistic mode: {statistic_mode}")
    floor_bits = _float64_bits(floor)
    targetids = sample.targetids
    binary: list[NDArray[np.float64]] = []
    nonbinary: list[NDArray[np.float64]] = []
    ids = sample.arrays["TARGETID"]
    for targetid in targetids:
        rows = ids == targetid
        velocity = np.asarray(sample.arrays["VRAD"][rows], dtype=np.float64)
        errors = np.asarray(sample.arrays["VRAD_ERR"][rows], dtype=np.float64)
        if velocity.size < 2:
            raise ValueError(f"TARGETID {targetid} must have at least two epochs")
        binary.append(
            _pair_beta(
                velocity,
                errors,
                floor,
                _child_rng(
                    seed,
                    sample.sample_index,
                    int(targetid),
                    "binary-noise",
                    float_bits=floor_bits,
                ),
            )
        )
        nonbinary.append(
            _pair_beta(
                np.zeros_like(velocity),
                errors,
                floor,
                _child_rng(
                    seed,
                    sample.sample_index,
                    int(targetid),
                    "nonbinary-noise",
                    float_bits=floor_bits,
                ),
            )
        )
    return BetaSample(
        sample_index=sample.sample_index,
        error_floor_kms=floor,
        statistic_mode=statistic_mode,
        targetids=targetids,
        binary_by_star=tuple(binary),
        nonbinary_by_star=tuple(nonbinary),
        binary_statistics=star_statistic(binary, mode=statistic_mode),
        nonbinary_statistics=star_statistic(nonbinary, mode=statistic_mode),
    )
