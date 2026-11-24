"""Production model dispatch and content-based model provenance."""

from __future__ import annotations

import warnings

from rvbinary.config import MS17Config, RVBinaryConfig
from rvbinary.io.observations import EpochData
from rvbinary.models.base import OrbitModel
from rvbinary.models.dm91 import DM91Model
from rvbinary.models.ms17 import MS17_IMPLEMENTATION, MS17Model, PeriodTable


def make_model(config: RVBinaryConfig) -> OrbitModel:
    if isinstance(config.model, MS17Config):
        warnings.warn(
            "MS17 trial only: 0.8 <= M1 < 1.2 solar masses; no extrapolation", stacklevel=2
        )
        return MS17Model(PeriodTable.load(config.model.period_table), config.model.environment)
    return DM91Model(config.model.log10_period_mean, config.model.log10_period_sigma)


def model_payload(config: RVBinaryConfig) -> dict[str, object]:
    payload = config.model.model_dump(mode="json")
    if isinstance(config.model, MS17Config):
        table = PeriodTable.load(config.model.period_table)
        payload.pop("period_table")
        payload.update(
            implementation=MS17_IMPLEMENTATION,
            table_sha256=table.sha256,
            trial_mass_bounds=[0.8, 1.2],
            mass_upper_exclusive=True,
            table_mass_bounds=table.mass_bounds,
            table_period_bounds=table.period_bounds,
            interpolation="nearest-logmass-logperiod-voronoi",
        )
    return payload


def validate_model_stars(config: RVBinaryConfig, data: EpochData) -> None:
    if isinstance(config.model, MS17Config):
        table = PeriodTable.load(config.model.period_table)
        for target, mass in zip(data.targetid, data.mass_solar, strict=True):
            try:
                table.validate_mass(float(mass))
            except ValueError as exc:
                raise ValueError(f"TARGETID {target}: {exc}") from exc
