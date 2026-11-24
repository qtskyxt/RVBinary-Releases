"""Strict typed configuration loaded from ``rvbinary.yaml`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    output: Path
    random_seed: int


class SubsampleConfig(StrictModel):
    name: str
    members: Path
    epochs: Path | None = None


class ObservationsConfig(StrictModel):
    mock_epochs: Path
    subsamples: list[SubsampleConfig]


GridValue = TypeVar("GridValue", bound=float)


def _sorted_unique(values: list[GridValue]) -> list[GridValue]:
    return sorted(set(values))


class DM91Config(StrictModel):
    name: Literal["dm91"] = "dm91"
    log10_period_mean: float = 4.8
    log10_period_sigma: PositiveFloat = 2.3
    include_unrestricted: bool = True
    period_limits_days: list[PositiveFloat] = Field(default_factory=list)

    _normalize_period_limits = field_validator("period_limits_days")(_sorted_unique)


class MS17Config(StrictModel):
    name: Literal["ms17"] = "ms17"
    period_table: Path | None = None
    environment: Literal["none", "umi_legacy"] = "none"
    include_unrestricted: bool = True
    period_limits_days: list[PositiveFloat] = Field(default_factory=list)

    _normalize_period_limits = field_validator("period_limits_days")(_sorted_unique)


class ErrorFloorConfig(StrictModel):
    values_kms: list[NonNegativeFloat] = Field(default_factory=lambda: [0.0])
    prior: Literal["uniform"] = "uniform"

    _normalize_values = field_validator("values_kms")(_sorted_unique)


class PrecomputedMockConfig(StrictModel):
    format: Literal["binary2"] = "binary2"
    directory: Path
    error_floor_mode: Literal["per_star_baked_in"] = "per_star_baked_in"


class MockConfig(StrictModel):
    samples: PositiveInt = 10000
    error_floor: ErrorFloorConfig = Field(default_factory=ErrorFloorConfig)
    precomputed: PrecomputedMockConfig | None = None


class PDFConfig(StrictModel):
    bins: PositiveInt = 100
    export_legacy_npy: bool = False


class StatisticsConfig(StrictModel):
    mode: Literal["consistent", "legacy"] = "consistent"


CredibleInterval = Annotated[float, Field(gt=0, le=1)]


class MCMCConfig(StrictModel):
    walkers: PositiveInt = 32
    steps: PositiveInt = 5000
    burn_in: NonNegativeInt = 1000
    credible_intervals: list[CredibleInterval] = Field(default_factory=lambda: [0.68, 0.95])
    allow_unconverged: bool = False

    _normalize_credible_intervals = field_validator("credible_intervals")(_sorted_unique)


class ExecutionConfig(StrictModel):
    mpi_ranks: PositiveInt = 32
    hdf5_mode: Literal["shards", "mpio"] = "shards"
    resume: bool = True


class SchedulerConfig(StrictModel):
    walltime: str = "24:00:00"
    memory: str = "64G"
    queue: str = "normal"
    account: str | None = None
    conda_environment: str = Field(default="rvbinary", min_length=1)
    conda_init: Path | None = None


class RVBinaryConfig(StrictModel):
    project: ProjectConfig
    observations: ObservationsConfig
    model: Annotated[DM91Config | MS17Config, Field(discriminator="name")] = Field(
        default_factory=DM91Config
    )
    mock: MockConfig = Field(default_factory=MockConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    mcmc: MCMCConfig = Field(default_factory=MCMCConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @field_validator("model", mode="before")
    @classmethod
    def _default_model_name(cls, value: object) -> object:
        if isinstance(value, dict) and "name" not in value:
            return {"name": "dm91", **value}
        return value

    @model_validator(mode="after")
    def _validate_precomputed_binary2(self) -> RVBinaryConfig:
        if self.mock.precomputed is None:
            return self
        if self.model.name != "dm91":
            raise ValueError("precomputed Binary2 data cannot be relabelled as MS17")
        if (
            not self.model.include_unrestricted
            or self.model.period_limits_days
            or [float(value) for value in self.mock.error_floor.values_kms] != [0.0]
        ):
            raise ValueError(
                "precomputed Binary2 mocks require unrestricted periods and no "
                "additional error-floor grid"
            )
        return self


def _resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> RVBinaryConfig:
    """Load, validate, and resolve paths in an RVBinary YAML configuration."""
    config_path = path.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text())
    config = RVBinaryConfig.model_validate(raw)
    base = config_path.parent

    if isinstance(config.model, MS17Config) and config.model.period_table is not None:
        config.model.period_table = _resolve_path(config.model.period_table, base)
    config.project.output = _resolve_path(config.project.output, base)
    config.observations.mock_epochs = _resolve_path(config.observations.mock_epochs, base)
    if config.mock.precomputed is not None:
        config.mock.precomputed.directory = _resolve_path(
            config.mock.precomputed.directory, base
        )
    if config.scheduler.conda_init is not None:
        config.scheduler.conda_init = _resolve_path(config.scheduler.conda_init, base)
    for subsample in config.observations.subsamples:
        subsample.members = _resolve_path(subsample.members, base)
        if subsample.epochs is not None:
            subsample.epochs = _resolve_path(subsample.epochs, base)

    return config
