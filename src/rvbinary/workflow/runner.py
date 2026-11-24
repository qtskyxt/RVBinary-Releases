"""Resumable end-to-end RVBinary workflow orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import yaml

from rvbinary.config import RVBinaryConfig
from rvbinary.inference.evidence import (
    discrete_credible_interval,
    marginal_log_evidence,
    normalize_discrete_posterior,
)
from rvbinary.inference.results import write_posterior
from rvbinary.inference.sampler import BinaryFractionProblem, run_binary_fraction_mcmc
from rvbinary.io.observations import read_epochs, read_member_ids, select_members
from rvbinary.mock.simulation import NOISE_MODEL
from rvbinary.mock.writer import (
    ArtifactSet,
    _all_indexes,
    _artifact_set,
    generate_mock,
    rebuild_mock_indexes,
)
from rvbinary.models.factory import model_payload
from rvbinary.pdf.builder import (
    PDFArtifact,
    _legacy_paths,
    _legacy_source_digests,
    _preflight,
    build_pdf_grid,
    build_precomputed_pdf_grid,
)
from rvbinary.pdf.histogram import PDFModel
from rvbinary.plotting import (
    combined_sensitivity_plot,
    error_floor_plots,
    period_sensitivity_plot,
    posterior_plot,
)
from rvbinary.statistics import pairwise_beta, star_statistic
from rvbinary.workflow.manifest import ArtifactManifest, file_digest


@dataclass(frozen=True)
class RunResult:
    """Paths, grid coordinates, summaries, and final workflow status."""

    mock_artifacts: ArtifactSet | None
    pdf_artifacts: tuple[PDFArtifact, ...]
    posterior_paths: tuple[Path, ...]
    summary_csv: Path
    plot_paths: tuple[Path, ...]
    period_limits: tuple[float | None, ...]
    error_floors: tuple[float | None, ...]
    rows: tuple[dict[str, object], ...]
    converged: bool
    exit_status: int


def _config_digest(config: RVBinaryConfig) -> str:
    resolved = config.model_dump(mode="json")
    if config.model.name == "ms17":
        resolved["model"] = model_payload(config)
    if config.mock.precomputed is None:
        resolved["mock_noise_model"] = NOISE_MODEL
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_pdf(path: Path) -> PDFModel:
    with np.load(path, allow_pickle=False) as stored:
        return PDFModel(
            edges=np.asarray(stored["edges"], dtype=np.float64),
            binary_density=np.asarray(stored["binary_density"], dtype=np.float64),
            nonbinary_density=np.asarray(stored["nonbinary_density"], dtype=np.float64),
            binary_counts=np.asarray(stored["binary_counts"], dtype=np.int64),
            nonbinary_counts=np.asarray(stored["nonbinary_counts"], dtype=np.int64),
        )


def _observed_statistics(config: RVBinaryConfig) -> dict[str, np.ndarray]:
    output = {}
    master = None
    for subsample in config.observations.subsamples:
        members = read_member_ids(subsample.members)
        if subsample.epochs is None:
            if master is None:
                master = read_epochs(config.observations.mock_epochs)
            epochs = master
        else:
            epochs = read_epochs(subsample.epochs)
        selected = select_members(epochs, members)
        output[subsample.name] = star_statistic(
            pairwise_beta(selected), mode=config.statistics.mode
        )
    return output


def _posterior_summary(path: Path) -> tuple[dict[str, object], np.ndarray]:
    with h5py.File(path, "r") as handle:
        masses = np.asarray(handle["summary/credible_mass"], dtype=np.float64)
        lower = np.asarray(handle["summary/lower"], dtype=np.float64)
        upper = np.asarray(handle["summary/upper"], dtype=np.float64)
        intervals = {
            str(float(mass)): [float(low), float(high)]
            for mass, low, high in zip(masses, lower, upper, strict=True)
        }
        summary = {
            "binary_fraction_median": float(handle["summary"].attrs["median"]),
            "credible_intervals": intervals,
            "converged": bool(handle.attrs["converged"]),
            "exit_status": int(handle.attrs["exit_status"]),
            "acceptance_mean": float(handle["diagnostics"].attrs["acceptance_mean"]),
            "burn_in": int(handle["diagnostics"].attrs["burn_in"]),
            "thin": int(handle["diagnostics"].attrs["thin"]),
        }
        samples = np.asarray(handle["samples"], dtype=np.float64)
    return summary, samples


def _posterior_matches_pdf(path: Path, pdf_path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            metadata = json.loads(str(handle.attrs["metadata"]))
            return (
                bool(handle.attrs.get("completion", False))
                and metadata.get("pdf_sha256") == file_digest(pdf_path)
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _record_mock(manifest: ArtifactManifest, artifacts: ArtifactSet) -> None:
    for index in [
        artifacts.table_index,
        *artifacts.binary_beta_indexes.values(),
        *artifacts.nonbinary_beta_indexes.values(),
    ]:
        manifest.record(index, "mock-index")
    for shard in artifacts.shards:
        manifest.record(shard, "mock-shard")
    for source in artifacts.source_batches:
        manifest.record(source, "mock-source")


def _mock_stage(config: RVBinaryConfig, manifest: ArtifactManifest) -> ArtifactSet:
    indexes = _all_indexes(config)
    if all(manifest.validate(path, "mock-index") for path in indexes):
        return _artifact_set(config)
    if all(path.is_file() for path in indexes):
        from rvbinary.mock.writer import _validate_existing

        _validate_existing(config, None)
        artifacts = _artifact_set(config)
        _record_mock(manifest, artifacts)
        manifest.state["stage"] = "mock"
        manifest.write()
        return artifacts
    table_shards = list((Path(config.project.output) / "mock" / "tables").glob("*.h5"))
    artifacts = rebuild_mock_indexes(config) if table_shards else generate_mock(config)
    _record_mock(manifest, artifacts)
    manifest.state["stage"] = "mock"
    manifest.write()
    return artifacts


def _discover_pdf_artifacts(
    config: RVBinaryConfig, mocks: ArtifactSet
) -> list[PDFArtifact]:
    root = Path(config.project.output) / "pdf"
    paths = sorted(root.glob("*/*/period-*/pdf.npz"))
    period_limits = [
        *([None] if config.model.include_unrestricted else []),
        *[float(value) for value in config.model.period_limits_days],
    ]
    expected = {
        (subsample.name, float(floor), limit)
        for subsample in config.observations.subsamples
        for floor in config.mock.error_floor.values_kms
        for limit in period_limits
    }
    source_digests = _preflight(config, mocks)
    artifacts = []
    coordinates = set()
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as stored:
                mode = str(stored["statistic_mode"])
                floor = float(stored["error_floor_kms"])
                raw_limit = float(stored["period_limit_days"])
                limit = None if np.isnan(raw_limit) else raw_limit
                stored_digests = json.loads(str(stored["source_digests"]))
                model = PDFModel(
                    edges=np.asarray(stored["edges"], dtype=np.float64),
                    binary_density=np.asarray(stored["binary_density"], dtype=np.float64),
                    nonbinary_density=np.asarray(
                        stored["nonbinary_density"], dtype=np.float64
                    ),
                    binary_counts=np.asarray(stored["binary_counts"], dtype=np.int64),
                    nonbinary_counts=np.asarray(
                        stored["nonbinary_counts"], dtype=np.int64
                    ),
                )
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return []
        widths = np.diff(model.edges)
        if (
            mode != config.statistics.mode
            or stored_digests != source_digests
            or np.any(widths <= 0.0)
            or not np.isclose(np.sum(model.binary_density * widths), 1.0)
            or not np.isclose(np.sum(model.nonbinary_density * widths), 1.0)
        ):
            return []
        subsample = path.relative_to(root).parts[0]
        coordinates.add((subsample, floor, limit))
        artifacts.append(PDFArtifact(path, subsample, floor, limit))
    return artifacts if coordinates == expected and len(artifacts) == len(expected) else []


def _pdf_stage(
    config: RVBinaryConfig,
    manifest: ArtifactManifest,
    mocks: ArtifactSet,
    *,
    force: bool = False,
) -> list[PDFArtifact]:
    saved = manifest.state.get("pdf_artifacts", [])
    if not force and isinstance(saved, list) and saved:
        artifacts = [
            PDFArtifact(
                Path(item["path"]),
                str(item["subsample"]),
                float(item["error_floor_kms"]),
                None if item["period_limit_days"] is None else float(item["period_limit_days"]),
            )
            for item in saved
        ]
        if all(manifest.validate(artifact.path, "pdf") for artifact in artifacts):
            return artifacts
    artifacts = (
        [] if force else _discover_pdf_artifacts(config, mocks)
    ) or build_pdf_grid(config, mocks)
    for artifact in artifacts:
        manifest.record(artifact.path, "pdf")
    manifest.state["pdf_artifacts"] = [
        {
            "path": str(artifact.path),
            "subsample": artifact.subsample,
            "error_floor_kms": artifact.error_floor_kms,
            "period_limit_days": artifact.period_limit_days,
        }
        for artifact in artifacts
    ]
    manifest.state["stage"] = "pdf"
    manifest.write()
    return artifacts


def _discover_precomputed_pdf_artifacts(config: RVBinaryConfig) -> list[PDFArtifact]:
    source_digests = _legacy_source_digests(_legacy_paths(config))
    artifacts = []
    for subsample in config.observations.subsamples:
        path = (
            Path(config.project.output)
            / "pdf"
            / subsample.name
            / "errorfloor-per-star"
            / "period-unrestricted"
            / "pdf.npz"
        )
        if not path.is_file():
            return []
        try:
            with np.load(path, allow_pickle=False) as stored:
                mode = str(stored["statistic_mode"])
                floor_mode = str(stored["error_floor_mode"])
                sample_count = int(stored["sample_count"])
                stored_digests = json.loads(str(stored["source_digests"]))
                model = PDFModel(
                    edges=np.asarray(stored["edges"], dtype=np.float64),
                    binary_density=np.asarray(stored["binary_density"], dtype=np.float64),
                    nonbinary_density=np.asarray(
                        stored["nonbinary_density"], dtype=np.float64
                    ),
                    binary_counts=np.asarray(stored["binary_counts"], dtype=np.int64),
                    nonbinary_counts=np.asarray(
                        stored["nonbinary_counts"], dtype=np.int64
                    ),
                )
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return []
        widths = np.diff(model.edges)
        if (
            mode != config.statistics.mode
            or floor_mode != "per_star_baked_in"
            or sample_count != config.mock.samples
            or stored_digests != source_digests
            or np.any(widths <= 0.0)
            or not np.isclose(np.sum(model.binary_density * widths), 1.0)
            or not np.isclose(np.sum(model.nonbinary_density * widths), 1.0)
        ):
            return []
        artifacts.append(PDFArtifact(path, subsample.name, None, None))
    return artifacts


def _precomputed_pdf_stage(
    config: RVBinaryConfig,
    manifest: ArtifactManifest,
    *,
    force: bool = False,
) -> list[PDFArtifact]:
    artifacts = [] if force else _discover_precomputed_pdf_artifacts(config)
    if not artifacts:
        artifacts = build_precomputed_pdf_grid(config)
    for artifact in artifacts:
        manifest.record(artifact.path, "pdf")
    manifest.state["pdf_artifacts"] = [
        {
            "path": str(artifact.path),
            "subsample": artifact.subsample,
            "error_floor_kms": None,
            "period_limit_days": None,
        }
        for artifact in artifacts
    ]
    manifest.state["stage"] = "pdf"
    manifest.write()
    return artifacts


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "subsample",
        "period_limit_days",
        "error_floor_kms",
        "error_floor_mode",
        "binary_fraction_median",
        "credible_lower",
        "credible_upper",
        "log_evidence",
        "error_floor_probability",
        "error_floor_interval_lower",
        "error_floor_interval_upper",
        "converged",
        "posterior_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_global_summaries(run_result: RunResult) -> list[Path]:
    """Write the global CSV and per-grid JSON summaries atomically."""
    _write_summary_csv(run_result.summary_csv, list(run_result.rows))
    paths = [run_result.summary_csv]
    for row in run_result.rows:
        summary = Path(str(row["posterior_path"])).with_name("summary.json")
        _atomic_text(summary, json.dumps(row, indent=2, sort_keys=True, default=str) + "\n")
        paths.append(summary)
    return paths


def _restore_complete(config: RVBinaryConfig, manifest: ArtifactManifest) -> RunResult | None:
    state = manifest.state
    if state.get("complete") is not True or not manifest.validate_all():
        return None
    mocks = None if config.mock.precomputed is not None else _artifact_set(config)
    pdf_artifacts = tuple(
        PDFArtifact(
            Path(item["path"]),
            str(item["subsample"]),
            None
            if item["error_floor_kms"] is None
            else float(item["error_floor_kms"]),
            None if item["period_limit_days"] is None else float(item["period_limit_days"]),
        )
        for item in state["pdf_artifacts"]
    )
    return RunResult(
        mock_artifacts=mocks,
        pdf_artifacts=pdf_artifacts,
        posterior_paths=tuple(Path(path) for path in state["posterior_paths"]),
        summary_csv=Path(state["summary_csv"]),
        plot_paths=tuple(Path(path) for path in state["plot_paths"]),
        period_limits=tuple(state["period_limits"]),
        error_floors=tuple(
            None if value is None else float(value) for value in state["error_floors"]
        ),
        rows=tuple(state["rows"]),
        converged=bool(state["converged"]),
        exit_status=int(state["exit_status"]),
    )


def run_pipeline(
    config: RVBinaryConfig,
    stages: tuple[str, ...] = ("mock", "pdf", "mcmc"),
    *,
    force: bool = False,
) -> RunResult:
    """Run or resume the configured workflow in dependency order."""
    if not stages:
        raise ValueError("at least one workflow stage is required")
    unknown = set(stages) - {"mock", "pdf", "mcmc"}
    if unknown:
        raise ValueError(f"unknown workflow stages: {sorted(unknown)}")
    output = Path(config.project.output)
    manifest_path = output / "manifest.json"
    digest = _config_digest(config)
    if manifest_path.exists() and not force:
        manifest = ArtifactManifest.load(manifest_path)
        if manifest.config_digest != digest:
            raise ValueError("existing manifest does not match the resolved configuration")
        if stages == ("mock", "pdf", "mcmc"):
            sources_current = (
                bool(_discover_precomputed_pdf_artifacts(config))
                if config.mock.precomputed is not None
                else True
            )
            restored = _restore_complete(config, manifest) if sources_current else None
            if restored is not None:
                return restored
    else:
        manifest = ArtifactManifest(manifest_path, digest)

    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(output / "run.yaml", yaml.safe_dump(config.model_dump(mode="json")))
    precomputed = config.mock.precomputed is not None
    mocks = None if precomputed else _mock_stage(config, manifest)
    if "pdf" in stages or "mcmc" in stages:
        pdf_artifacts = (
            _precomputed_pdf_stage(config, manifest, force=force)
            if precomputed
            else _pdf_stage(config, manifest, mocks, force=force)  # type: ignore[arg-type]
        )
    else:
        pdf_artifacts = []
    observed = _observed_statistics(config) if "mcmc" in stages else {}
    rows: list[dict[str, object]] = []
    samples_by_path: dict[Path, np.ndarray] = {}
    posterior_paths: list[Path] = []

    if "mcmc" in stages:
        for artifact in pdf_artifacts:
            pdf = _load_pdf(artifact.path)
            statistics = observed[artifact.subsample]
            evidence = marginal_log_evidence(statistics, pdf, "uniform")
            posterior_path = artifact.path.with_name("posterior.h5")
            if (
                not force
                and manifest.validate(posterior_path, "posterior")
                and _posterior_matches_pdf(posterior_path, artifact.path)
            ):
                summary, samples = _posterior_summary(posterior_path)
            else:
                result = run_binary_fraction_mcmc(
                    BinaryFractionProblem(statistics, pdf),
                    config.mcmc,
                    seed=config.project.random_seed,
                )
                write_posterior(
                    posterior_path,
                    result,
                    {
                        "subsample": artifact.subsample,
                        "period_limit_days": artifact.period_limit_days,
                        "error_floor_kms": artifact.error_floor_kms,
                        "pdf_path": str(artifact.path),
                        "pdf_sha256": file_digest(artifact.path),
                        "statistic_mode": config.statistics.mode,
                    },
                )
                manifest.record(posterior_path, "posterior")
                manifest.write()
                summary, samples = _posterior_summary(posterior_path)
            mass = min(
                (float(value) for value in summary["credible_intervals"]),
                key=lambda value: abs(value - 0.68),
            )
            lower, upper = summary["credible_intervals"][str(mass)]
            rows.append(
                {
                    "subsample": artifact.subsample,
                    "period_limit_days": artifact.period_limit_days,
                    "error_floor_kms": artifact.error_floor_kms,
                    "error_floor_mode": (
                        "per_star_baked_in"
                        if artifact.error_floor_kms is None
                        else "grid"
                    ),
                    **summary,
                    "credible_lower": lower,
                    "credible_upper": upper,
                    "log_evidence": evidence,
                    "posterior_path": str(posterior_path),
                }
            )
            posterior_paths.append(posterior_path)
            samples_by_path[posterior_path] = samples

        groups = {(str(row["subsample"]), row["period_limit_days"]) for row in rows}
        for subsample, period_limit in groups:
            selected = [
                row
                for row in rows
                if row["subsample"] == subsample and row["period_limit_days"] == period_limit
            ]
            if selected[0]["error_floor_kms"] is None:
                selected[0]["error_floor_probability"] = 1.0
                selected[0]["error_floor_interval_lower"] = None
                selected[0]["error_floor_interval_upper"] = None
                continue
            selected.sort(key=lambda row: float(row["error_floor_kms"]))
            probability = normalize_discrete_posterior(
                np.array([float(row["log_evidence"]) for row in selected]),
                np.ones(len(selected)),
            )
            grid = np.array([float(row["error_floor_kms"]) for row in selected])
            interval = discrete_credible_interval(grid, probability, 0.68)
            for row, value in zip(selected, probability, strict=True):
                row["error_floor_probability"] = float(value)
                row["error_floor_interval_lower"] = interval[0]
                row["error_floor_interval_upper"] = interval[1]

    periods = tuple(
        [None] * int(config.model.include_unrestricted)
        + [float(value) for value in config.model.period_limits_days]
    )
    floors: tuple[float | None, ...] = (
        (None,)
        if precomputed
        else tuple(float(value) for value in config.mock.error_floor.values_kms)
    )
    summary_csv = output / "summary.csv"
    converged = bool(rows) and all(bool(row["converged"]) for row in rows)
    exit_status = 0 if converged or config.mcmc.allow_unconverged or not rows else 1
    provisional = RunResult(
        mocks,
        tuple(pdf_artifacts),
        tuple(posterior_paths),
        summary_csv,
        (),
        periods,
        floors,
        tuple(rows),
        converged,
        exit_status,
    )
    summary_paths = write_global_summaries(provisional) if rows else []
    plot_paths: list[Path] = []
    for row in rows:
        posterior_path = Path(str(row["posterior_path"]))
        plot_paths.append(
            posterior_plot(
                posterior_path.with_name("posterior.png"),
                samples_by_path[posterior_path],
                f"{row['subsample']} binary fraction",
            )
        )
    for subsample, period_limit in {
        (str(row["subsample"]), row["period_limit_days"]) for row in rows
    }:
        selected = [
            row
            for row in rows
            if row["subsample"] == subsample and row["period_limit_days"] == period_limit
        ]
        if selected[0]["error_floor_kms"] is None:
            continue
        if period_limit is None:
            period_label = "unrestricted"
        else:
            bits = int(np.asarray(float(period_limit), dtype=np.float64).view(np.uint64).item())
            period_label = f"days-float64-{bits:016x}"
        directory = output / "pdf" / subsample / "sensitivity" / f"period-{period_label}"
        plot_paths.extend(error_floor_plots(directory, selected))
    for subsample in {str(row["subsample"]) for row in rows} if not precomputed else set():
        selected = [row for row in rows if row["subsample"] == subsample]
        plot_paths.append(
            period_sensitivity_plot(output / "pdf" / subsample / "period_sensitivity.png", selected)
        )
    if rows and not precomputed:
        plot_paths.append(
            combined_sensitivity_plot(output / "period_errorfloor_sensitivity.png", rows)
        )

    result = RunResult(
        mocks,
        tuple(pdf_artifacts),
        tuple(posterior_paths),
        summary_csv,
        tuple(plot_paths),
        periods,
        floors,
        tuple(rows),
        converged,
        exit_status,
    )
    for path in [output / "run.yaml", *summary_paths, *plot_paths]:
        manifest.record(path, "summary" if path.suffix != ".png" else "plot")
    manifest.state.update(
        {
            "complete": stages == ("mock", "pdf", "mcmc"),
            "stage": "complete" if stages == ("mock", "pdf", "mcmc") else stages[-1],
            "posterior_paths": [str(path) for path in posterior_paths],
            "summary_csv": str(summary_csv),
            "plot_paths": [str(path) for path in plot_paths],
            "period_limits": list(periods),
            "error_floors": list(floors),
            "rows": rows,
            "converged": converged,
            "exit_status": exit_status,
        }
    )
    manifest.write()
    return result
