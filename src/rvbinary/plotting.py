"""Non-interactive summary plots for RVBinary results."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(path: Path, draw: Callable[[object], None]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    draw(axes)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        figure.savefig(temporary, format="png", dpi=120)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return path


def posterior_plot(path: Path, samples: np.ndarray, title: str) -> Path:
    """Save one binary-fraction posterior histogram."""
    values = np.asarray(samples, dtype=np.float64).reshape(-1)

    def draw(axes: object) -> None:
        axes.hist(values, bins=30, density=True, color="#2d6a9f", alpha=0.85)
        axes.axvline(np.median(values), color="#b23a48", linewidth=1.8)
        axes.set(xlim=(0.0, 1.0), xlabel="Binary fraction", ylabel="Density", title=title)

    return _save_figure(path, draw)


def error_floor_plots(directory: Path, rows: list[dict[str, object]]) -> list[Path]:
    """Save error-floor posterior and binary-fraction sensitivity figures."""
    ordered = sorted(rows, key=lambda row: float(row["error_floor_kms"]))
    floors = np.array([float(row["error_floor_kms"]) for row in ordered])
    probability = np.array([float(row["error_floor_probability"]) for row in ordered])
    median = np.array([float(row["binary_fraction_median"]) for row in ordered])

    def draw_probability(axes: object) -> None:
        axes.plot(floors, probability, marker="o", color="#2d6a9f")
        axes.set(xlabel="Error floor (km/s)", ylabel="Posterior probability")

    def draw_fraction(axes: object) -> None:
        axes.plot(floors, median, marker="o", color="#b23a48")
        axes.set(
            xlabel="Error floor (km/s)",
            ylabel="Binary fraction",
            ylim=(0.0, 1.0),
        )

    return [
        _save_figure(directory / "errorfloor_posterior.png", draw_probability),
        _save_figure(directory / "binary_fraction_vs_errorfloor.png", draw_fraction),
    ]


def period_sensitivity_plot(path: Path, rows: list[dict[str, object]]) -> Path:
    """Save median binary fraction versus finite period limits."""
    finite = [row for row in rows if row["period_limit_days"] is not None]

    def draw(axes: object) -> None:
        if finite:
            floors = sorted({float(row["error_floor_kms"]) for row in finite})
            for floor in floors:
                selected = sorted(
                    (row for row in finite if float(row["error_floor_kms"]) == floor),
                    key=lambda row: float(row["period_limit_days"]),
                )
                axes.plot(
                    [float(row["period_limit_days"]) for row in selected],
                    [float(row["binary_fraction_median"]) for row in selected],
                    marker="o",
                    label=f"{floor:g} km/s",
                )
            axes.set_xscale("log")
            axes.legend()
        else:
            axes.text(0.5, 0.5, "Unrestricted model only", ha="center", va="center")
        axes.set(xlabel="Period upper limit (days)", ylabel="Binary fraction", ylim=(0, 1))

    return _save_figure(path, draw)


def combined_sensitivity_plot(path: Path, rows: list[dict[str, object]]) -> Path:
    """Save a combined period/error-floor posterior summary."""

    def draw(axes: object) -> None:
        x = np.array([float(row["error_floor_kms"]) for row in rows])
        y = np.array(
            [
                -1.0
                if row["period_limit_days"] is None
                else np.log10(float(row["period_limit_days"]))
                for row in rows
            ]
        )
        color = np.array([float(row["binary_fraction_median"]) for row in rows])
        plotted = axes.scatter(x, y, c=color, cmap="viridis", vmin=0.0, vmax=1.0, s=70)
        axes.set(
            xlabel="Error floor (km/s)",
            ylabel="log10 period limit (days); -1 = unrestricted",
        )
        axes.figure.colorbar(plotted, ax=axes, label="Binary fraction")

    return _save_figure(path, draw)
