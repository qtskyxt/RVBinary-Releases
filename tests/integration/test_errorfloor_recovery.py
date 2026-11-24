from __future__ import annotations

import numpy as np

from rvbinary.inference.evidence import marginal_log_evidence, normalize_discrete_posterior
from rvbinary.pdf.histogram import PDFModel


def _model(binary_density: list[float], nonbinary_density: list[float]) -> PDFModel:
    return PDFModel(
        edges=np.array([1.0, 2.0, 3.0]),
        binary_density=np.asarray(binary_density),
        nonbinary_density=np.asarray(nonbinary_density),
        binary_counts=np.array([1, 1]),
        nonbinary_counts=np.array([1, 1]),
    )


def test_injected_errorfloor_grid_point_is_recovered() -> None:
    floors = np.array([0.0, 0.4, 0.8])
    observed = np.full(20, 1.25)
    models = [
        _model([0.10, 0.90], [0.10, 0.90]),
        _model([0.98, 0.02], [0.98, 0.02]),
        _model([0.25, 0.75], [0.25, 0.75]),
    ]

    evidence = np.array([marginal_log_evidence(observed, model) for model in models])
    probability = normalize_discrete_posterior(evidence, np.ones(floors.size))

    assert floors[np.argmax(probability)] == 0.4
    assert probability[1] > 0.99
