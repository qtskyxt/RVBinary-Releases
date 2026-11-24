"""Render editable Slurm and PBS scripts without submitting jobs."""

from __future__ import annotations

import os
import re
import shlex
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Literal

from jinja2 import Environment, StrictUndefined

from rvbinary.config import RVBinaryConfig

SchedulerKind = Literal["slurm", "pbs"]


def _directive(value: str, name: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
        raise ValueError(f"scheduler {name} contains unsupported characters")
    return value


def render_scheduler(
    config: RVBinaryConfig,
    kind: SchedulerKind,
    config_path: Path = Path("rvbinary.yaml"),
) -> str:
    """Render a scheduler script as text without executing submission tools."""
    if kind not in ("slurm", "pbs"):
        raise ValueError(f"unsupported scheduler: {kind}")
    template_text = files("rvbinary.templates").joinpath(f"{kind}.sh.j2").read_text()
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(
        template_text
    )
    scheduler = config.scheduler
    job_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.project.name).strip("-")
    return template.render(
        job_name=job_name or "rvbinary",
        ranks=config.execution.mpi_ranks,
        walltime=_directive(scheduler.walltime, "walltime"),
        memory=_directive(scheduler.memory, "memory"),
        queue=_directive(scheduler.queue, "queue"),
        account=None
        if scheduler.account is None
        else _directive(scheduler.account, "account"),
        log_dir=shlex.quote(str(Path(config.project.output) / "logs")),
        conda_environment=shlex.quote(scheduler.conda_environment),
        conda_init=(
            None
            if scheduler.conda_init is None
            else shlex.quote(str(scheduler.conda_init))
        ),
        config_path=shlex.quote(str(Path(config_path))),
        precomputed=config.mock.precomputed is not None,
    ).rstrip() + "\n"


def render_to_file(
    config: RVBinaryConfig,
    kind: SchedulerKind,
    output: Path,
    config_path: Path = Path("rvbinary.yaml"),
) -> Path:
    """Atomically write an executable, editable scheduler script."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(render_scheduler(config, kind, config_path))
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
