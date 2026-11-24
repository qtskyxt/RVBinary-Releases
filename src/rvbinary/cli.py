"""Command-line interface for RVBinary workflows."""

import json
from pathlib import Path

import typer

from rvbinary.config import load_config

app = typer.Typer(no_args_is_help=True)


@app.command()
def gui(port: int = typer.Option(8501, min=1, max=65535)) -> None:
    """Open the optional lightweight orbit generator in a browser."""
    from rvbinary.gui.launcher import launch_gui

    try:
        status = launch_gui(port)
    except (RuntimeError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    raise typer.Exit(status)


@app.command()
def validate(config: Path) -> None:
    """Validate and resolve a configuration file."""
    load_config(config)
    typer.echo(f"Valid configuration: {config}")


@app.command()
def mock(
    config: Path,
    extend_stars: Path | None = typer.Option(None, "--extend-stars"),
    merge_only: bool = typer.Option(False, "--merge-only"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Generate reusable binary and non-binary mock velocities."""
    resolved = load_config(config)
    if resolved.mock.precomputed is not None:
        if merge_only or extend_stars is not None:
            raise typer.BadParameter(
                "--merge-only and --extend-stars are unavailable for precomputed mocks"
            )
        typer.echo(f"Using precomputed mocks: {resolved.mock.precomputed.directory}")
        return
    if merge_only and extend_stars is not None:
        raise typer.BadParameter("--merge-only and --extend-stars cannot be combined")
    if dry_run:
        action = "rebuild mock indexes" if merge_only else "generate mock samples"
        typer.echo(f"Dry run: {action} in {resolved.project.output}")
        return

    from rvbinary.io.observations import read_epochs
    from rvbinary.mock.writer import rebuild_mock_indexes

    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None

    if merge_only:
        if MPI is None:
            artifacts = rebuild_mock_indexes(resolved)
        else:
            state = None
            artifacts = None
            if MPI.COMM_WORLD.Get_rank() == 0:
                try:
                    artifacts = rebuild_mock_indexes(resolved)
                    state = (True, None)
                except BaseException as exc:
                    state = (False, f"{type(exc).__name__}: {exc}")
            ok, payload = MPI.COMM_WORLD.bcast(state, root=0)
            if not ok:
                raise RuntimeError(f"merge-only failed: {payload}")
            if MPI.COMM_WORLD.Get_rank() != 0:
                return
            assert artifacts is not None
    elif extend_stars is not None:
        new_epochs = read_epochs(extend_stars)
        if MPI is None:
            from rvbinary.mock.writer import extend_mock_stars

            artifacts = extend_mock_stars(resolved, new_epochs)
        else:
            from rvbinary.parallel import extend_mock_stars_mpi

            artifacts = extend_mock_stars_mpi(resolved, new_epochs, MPI.COMM_WORLD)
            if MPI.COMM_WORLD.Get_rank() != 0:
                return
    else:
        if MPI is None or MPI.COMM_WORLD.Get_size() == 1:
            if resolved.execution.hdf5_mode == "mpio":
                raise RuntimeError("hdf5_mode=mpio requires mpi4py") from None
            from rvbinary.workflow.runner import run_pipeline

            artifacts = run_pipeline(resolved, stages=("mock",)).mock_artifacts
        else:
            from rvbinary.parallel import generate_mock_mpi

            artifacts = generate_mock_mpi(resolved, MPI.COMM_WORLD)
            if MPI.COMM_WORLD.Get_rank() != 0:
                return
    typer.echo(f"Mock index: {artifacts.table_index}")


@app.command()
def pdf(
    config: Path,
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build probability-density grids from mock velocities."""
    resolved = load_config(config)
    if dry_run:
        typer.echo(f"Dry run: build PDF grid in {resolved.project.output}")
        return
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    if MPI is None or MPI.COMM_WORLD.Get_size() == 1:
        from rvbinary.workflow.runner import run_pipeline

        result = run_pipeline(resolved, stages=("pdf",), force=force)
        artifacts = result.pdf_artifacts
    else:
        if resolved.mock.precomputed is not None:
            from rvbinary.pdf.builder import build_precomputed_pdf_grid

            artifacts = build_precomputed_pdf_grid(resolved, MPI.COMM_WORLD)
        else:
            from rvbinary.mock.writer import _artifact_set
            from rvbinary.pdf.builder import build_pdf_grid

            artifacts = build_pdf_grid(resolved, _artifact_set(resolved), MPI.COMM_WORLD)
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
    typer.echo(f"PDF artifacts: {len(artifacts)}")


@app.command()
def mcmc(
    config: Path,
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Infer binary fractions over the configured grids."""
    resolved = load_config(config)
    if dry_run:
        typer.echo(f"Dry run: infer posterior grid in {resolved.project.output}")
        return
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    if MPI is not None and MPI.COMM_WORLD.Get_rank() != 0:
        return
    from rvbinary.workflow.runner import run_pipeline

    result = run_pipeline(resolved, stages=("mcmc",), force=force)
    typer.echo(f"Posterior artifacts: {len(result.posterior_paths)}")
    if result.exit_status:
        raise typer.Exit(result.exit_status)


@app.command()
def run(
    config: Path,
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Execute all workflow stages in dependency order."""
    resolved = load_config(config)
    if dry_run:
        typer.echo(f"Dry run: execute full pipeline in {resolved.project.output}")
        return
    try:
        from mpi4py import MPI
    except ImportError:
        MPI = None
    if MPI is not None and MPI.COMM_WORLD.Get_rank() != 0:
        return
    from rvbinary.workflow.runner import run_pipeline

    result = run_pipeline(resolved, force=force)
    typer.echo(f"Summary: {result.summary_csv}")
    if result.exit_status:
        raise typer.Exit(result.exit_status)


@app.command("inspect")
def inspect_file(file: Path) -> None:
    """Inspect an RVBinary or legacy HDF5 artifact."""
    from rvbinary.io.legacy import inspect_hdf5

    report = inspect_hdf5(file)
    typer.echo(
        json.dumps(
            {
                "path": str(report.path),
                "schema": report.schema,
                "sample_ids": report.sample_ids,
                "completion": report.completion,
                "resumable": report.resumable,
                "usable": report.usable,
                "broken_links": report.broken_links,
                "duplicate_sample_ids": report.duplicate_sample_ids,
                "malformed": report.malformed,
                "dataset_shapes": {
                    name: list(shape) for name, shape in report.dataset_shapes.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command()
def compact(index: Path, output: Path) -> None:
    """Compact an external-link index into one physical HDF5 file."""
    from rvbinary.io.hdf5 import compact_index

    compact_index(index, output)
    typer.echo(f"Compacted artifact: {output}")


@app.command()
def submit(
    config: Path,
    scheduler: str = typer.Option(..., "--scheduler"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Render an editable scheduler script without submitting it."""
    resolved = load_config(config)
    if scheduler not in {"slurm", "pbs"}:
        raise typer.BadParameter("--scheduler must be 'slurm' or 'pbs'")
    destination = output or config.with_name(f"rvbinary-{scheduler}.sh")
    from rvbinary.workflow.scheduler import render_to_file

    path = render_to_file(resolved, scheduler, destination, config.resolve())
    typer.echo(f"Rendered scheduler script: {path}")
