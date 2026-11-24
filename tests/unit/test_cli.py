import re

import pytest
from typer.testing import CliRunner

from rvbinary.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert all(name in result.stdout for name in ("mock", "pdf", "mcmc", "run"))


def test_cli_help_lists_complete_command_tree() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    commands = ("validate", "mock", "pdf", "mcmc", "run", "inspect", "compact", "submit")
    assert all(name in result.stdout for name in commands)


@pytest.mark.parametrize("color", [False, True])
def test_mock_help_exposes_merge_only_recovery(color: bool) -> None:
    result = CliRunner().invoke(app, ["mock", "--help"], color=color)

    assert result.exit_code == 0
    assert "--merge-only" in re.sub(r"\[[0-9;]*m", "", result.stdout)
