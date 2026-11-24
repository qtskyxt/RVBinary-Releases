from __future__ import annotations

from pathlib import Path

from rvbinary.workflow.manifest import ArtifactManifest


def test_manifest_round_trip_and_digest_validation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.dat"
    artifact.write_bytes(b"original")
    path = tmp_path / "manifest.json"
    manifest = ArtifactManifest(path, config_digest="abc123")
    manifest.record(artifact, "test", {"grid": 2})
    manifest.state["complete"] = True

    manifest.write()
    loaded = ArtifactManifest.load(path)

    assert loaded.config_digest == "abc123"
    assert loaded.validate(artifact, "test")
    assert loaded.state["complete"] is True
    artifact.write_bytes(b"changed")
    assert not loaded.validate(artifact, "test")


def test_manifest_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema":"future/v9"}')

    try:
        ArtifactManifest.load(path)
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("unknown manifest schema was accepted")
