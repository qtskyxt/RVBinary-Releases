"""Atomic artifact manifest with content-digest validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_SCHEMA = "rvbinary.manifest/v1"


def file_digest(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class ArtifactManifest:
    """Content-addressed run state persisted beside project outputs."""

    path: Path
    config_digest: str
    artifacts: dict[str, dict[str, object]] = field(default_factory=dict)
    state: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.config_digest:
            raise ValueError("manifest config_digest must be non-empty")

    def _key(self, path: Path) -> str:
        return os.path.relpath(Path(path).resolve(), self.path.parent.resolve())

    def record(
        self, path: Path, kind: str, metadata: dict[str, object] | None = None
    ) -> None:
        """Record the current digest of one complete artifact."""
        artifact = Path(path)
        if not artifact.is_file():
            raise ValueError(f"cannot record missing artifact: {artifact}")
        self.artifacts[self._key(artifact)] = {
            "kind": kind,
            "sha256": file_digest(artifact),
            "metadata": dict(metadata or {}),
        }

    def validate(self, path: Path, kind: str | None = None) -> bool:
        """Return whether a file still matches its manifest record."""
        artifact = Path(path)
        record = self.artifacts.get(self._key(artifact))
        if record is None or not artifact.is_file():
            return False
        if kind is not None and record.get("kind") != kind:
            return False
        return record.get("sha256") == file_digest(artifact)

    def validate_all(self) -> bool:
        """Validate every recorded artifact without changing state."""
        for relative, record in self.artifacts.items():
            path = (self.path.parent / relative).resolve()
            if not path.is_file() or record.get("sha256") != file_digest(path):
                return False
        return bool(self.artifacts)

    def write(self) -> None:
        """Atomically persist sorted, portable JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": MANIFEST_SCHEMA,
            "config_digest": self.config_digest,
            "artifacts": self.artifacts,
            "state": self.state,
        }
        descriptor, name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
            )
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> ArtifactManifest:
        """Load and structurally validate an existing manifest."""
        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to load manifest {manifest_path}: {exc}") from exc
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(f"unsupported manifest schema: {payload.get('schema')!r}")
        artifacts = payload.get("artifacts")
        state = payload.get("state")
        if not isinstance(artifacts, dict) or not isinstance(state, dict):
            raise ValueError("manifest artifacts and state must be mappings")
        return cls(
            manifest_path,
            str(payload.get("config_digest", "")),
            artifacts=artifacts,
            state=state,
        )
