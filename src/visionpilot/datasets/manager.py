"""Dataset lifecycle guards shared by download and preparation commands."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class StoragePlan:
    archive_bytes: int
    expansion_multiplier: float = 2.5

    @property
    def required_free_bytes(self) -> int:
        return int(self.archive_bytes * self.expansion_multiplier)

    def ensure_available(self, destination: str | Path) -> None:
        free = shutil.disk_usage(Path(destination)).free
        if free < self.required_free_bytes:
            raise OSError(
                f"Need {self.required_free_bytes / 1024**3:.1f} GiB free for archive, extraction, and cache; "
                f"only {free / 1024**3:.1f} GiB available."
            )


class DatasetManager:
    """Safe archive validation/extraction; network transfer stays an explicit CLI action."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def verified(self, archive: str | Path, expected_sha256: str) -> bool:
        path = Path(archive)
        return path.is_file() and sha256_file(path).lower() == expected_sha256.lower()

    @staticmethod
    def _validate_member(destination: Path, member_name: str) -> None:
        target = (destination / member_name).resolve()
        if destination.resolve() not in target.parents and target != destination.resolve():
            raise ValueError(f"Unsafe archive path rejected: {member_name}")

    def extract(self, archive: str | Path, destination: str | Path) -> list[Path]:
        """Extract a verified zip/tar archive after checking every member for path traversal."""
        archive_path, destination_path = Path(archive), Path(destination)
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        destination_path.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as payload:
                for member in payload.infolist():
                    self._validate_member(destination_path, member.filename)
                payload.extractall(destination_path)
                return [destination_path / member.filename for member in payload.infolist() if not member.is_dir()]
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as payload:
                members = payload.getmembers()
                for member in members:
                    if not (member.isfile() or member.isdir()):
                        raise ValueError(f"Non-file archive member rejected: {member.name}")
                    self._validate_member(destination_path, member.name)
                payload.extractall(destination_path, members=members)
                return [destination_path / member.name for member in members if member.isfile()]
        raise ValueError(f"Unsupported archive type: {archive_path}; use .zip, .tar, .tar.gz, or .tgz")
