"""BridgeData V2 TFDS directory manifest, inventory, and download-plan tools.

The official OpenVLA documentation describes BridgeData V2 as a recursively
downloaded TFDS directory renamed ``bridge_orig``.  This module encodes the
expected layout, scans an existing directory, validates completeness, and
generates a download plan — all without executing any actual download.

Archive-mode (``download_bridge.py --mode archive``) is kept for alternate
single-archive workflows.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Manifest: describes the *expected* BridgeData V2 TFDS directory layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BridgeShardSpec:
    """One expected shard file within a TFDS split."""
    relative_path: str
    expected_size_bytes: int | None = None  # None = unknown until scanned


@dataclass(frozen=True)
class BridgeSplitSpec:
    """Expected layout for one TFDS split (train / validation / test)."""
    name: str
    shard_count: int
    shard_pattern: str  # e.g. "bridge_orig-train.tfrecord-{index:05d}-of-{count:05d}"
    metadata_files: tuple[str, ...] = ()

    def expected_shards(self) -> list[str]:
        """Generate expected shard filenames from the pattern."""
        results = []
        for i in range(self.shard_count):
            results.append(
                self.shard_pattern.format(index=i, count=self.shard_count)
            )
        return results


@dataclass(frozen=True)
class BridgeManifest:
    """Full expected layout of the bridge_orig TFDS directory.

    Shard counts and patterns follow the published BridgeData V2 TFDS release.
    Adjust ``from_defaults()`` if the upstream release changes.
    """
    dataset_name: str
    splits: tuple[BridgeSplitSpec, ...]
    top_level_files: tuple[str, ...] = ("dataset_info.json", "features.json")
    estimated_total_bytes: int | None = None

    @classmethod
    def from_defaults(cls) -> "BridgeManifest":
        """Conservative defaults based on published BridgeData V2 documentation.

        The exact shard counts are placeholders until verified against the
        actual GCS bucket listing.  Users should run ``scan_bridge_directory``
        on a partially-downloaded copy to reconcile.
        """
        return cls(
            dataset_name="bridge_orig",
            splits=(
                BridgeSplitSpec(
                    name="train",
                    shard_count=256,
                    shard_pattern="bridge_orig-train.tfrecord-{index:05d}-of-{count:05d}",
                    metadata_files=("train.metadata",),
                ),
                BridgeSplitSpec(
                    name="val",
                    shard_count=16,
                    shard_pattern="bridge_orig-val.tfrecord-{index:05d}-of-{count:05d}",
                    metadata_files=("val.metadata",),
                ),
            ),
            top_level_files=("dataset_info.json", "features.json"),
            estimated_total_bytes=124 * 1024**3,  # ~124 GiB
        )

    def all_expected_files(self) -> list[str]:
        """Return all expected relative paths."""
        files = list(self.top_level_files)
        for split in self.splits:
            files.extend(split.expected_shards())
            files.extend(split.metadata_files)
        return files


# ---------------------------------------------------------------------------
# Inventory: what actually exists on disk
# ---------------------------------------------------------------------------

@dataclass
class BridgeFileInfo:
    """One file found during a directory scan."""
    relative_path: str
    size_bytes: int


@dataclass
class BridgeInventory:
    """Result of scanning a bridge_orig directory."""
    root: str
    files: list[BridgeFileInfo] = field(default_factory=list)
    total_bytes: int = 0
    file_count: int = 0

    def relative_paths(self) -> set[str]:
        return {f.relative_path for f in self.files}


def scan_bridge_directory(root: str | Path) -> BridgeInventory:
    """Walk a directory and catalogue every file found."""
    root_path = Path(root).resolve()
    inventory = BridgeInventory(root=str(root_path))
    if not root_path.is_dir():
        return inventory
    for child in sorted(root_path.rglob("*")):
        if child.is_file():
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            rel = str(child.relative_to(root_path))
            inventory.files.append(BridgeFileInfo(rel, size))
            inventory.total_bytes += size
            inventory.file_count += 1
    return inventory


# ---------------------------------------------------------------------------
# Validation: compare inventory against manifest
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Result of validating an inventory against a manifest."""
    complete: bool = False
    expected_files: int = 0
    found_files: int = 0
    missing_files: list[str] = field(default_factory=list)
    extra_files: list[str] = field(default_factory=list)
    total_found_bytes: int = 0
    estimated_total_bytes: int | None = None
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "expected_files": self.expected_files,
            "found_files": self.found_files,
            "missing_count": len(self.missing_files),
            "missing_files": self.missing_files[:50],  # cap for readability
            "extra_files_count": len(self.extra_files),
            "total_found_bytes": self.total_found_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
            "messages": self.messages,
        }


def validate_bridge_inventory(
    inventory: BridgeInventory,
    manifest: BridgeManifest | None = None,
) -> ValidationReport:
    """Check an inventory against a manifest and report completeness."""
    if manifest is None:
        manifest = BridgeManifest.from_defaults()

    expected = set(manifest.all_expected_files())
    found = inventory.relative_paths()

    missing = sorted(expected - found)
    extra = sorted(found - expected)

    report = ValidationReport(
        complete=len(missing) == 0,
        expected_files=len(expected),
        found_files=len(found & expected),
        missing_files=missing,
        extra_files=extra,
        total_found_bytes=inventory.total_bytes,
        estimated_total_bytes=manifest.estimated_total_bytes,
    )

    if report.complete:
        report.messages.append("All expected files are present.")
    else:
        report.messages.append(
            f"{len(missing)} of {len(expected)} expected files are missing."
        )
    if extra:
        report.messages.append(
            f"{len(extra)} unexpected files found (may be valid auxiliary data)."
        )
    return report


# ---------------------------------------------------------------------------
# Download plan: what still needs to be fetched
# ---------------------------------------------------------------------------

@dataclass
class DownloadPlan:
    """Describes what files are still needed and estimated remaining bytes."""
    files_needed: list[str] = field(default_factory=list)
    files_already_present: int = 0
    estimated_remaining_bytes: int | None = None
    storage_available_bytes: int | None = None
    storage_sufficient: bool | None = None
    recommended_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_needed_count": len(self.files_needed),
            "files_needed": self.files_needed[:50],
            "files_already_present": self.files_already_present,
            "estimated_remaining_bytes": self.estimated_remaining_bytes,
            "storage_available_bytes": self.storage_available_bytes,
            "storage_sufficient": self.storage_sufficient,
            "recommended_command": self.recommended_command,
        }


def generate_download_plan(
    destination: str | Path,
    manifest: BridgeManifest | None = None,
) -> DownloadPlan:
    """Generate a plan for what still needs to be downloaded.

    Does NOT execute any download.  Returns a plan with:
      - list of missing files
      - estimated remaining bytes
      - whether disk has enough space
      - recommended gsutil/rclone command template
    """
    if manifest is None:
        manifest = BridgeManifest.from_defaults()

    dest_path = Path(destination).resolve()
    inventory = scan_bridge_directory(dest_path)
    validation = validate_bridge_inventory(inventory, manifest)

    # Estimate remaining bytes proportionally
    if manifest.estimated_total_bytes and validation.expected_files > 0:
        per_file = manifest.estimated_total_bytes / validation.expected_files
        remaining = int(per_file * len(validation.missing_files))
    else:
        remaining = None

    # Check available disk space (need remaining + 20% safety margin)
    try:
        disk_free = shutil.disk_usage(dest_path if dest_path.exists() else dest_path.parent).free
    except OSError:
        disk_free = None

    storage_ok = None
    if remaining is not None and disk_free is not None:
        storage_ok = disk_free >= int(remaining * 1.2)

    plan = DownloadPlan(
        files_needed=validation.missing_files,
        files_already_present=validation.found_files,
        estimated_remaining_bytes=remaining,
        storage_available_bytes=disk_free,
        storage_sufficient=storage_ok,
        recommended_command=(
            f"# Official TFDS directory download — run manually:\n"
            f"# gsutil -m cp -r gs://gresearch/robotics/{manifest.dataset_name}/ {dest_path}/\n"
            f"# Or use rclone for resumability:\n"
            f"# rclone copy --progress gcs:gresearch/robotics/{manifest.dataset_name}/ {dest_path}/"
        ),
    )
    return plan


def estimate_storage_requirements(
    manifest: BridgeManifest | None = None,
) -> dict[str, int | None]:
    """Estimate storage needed for the full BridgeData V2 workflow."""
    if manifest is None:
        manifest = BridgeManifest.from_defaults()
    raw = manifest.estimated_total_bytes or 0
    return {
        "raw_dataset_bytes": raw,
        "processed_cache_bytes": int(raw * 0.3),       # normalized JSONL + images
        "checkpoint_bytes": 15 * 1024**3,               # ~15 GiB for OpenVLA LoRA
        "total_estimated_bytes": int(raw * 1.3) + 15 * 1024**3,
        "total_estimated_gib": round((int(raw * 1.3) + 15 * 1024**3) / 1024**3, 1),
    }


def write_download_plan(
    plan: DownloadPlan,
    output: str | Path,
) -> Path:
    """Write a download plan to a JSON file."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
