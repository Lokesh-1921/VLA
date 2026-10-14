#!/usr/bin/env python3
"""Explicit, resumable BridgeData downloader with storage and checksum gates.

Supports two modes:
  archive       — single verified archive URL + SHA-256 (original mode)
  tfds-directory — official TFDS directory-style release (scan/plan/validate only)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
from visionpilot.datasets.manager import DatasetManager, StoragePlan, sha256_file
from visionpilot.utils.logging import write_json


def archive_download(args: argparse.Namespace) -> int:
    """Original archive-mode download with resume and checksum."""
    if not args.source or not args.sha256:
        print("ERROR: --source and --sha256 are required for archive mode.")
        return 1
    if args.expected_gb <= 0 or args.required_free_multiplier < 1:
        print("ERROR: expected size must be positive and free-space multiplier at least 1")
        return 1
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manager = DatasetManager(output)
    final, partial = output / args.filename, output / f"{args.filename}.part"
    plan = StoragePlan(int(args.expected_gb * 1024**3), args.required_free_multiplier)
    try:
        plan.ensure_available(output)
    except OSError as exc:
        print(f"ERROR: {exc}")
        return 1
    if manager.verified(final, args.sha256):
        print(f"Verified archive already exists: {final}")
        if args.extract_to:
            extracted = manager.extract(final, args.extract_to)
            print(f"Safely extracted {len(extracted)} files to {args.extract_to}")
        return 0
    start_byte = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(args.source)
    if start_byte:
        request.add_header("Range", f"bytes={start_byte}-")
    print(f"Downloading to {partial} ({'resuming at ' + str(start_byte) if start_byte else 'starting fresh'})")
    started = time.monotonic()
    with urllib.request.urlopen(request) as response:
        resumed = start_byte > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if start_byte and not resumed:
            print("Server did not accept range request; restarting the partial file safely.")
        with partial.open(mode) as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
                received = handle.tell()
                elapsed = max(time.monotonic() - started, 0.001)
                print(f"\rreceived {received / 1024**3:.2f} GiB ({received / 1024**2 / elapsed:.1f} MiB/s)", end="", flush=True)
    print()
    actual = sha256_file(partial)
    if actual.lower() != args.sha256.lower():
        raise RuntimeError(f"Checksum mismatch; partial retained for inspection/resume. expected={args.sha256}, actual={actual}")
    partial.replace(final)
    write_json(output / "download_metadata.json", {"source": args.source, "archive": str(final), "sha256": actual, "bytes": final.stat().st_size, "downloaded_at_unix": time.time()})
    if args.extract_to:
        extracted = manager.extract(final, args.extract_to)
        write_json(output / "extraction_metadata.json", {"archive": str(final), "destination": str(args.extract_to), "file_count": len(extracted)})
        print(f"Safely extracted {len(extracted)} files to {args.extract_to}")
    print(f"Verified archive saved at {final}. Inspect before extracting/preprocessing.")
    return 0


def tfds_directory_mode(args: argparse.Namespace) -> int:
    """TFDS directory-style mode: scan, validate, plan — no actual download."""
    from visionpilot.datasets.bridge_manifest import (
        BridgeManifest,
        estimate_storage_requirements,
        generate_download_plan,
        scan_bridge_directory,
        validate_bridge_inventory,
        write_download_plan,
    )

    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = BridgeManifest.from_defaults()

    if args.validate_only:
        print(f"Scanning existing directory: {dest}")
        inventory = scan_bridge_directory(dest)
        report = validate_bridge_inventory(inventory, manifest)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        plan_path = _bootstrap.ROOT / "data" / "metadata" / "bridge_v2_validation.json"
        write_json(plan_path, report.to_dict())
        print(f"Validation report written to {plan_path}")
        return 0 if report.complete else 1

    # Generate download plan
    print(f"Generating download plan for TFDS directory at: {dest}")
    plan = generate_download_plan(dest, manifest)
    storage = estimate_storage_requirements(manifest)

    print(f"\nBridgeData V2 TFDS Download Plan")
    print(f"  Files already present: {plan.files_already_present}")
    print(f"  Files still needed:    {len(plan.files_needed)}")
    if plan.estimated_remaining_bytes:
        print(f"  Estimated remaining:   {plan.estimated_remaining_bytes / 1024**3:.1f} GiB")
    if plan.storage_available_bytes:
        print(f"  Disk free:             {plan.storage_available_bytes / 1024**3:.1f} GiB")
    if plan.storage_sufficient is not None:
        print(f"  Storage sufficient:    {plan.storage_sufficient}")
    print(f"\nTotal storage estimate (raw + processed + checkpoints): {storage['total_estimated_gib']} GiB")
    print(f"\n{plan.recommended_command}")
    print("\nThis command was NOT executed. Run it manually after verifying disk space.")

    plan_path = _bootstrap.ROOT / "data" / "metadata" / "bridge_v2_download_plan.json"
    write_download_plan(plan, plan_path)
    write_json(
        _bootstrap.ROOT / "data" / "metadata" / "bridge_v2_storage_estimate.json",
        storage,
    )
    print(f"\nDownload plan written to {plan_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("archive", "tfds-directory"), default="archive",
                        help="archive: single URL+SHA-256 download; tfds-directory: official TFDS directory scan/plan")
    parser.add_argument("--source", help="verified archive URL (archive mode only)")
    parser.add_argument("--sha256", help="verified archive SHA-256 (archive mode only)")
    parser.add_argument("--output-dir", default="data/raw/bridge_v2")
    parser.add_argument("--filename", default="bridge_v2.archive")
    parser.add_argument("--expected-gb", type=float, default=124.0)
    parser.add_argument("--required-free-multiplier", type=float, default=2.5,
                        help="archive + extraction + preprocessing safety factor")
    parser.add_argument("--extract-to",
                        help="optional explicit destination for safe extraction after checksum validation (archive mode)")
    parser.add_argument("--validate-only", action="store_true",
                        help="scan and validate an existing directory without generating a download plan (tfds-directory mode)")
    args = parser.parse_args()

    if args.mode == "archive":
        return archive_download(args)
    else:
        return tfds_directory_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
