# Data layer

Every dataset becomes a `RobotSample`: image reference or in-memory image, instruction, optional state, native action, episode ID, timestep, source, embodiment, and optional next-frame reference. The JSONL wire format preserves these fields and makes sample-level inspection possible before preprocessing large data.

`BridgeV2Dataset` accepts an inspected normalized JSONL export. It intentionally does not guess the evolving upstream archive layout. `RLDSDataset` is a documented adapter boundary; use it after pinning an RLDS/TensorFlow version. `CustomVisionPilotDataset` reads records written by the webcam collector. Deterministic episode-level splits are stored separately from raw data.

## BridgeData V2 download modes

### Mode 1: TFDS directory (official release)

The official BridgeData V2 release is a TFDS (TensorFlow Datasets) directory hosted on GCS, conventionally renamed `bridge_orig`. The expected layout is:

```text
bridge_orig/
├── dataset_info.json
├── features.json
├── bridge_orig-train.tfrecord-00000-of-00256
├── bridge_orig-train.tfrecord-00001-of-00256
├── ...
├── train.metadata
├── bridge_orig-val.tfrecord-00000-of-00016
├── ...
└── val.metadata
```

The `bridge_manifest.py` module provides:

- **`BridgeManifest`** — encodes the expected directory structure
- **`scan_bridge_directory(root)`** — walks an existing directory and catalogues files
- **`validate_bridge_inventory(inventory, manifest)`** — checks completeness against expectations
- **`generate_download_plan(destination)`** — lists missing files, estimates remaining bytes, prints recommended gsutil/rclone commands
- **`estimate_storage_requirements()`** — estimates raw + processed + checkpoint storage

Use `download_bridge.py --mode tfds-directory` for this workflow. The script generates a download plan and validates existing files but **does not execute any download**.

### Mode 2: Single archive (alternative)

The downloader also supports a single verified archive URL + SHA-256 for alternative distribution channels. This mode takes an explicit approved source URL and SHA-256, checks a conservative storage budget, supports HTTP range resume, and skips an already verified archive.

Use `download_bridge.py --mode archive --source URL --sha256 CHECKSUM` for this workflow.
