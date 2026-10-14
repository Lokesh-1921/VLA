from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_markdown_report(path: str | Path, title: str, metrics: dict[str, Any], metadata: dict[str, Any] | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Metrics", ""]
    for key, value in metrics.items():
        rendered = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        lines.append(f"- `{key}`: {rendered}")
    if metadata:
        lines.extend(["", "## Metadata", ""])
        for key, value in metadata.items():
            lines.append(f"- `{key}`: {value}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
