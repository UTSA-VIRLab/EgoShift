"""Small generated-table hook for registered results."""

from __future__ import annotations

from typing import Any


def markdown_summary(rows: list[dict[str, Any]]) -> str:
    header = "| Model | Task | Items | Accuracy | Result ID |\n|---|---:|---:|---:|---|"
    body: list[str] = []
    for result in rows:
        model = result.get("model", {}).get("id", "unknown")
        result_id = str(result.get("result_id", ""))[:12]
        for task, values in sorted(result.get("metrics", {}).get("tasks", {}).items()):
            accuracy = values.get("accuracy")
            formatted = "n/a" if accuracy is None else f"{100 * accuracy:.2f}%"
            body.append(f"| {model} | {task} | {values.get('items', 0)} | {formatted} | `{result_id}` |")
    return "\n".join([header, *body]) + "\n"

