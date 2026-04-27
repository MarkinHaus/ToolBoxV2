"""
Load Tasks and Suites from YAML files.
Supports filtering by tags and model multimodal capabilities.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from toolboxv2.mods.isaa.base.bench .core import Attachment, Check, Suite, Task

try:
    import yaml
except ImportError:
    yaml = None  # fallback to json


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml:
        return yaml.safe_load(text)
    # Fallback: try JSON (allows .json task files)
    import json
    return json.loads(text)


def load_task(path: Path) -> Task:
    """Load a single Task from a YAML/JSON file."""
    d = _load_yaml(path)
    if d is None:
        print(f"[bench] skipping {path}")
        return None
    checks = [
        Check(type=c["type"], params={k: v for k, v in c.items() if k != "type"})
        for c in d.get("checks", [])
    ]
    attachments = [
        Attachment(type=a["type"], path=a["path"])
        for a in d.get("attachments", [])
    ]
    return Task(
        id=d["id"],
        complexity=d.get("complexity", "tutorial"),
        modality=d.get("modality", ["text"]),
        prompt=d["prompt"],
        checks=checks,
        tags=d.get("tags", []),
        attachments=attachments,
        ground_truth=d.get("ground_truth"),
    )


def load_tasks_from_dir(task_dir: Path) -> list[Task]:
    """Recursively load all .yaml/.yml/.json task files from a directory."""
    tasks = []
    if not task_dir.exists():
        return tasks
    for ext in ("*.yaml", "*.yml", "*.json"):
        for path in sorted(task_dir.rglob(ext)):
            try:
                t = load_task(path)
                if t is None:
                    continue
                tasks.append(t)
            except KeyError:
                pass  # not a task file (suite, config, etc.)
            except (TypeError, ValueError) as e:
                print(f"[bench] skipping {path}: {e}")
    return tasks


def load_suite(path: Path) -> Suite:
    """Load a Suite definition from YAML."""
    d = _load_yaml(path)
    return Suite(
        id=d["id"],
        name=d.get("name", d["id"]),
        description=d.get("description", ""),
        tasks=d.get("tasks", []),
        task_pattern=d.get("task_pattern", ""),
        tags_filter=d.get("tags_filter", []),
    )


def resolve_suite(suite: Suite, all_tasks: list[Task]) -> list[Task]:
    """Resolve a Suite to its list of Tasks."""
    matched = []

    if suite.tasks:
        # Explicit task IDs
        task_map = {t.id: t for t in all_tasks}
        for tid in suite.tasks:
            if tid in task_map:
                matched.append(task_map[tid])

    if suite.task_pattern:
        # Glob match on task IDs
        for t in all_tasks:
            if fnmatch.fnmatch(t.id, suite.task_pattern) and t not in matched:
                matched.append(t)

    if suite.tags_filter:
        # Filter by tags (task must have ALL specified tags)
        tag_set = set(suite.tags_filter)
        for t in all_tasks:
            if tag_set.issubset(set(t.tags)) and t not in matched:
                matched.append(t)

    # If nothing specified, return all
    if not suite.tasks and not suite.task_pattern and not suite.tags_filter:
        matched = list(all_tasks)

    return matched


def filter_by_modality(tasks: list[Task], model_modalities: list[str]) -> list[Task]:
    """Filter out tasks the model can't handle.

    model_modalities: e.g. ["text", "image"] — what the model supports.
    Tasks requiring unsupported modalities are skipped.
    """
    supported = set(model_modalities)
    result = []
    for t in tasks:
        required = set(t.modality)
        if required.issubset(supported):
            result.append(t)
    return result
