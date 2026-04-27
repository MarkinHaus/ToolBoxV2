"""
isaa.base.bench — Binary benchmark framework.
All scoring is pass/fail. No scales, no vibes.

Usage:
    # Direct
    from bench import RowModelAdapter
    adapter = RowModelAdapter("openrouter/google/gemini-2.5-flash", task_dir="tasks/")
    report = await adapter.benchmark("gemini-flash")

    # CLI
    python -m bench run --task-dir tasks/ --model openrouter/google/gemini-2.5-flash
    python -m bench calibrate --task-dir tasks/calibration/
    python -m bench dashboard reports/*.json -o comparison.html
"""

from toolboxv2.mods.isaa.base.bench.core import (
    Attachment,
    Check,
    CheckResult,
    JudgeProfile,
    Report,
    Suite,
    Task,
    TaskContext,
    TaskResult,
)
from toolboxv2.mods.isaa.base.bench.loader import (
    filter_by_modality,
    load_suite,
    load_task,
    load_tasks_from_dir,
    resolve_suite,
)
from toolboxv2.mods.isaa.base.bench.runner import run_tasks, run_tasks_sync
from toolboxv2.mods.isaa.base.bench.adapters import (
    AgentAdapter,
    AgentStreamAdapter,
    RowModelAdapter,
)
from toolboxv2.mods.isaa.base.bench.calibrator import calibrate

# Ensure builtins + judge validators are registered on import
import toolboxv2.mods.isaa.base.bench.validators.builtins  # noqa: F401
import toolboxv2.mods.isaa.base.bench.validators.judge  # noqa: F401
import toolboxv2.mods.isaa.base.bench.honesty  # noqa: F401

__all__ = [
    "Attachment", "Check", "CheckResult", "JudgeProfile",
    "Report", "Suite", "Task", "TaskContext", "TaskResult",
    "filter_by_modality", "load_suite", "load_task",
    "load_tasks_from_dir", "resolve_suite",
    "run_tasks", "run_tasks_sync",
    "AgentAdapter", "AgentStreamAdapter", "RowModelAdapter",
    "calibrate",
]

if __name__ == "__main__":
    import os
    import re


    def split_yaml_tasks(input_file="tasks.yaml", output_dir="tasks"):
        # Output-Ordner erstellen
        os.makedirs(output_dir, exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Am Trennzeichen '---' aufsplitten
        chunks = re.split(r'^---\s*$', content, flags=re.MULTILINE)

        saved_count = 0
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # ID für den Dateinamen extrahieren
            match = re.search(r'^id:\s*([^\s]+)', chunk, re.MULTILINE)
            if match:
                task_id = match.group(1).strip()
                filepath = os.path.join(output_dir, f"{task_id}.yaml")

                # Datei schreiben
                with open(filepath, 'w', encoding='utf-8') as out_f:
                    out_f.write(chunk + '\n')

                print(f"✓ {filepath}")
                saved_count += 1
            else:
                print("⚠️ Warnung: Block ohne 'id:' gefunden. Übersprungen.")

        print(f"\nErfolgreich {saved_count} YAML-Dateien in ./{output_dir}/ erstellt.")


    # split_yaml_tasks(r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\isaa\base\bench\tasks\collection_002.yaml",
    #                  r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\isaa\base\bench\tasks\calc")
