# toolboxv2/flows/notebooks.py
# Jupyter Notebook manager — create, run, serve, manage
# Entry: tb -m notebooks <command> [args]

import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from toolboxv2 import App, get_app

NAME = "notebooks"

# ══════════════════════════════════════════════════════════════════════════════
#  Dependency helpers
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_CORE = ["nbformat"]
_REQUIRED_EXEC = ["nbconvert", "ipykernel"]
_REQUIRED_SERVE = ["jupyter", "notebook"]  # `jupyter notebook` or `jupyter lab`


def _pip_cmd() -> list[str]:
    """Return the best available installer command."""
    # prefer uv if available
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install"]
    # fallback: pip via sys.executable
    return [sys.executable, "-m", "pip", "install"]


def _check_import(module: str) -> bool:
    """Check if a module is importable."""
    import importlib
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _ensure_deps(packages: list[str], purpose: str) -> bool:
    """Check deps, prompt to install if missing. Returns True if all available."""
    missing = [p for p in packages if not _check_import(p)]
    if not missing:
        return True

    names = ", ".join(missing)
    print(f"⚠  Missing dependencies for {purpose}: {names}")
    answer = input(f"   Install now? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes", "j", "ja"):
        cmd = _pip_cmd() + missing
        print(f"   Running: {' '.join(cmd)}")
        try:
            subprocess.check_call(cmd)
            # verify
            still_missing = [p for p in missing if not _check_import(p)]
            if still_missing:
                print(f"❌ Still missing after install: {', '.join(still_missing)}")
                return False
            print("✅ Installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Install failed: {e}")
            return False
    else:
        print("   Skipped. Feature unavailable.")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Default template
# ══════════════════════════════════════════════════════════════════════════════

def _default_template(notebook_name: str) -> dict:
    """Return a nbformat-v4 notebook dict with the ToolBoxV2 bootstrap cell."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            "toolboxv2": {
                "created": datetime.now().isoformat(),
                "name": notebook_name,
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {notebook_name}\n",
                    f"*Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
                    "\n",
                    "---",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"tags": ["setup"]},
                "source": [
                    "# ── ToolBoxV2 bootstrap ──────────────────────────────────\n",
                    "from toolboxv2 import get_app, App\n",
                    "\n",
                    "app: App = get_app()\n",
                    "print(f\"ToolBoxV2 {app.version} ready  |  {app.id}\")\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# your code here\n",
                ],
                "execution_count": None,
                "outputs": [],
            },
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Notebook manager
# ══════════════════════════════════════════════════════════════════════════════

class NotebookManager:
    def __init__(self, app: App):
        self.app = app
        self.nb_dir = Path(app.data_dir) / "notebooks"
        self.nb_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, name: str) -> Path:
        """Resolve notebook name to .ipynb path."""
        if not name.endswith(".ipynb"):
            name += ".ipynb"
        return self.nb_dir / name

    # ── create ────────────────────────────────────────────────────────────────

    def create(self, name: str, template: Optional[str] = None) -> Path:
        """Create a new notebook with default template."""
        path = self._resolve(name)
        if path.exists():
            print(f"⚠  Already exists: {path.name}")
            answer = input("   Overwrite? [y/N] ").strip().lower()
            if answer not in ("y", "yes", "j", "ja"):
                print("   Cancelled.")
                return path

        if template and Path(template).exists():
            # use custom template file
            nb_data = json.loads(Path(template).read_text(encoding="utf-8"))
        else:
            clean_name = name.replace(".ipynb", "").replace("_", " ").replace("-", " ").title()
            nb_data = _default_template(clean_name)

        path.write_text(json.dumps(nb_data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Created: {path.name}")
        return path

    # ── list ──────────────────────────────────────────────────────────────────

    def list_notebooks(self):
        """List all notebooks in the directory."""
        nbs = sorted(self.nb_dir.glob("*.ipynb"))
        if not nbs:
            print("   (no notebooks yet — use 'create <name>' to start)")
            return

        print(f"\n  📂 {self.nb_dir}\n")
        for nb in nbs:
            stat = nb.stat()
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            # count cells
            try:
                data = json.loads(nb.read_text(encoding="utf-8"))
                n_cells = len(data.get("cells", []))
            except Exception:
                n_cells = "?"
            print(f"  📓 {nb.stem:30s}  {n_cells:>3} cells  {size_kb:6.1f}kB  {mtime}")
        print()

    # ── run (headless) ────────────────────────────────────────────────────────

    async def run(self, name: str, timeout: int = 600, inplace: bool = False):
        """Execute a notebook headless via nbconvert."""
        if not _ensure_deps(_REQUIRED_EXEC, "notebook execution"):
            return

        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        path = self._resolve(name)
        if not path.exists():
            print(f"❌ Not found: {path.name}")
            return

        print(f"▶  Running {path.name} ...")
        nb = nbformat.read(str(path), as_version=4)

        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name="python3",
        )

        try:
            ep.preprocess(nb, {"metadata": {"path": str(self.nb_dir)}})
            print("✅ Execution finished.")
        except Exception as e:
            print(f"❌ Execution failed: {e}")

        # save output
        if inplace:
            out_path = path
        else:
            out_path = path.with_stem(path.stem + "_out")
        nbformat.write(nb, str(out_path))
        print(f"💾 Output: {out_path.name}")

        # print cell outputs
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code" or not cell.outputs:
                continue
            for out in cell.outputs:
                text = out.get("text", "")
                if not text and "data" in out:
                    text = out["data"].get("text/plain", "")
                if text:
                    print(f"  Cell[{i}]: {text.rstrip()}")

    # ── serve ─────────────────────────────────────────────────────────────────

    def serve(self, port: int = 8888, lab: bool = False):
        """Start a Jupyter server."""
        pkg = "jupyterlab" if lab else "notebook"
        if not _ensure_deps([pkg], f"Jupyter {'Lab' if lab else 'Notebook'} server"):
            return

        cmd_name = "lab" if lab else "notebook"
        cmd = [
            sys.executable, "-m", "jupyter", cmd_name,
            f"--notebook-dir={self.nb_dir}",
            f"--port={port}",
            "--no-browser",
        ]
        print(f"🚀 Starting Jupyter {cmd_name} on port {port} ...")
        print(f"   Dir: {self.nb_dir}")
        print(f"   Ctrl+C to stop\n")

        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n🛑 Jupyter server stopped.")

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, name: str):
        path = self._resolve(name)
        if not path.exists():
            print(f"❌ Not found: {path.name}")
            return
        answer = input(f"   Delete {path.name}? [y/N] ").strip().lower()
        if answer in ("y", "yes", "j", "ja"):
            path.unlink()
            print(f"🗑  Deleted: {path.name}")
        else:
            print("   Cancelled.")

    # ── info ──────────────────────────────────────────────────────────────────

    def info(self, name: str):
        path = self._resolve(name)
        if not path.exists():
            print(f"❌ Not found: {path.name}")
            return

        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        tb_meta = meta.get("toolboxv2", {})
        cells = data.get("cells", [])

        code_cells = [c for c in cells if c["cell_type"] == "code"]
        md_cells = [c for c in cells if c["cell_type"] == "markdown"]

        print(f"\n  📓 {path.name}")
        print(f"     Created:    {tb_meta.get('created', '?')}")
        print(f"     Modified:   {datetime.fromtimestamp(path.stat().st_mtime).isoformat()}")
        print(f"     Cells:      {len(cells)} total ({len(code_cells)} code, {len(md_cells)} markdown)")
        print(f"     Size:       {path.stat().st_size / 1024:.1f} kB")
        print(f"     Kernel:     {meta.get('kernelspec', {}).get('display_name', '?')}")
        print()

    # ── export ────────────────────────────────────────────────────────────────

    def export(self, name: str, fmt: str = "html"):
        """Export notebook to html/pdf/py/md via nbconvert."""
        if not _ensure_deps(["nbconvert"], "notebook export"):
            return

        path = self._resolve(name)
        if not path.exists():
            print(f"❌ Not found: {path.name}")
            return

        cmd = [sys.executable, "-m", "nbconvert", f"--to={fmt}", str(path)]
        print(f"📤 Exporting {path.name} → {fmt} ...")
        try:
            subprocess.check_call(cmd)
            print("✅ Done.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Export failed: {e}")

    # ── duplicate ─────────────────────────────────────────────────────────────

    def duplicate(self, name: str, new_name: Optional[str] = None):
        path = self._resolve(name)
        if not path.exists():
            print(f"❌ Not found: {path.name}")
            return

        if not new_name:
            new_name = path.stem + "_copy"
        dest = self._resolve(new_name)
        shutil.copy2(path, dest)
        print(f"📋 Duplicated: {path.name} → {dest.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI parser + entry point
# ══════════════════════════════════════════════════════════════════════════════

USAGE = """\
\x1b[1;36m  ╔══════════════════════════════════════════╗
  ║   Notebooks  ·  ToolBoxV2                ║
  ║   Jupyter Notebook Manager               ║
  ╚══════════════════════════════════════════╝\x1b[0m

  \x1b[1mCommands:\x1b[0m
    create  <name> [--template <path>]   Create notebook with default template
    list                                 List all notebooks
    run     <name> [--timeout N]         Execute headless (nbconvert)
                   [--inplace]
    serve   [--port N] [--lab]           Start Jupyter server
    info    <name>                       Show notebook metadata
    export  <name> [--format html|pdf|py|md]
    delete  <name>                       Delete notebook
    dup     <name> [new_name]            Duplicate notebook
"""


def _parse_args(args) -> tuple[str, dict]:
    """Parse CLI args into (command, kwargs)."""
    if hasattr(args, '__iter__') and not isinstance(args, str):
        argv = list(args) if args else []
    else:
        argv = str(args).split() if args else []

    if not argv:
        return "help", {}

    cmd = argv[0].lower()
    rest = argv[1:]
    kwargs = {}

    if cmd in ("create", "new"):
        cmd = "create"
        if rest:
            kwargs["name"] = rest[0]
        if "--template" in rest:
            idx = rest.index("--template")
            if idx + 1 < len(rest):
                kwargs["template"] = rest[idx + 1]

    elif cmd in ("list", "ls"):
        cmd = "list"

    elif cmd in ("run", "exec"):
        cmd = "run"
        if rest:
            kwargs["name"] = rest[0]
        if "--timeout" in rest:
            idx = rest.index("--timeout")
            if idx + 1 < len(rest):
                kwargs["timeout"] = int(rest[idx + 1])
        if "--inplace" in rest:
            kwargs["inplace"] = True

    elif cmd == "serve":
        if "--port" in rest:
            idx = rest.index("--port")
            if idx + 1 < len(rest):
                kwargs["port"] = int(rest[idx + 1])
        if "--lab" in rest:
            kwargs["lab"] = True

    elif cmd == "info":
        if rest:
            kwargs["name"] = rest[0]

    elif cmd == "export":
        if rest:
            kwargs["name"] = rest[0]
        if "--format" in rest:
            idx = rest.index("--format")
            if idx + 1 < len(rest):
                kwargs["fmt"] = rest[idx + 1]

    elif cmd in ("delete", "del", "rm"):
        cmd = "delete"
        if rest:
            kwargs["name"] = rest[0]

    elif cmd in ("dup", "duplicate", "copy", "cp"):
        cmd = "dup"
        if rest:
            kwargs["name"] = rest[0]
        if len(rest) > 1:
            kwargs["new_name"] = rest[1]

    else:
        cmd = "help"

    return cmd, kwargs


async def run(app: App, args):
    if not _ensure_deps(_REQUIRED_CORE, "notebook management"):
        return

    mgr = NotebookManager(app)
    cmd, kwargs = _parse_args(sys.argv[3:])

    if cmd == "help":
        print(USAGE)

    elif cmd == "create":
        if "name" not in kwargs:
            print("❌ Usage: create <name> [--template <path>]")
            return
        mgr.create(**kwargs)

    elif cmd == "list":
        mgr.list_notebooks()

    elif cmd == "run":
        if "name" not in kwargs:
            print("❌ Usage: run <name> [--timeout N] [--inplace]")
            return
        await mgr.run(**kwargs)

    elif cmd == "serve":
        mgr.serve(**kwargs)

    elif cmd == "info":
        if "name" not in kwargs:
            print("❌ Usage: info <name>")
            return
        mgr.info(**kwargs)

    elif cmd == "export":
        if "name" not in kwargs:
            print("❌ Usage: export <name> [--format html|pdf|py|md]")
            return
        mgr.export(**kwargs)

    elif cmd == "delete":
        if "name" not in kwargs:
            print("❌ Usage: delete <name>")
            return
        mgr.delete(**kwargs)

    elif cmd == "dup":
        if "name" not in kwargs:
            print("❌ Usage: dup <name> [new_name]")
            return
        mgr.duplicate(**kwargs)

    await app.a_exit()
