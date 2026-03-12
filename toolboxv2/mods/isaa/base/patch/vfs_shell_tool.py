"""
vfs_shell_tool.py
=================
Unix-like shell interface and context manager for VirtualFileSystemV2.

Provides exactly TWO primary agent tools that replace ~18 individual VFS tools:

  vfs_shell(command)      — All filesystem operations via unix-like commands
  vfs_view(path, ...)     — Scroll / focus control for the context window

Usage in init_session_tools():
    from toolboxv2.mods.isaa.base.Agent.vfs_shell_tool import make_vfs_shell, make_vfs_view

    vfs_shell = make_vfs_shell(session)
    vfs_view  = make_vfs_view(session)
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.agent_session_v2 import AgentSessionV2

from toolboxv2.mods.isaa.base.patch.power_vfs import find_files, grep_vfs


# =============================================================================
# HELPERS
# =============================================================================

def _ok(stdout: str = "", stderr: str = "", returncode=0) -> dict:
    return {"success": returncode==0,  "stdout": stdout, "stderr": stderr, "returncode": returncode}

def _err(stderr: Any, returncode: int = 1) -> dict:
    return {"success": False, "stdout": "", "stderr": str(stderr), "returncode": returncode}


def _parse_n_flag(args: list[str], default: int = 10) -> tuple[int, list[str]]:
    """Extract -n N / -nN from args, return (n, remaining_args)."""
    n = default
    rest: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-n" and i + 1 < len(args):
            try:
                n = int(args[i + 1])
                i += 2
                continue
            except ValueError:
                pass
        elif a.startswith("-n") and len(a) > 2:
            try:
                n = int(a[2:])
                i += 1
                continue
            except ValueError:
                pass
        rest.append(a)
        i += 1
    return n, rest


# =============================================================================
# VFS SHELL FACTORY
# =============================================================================

def make_vfs_shell(session: "AgentSessionV2"):
    """
    Factory — returns a vfs_shell closure bound to *session*.

    Call once in init_session_tools() and register the returned function as a tool.
    """
    vfs = session.vfs

    def vfs_shell(command: str) -> dict:
        """
        Unix-like shell interface for VFS operations.

        Supported commands
        ------------------
        NAVIGATION  ls [-la] [-R] [path]  |  pwd  |  tree [path] [-L depth]
        READ        cat <path>  |  head [-n N] <path>  |  tail [-n N] <path>
                    wc [-lwc] <path>  |  stat <path>
        SEARCH      find [path] [-name pattern] [-type f|d]
                    grep [-rniIlC N] <pattern> [path|file_pattern]
        WRITE       touch <path>
                    echo "text" > <path>        (overwrite)
                    echo "text" >> <path>       (append)
                    write <path> "content"      (multi-line, \\n supported)
                    edit <path> <start> <end> "new_content"   (line-range replace)
                    mkdir [-p] <path>
                    rm [-rf] <path>
                    mv <src> <dst>
                    cp <src> <dst>
        CONTEXT     close <path>               (remove from context window)
        EXECUTE     exec <path> [args...]

        Returns
        -------
        {"success": bool, "stdout": str, "stderr": str, "returncode": int}

        Examples
        --------
        vfs_shell("ls -la /src")
        vfs_shell("grep -rn 'def train' /src")
        vfs_shell("write /src/config.py 'HOST = \\"localhost\\"\\nPORT = 8080'")
        vfs_shell("edit /src/main.py 10 14 'def main():\\n    pass'")
        vfs_shell("close /src/utils.py")
        """
        command = command.strip()
        if not command:
            return _err("empty command")

        # ── Special-case: echo with shell redirection ──────────────────────
        # Matches:  echo "..." > path   OR   echo "..." >> path
        echo_m = re.match(
            r"^echo\s+(.*?)\s*(>>|>)\s*(\S+)\s*$", command, re.DOTALL
        )
        if echo_m:
            raw_content, op, path = echo_m.groups()
            # Strip a single surrounding quote layer
            if len(raw_content) >= 2 and raw_content[0] == raw_content[-1] and raw_content[0] in ('"', "'"):
                raw_content = raw_content[1:-1]
            content = raw_content.replace("\\n", "\n").replace("\\t", "\t")
            r = vfs.write(path, content) if op == ">" else vfs.append(path, content)
            return _ok() if r.get("success") else _err(r.get("error", "write failed"))

        # ── Parse command ──────────────────────────────────────────────────
        try:
            args = shlex.split(command)
        except ValueError as e:
            return _err(f"parse error: {e}")

        if not args:
            return _err("empty command")

        cmd, rest = args[0].lower(), args[1:]

        # ═══════════════════════════════════════════════════════════════════
        # pwd
        # ═══════════════════════════════════════════════════════════════════
        if cmd == "pwd":
            return _ok("/")

        # ═══════════════════════════════════════════════════════════════════
        # ls [-la] [-R] [path]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "ls":
            long_fmt = recursive = False
            path = "/"
            for a in rest:
                if a.startswith("-"):
                    if any(c in a for c in ("l", "a")):
                        long_fmt = True
                    if "R" in a:
                        recursive = True
                else:
                    path = a

            r = vfs.ls(path, recursive=recursive, show_hidden=True)
            if not r.get("success"):
                return _err(r.get("error", "ls failed"))

            lines: list[str] = []
            for item in r.get("contents", []):
                is_dir = item["type"] == "directory"
                name = item["name"] + ("/" if is_dir else "")
                if long_fmt:
                    size  = item.get("size", 0)
                    state = f" [{item['state']}]" if item.get("state") else ""
                    ftype = item.get("file_type", "")
                    prefix = "d" if is_dir else "-"
                    lines.append(f"{prefix}  {size:>8}  {name:<40}{state}  {ftype}")
                else:
                    lines.append(name)

            return _ok("\n".join(lines) if lines else "(empty directory)")

        # ═══════════════════════════════════════════════════════════════════
        # tree [path] [-L depth]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "tree":
            path = "/"
            depth = 4
            i = 0
            while i < len(rest):
                a = rest[i]
                if a in ("-L", "-d") and i + 1 < len(rest):
                    try:
                        depth = int(rest[i + 1])
                    except ValueError:
                        pass
                    i += 2
                elif a.startswith(("-L", "-d")) and len(a) > 2:
                    try:
                        depth = int(a[2:])
                    except ValueError:
                        pass
                    i += 1
                elif not a.startswith("-"):
                    path = a
                    i += 1
                else:
                    i += 1
            tree = vfs._build_tree_string(path, max_depth=depth)
            return _ok(f"{path}\n{tree}" if tree else f"{path}\n(empty)")

        # ═══════════════════════════════════════════════════════════════════
        # cat <path> [path ...]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "cat":
            if not rest:
                return _err("cat: missing file operand")
            parts: list[str] = []
            for p in rest:
                r = vfs.read(p)
                if not r.get("success"):
                    parts.append(f"cat: {p}: {r.get('error', 'no such file')}")
                else:
                    parts.append(r["content"])
            return _ok("\n".join(parts))

        # ═══════════════════════════════════════════════════════════════════
        # head [-n N] <path>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "head":
            n, paths = _parse_n_flag(rest, default=10)
            paths = [p for p in paths if not p.startswith("-")]
            if not paths:
                return _err("head: missing file operand")
            parts = []
            for p in paths:
                r = vfs.read(p)
                if not r.get("success"):
                    parts.append(f"head: {p}: {r.get('error', 'no such file')}")
                else:
                    parts.append("\n".join(r["content"].split("\n")[:n]))
            return _ok("\n".join(parts))

        # ═══════════════════════════════════════════════════════════════════
        # tail [-n N] <path>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "tail":
            n, paths = _parse_n_flag(rest, default=10)
            paths = [p for p in paths if not p.startswith("-")]
            if not paths:
                return _err("tail: missing file operand")
            parts = []
            for p in paths:
                r = vfs.read(p)
                if not r.get("success"):
                    parts.append(f"tail: {p}: {r.get('error', 'no such file')}")
                else:
                    parts.append("\n".join(r["content"].split("\n")[-n:]))
            return _ok("\n".join(parts))

        # ═══════════════════════════════════════════════════════════════════
        # wc [-l | -w | -c] <path>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "wc":
            flags = [a for a in rest if a.startswith("-")]
            paths = [a for a in rest if not a.startswith("-")]
            if not paths:
                return _err("wc: missing file operand")
            lines_out = []
            for p in paths:
                r = vfs.read(p)
                if not r.get("success"):
                    lines_out.append(f"wc: {p}: {r.get('error', 'no such file')}")
                    continue
                c = r["content"]
                lc, wc, cc = len(c.splitlines()), len(c.split()), len(c)
                if "-l" in flags:
                    lines_out.append(f"{lc:>8} {p}")
                elif "-w" in flags:
                    lines_out.append(f"{wc:>8} {p}")
                elif "-c" in flags:
                    lines_out.append(f"{cc:>8} {p}")
                else:
                    lines_out.append(f"{lc:>8} {wc:>8} {cc:>8} {p}")
            return _ok("\n".join(lines_out))

        # ═══════════════════════════════════════════════════════════════════
        # stat / info <path>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd in ("stat", "info"):
            if not rest:
                return _err(f"{cmd}: missing operand")
            p = next((a for a in rest if not a.startswith("-")), None)
            if not p:
                return _err(f"{cmd}: missing operand")
            r = vfs.get_file_info(p)
            if not r.get("success"):
                return _err(r.get("error", "not found"))
            lines_out = [f"  File: {p}"]
            for k, v in r.items():
                if k not in ("success", "path"):
                    lines_out.append(f"  {k:20}: {v}")
            return _ok("\n".join(lines_out))

        # ═══════════════════════════════════════════════════════════════════
        # find [path] [-name pattern] [-type f|d]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "find":
            path = "/"
            name_pattern = "*"
            file_type: str | None = None
            i = 0
            while i < len(rest):
                a = rest[i]
                if a == "-name" and i + 1 < len(rest):
                    name_pattern = rest[i + 1]; i += 2
                elif a == "-type" and i + 1 < len(rest):
                    file_type = rest[i + 1]; i += 2
                elif not a.startswith("-"):
                    path = a; i += 1
                else:
                    i += 1

            results: list[str] = []
            if file_type != "d":
                results.extend(find_files(vfs, name_pattern, path))
            if file_type != "f":
                prefix = path.rstrip("/") + "/"
                for dp in sorted(vfs.directories):
                    if dp == path:
                        continue
                    if dp.startswith(prefix) or path == "/":
                        dname = os.path.basename(dp)
                        if fnmatch.fnmatch(dname, name_pattern):
                            results.append(dp + "/")
            results.sort()
            return _ok("\n".join(results) if results else "(no matches)", returncode=0 if results else 1)

        # ═══════════════════════════════════════════════════════════════════
        # grep [-r] [-i] [-n] [-l] [-C N] <pattern> [path]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "grep":
            recursive = case_insensitive = show_lineno = files_only = False
            context_n = 0
            pattern: str | None = None
            targets: list[str] = []
            i = 0
            while i < len(rest):
                a = rest[i]
                if a.startswith("-") and not re.match(r"^-\d+$", a):
                    if "r" in a or "R" in a: recursive = True
                    if "i" in a:             case_insensitive = True
                    if "n" in a:             show_lineno = True
                    if "l" in a:             files_only = True
                    if "C" in a:
                        # -C N or -CN
                        tail_part = a[a.index("C") + 1:]
                        if tail_part.isdigit():
                            context_n = int(tail_part)
                        elif i + 1 < len(rest) and rest[i + 1].isdigit():
                            context_n = int(rest[i + 1]); i += 1
                    i += 1
                else:
                    if pattern is None:
                        pattern = a
                    else:
                        targets.append(a)
                    i += 1

            if pattern is None:
                return _err("grep: missing pattern")

            # Build case-insensitive pattern if needed
            grep_pattern = f"(?i){pattern}" if case_insensitive else pattern

            # Resolve search path and file_pattern
            if not targets:
                search_path, file_pattern = ("/", "*") if recursive else ("", "")
                if not recursive:
                    return _err("grep: no file specified (use -r for recursive search)")
            else:
                t = targets[-1]
                if vfs._is_directory(t):
                    search_path, file_pattern = t, "*"
                elif vfs._is_file(t):
                    search_path = str(os.path.dirname(vfs._normalize_path(t))) or "/"
                    file_pattern = os.path.basename(t)
                else:
                    # Treat as file glob pattern, search from /
                    search_path, file_pattern = "/", t

            grep_results = grep_vfs(
                vfs=vfs,
                pattern=grep_pattern,
                file_pattern=file_pattern,
                path=search_path,
                context_lines=context_n,
            )

            if not grep_results:
                return _ok("(no matches)", returncode=1)

            seen_files: list[str] = []
            out_lines: list[str] = []
            for m in grep_results:
                if files_only:
                    if m["file"] not in seen_files:
                        seen_files.append(m["file"])
                    continue
                prefix = f"{m['file']}:"
                if show_lineno:
                    prefix += f"{m['line']}:"
                out_lines.append(f"{prefix}{m['match']}")
                if context_n and "context" in m:
                    for cl in m["context"]:
                        out_lines.append(f"  {cl}")
                    out_lines.append("--")

            final = "\n".join(seen_files if files_only else out_lines)
            return _ok(final)

        # ═══════════════════════════════════════════════════════════════════
        # touch <path> [path ...]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "touch":
            if not rest:
                return _err("touch: missing file operand")
            msgs: list[str] = []
            for p in rest:
                np = vfs._normalize_path(p)
                if not vfs._is_file(np):
                    r = vfs.create(p, "")
                    if not r.get("success"):
                        return _err(r.get("error", "touch failed"))
                    msgs.append(f"created {p}")
                else:
                    vfs.files[np].updated_at = datetime.now().isoformat()
                    msgs.append(f"touched {p}")
            return _ok("\n".join(msgs))

        # ═══════════════════════════════════════════════════════════════════
        # write <path> <content>   (multi-line aware, \n \t processed)
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "write":
            if len(rest) < 2:
                return _err("write: usage: write <path> <content>")
            path = rest[0]
            # Extract content from raw command to preserve \n before any tokenizer eats it
            raw = re.match(r"write\s+\S+\s+(.*)", command, re.DOTALL)
            raw_content = raw.group(1) if raw else " ".join(rest[1:])
            if len(raw_content) >= 2 and raw_content[0] == raw_content[-1] and raw_content[0] in ('"', "'"):
                raw_content = raw_content[1:-1]
            content = raw_content.replace("\\n", "\n").replace("\\t", "\t")
            r = vfs.write(path, content)
            return _ok(f"written: {path} ({len(content)} chars)") if r.get("success") else _err(r.get("error", ""))
        # ═══════════════════════════════════════════════════════════════════
        # edit <path> <line_start> <line_end> <new_content>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "edit":
            if len(rest) < 4:
                return _err("edit: usage: edit <path> <line_start> <line_end> <new_content>")
            path = rest[0]
            try:
                line_start, line_end = int(rest[1]), int(rest[2])
            except ValueError:
                return _err("edit: line_start and line_end must be integers")
            new_content = " ".join(rest[3:]).replace("\\n", "\n").replace("\\t", "\t")
            r = vfs.edit(path, line_start, line_end, new_content)
            return _ok(r.get("message", "ok")) if r.get("success") else _err(r.get("error", ""))

        # ═══════════════════════════════════════════════════════════════════
        # mkdir [-p] <path> [path ...]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "mkdir":
            if not rest:
                return _err("mkdir: missing operand")
            parents = any(a in ("-p", "--parents") for a in rest)
            paths = [a for a in rest if not a.startswith("-")]
            if not paths:
                return _err("mkdir: missing path operand")
            msgs: list[str] = []
            for p in paths:
                r = vfs.mkdir(p, parents=parents)
                if not r.get("success"):
                    return _err(r.get("error", "mkdir failed"))
                msgs.append(r.get("message", f"created {p}"))
            return _ok("\n".join(msgs))

        # ═══════════════════════════════════════════════════════════════════
        # rm [-rf] <path> [path ...]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "rm":
            if not rest:
                return _err("rm: missing operand")
            flag_str = "".join(a for a in rest if a.startswith("-"))
            recursive_rm = "r" in flag_str or "R" in flag_str
            force_rm     = "f" in flag_str
            paths = [a for a in rest if not a.startswith("-")]
            if not paths:
                return _err("rm: missing path operand")
            msgs: list[str] = []
            for p in paths:
                if vfs._is_directory(p):
                    if not recursive_rm and not force_rm:
                        return _err(f"rm: {p}: is a directory (use -rf)")
                    r = vfs.rmdir(p, force=True)
                else:
                    r = vfs.delete(p)
                if not r.get("success"):
                    if not force_rm:
                        return _err(r.get("error", f"rm: cannot remove {p}"))
                else:
                    msgs.append(r.get("message", f"removed {p}"))
            return _ok("\n".join(msgs))

        # ═══════════════════════════════════════════════════════════════════
        # mv <src> <dst>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "mv":
            non_flag = [a for a in rest if not a.startswith("-")]
            if len(non_flag) < 2:
                return _err("mv: missing destination operand")
            r = vfs.mv(non_flag[0], non_flag[1])
            return _ok(r.get("message", "")) if r.get("success") else _err(r.get("error", "mv failed"))

        # ═══════════════════════════════════════════════════════════════════
        # cp <src> <dst>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "cp":
            non_flag = [a for a in rest if not a.startswith("-")]
            if len(non_flag) < 2:
                return _err("cp: missing destination operand")
            src, dst = non_flag[0], non_flag[1]
            rr = vfs.read(src)
            if not rr.get("success"):
                return _err(f"cp: cannot read {src}: {rr.get('error', '')}")
            # If dst is a directory → copy into it
            if vfs._is_directory(dst):
                dst = dst.rstrip("/") + "/" + os.path.basename(src)
            wr = vfs.write(dst, rr["content"])
            if not wr.get("success"):
                wr = vfs.create(dst, rr["content"])
            return _ok(f"copied {src} → {dst}") if wr.get("success") else _err(wr.get("error", "cp failed"))

        # ═══════════════════════════════════════════════════════════════════
        # close <path>  — remove file from context window (sync, no summary)
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "close":
            if not rest:
                return _err("close: missing file operand")
            msgs: list[str] = []
            for p in rest:
                np = vfs._normalize_path(p)
                if not vfs._is_file(np):
                    return _err(f"close: {p}: no such file")
                f = vfs.files[np]
                if f.readonly:
                    return _err(f"close: {p}: cannot close system file")
                f.state = "closed"
                f.mini_summary = f"[{f.size} chars — closed via vfs_shell]"
                vfs._dirty = True
                msgs.append(f"closed: {p}")
            return _ok("\n".join(msgs))

        # ═══════════════════════════════════════════════════════════════════
        # exec <path> [args...]
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "exec":
            if not rest:
                return _err("exec: missing file operand")
            path = rest[0]
            exec_args = rest[1:] if len(rest) > 1 else None
            r = vfs.execute(path, args=exec_args)
            if not r.get("success"):
                return _err(
                    r.get("stderr", r.get("error", "exec failed")),
                    returncode=r.get("return_code", 1),
                )
            return _ok(r.get("stdout", ""), r.get("stderr", ""))

        # ═══════════════════════════════════════════════════════════════════
        # Unknown
        # ═══════════════════════════════════════════════════════════════════
        else:
            return _err(
                f"vfs_shell: {cmd}: command not found\n"
                "Try: ls cat head tail wc stat tree find grep "
                "touch write edit echo mkdir rm mv cp close exec"
            )

    return vfs_shell


# =============================================================================
# VFS VIEW FACTORY  (context window / scroll tool)
# =============================================================================

def make_vfs_view(session: "AgentSessionV2"):
    """
    Factory — returns a vfs_view closure bound to *session*.

    Call once in init_session_tools() and register the returned function as a tool.
    """
    vfs = session.vfs

    def vfs_view(
        path: str,
        line_start: int = 1,
        line_end: int = -1,
        scroll_to: str | None = None,
        context_lines: int = 40,
        close_others: bool = False,
    ) -> dict:
        """
        Open or scroll to a specific section of a file in the VFS context window.

        Files opened here are **permanently visible** in every subsequent prompt
        until explicitly closed (via `vfs_shell("close <path>")` or close_others=True).

        Core Workflow — Finding two related things x and y
        ---------------------------------------------------
        # 1. Locate x
        vfs_shell("grep -rn 'ClassX' /src")
        # → /src/models.py:42:class ClassX:

        # 2. Focus on x  →  opens models.py, shows ~22 lines around ClassX
        vfs_view("/src/models.py", scroll_to="ClassX", context_lines=60)

        # 3. Locate y
        vfs_shell("grep -rn 'method_y' /src")
        # → /src/services.py:88:    def method_y(self):

        # 4. Add y to context  →  now BOTH sections are visible
        vfs_view("/src/services.py", scroll_to="method_y", context_lines=40)

        # 5. Answer precisely — you see exactly x and y, nothing else bloating context

        # 6. Reset for next task  →  close everything, open fresh
        vfs_view("/src/new_file.py", scroll_to="...", close_others=True)

        Parameters
        ----------
        path          : File to open/scroll (required).
        line_start    : First line to show (1-indexed). Ignored when scroll_to is set.
        line_end      : Last line to show  (-1 = EOF).   Ignored when scroll_to is set.
        scroll_to     : Regex/text pattern.  Finds first match and centers the view
                        within ±context_lines//2 lines around it.
        context_lines : Lines to show around the scroll_to match (default: 40).
        close_others  : If True, close ALL other open files first → clean focused context.

        Returns
        -------
        {
          "success": bool,
          "path": str,
          "content": str,          # visible text
          "showing": "lines X-Y of Z",
          "total_lines": int,
          "file_type": str,
          "match": {"matched_line": int, "pattern": str}   # only when scroll_to used
        }
        """
        import re as _re

        try:
            np = vfs._normalize_path(path)

            if not vfs._is_file(np):
                return {"success": False, "error": f"file not found: {path}"}

            # ── Optional: close all other open files ──────────────────────
            if close_others:
                for p, f in vfs.files.items():
                    if p != np and f.state == "open" and not f.readonly:
                        f.state = "closed"
                        f.mini_summary = f"[auto-closed by vfs_view close_others]"
                vfs._dirty = True

            f = vfs.files[np]

            # ── Lazy-load shadow files ─────────────────────────────────────
            try:
                from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType
                if (
                    hasattr(f, "backing_type")
                    and f.backing_type == FileBackingType.SHADOW
                    and not f.is_loaded
                ):
                    lr = vfs._load_shadow_content(np)
                    if not lr.get("success"):
                        return lr
            except ImportError:
                pass

            content = f.content
            all_lines = content.split("\n")
            total = len(all_lines)
            match_info: dict | None = None

            # ── scroll_to: find first match and center ────────────────────
            if scroll_to:
                try:
                    pat = _re.compile(scroll_to, _re.IGNORECASE)
                except _re.error:
                    pat = _re.compile(_re.escape(scroll_to), _re.IGNORECASE)

                found_line: int | None = None
                for idx, ln in enumerate(all_lines, 1):
                    if pat.search(ln):
                        found_line = idx
                        break

                if found_line is None:
                    return {
                        "success": False,
                        "error": f"pattern '{scroll_to}' not found in {path}",
                        "hint": (
                            f"File has {total} lines. "
                            f"Try vfs_shell(\"grep -n '{scroll_to}' {path}\") first."
                        ),
                    }

                half = context_lines // 2
                line_start = max(1, found_line - half)
                line_end   = min(total, found_line + half)
                match_info = {"matched_line": found_line, "pattern": scroll_to}

            # ── Apply view window ─────────────────────────────────────────
            f.state      = "open"
            f.view_start = max(0, line_start - 1)
            f.view_end   = line_end
            vfs._dirty   = True

            end_idx      = line_end if line_end > 0 else total
            visible      = all_lines[f.view_start:end_idx]

            result: dict = {
                "success":     True,
                "path":        np,
                "content":     "\n".join(visible),
                "showing":     f"lines {line_start}–{end_idx} of {total}",
                "total_lines": total,
                "file_type":   f.file_type.description if f.file_type else "Unknown",
            }
            if match_info:
                result["match"] = match_info
            if close_others:
                result["note"] = "all other files closed"
            return result

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    return vfs_view
