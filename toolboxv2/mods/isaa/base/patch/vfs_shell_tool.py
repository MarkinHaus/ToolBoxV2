"""
vfs_shell_tool.py
=================
Unix-like shell interface and context manager for VirtualFileSystemV2.

Provides exactly TWO primary agent tools that replace ~18 individual VFS tools:

  vfs_shell(reason, command)      — All filesystem operations via unix-like commands
  vfs_view(path, ...)     — Scroll / focus control for the context window

Usage in init_session_tools():
    from toolboxv2.mods.isaa.base.Agent.vfs_shell_tool import make_vfs_shell, make_vfs_view

    vfs_shell = make_vfs_shell(session)
    vfs_view  = make_vfs_view(session)
"""

from __future__ import annotations

import fnmatch
import json
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


def _decode_content(raw: str) -> str:
    """
    Normalisiert Content aus Agent-Tool-Calls.

    Versteht alle drei Schreibweisen:
      "line1\\nline2"   <- JSON-escape (Standard)
      "line1\nline2"    <- echter Newline (auch ok)
      r"line1\\nline2"  <- r-Praefix wurde bereits von _strip_quotes entfernt,
                           \\n bleibt dann als Literal (fuer Regex, Windows-Pfade)

    Idempotent: mehrfaches Aufrufen = selbes Ergebnis.
    NIEMALS selbst \\\\n schreiben — das erzeugt Backslash+n im File.
    """
    if True:
        return raw
    # Schritt 1: double-escaped abbauen (\\\\n -> \\n), max 3 Durchlaeufe
    for _ in range(3):
        if '\\\\' not in raw:
            break
        raw = raw.replace('\\\\n', '\\n')
        raw = raw.replace('\\\\t', '\\t')
        raw = raw.replace('\\\\r', '\\r')
        new = re.sub(r'\\\\(.)', r'\\\1', raw)
        if new == raw:
            break
        raw = new

    # Schritt 2: JSON-style escapes -> echte Zeichen
    raw = raw.replace('\\n', '\n')
    raw = raw.replace('\\t', '\t')
    raw = raw.replace('\\r', '\r')
    raw = raw.replace('\\"', '"')
    raw = raw.replace("\\'", "'")
    return raw


def _strip_quotes(s: str) -> str:
    """
    Entfernt aeussere Anfuehrungszeichen inkl. r-Praefix.

    Unterstuetzte Formen:
      "..."   '...'   r"..."   r'...'   \"\"\"...\"\"\"   '''...'''
    """
    s = s.strip()
    is_raw = s.startswith(('r"', "r'"))
    if is_raw:
        s = s[1:]   # r-Praefix entfernen, Rest normal behandeln
    for q in ('"""', "'''"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 6:
            return s[3:-3]
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s
# =============================================================================
# VFS SHELL FACTORY
# =============================================================================

def _split_compound(command: str) -> list[tuple[str, str]]:
    """
    Tokenise a compound shell command into (operator, cmd_text) pairs.

    Operators (longest match first, quoting respected):
        '&&'  AND  — next runs only if previous SUCCEEDED
        '||'  OR   — next runs only if previous FAILED
        '|'   PIPE — always runs, previous stdout becomes stdin
        ';'   SEQ  — always runs

    NOTE: Newline is intentionally NOT an operator.
    Agents send multi-line content via write_mini/echo as actual newlines,
    e.g.  write_mini /f.py "line1\nline2"  — treating \n as a separator
    would silently truncate content or produce 'command not found'.
    Use ';' or '&&' to batch multiple commands.
    """
    segments: list[tuple[str, str]] = []
    buf: list[str] = []
    pending_op = 'seq'
    in_single = False
    in_double = False
    i = 0
    n = len(command)

    while i < n:
        c = command[i]

        if in_single:
            buf.append(c)
            if c == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            buf.append(c)
            if c == '"':
                in_double = False
            i += 1
            continue

        if c == "'":
            in_single = True
            buf.append(c)
            i += 1
            continue

        if c == '"':
            in_double = True
            buf.append(c)
            i += 1
            continue

        two = command[i:i + 2]

        if two == '&&':
            seg = ''.join(buf).strip()
            if seg:
                segments.append((pending_op, seg))
            buf = []
            pending_op = '&&'
            i += 2
            continue

        if two == '||':
            seg = ''.join(buf).strip()
            if seg:
                segments.append((pending_op, seg))
            buf = []
            pending_op = '||'
            i += 2
            continue

        if c == '|':
            seg = ''.join(buf).strip()
            if seg:
                segments.append((pending_op, seg))
            buf = []
            pending_op = '|'
            i += 1
            continue

        if c == ';':
            seg = ''.join(buf).strip()
            if seg:
                segments.append((pending_op, seg))
            buf = []
            pending_op = 'seq'
            i += 1
            continue

        buf.append(c)
        i += 1

    seg = ''.join(buf).strip()
    if seg:
        segments.append((pending_op, seg))

    return segments

def _pipe_exec(cmd: str, stdin_text: str) -> dict:
    """
    Execute *cmd* against *stdin_text* (piped input).

    Pipe-aware: grep, wc, head, tail, sort, uniq, cat (no args).
    Everything else returns __needs_dispatch so _run_compound can
    forward it to the full vfs_shell dispatcher.
    """
    import re as _re
    try:
        args = shlex.split(cmd)
    except ValueError as e:
        return {"success": False, "stdout": "", "stderr": f"pipe parse error: {e}", "returncode": 1}

    if not args:
        return {"success": False, "stdout": "", "stderr": "empty piped command", "returncode": 1}

    cmd_name = args[0].lower()
    rest = args[1:]

    # ── grep ──────────────────────────────────────────────────────────────
    if cmd_name == 'grep':
        flag_args = [a for a in rest if a.startswith('-')]
        non_flags = [a for a in rest if not a.startswith('-')]
        if not non_flags:
            return {"success": False, "stdout": "", "stderr": "grep: missing pattern", "returncode": 1}

        pattern      = non_flags[0]
        case_i       = any('i' in f for f in flag_args)
        show_n       = any('n' in f for f in flag_args)
        invert       = any('v' in f for f in flag_args)
        context_n    = 0

        for j, a in enumerate(flag_args):
            if 'C' in a:
                tail = a[a.index('C') + 1:]
                if tail.isdigit():
                    context_n = int(tail)
                elif j + 1 < len(flag_args) and flag_args[j + 1].isdigit():
                    context_n = int(flag_args[j + 1])

        try:
            rx = _re.compile(f"(?i){pattern}" if case_i else pattern)
        except _re.error:
            rx = _re.compile(_re.escape(pattern))

        lines = stdin_text.splitlines()
        matches: list[str] = []
        for lineno, line in enumerate(lines, 1):
            hit = bool(rx.search(line))
            if invert:
                hit = not hit
            if hit:
                prefix = f"{lineno}:" if show_n else ""
                matches.append(f"{prefix}{line}")
                if context_n:
                    start = max(0, lineno - 1 - context_n)
                    end   = min(len(lines), lineno + context_n)
                    ctx   = [f"  {lines[k]}" for k in range(start, end) if k != lineno - 1]
                    matches.extend(ctx)
                    matches.append("--")

        if matches:
            return {"success": True, "stdout": '\n'.join(matches), "stderr": "", "returncode": 0}
        return {"success": False, "stdout": "(no matches)", "stderr": "", "returncode": 1}

    # ── wc ────────────────────────────────────────────────────────────────
    if cmd_name == 'wc':
        flag_args = [a for a in rest if a.startswith('-')]
        lc = len(stdin_text.splitlines())
        wc = len(stdin_text.split())
        cc = len(stdin_text)
        if '-l' in flag_args:
            return {"success": True, "stdout": str(lc), "stderr": "", "returncode": 0}
        if '-w' in flag_args:
            return {"success": True, "stdout": str(wc), "stderr": "", "returncode": 0}
        if '-c' in flag_args:
            return {"success": True, "stdout": str(cc), "stderr": "", "returncode": 0}
        return {"success": True, "stdout": f"{lc:>8} {wc:>8} {cc:>8}", "stderr": "", "returncode": 0}

    # ── head ──────────────────────────────────────────────────────────────
    if cmd_name == 'head':
        n_lines, _ = _parse_n_flag(rest, default=10)
        out = '\n'.join(stdin_text.splitlines()[:n_lines])
        return {"success": True, "stdout": out, "stderr": "", "returncode": 0}

    # ── tail ──────────────────────────────────────────────────────────────
    if cmd_name == 'tail':
        n_lines, _ = _parse_n_flag(rest, default=10)
        out = '\n'.join(stdin_text.splitlines()[-n_lines:])
        return {"success": True, "stdout": out, "stderr": "", "returncode": 0}

    # ── sort ──────────────────────────────────────────────────────────────
    if cmd_name == 'sort':
        reverse = '-r' in rest
        sorted_lines = sorted(stdin_text.splitlines(), reverse=reverse)
        return {"success": True, "stdout": '\n'.join(sorted_lines), "stderr": "", "returncode": 0}

    # ── uniq ──────────────────────────────────────────────────────────────
    if cmd_name == 'uniq':
        seen: list[str] = []
        for line in stdin_text.splitlines():
            if not seen or seen[-1] != line:
                seen.append(line)
        return {"success": True, "stdout": '\n'.join(seen), "stderr": "", "returncode": 0}

    # ── cat passthrough ───────────────────────────────────────────────────
    if cmd_name == 'cat' and not rest:
        return {"success": True, "stdout": stdin_text, "stderr": "", "returncode": 0}

    # ── Unknown: dispatch normally via vfs_shell ──────────────────────────
    return {"success": None, "__needs_dispatch": True, "__cmd": cmd}

def _run_compound(command: str, _single_fn) -> dict:
    """
    Execute a compound command string, handling all batch operators.

    Operator semantics
    ------------------
    seq / ;  : always execute, stdout appended to accumulated output
    &&       : execute ONLY if previous succeeded (returncode == 0)
    ||       : execute ONLY if previous failed   (returncode != 0)
    |        : always execute; previous stdout becomes this cmd's stdin.
               Intermediate pipe stages are NOT added to accumulated output —
               only the final stage of each pipe chain contributes to stdout.
    """
    segments = _split_compound(command)

    single_fn = lambda x:_single_fn("",x)

    if not segments:
        return {"success": False, "stdout": "", "stderr": "empty command", "returncode": 1}

    if len(segments) == 1:
        return single_fn(segments[0][1])

    all_stdouts: list[str] = []
    last: dict = {"success": True, "stdout": "", "stderr": "", "returncode": 0}

    for idx, (op, cmd) in enumerate(segments):

        if op == '&&' and not last.get('success', False):
            continue

        if op == '||' and last.get('success', True):
            continue

        if op == '|':
            pipe_stdin = last.get('stdout') or ''
            result = _pipe_exec(cmd, pipe_stdin)
            if result.get('success') is None and result.get('__needs_dispatch'):
                result = single_fn(result['__cmd'])
        else:
            result = single_fn(cmd)

        last = result

        # Only accumulate stdout when this segment does NOT feed into a pipe.
        # If the NEXT segment's operator is '|', this segment's output is
        # stdin for the next stage — do not add it to the visible output.
        is_last = (idx == len(segments) - 1)
        next_op = segments[idx + 1][0] if not is_last else None
        if next_op != '|':
            stdout = result.get('stdout', '')
            if stdout:
                all_stdouts.append(stdout)

    return {
        'success':    last.get('success', False),
        'stdout':     '\n'.join(s for s in all_stdouts if s),
        'stderr':     last.get('stderr', ''),
        'returncode': last.get('returncode', 0),
    }

def make_vfs_shell(session: "AgentSessionV2"):
    """
    Factory — returns a vfs_shell closure bound to *session*.

    Call once in init_session_tools() and register the returned function as a tool.
    """
    vfs = session.vfs

    def vfs_shell(reason: str, command: str) -> dict:
        """
        Unix-like shell interface for VFS operations.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        WICHTIG: SCHREIB-LIMITS & CHUNK-PROTOKOLL
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Ein einzelner Tool-Call darf max ~40 Zeilen Content enthalten.
        Grund: JSON-Tool-Calls können bei Längenbeschränkung abbrechen.
        Ein abgebrochener Call = keine Wirkung. Kein partial-write.

        REGEL:
          < 40 Zeilen  →  write <path> "..."   (ein Call)
          ≥ 40 Zeilen  →  write_chunk          (ein Call pro Block)

        CHUNK-PROTOKOLL:
          write_chunk <path> 0 <N> "<block_0_content>"   ← erzeugt/überschreibt Datei
          write_chunk <path> 1 <N> "<block_1_content>"   ← hängt an
          ...
          write_chunk <path> N-1 <N> "<block_N-1>"       ← finalisiert

        Nach Abbruch / Wiederaufnahme:
          write_chunk_status <path>   →  zeigt welche Blöcke fehlen
          Dann nur die fehlenden Blöcke erneut senden.

        Content-Encoding (alle Varianten funktionieren):
          "line1\\nline2"   ← \\n wird zu Newline (Standard in JSON)
          "line1\nline2"    ← echter Newline (auch ok)
          NIEMALS: \\\\n   ← das erzeugt einen Backslash + n im File!
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Reason for the action
        ------------------
        Supported commands
        ------------------
        NAVIGATION  ls [-la] [-R] [path]  |  pwd  |  tree [path] [-L depth]
        READ        cat <path>  |  head [-n N] <path>  |  tail [-n N] <path>  |  sed -n 'X,Yp' <path>
                    wc [-lwc] <path>  |  stat <path>
        SEARCH      find [path] [-name pattern] [-type f|d]
                    grep [-rniIlC N] <pattern> [path|file_pattern]
        WRITE       touch <path>
                    echo "text" > <path>        (overwrite)
                    echo "text" >> <path>       (append)
                    write_mini <path> "content"      (multi-line, \\n supported) # only for small files
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
        vfs_shell("get an initial vow of the project", "ls -la /src")
        vfs_shell("i need to edit the train logic therefor i search for train", "grep -rn 'def train' /src")
        vfs_shell("setting the app config to finalize the work", "write /src/config.py 'HOST = \\"localhost\\"\\nPORT = 8080'")
        vfs_shell("the main function is an artefact for other modules to entry", "edit /src/main.py 10 14 'def main():\\n    pass'")
        vfs_shell("read specific line range for analysis", "sed -n '100,150p' /src/app.py")
        vfs_shell("the close file i am don working on it. closing its keeps my context clean and im mor focused. this is good", "close /src/utils.py")
        """
        command = command.strip()
        if not command:
            return _err("empty command")

        # Strip unsupported shell redirections before any parsing
        # Handles: 2>/dev/null  2>&1  >/dev/null  1>/dev/null  >file
        command = re.sub(r'\s+\d+>[&>]?\S*', '', command).strip()

        # ── Multi-command batch dispatch ──────────────────────────────────────
        # Operators: && || | ;   (newline is NOT a separator — content safety)
        if len(_split_compound(command)) > 1:
            return _run_compound(command, vfs_shell)

        # ── Special-case: echo with shell redirection ──────────────────────
        # Matches:  echo "..." > path   OR   echo "..." >> path
        echo_m = re.match(
            r"^echo\s+(.*\S)\s+(>>|>)\s+(\S+)\s*$", command, re.DOTALL
        )
        if echo_m:
            raw_content, op, path = echo_m.groups()
            content = _decode_content(_strip_quotes(raw_content))
            r = vfs.write(path, content) if op == ">" else vfs.append(path, content)
            return _ok() if r.get("success") else _err(r.get("error", "write failed"))

        # ── Parse command ──────────────────────────────────────────────────
        try:
            args = shlex.split(command)
        except ValueError as e:
            # Fallback for raw-content commands with unbalanced quotes
            parts = command.split(maxsplit=1)
            if parts and parts[0].lower() in ("write", "write_mini", "write_chunk", "edit"):
                cmd_name = parts[0].lower()
                if cmd_name in ("write", "write_mini"):
                    args = command.split(maxsplit=2)
                else:
                    args = command.split(maxsplit=4)
            else:
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
            any_error = False
            for p in rest:
                r = vfs.read(p)
                if not r.get("success"):
                    parts.append(f"cat: {p}: no such file or directory")
                    any_error = True
                else:
                    parts.append(r["content"])
            stdout = "\n".join(parts)
            if any_error and len(rest) == 1:
                # single missing file → failure (enables || fallback)
                return {"success": False, "stdout": stdout, "stderr": "", "returncode": 1}
            return _ok(stdout)

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
        # sed -n 'X,Yp' <path>
        # ═══════════════════════════════════════════════════════════════════
        elif cmd == "sed":
            # Simple support for sed -n 'start,endp' path
            path = rest[-1] if rest else None
            script = next((a for a in rest if 'p' in a and not a.startswith('-')), "")

            if not path or not script:
                return _err("sed: usage: sed -n 'X,Yp' <path>")

            # Extract numbers from '100,154p'
            match = re.search(r"(\d+),(\d+)p", script)
            if not match:
                # Fallback: check for single line '100p'
                match = re.search(r"(\d+)p", script)
                if not match:
                    return _err("sed: only line range printing supported, e.g., -n '10,20p'")
                start = end = int(match.group(1))
            else:
                start, end = int(match.group(1)), int(match.group(2))

            r = vfs.read(path)
            if not r.get("success"):
                return _err(f"sed: {path}: {r.get('error', 'not found')}")

            lines = r["content"].splitlines()
            # sed is 1-indexed, inclusive
            sliced = lines[max(0, start - 1):end]
            return _ok("\n".join(sliced))
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
        elif cmd in ("write", "write_mini"):
            if len(rest) < 2:
                return _err("write: usage: write <path> <content>")
            path = rest[0]
            # Extract content from raw command to preserve \n before any tokenizer eats it
            raw = re.match(r"write(?:_mini)?\s+\S+\s+(.*)", command, re.DOTALL)
            raw_content = raw.group(1) if raw else " ".join(rest[1:])
            content = _decode_content(_strip_quotes(raw_content))
            r = vfs.write(path, content)
            return _ok(f"written: {path} ({len(content)} chars)") if r.get("success") else _err(r.get("error", ""))

        # In vfs_shell_tool.py

        elif cmd == "write_chunk":
            """
            Schreibe einen einzelnen Block einer größeren Datei.

            Syntax:
                write_chunk <path> <chunk_idx> <total_chunks> <content>

            Beispiel (3 Blöcke à ~50 Zeilen):
                write_chunk /src/big.py 0 3 "import os\nimport sys\n..."
                write_chunk /src/big.py 1 3 "def foo():\n    pass\n..."
                write_chunk /src/big.py 2 3 "if __name__ == '__main__':\n..."

            Bei chunk_idx == 0        → Datei wird neu angelegt (overwrite)
            Bei chunk_idx == total-1  → Datei wird finalisiert, Zeilencount gemeldet
            Abbruch mittendrin?       → write_chunk_status <path> zeigt welcher Block fehlt
            """
            if len(rest) < 4:
                return _err("write_chunk: usage: write_chunk <path> <idx> <total> <content>")

            path = rest[0]
            try:
                idx = int(rest[1])
                total = int(rest[2])
            except ValueError:
                return _err("write_chunk: idx and total must be integers")

            raw = re.match(r"write_chunk\s+\S+\s+\d+\s+\d+\s+(.*)", command, re.DOTALL)
            raw_content = raw.group(1) if raw else " ".join(rest[3:])
            content = _decode_content(_strip_quotes(raw_content))

            np = vfs._normalize_path(path)

            # Chunk-State im VFS als Sidecar speichern
            state_path = np + ".__chunks__"

            if idx == 0:
                # Start: Sidecar anlegen, Datei leeren
                state = {"total": total, "received": [], "size": 0}
                vfs.write(state_path, json.dumps(state))
                r = vfs.write(path, content  if content.endswith("\n") else content + "\n")
            else:
                # Folge-Block: Sidecar lesen, anhängen
                sr = vfs.read(state_path)
                if not sr.get("success"):
                    return _err(f"write_chunk: no active chunk session for {path} — start with idx=0")
                state = json.loads(sr["content"])

                if idx in state["received"]:
                    return _ok(f"chunk {idx}/{total - 1} already received, skipped")

                if state["total"] != total:
                    return _err(f"write_chunk: total mismatch (expected {state['total']}, got {total})")

                r = vfs.append(path, content  if content.endswith("\n") else content + "\n")

            if not r.get("success"):
                return _err(r.get("error", "write failed"))

            state["received"].append(idx)
            state["size"] += len(content)
            vfs.write(state_path, json.dumps(state))

            missing = [i for i in range(total) if i not in state["received"]]

            if not missing:
                # Finalisiert — Sidecar löschen
                vfs.delete(state_path)
                fc = vfs.read(path)
                lines = len(fc["content"].splitlines()) if fc.get("success") else "?"
                return _ok(f"✓ {path} complete — {total} chunks, {state['size']} chars, {lines} lines")

            return _ok(f"chunk {idx}/{total - 1} ok — missing: {missing}")


        elif cmd == "write_chunk_status":
            """Zeigt Status einer laufenden Chunk-Session."""
            if not rest:
                return _err("write_chunk_status: missing path")
            path = rest[0]
            state_path = vfs._normalize_path(path) + ".__chunks__"
            sr = vfs.read(state_path)
            if not sr.get("success"):
                return _ok(f"No active chunk session for {path}")
            state = json.loads(sr["content"])
            missing = [i for i in range(state["total"]) if i not in state["received"]]
            return _ok(
                f"path={path}\n"
                f"total={state['total']}  received={sorted(state['received'])}  missing={missing}\n"
                f"size_so_far={state['size']} chars"
            )
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
            new_content = _decode_content(_strip_quotes(" ".join(rest[3:])))
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

        elif cmd == "sync":
            result = vfs.sync_all()
            if result["success"]:
                n_synced = len(result.get("synced", []))
                return _ok(f"synced {n_synced} file(s) to disk")
            else:
                errors = "; ".join(result.get("errors", ["unknown error"]))
                return _err(f"sync errors: {errors}")

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
                "Try: ls cat head tail wc stat tree find grep sync "
                "touch write edit echo mkdir rm mv cp close exec write_chunk write_chunk_status sed"
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
        is_media: bool = False,
        focus_on_media_section: str | None = None,
    ) -> dict:
        """
        Open or scroll to a specific section of a file in the VFS context window.

        Files opened here are **permanently visible** in every subsequent prompt
        until explicitly closed (via `vfs_shell("to keep my context cleen a focused, I only close unimportant files", "close <path>")` or close_others=True).

        Core Workflow — Finding two related things x and y
        ---------------------------------------------------
        # 1. Locate x
        vfs_shell("find the specific class X so i understand its connections to Class Y", "grep -rn 'class X' /src")
        # → /src/models.py:42:class ClassX:

        # 2. Focus on x  →  opens models.py, shows ~22 lines around ClassX
        vfs_view("/src/models.py", scroll_to="ClassX", context_lines=60)

        # 3. Locate y
        vfs_shell("initial find locations and information abut method_y", "grep -rn 'method_y' /src")
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

        if isinstance(line_start, str):
            line_start = int(line_start.strip())
        if isinstance(line_end, str):
            line_end = int(line_end.strip())
        if isinstance(context_lines, str):
            context_lines = int(context_lines.strip())

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

            # ── MEDIA / VISION MODEL PIPELINE ──────────────────────────────
            ext = os.path.splitext(f.filename)[1].lower()
            is_media_auto = is_media or ext in [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"]

            force_reanalyze = is_media_auto and focus_on_media_section is not None
            already_analyzed = f._content and f._content.startswith("--- MEDIA ANALYSIS")

            if is_media_auto and (force_reanalyze or not already_analyzed):
                local_path = getattr(f, "local_path", None)
                if not local_path or not os.path.exists(local_path):
                    return {"success": False, "error": f"Media file needs a local backing file: {path}"}

                import litellm
                import base64

                model = os.getenv("VISIONMODEL", "openrouter/openai/gpt-4.1-mini")
                prompt = focus_on_media_section or "Please describe this media file in detail. Check state, content, and anomalies."

                try:
                    text_result = ""
                    if ext == ".pdf":
                        # Attempt 1: Native PDF support via litellm (z.B. für Gemini/Anthropic)
                        try:
                            with open(local_path, "rb") as pf:
                                b64_pdf = base64.b64encode(pf.read()).decode("utf-8")
                            messages = [{"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{b64_pdf}"}}
                            ]}]
                            resp = litellm.completion(model=model, messages=messages)
                            text_result = resp.choices[0].message.content
                        except Exception as native_err:
                            # Attempt 2: Fallback to PyMuPDF (fitz) converting pages to images
                            import fitz
                            doc = fitz.open(local_path)

                            # Map line_start/end to PDF pages (0-indexed)
                            p_start = max(0, int(line_start) - 1)
                            p_end = int(line_end) if int(line_end) > 0 else len(doc)

                            content_arr = [{"type": "text", "text": prompt}]
                            for i in range(p_start, min(p_end, len(doc))):
                                page = doc.load_page(i)
                                pix = page.get_pixmap()
                                img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                                content_arr.append(
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

                            messages = [{"role": "user", "content": content_arr}]
                            resp = litellm.completion(model=model, messages=messages)
                            text_result = resp.choices[0].message.content
                    else:
                        # Standard Images
                        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else f"image/{ext.strip('.')}"
                        with open(local_path, "rb") as imf:
                            img_b64 = base64.b64encode(imf.read()).decode("utf-8")
                        messages = [{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                        ]}]
                        resp = litellm.completion(model=model, messages=messages)
                        text_result = resp.choices[0].message.content

                    # Overwrite VFS memory with the text analysis directly
                    # (bypass setter so it doesn't set is_dirty=True and corrupt local disk on sync)
                    f._content = f"--- MEDIA ANALYSIS RESULT ({model}) ---\nPrompt: {prompt}\n\n{text_result}"
                    f.size_bytes = len(f._content)
                    f.line_count = len(f._content.splitlines())

                except ImportError as ie:
                    return {"success": False,
                            "error": f"Missing library for media processing: {ie} (try: pip install pymupdf litellm)"}
                except Exception as e:
                    return {"success": False, "error": f"Vision Model Error: {e}"}

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
                            f"Try vfs_shell(\"find specif section to work focussed and persist.\", \"grep -n '{scroll_to}' {path}\") first."
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


"""

 → Input / Arguments:
                                │     {"command":"find / -type f 2>/dev/null | head -50","reason":"List all
                                │     files in VFS to find available content"}
                                │   ────────────────────────────────────────────────────────────────
                                │   ← Result [Fokus]:
                                │     Error executing vfs_shell: make_vfs_shell.<locals>.vfs_shell() missing 1
                                │     required positional argument: 'command'
                                │   (no result data)


●
vfs_shell
0.05s
▼
status success · 0.049s
command
copy
{"command":"write /global/fix_display_templates.patch '# icli_web Panel System — Bug Fixes\n# =========================================\n# Applies to two files:\n#   1. toolboxv2/mods/icli_web/_icli_web_tool.py  (Python backend)\n#   2. toolboxv2/mods/icli_web/static/orb.html     (JS frontend)\n#\n# Bugs fixed:\n#   1. HTML template shows raw HTML instead of rendering it\n#   2. Markdown template falls through to JSON.stringify() dump\n#   3. Docstring missing \"markdown\" and \"file\" template types\n#\n# Checked but NOT a bug:\n#   - \"choice\" vs \"choices\" naming is consistent throughout both files\n#\n# Apply with:  patch -p1 < fix_display_templates.patch\n\ndiff --git a/toolboxv2/mods/icli_web/_icli_web_tool.py b/toolboxv2/mods/icli_web/_icli_web_tool.py\n--- a/toolboxv2/mods/icli_web/_icli_web_tool.py\n+++ b/toolboxv2/mods/icli_web/_icli_web_tool.py\n@@ -167,7 +167,7 @@\n         \"html_panel\": {\n         \"template\": \"html\",\n         \"category\": \"info\",\n         \"description\": (\n             \"Arbitrary HTML. SECURITY: set trusted=True only if the HTML \"\n             \"came from agent-generated code, never from untrusted sources.\"\n         ),\n         \"content\": {\n             \"title\": \"HTML\",\n-            \"trusted\": False,\n+            \"trusted\": True,\n             \"html\": \"<p>Replace this with your HTML content.</p>\",\n         },\n     },\n\n@@ -540,7 +540,7 @@\n         template: \"form\" | \"choice\" | \"info\" | \"html\"\n+        template: \"form\" | \"choice\" | \"info\" | \"html\" | \"markdown\" | \"file\"\n         content:  Template-specific payload (see list_panel_templates).\n         state:    Optional initial field values.\n         panel_id: Stable id to update an existing panel. Auto-generated if omitted.\n\ndiff --git a/toolboxv2/mods/icli_web/static/orb.html b/toolboxv2/mods/icli_web/static/orb.html\n--- a/toolboxv2/mods/icli_web/static/orb.html\n+++ b/toolboxv2/mods/icli_web/static/orb.html\n@@ -2800,6 +2800,7 @@\n   //   \"form\"   — content.fields: [{name,label,type, options?, value?}]\n   //   \"choice\" — content.prompt, content.options: [{value,label}]\n   //   \"info\"   — content.title, content.body (sanitized text)\n   //   \"html\"   — content.html (raw; needs trusted=true or it renders as text)\n+  //   \"markdown\" — content.md (or content.markdown); safe escaped-first renderer\n   //   \"file\"   — content.prompt, content.max_size_mb, content.vfs_target\n   //\n   // Unknown templates fall back to a JSON dump so nothing is lost.\n\n@@ -2895,6 +2896,30 @@\n           cbody.appendChild(pre);\n         }\n+      } else if (msg.template === \"markdown\") {\n+        renderMarkdownTemplate(cbody, msg.content);\n       } else {\n         const pre = document.createElement(\"pre\");\n         pre.style.whiteSpace = \"pre-wrap\";\n\n@@ -3031,6 +3056,64 @@\n   function renderInfoTemplate(el, content) {\n     const body = document.createElement(\"div\");\n     body.textContent = content?.body || \"\";\n     body.style.whiteSpace = \"pre-wrap\";\n     body.style.fontSize = \"14px\";\n     el.appendChild(body);\n   }\n+\n+  // ── Markdown renderer (safe, escaped-first) ───────────────────────\n+  // Escapes all HTML first, then applies markdown patterns.\n+  // Supports: headings, bold, italic, inline code, fenced code blocks,\n+  // links, unordered/ordered lists, blockquotes, paragraphs.\n+  function renderMarkdownTemplate(el, content) {\n+    const md = content?.md || content?.markdown || \"\";\n+    const wrap = document.createElement(\"div\");\n+    wrap.style.fontSize = \"14px\";\n+    wrap.style.lineHeight = \"1.6\";\n+    wrap.style.whiteSpace = \"pre-wrap\";\n+    wrap.innerHTML = parseMarkdown(md);\n+    el.appendChild(wrap);\n+  }\n+\n+  function parseMarkdown(src) {\n+    // 1) Escape HTML to prevent injection\n+    let html = src\n+      .replace(/&/g, \"&amp;\")\n+      .replace(/</g, \"&lt;\")\n+      .replace(/>/g, \"&gt;\");\n+\n+    // 2) Fenced code blocks (``` ... ```) — must run before other rules\n+    html = html.replace(/```(\\\\w*)\\\\n([\\\\s\\\\S]*?)```/g,\n+      (_, lang, code) => {\n+        const escaped = code\n+          .replace(/&/g, \"&amp;\")\n+          .replace(/</g, \"&lt;\")\n+          .replace(/>/g, \"&gt;\");\n+        return \\'<pre style=\"background:var(--paper-sunken);padding:10px;overflow-x:auto;border:1px solid var(--ink)\"><code>\\' + escaped + \\'</code></pre>\\';\n+      }\n+    );\n+\n+    // 3) Inline patterns\n+    html = html.replace(/&lt;!--[\\s\\S]*?--&gt;/g, \"\");  // strip HTML comments\n+    html = html.replace(/^### (.+)$/gm, \"<h6>$1</h6>\");\n+    html = html.replace(/^## (.+)$/gm, \"<h5>$1</h5>\");\n+    html = html.replace(/^# (.+)$/gm, \"<h4>$1</h4>\");\n+    html = html.replace(/\\\\*\\\\*(.+?)\\\\*\\\\*/g, \"<strong>$1</strong>\");\n+    html = html.replace(/\\\\*(.+?)\\\\*/g, \"<em>$1</em>\");\n+    html = html.replace(/`([^`]+)`/g, \"<code style=\\\"background:var(--paper-sunken);padding:1px 4px\\\">$1</code>\");\n+    html = html.replace(/\\\\[([^\\\\]]+)\\\\]\\\\(([^)]+)\\\\)/g, \"<a href=\\\"$2\\\" target=\\\"_blank\\\" style=\\\"color:var(--primary)\\\">$1</a>\");\n+\n+    // 4) Block-level: blockquotes, lists, paragraphs\n+    html = html.replace(/^&gt; (.+)$/gm, \"<blockquote style=\\\"border-left:3px solid var(--ink);padding-left:10px;color:var(--ink-muted)\\\">$1</blockquote>\");\n+    html = html.replace(/^- (.+)$/gm, \"<li>$1</li>\");\n+    html = html.replace(/^\\\\d+\\\\. (.+)$/gm, \"<li>$1</li>\");\n+\n+    // Wrap consecutive <li> in <ul>\n+    html = html.replace(/((?:<li>.*<\\\\/li>\\\\n?)+)/g, \"<ul style=\\\"padding-left:20px\\\">$1</ul>\");\n+\n+    // 5) Paragraphs: double-newline separated blocks\n+    html = html.replace(/\\\\n\\\\n+/g, \"</p><p>\");\n+    html = \"<p>\" + html + \"</p>\";\n+    html = html.replace(/<p>\\\\s*<(h[456]|pre|ul|ol|blockquote|li)/g, \"<$1\");\n+    html = html.replace(/<\\\\/li>\\\\s*<\\\\/p>/g, \"</li>\");\n+    html = html.replace(/<p>\\\\s*<\\\\/p>/g, \"\");\n+\n+    return html;\n+  }\n'","reason":"Write the patch file with all three fixes"}
stdout rc=1
copy
(empty)
stderr <span class="vfs-rc nonzero">rc 1</span>
copy
parse error: No closing quotation
"""
