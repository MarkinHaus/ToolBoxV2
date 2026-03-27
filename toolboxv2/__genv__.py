#!/usr/bin/env python3
"""
tb_env.py — ToolBoxV2 global PATH manager
Kompatibel mit installer.sh v1.0.0+ und installer.ps1 v1.0.0+

Erkennt & verwaltet:
  Linux/macOS:
    • installer.sh-Block  (# ToolBoxV2 / TOOLBOX_HOME / PATH)
    • Eigener Marker-Block (# >>> ToolBoxV2 PATH >>>)
    • Symlinks in /usr/local/bin/tb, ~/.local/bin/tb
  Windows:
    • PATH-Eintrag  (HKCU\\Environment\\PATH)  — installer.ps1 & eigener
    • TOOLBOX_HOME  (HKCU\\Environment\\TOOLBOX_HOME)  — installer.ps1

Usage:
    python tb_env.py install
    python tb_env.py uninstall
    python tb_env.py status
"""
import sys, os, platform, re

# ── Basiswerte ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM = platform.system()  # "Windows" | "Linux" | "Darwin"
BIN_DIR = os.path.join(SCRIPT_DIR, "Scripts" if SYSTEM == "Windows" else "bin")

# ── install.manifest Suchpfade ────────────────────────────────────────────────
_lad = os.environ.get("LOCALAPPDATA", "")
_app = os.environ.get("APPDATA", "")

MANIFEST_SEARCH = [
    # Linux / macOS (installer.sh defaults)
    "/opt/toolboxv2/install.manifest",
    os.path.expanduser("~/.local/share/toolboxv2/install.manifest"),
    os.path.expanduser("~/.toolboxv2/install.manifest"),
    os.path.expanduser("~/Library/Application Support/toolboxv2/install.manifest"),
    # Windows (installer.ps1 defaults)
    os.path.join(_lad, "toolboxv2", "install.manifest") if _lad else "",
    os.path.join(_app, "toolboxv2", "install.manifest") if _app else "",
    r"C:\toolboxv2\install.manifest",
    # Neben diesem Script (source-install direkt im repo)
    os.path.join(SCRIPT_DIR, "install.manifest"),
]

# Symlink-Ziele des Installers (Linux/macOS)
INSTALLER_SYMLINKS = [
    "/usr/local/bin/tb",
    os.path.expanduser("~/.local/bin/tb"),
]

# Shell-Dateien (Linux/macOS)
SHELL_FILES = [
    os.path.expanduser("~/.bashrc"),
    os.path.expanduser("~/.zshrc"),
    os.path.expanduser("~/.profile"),
]

# ── Marker (eigenes Format, Linux/macOS) ─────────────────────────────────────
MY_MARKER = "# >>> ToolBoxV2 PATH >>>"
MY_MARKER_END = "# <<< ToolBoxV2 PATH <<<"
MY_BLOCK = f"""{MY_MARKER}
export TOOLBOX_HOME="{SCRIPT_DIR}"
export PATH="{BIN_DIR}:$PATH"
{MY_MARKER_END}"""


# ── YAML mini-parser ──────────────────────────────────────────────────────────
def _yaml_get(path: str, key: str, default: str = "") -> str:
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.match(rf'^{re.escape(key)}:\s*"?([^"\n]*)"?\s*$', line)
            if m:
                return m.group(1).strip()
    except OSError:
        pass
    return default


def find_manifests() -> list[dict]:
    results, seen = [], set()
    # Auch TOOLBOX_HOME / TB_INSTALL_DIR aus aktueller Env prüfen
    extra = [
        os.path.join(os.environ.get("TOOLBOX_HOME", ""), "install.manifest"),
        os.path.join(os.environ.get("TB_INSTALL_DIR", ""), "install.manifest"),
    ]
    for mp in MANIFEST_SEARCH + extra:
        if not mp:
            continue
        mp = os.path.normpath(mp)
        if not os.path.isfile(mp) or mp in seen:
            continue
        seen.add(mp)
        results.append({
            "manifest_path": mp,
            "toolbox_home": _yaml_get(mp, "toolbox_home"),
            "bin_path": _yaml_get(mp, "bin_path"),
            "install_mode": _yaml_get(mp, "install_mode"),
            "tb_version": _yaml_get(mp, "tb_version"),
            "runtime": _yaml_get(mp, "runtime"),
            "installed_at": _yaml_get(mp, "installed_at"),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# WINDOWS — Registry-Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _win_get_env(name: str) -> str:
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ)
        val, _ = winreg.QueryValueEx(key, name)
        winreg.CloseKey(key)
        return val or ""
    except FileNotFoundError:
        return ""


def _win_set_env(name: str, value: str | None) -> None:
    """Setzt oder löscht (value=None) eine User-Umgebungsvariable."""
    import winreg, ctypes
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE)
    if value is None:
        try:
            winreg.DeleteValue(key, name)
        except FileNotFoundError:
            pass
    else:
        winreg.SetValueEx(key, name, 0, winreg.REG_EXPAND_SZ, value)
    winreg.CloseKey(key)
    # WM_SETTINGCHANGE → laufende Shells übernehmen es sofort
    ctypes.windll.user32.SendMessageTimeoutW(
        0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, None
    )


def _win_path_entries() -> list[str]:
    return [p for p in _win_get_env("PATH").split(";") if p]


def _win_path_contains(entry: str) -> bool:
    entry_norm = os.path.normcase(entry)
    return any(os.path.normcase(p) == entry_norm for p in _win_path_entries())


def _win_path_add(entry: str) -> None:
    parts = _win_path_entries()
    _win_set_env("PATH", ";".join([entry] + parts))


def _win_path_remove(entries: list[str]) -> bool:
    """Entfernt alle Einträge aus der Liste. Gibt True zurück wenn etwas entfernt wurde."""
    norm = {os.path.normcase(e) for e in entries}
    parts = _win_path_entries()
    filtered = [p for p in parts if os.path.normcase(p) not in norm]
    if len(filtered) == len(parts):
        return False
    _win_set_env("PATH", ";".join(filtered))
    return True


# ══════════════════════════════════════════════════════════════════════════════
# LINUX / MACOS — Shell-File-Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _my_block_present(content: str) -> bool:
    return MY_MARKER in content


def _installer_block_present(content: str) -> bool:
    return bool(re.search(r'#\s*ToolBoxV2\s*\n\s*export TOOLBOX_HOME=', content))


def _remove_my_block(content: str) -> str:
    lines, inside, out = content.splitlines(keepends=True), False, []
    for line in lines:
        if line.strip() == MY_MARKER:
            inside = True
        elif line.strip() == MY_MARKER_END:
            inside = False
        elif not inside:
            out.append(line)
    return "".join(out)


def _remove_installer_block(content: str) -> str:
    return re.sub(
        r'\n?[ \t]*#\s*ToolBoxV2\s*\n'
        r'[ \t]*export TOOLBOX_HOME=[^\n]*\n'
        r'[ \t]*export PATH=[^\n]*\n?',
        '\n',
        content,
    )


def _symlink_status() -> list[tuple[str, str | None]]:
    result = []
    for sl in INSTALLER_SYMLINKS:
        if os.path.islink(sl):
            result.append((sl, os.readlink(sl)))
        elif os.path.exists(sl):
            result.append((sl, "(keine Symlink, reguläre Datei)"))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ACTIONS
# ══════════════════════════════════════════════════════════════════════════════
def install() -> None:
    if not os.path.isdir(BIN_DIR):
        print(f"[ERROR] venv/bin nicht gefunden: {BIN_DIR}", file=sys.stderr)
        sys.exit(1)

    if SYSTEM == "Windows":
        changed = False
        # PATH
        if _win_path_contains(BIN_DIR):
            print(f"[SKIP] Bereits in PATH: {BIN_DIR}")
        else:
            _win_path_add(BIN_DIR)
            print(f"[OK] PATH ← {BIN_DIR}")
            changed = True
        # TOOLBOX_HOME
        current_home = _win_get_env("TOOLBOX_HOME")
        if current_home and os.path.normcase(current_home) == os.path.normcase(SCRIPT_DIR):
            print(f"[SKIP] TOOLBOX_HOME bereits gesetzt: {current_home}")
        else:
            _win_set_env("TOOLBOX_HOME", SCRIPT_DIR)
            print(f"[OK] TOOLBOX_HOME ← {SCRIPT_DIR}")
            changed = True
        if changed:
            print("     Neues Terminal öffnen um `tb` zu nutzen.")
    else:
        modified = []
        for rc in SHELL_FILES:
            if not os.path.exists(rc):
                continue
            content = open(rc).read()
            if _installer_block_present(content):
                print(f"[SKIP] installer.sh-Block gefunden in {rc} — kein Duplikat nötig")
                continue
            if _my_block_present(content):
                print(f"[SKIP] Bereits eingetragen in {rc}")
                continue
            with open(rc, "a") as f:
                f.write(f"\n{MY_BLOCK}\n")
            modified.append(rc)
        if modified:
            print(f"[OK] Eingetragen in: {', '.join(modified)}")
            print("     `source ~/.bashrc` ausführen oder Terminal neu starten.")
        else:
            print("[INFO] Bereits in allen Shell-Dateien vorhanden.")


def uninstall() -> None:
    if SYSTEM == "Windows":
        manifests = find_manifests()

        # Alle bekannten TB-bin-Einträge sammeln (eigener + alle aus Manifests)
        to_remove = [BIN_DIR]
        for m in manifests:
            if m["toolbox_home"]:
                to_remove.append(os.path.join(m["toolbox_home"], "bin"))
            if m["bin_path"]:
                to_remove.append(os.path.dirname(m["bin_path"]))

        removed = _win_path_remove(list(dict.fromkeys(to_remove)))  # dedupliziert
        if removed:
            print("[OK] TB-Einträge aus User-PATH entfernt.")
        else:
            print("[INFO] Kein TB-Eintrag in PATH — nichts zu tun.")

        # TOOLBOX_HOME entfernen (installer.ps1 setzt es, wir räumen es auf)
        if _win_get_env("TOOLBOX_HOME"):
            _win_set_env("TOOLBOX_HOME", None)
            print("[OK] TOOLBOX_HOME aus User-Umgebung entfernt.")
        else:
            print("[INFO] TOOLBOX_HOME nicht gesetzt — nichts zu tun.")

        if removed:
            print("     Neues Terminal öffnen um Änderungen zu übernehmen.")
        return

    # Linux / macOS
    modified = []
    for rc in SHELL_FILES:
        if not os.path.exists(rc):
            continue
        content = open(rc).read()
        new = content
        if _installer_block_present(new):
            new = _remove_installer_block(new)
            print(f"[OK] installer.sh-Block entfernt aus {rc}")
        if _my_block_present(new):
            new = _remove_my_block(new)
            print(f"[OK] Eigener Marker-Block entfernt aus {rc}")
        if new != content:
            open(rc, "w").write(new)
            modified.append(rc)

    if not modified:
        print("[INFO] Kein TB-Eintrag in Shell-Dateien gefunden.")

    for sl, target in _symlink_status():
        try:
            os.remove(sl)
            print(f"[OK] Symlink entfernt: {sl} → {target}")
        except PermissionError:
            print(f"[WARN] Kein Zugriff auf {sl} — ggf. sudo nötig")

    if modified:
        print("     Terminal neu starten um Änderungen zu übernehmen.")


def status() -> None:
    W = 60
    print("=" * W)
    print("  ToolBoxV2 PATH Status")
    print("=" * W)

    # ── venv/bin ──────────────────────────────────────────────
    print(f"\n[venv/bin]")
    print(f"  Pfad    : {BIN_DIR}")
    print(f"  Existiert: {'ja' if os.path.isdir(BIN_DIR) else 'NEIN'}")
    tb_bin = os.path.join(BIN_DIR, "tb.exe" if SYSTEM == "Windows" else "tb")
    print(f"  tb-Binary: {'ja' if os.path.isfile(tb_bin) else 'nicht gefunden'}")

    # ── install.manifest ──────────────────────────────────────
    manifests = find_manifests()
    print(f"\n[Installer-Installationen ({len(manifests)} gefunden)]")
    if manifests:
        for m in manifests:
            print(f"  • {m['toolbox_home'] or '(unbekannt)'}")
            print(f"      Mode       : {m['install_mode']}")
            print(f"      Version    : {m['tb_version']}")
            print(f"      Runtime    : {m['runtime']}")
            print(f"      Installiert: {m['installed_at']}")
            print(f"      Manifest   : {m['manifest_path']}")
    else:
        print("  (keine install.manifest gefunden)")

    if SYSTEM == "Windows":
        # ── Windows PATH ──────────────────────────────────────
        print(f"\n[Windows User-PATH]")
        tb_entries = [p for p in _win_path_entries()
                      if "toolbox" in p.lower() or "toolbox" in p.lower()
                      or os.path.normcase(p) == os.path.normcase(BIN_DIR)]
        if tb_entries:
            for e in tb_entries:
                src = ""
                for m in manifests:
                    if m["toolbox_home"] and os.path.normcase(e) == os.path.normcase(
                        os.path.join(m["toolbox_home"], "bin")):
                        src = "  ← installer.ps1"
                if os.path.normcase(e) == os.path.normcase(BIN_DIR):
                    src += "  ← tb_env.py"
                print(f"  [x] {e}{src}")
        else:
            print("  [ ] Kein TB-Eintrag gefunden")

        # ── TOOLBOX_HOME ──────────────────────────────────────
        print(f"\n[TOOLBOX_HOME (User-Env)]")
        th = _win_get_env("TOOLBOX_HOME")
        if th:
            src = "← installer.ps1" if any(
                m["toolbox_home"] and os.path.normcase(m["toolbox_home"]) == os.path.normcase(th)
                for m in manifests
            ) else "← tb_env.py / manuell"
            print(f"  [x] {th}  {src}")
        else:
            print("  [ ] nicht gesetzt")

    else:
        # ── Symlinks ──────────────────────────────────────────
        print(f"\n[Symlinks]")
        symlinks = _symlink_status()
        if symlinks:
            for sl, target in symlinks:
                print(f"  [x] {sl} → {target}")
        else:
            print("  (keine TB-Symlinks gefunden)")

        # ── Shell-Dateien ─────────────────────────────────────
        print(f"\n[Shell-Einträge]")
        for rc in SHELL_FILES:
            if not os.path.exists(rc):
                continue
            content = open(rc).read()
            installer = _installer_block_present(content)
            own = _my_block_present(content)
            tag = ""
            if installer: tag += "[installer.sh] "
            if own:       tag += "[tb_env.py]"
            if not tag:   tag = "[ ]"
            print(f"  {tag} {rc}")

    print()


def main():
    # ── Entry ─────────────────────────────────────────────────────────────────────
    CMDS = {"install": install, "uninstall": uninstall, "status": status}

    if len(sys.argv) != 2 or sys.argv[1] not in CMDS:
        print(f"Usage: tb access [install|uninstall|status]")
        sys.argv.append("status")

    CMDS[sys.argv[1]]()

if __name__ == "__main__":
    main()
