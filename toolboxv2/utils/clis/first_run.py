"""
First-Run Onboarding — runs on the first bare 'tb' (manifest has no app.profile).

Max 3 questions:
  1. Profile    (local | server | developer)
  2. LLM key    (optional, validated, written to .env)
  3. Account    (optional, hands off to the login flow)

Non-interactive (no TTY, CI, AI assistants): apply_profile(profile) does the
same persistence without any prompt. Both paths return a plain dict result.
"""
import sys
from typing import Optional

from .cli_printing import print_box_header, print_box_footer, print_status, c_print, Colors
from toolboxv2.utils.manifest.schema import ProfileType, PROFILE_DEFAULTS, PROFILE_ALIASES

PROFILES = {
    "local":     ("👤 Local",     "Ich nutze die Toolbox für mich auf diesem Rechner."),
    "server":    ("🖥️  Server",    "Ich betreibe ein System / Infrastruktur (headless)."),
    "developer": ("🛠️  Developer", "Ich entwickle Mods, Features oder den Core."),
}

# Providers offered in question 2 (subset of config_wizard.LLM_PROVIDERS).
_LLM_CHOICES = ["openrouter", "anthropic", "openai", "gemini", "groq", "ollama"]


def normalize_profile(profile: Optional[str]) -> str:
    """Map any accepted name (incl. legacy aliases) to a ProfileType value."""
    p = (profile or "local").strip().lower()
    p = PROFILE_ALIASES.get(p, p)
    return ProfileType(p).value  # raises ValueError on unknown names


def onboarding_state() -> dict:
    """Single source for "first run?" (CLI, local web UI, wizard API):
    onboarding is done iff the manifest has app.profile."""
    try:
        from toolboxv2 import tb_root_dir
        from toolboxv2.utils.manifest import ManifestLoader
        from toolboxv2._state_dir import manifest_base_dir
        loader = ManifestLoader(manifest_base_dir())
        profile = loader.load().app.profile if loader.exists() else None
        profile = profile.value if profile is not None else None
    except Exception:
        profile = None
    return {"installed": True, "first_run": profile is None, "profile": profile}


def apply_profile(profile: str) -> dict:
    """Persist profile + its manifest defaults. No prompts. Idempotent."""
    import yaml
    from toolboxv2 import tb_root_dir
    from toolboxv2.utils.manifest import ManifestLoader
    from toolboxv2.utils.manifest.schema import TBManifest

    chosen = normalize_profile(profile)
    from toolboxv2._state_dir import manifest_base_dir
    loader = ManifestLoader(manifest_base_dir())
    if not loader.exists():
        loader.save(TBManifest())

    # Typed write (not manifest_cli.cmd_set: it stringifies lists).
    path = loader.manifest_path
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for dotted, value in {**PROFILE_DEFAULTS.get(chosen, {}), "app.profile": chosen}.items():
        node = data
        *parents, leaf = dotted.split(".")
        for k in parents:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[leaf] = value

    errors = []
    try:
        TBManifest.model_validate(data)  # never persist an invalid manifest
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception as e:
        errors.append(str(e))

    return {"ok": not errors, "profile": chosen, "errors": errors, "manifest": str(path)}


def _ask(prompt: str, default: str = "") -> str:
    try:
        return input(f"{Colors.CYAN}❯{Colors.RESET} {prompt}").strip() or default
    except (KeyboardInterrupt, EOFError):
        return default


def _ask_profile() -> str:
    keys = list(PROFILES)
    c_print(f"\n  {Colors.BOLD}1/3 · Wofür nutzt du die Toolbox?{Colors.RESET}\n")
    for i, key in enumerate(keys, 1):
        label, desc = PROFILES[key]
        c_print(f"  {Colors.CYAN}{i}){Colors.RESET} {label}  {Colors.DIM}{desc}{Colors.RESET}")
    raw = _ask(f"Choose [1-{len(keys)}] (default: 1): ", "1")
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(keys) else 0
    return keys[idx]


def _ask_llm_key() -> Optional[str]:
    """Question 2. Returns provider key if something was saved, else None."""
    from toolboxv2.utils.clis.config_wizard import LLM_PROVIDERS, validate_llm_key, save_env_file
    from toolboxv2.init_onboarding import _writable_env_path

    choices = [p for p in _LLM_CHOICES if p in LLM_PROVIDERS]
    c_print(f"\n  {Colors.BOLD}2/3 · LLM-Provider für ISAA (optional){Colors.RESET}\n")
    for i, key in enumerate(choices, 1):
        c_print(f"  {Colors.CYAN}{i}){Colors.RESET} {LLM_PROVIDERS[key]['name']}")
    raw = _ask("Choose (Enter = später): ")
    if not (raw.isdigit() and 1 <= int(raw) <= len(choices)):
        return None

    provider = choices[int(raw) - 1]
    spec = LLM_PROVIDERS[provider]
    env_vars = {}
    if spec.get("key_vars"):
        key = _ask(f"{spec['key_vars'][0]}: ")
        if not key:
            return None
        env_vars[spec["key_vars"][0]] = key
    ok, msg, _ = validate_llm_key(provider, env_vars.get((spec.get("key_vars") or [""])[0], ""))
    print_status(msg, "success" if ok else "warning")
    if spec.get("url_var"):
        env_vars.setdefault(spec["url_var"], spec.get("default_url", ""))
    save_env_file(_writable_env_path(), env_vars)  # saved even if offline-validation failed
    return provider


def run_first_run(interactive: Optional[bool] = None) -> dict:
    """Onboarding entry. Returns {"ok", "profile", "llm", "login", "errors", "manifest"}.

    interactive=None → auto (TTY on stdin and stdout).
    """
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if not interactive:
        res = apply_profile("local")
        res.update(llm=None, login=False)
        return res

    print_box_header("Welcome to ToolBoxV2", "🚀")
    chosen = _ask_profile()
    res = apply_profile(chosen)
    print_status(f"Profile: {PROFILES[chosen][0]}", "success" if res["ok"] else "warning")

    try:
        res["llm"] = _ask_llm_key()
    except Exception as e:
        res["llm"] = None
        print_status(f"LLM setup skipped ({e}) — later: tb -init config", "warning")

    c_print(f"\n  {Colors.BOLD}3/3 · Account anlegen / einloggen? (optional){Colors.RESET}")
    res["login"] = _ask("[y/N]: ", "n").lower() in ("y", "j", "yes", "ja")

    print_box_footer()
    if chosen == "server":
        c_print(f"  {Colors.DIM}Server setup: bash setup_tb_server.sh · Status: tb status{Colors.RESET}")
    c_print(f"  {Colors.DIM}Alle Einstellungen später: tb -init config{Colors.RESET}\n")
    return res
