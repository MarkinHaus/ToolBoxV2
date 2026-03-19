"""
First-Run Onboarding — wird beim ersten 'tb' ohne gesetztes app.profile ausgeführt.
"""
from .cli_printing import print_box_header, print_box_footer, print_status, c_print, Colors


PROFILES = {
    "consumer":  ("👤 Consumer",  "Ich nutze eine App / eine Mod. Einfach starten."),
    "homelab":   ("🏠 Homelab",   "Ich betreibe mehrere Mods, Features, Flows lokal."),
    "server":    ("🖥️  Server",    "Ich manage ein verteiltes System / IT-Infrastruktur."),
    "business":  ("💼 Business",  "Ich brauche einen schnellen Gesundheitsstatus."),
    "developer": ("🛠️  Developer", "Ich entwickle Mods, Features oder den Core."),
}


def run_first_run() -> str:
    """
    Zeigt Profil-Auswahl, schreibt app.profile ins Manifest, startet Config-Wizard.
    Gibt das gewählte Profil zurück.
    """
    import sys, subprocess
    from toolboxv2 import tb_root_dir

    print_box_header("Welcome to ToolBoxV2", "🚀")
    c_print(f"\n  {Colors.BOLD}First start detected — choose your profile:{Colors.RESET}\n")

    keys = list(PROFILES.keys())
    for i, key in enumerate(keys, 1):
        label, desc = PROFILES[key]
        c_print(f"  {Colors.CYAN}{i}){Colors.RESET} {label}")
        c_print(f"     {Colors.DIM}{desc}{Colors.RESET}")
        print()

    try:
        raw = input(f"{Colors.CYAN}❯{Colors.RESET} Choose [1-{len(keys)}] (default: 2): ").strip()
        idx = (int(raw) - 1) if raw.isdigit() and 1 <= int(raw) <= len(keys) else 1
    except (KeyboardInterrupt, EOFError):
        idx = 1

    chosen = keys[idx]
    label, _ = PROFILES[chosen]
    print_status(f"Profile set: {label}", "success")

    # Schreibe ins Manifest via manifest set (nutzt bestehende Logik)
    try:
        from toolboxv2.utils.clis.manifest_cli import cmd_set

        class _A:
            key = "app.profile"
            value = chosen

        # Manifest muss existieren — falls nicht, erst init
        from toolboxv2.utils.manifest import ManifestLoader
        loader = ManifestLoader(tb_root_dir)
        if not loader.exists():
            print_status("No manifest found — creating default...", "info")
            from toolboxv2.utils.manifest.schema import TBManifest
            loader.save(TBManifest())

        cmd_set(_A())
    except Exception as e:
        print_status(f"Could not persist profile ({e}) — continuing anyway", "warning")

    print_box_footer()

    # Config Wizard anbieten
    try:
        run_wiz = input(f"{Colors.CYAN}❯{Colors.RESET} Run config wizard now? [Y/n]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        run_wiz = "n"

    if run_wiz != "n":
        from toolboxv2.utils.clis.config_wizard import run_config_wizard
        run_config_wizard(tb_root_dir)

    if label == "server":
        c_print("Run the setup script:")
        c_print("  bash setup_tb_server.sh")
        c_print("")
        c_print("Or make it executable and run:")
        c_print("  chmod +x setup_tb_server.sh")
        c_print("  ./setup_tb_server.sh")
        c_print("")
        c_print("Optional:")
        c_print("  bash tb-registry/setup_registry.sh")

    return chosen
