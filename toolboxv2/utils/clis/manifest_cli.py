"""
Manifest CLI - Command line interface for tb-manifest.yaml management

Commands:
    tb manifest show      - Display current manifest
    tb manifest validate  - Validate manifest syntax and semantics
    tb manifest apply     - Generate sub-configs from manifest
    tb manifest init      - Create default manifest interactively
    tb manifest status    - Show service status vs manifest
    tb manifest sync      - Sync running services with manifest
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .cli_printing import (
    print_box_header,
    print_box_footer,
    print_status,
    print_separator,
    print_code_block,
    c_print,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for manifest CLI."""
    parser = argparse.ArgumentParser(
        prog="tb manifest",
        description="📋 ToolBoxV2 Manifest Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔════════════════════════════════════════════════════════════════════════════╗
║                         Manifest Commands                                  ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  tb manifest show              Show current manifest content               ║
║  tb manifest validate          Validate manifest syntax and semantics      ║
║  tb manifest apply             Generate sub-configs from manifest          ║
║  tb manifest init              Create default manifest interactively       ║
║  tb manifest status            Show service status vs manifest             ║
║  tb manifest sync              Sync running services with manifest         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Manifest commands")

    # show
    p_show = subparsers.add_parser("show", help="Display current manifest")
    p_show.add_argument("--json", action="store_true", help="Output as JSON")
    p_show.add_argument("--section", type=str, help="Show specific section (app, database, workers, etc.)")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate manifest")
    p_validate.add_argument("--strict", action="store_true", help="Strict validation mode")

    # apply
    p_apply = subparsers.add_parser("apply", help="Generate sub-configs from manifest")
    p_apply.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    p_apply.add_argument("--force", action="store_true", help="Overwrite existing files")
    p_apply.add_argument("--env", action="store_true", help="Also suggest missing env vars")

    # init
    p_init = subparsers.add_parser("init", help="Create default manifest")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing manifest")
    p_init.add_argument("--env", type=str, choices=["development", "production", "staging"],
                        default="development", help="Initial environment")

    # status
    subparsers.add_parser("status", help="Show service status vs manifest")

    # sync
    p_sync = subparsers.add_parser("sync", help="Sync running services with manifest")
    p_sync.add_argument("--dry-run", action="store_true", help="Show what would be done")
    p_sync.add_argument("--restart", action="store_true", help="Restart all manifest services")

    return parser


def cmd_show(args) -> int:
    """Show manifest content."""
    from toolboxv2.utils.manifest import ManifestLoader
    from toolboxv2 import tb_root_dir

    loader = ManifestLoader(tb_root_dir)

    if not loader.exists():
        print_status("No tb-manifest.yaml found", "error")
        print_status("Create one with: tb manifest init", "info")
        return 1

    try:
        if args.json:
            manifest = loader.load()
            import json
            if args.section:
                section_data = getattr(manifest, args.section, None)
                if section_data is None:
                    print_status(f"Unknown section: {args.section}", "error")
                    return 1
                print(json.dumps(section_data.model_dump(), indent=2, default=str))
            else:
                print(json.dumps(manifest.model_dump(), indent=2, default=str))
        else:
            # Show raw YAML
            with open(loader.manifest_path, "r", encoding="utf-8") as f:
                content = f.read()

            if args.section:
                # Extract section from YAML
                import yaml
                data = yaml.safe_load(content)
                if args.section not in data:
                    print_status(f"Unknown section: {args.section}", "error")
                    return 1
                print_code_block(yaml.dump({args.section: data[args.section]}, default_flow_style=False), "yaml")
            else:
                print_box_header("tb-manifest.yaml", "📋")
                print_code_block(content, "yaml", show_line_numbers=True)
                print_box_footer()

        return 0
    except Exception as e:
        print_status(f"Error reading manifest: {e}", "error")
        return 1


def cmd_validate(args) -> int:
    """Validate manifest."""
    from toolboxv2.utils.manifest import ManifestLoader
    from toolboxv2 import tb_root_dir

    loader = ManifestLoader(tb_root_dir)

    print_box_header("Manifest Validation", "✅")

    if not loader.exists():
        print_status("No tb-manifest.yaml found", "error")
        print_box_footer()
        return 1

    try:
        # Try to load (validates schema)
        manifest = loader.load()
        print_status("Schema validation: PASSED", "success")

        # Semantic validation
        is_valid, errors = loader.validate()

        if is_valid:
            print_status("Semantic validation: PASSED", "success")
        else:
            print_status("Semantic validation: FAILED", "error")
            for err in errors:
                print(f"  • {err}")

        print_box_footer()
        return 0 if is_valid else 1

    except Exception as e:
        print_status(f"Schema validation: FAILED", "error")
        print(f"  • {e}")
        print_box_footer()
        return 1


def cmd_apply(args) -> int:
    """Apply manifest to generate sub-configs."""
    from toolboxv2.utils.manifest import ManifestLoader, ConfigConverter
    from toolboxv2 import tb_root_dir

    loader = ManifestLoader(tb_root_dir)

    print_box_header("Apply Manifest", "🔄")

    if not loader.exists():
        print_status("No tb-manifest.yaml found", "error")
        print_status("Create one with: tb manifest init", "info")
        print_box_footer()
        return 1

    try:
        manifest = loader.load()
        converter = ConfigConverter(manifest, tb_root_dir)

        if args.dry_run:
            print_status("Dry run - no files will be written", "info")
            print()
            print("Would generate:")
            print(f"  • {tb_root_dir / '.config.yaml'}")
            print(f"  • {tb_root_dir / 'bin' / 'config.toml'}")
            print(f"  • {tb_root_dir / '.info' / 'services.json'}")

            if manifest.nginx.enabled:
                print(f"  • {tb_root_dir / 'nginx.conf'}")

            if args.env:
                suggestions = converter._suggest_env_vars()
                if suggestions:
                    print()
                    print_status("Missing environment variables:", "warning")
                    for var, default in suggestions.items():
                        print(f"  • {var}={default or '(needs value)'}")
        else:
            print_status("Generating configuration files...", "progress")
            generated = converter.apply_all()

            print()
            print_status("Generated files:", "success")
            for path in generated:
                print(f"  ✓ {path}")

            if args.env:
                suggestions = converter._suggest_env_vars()
                if suggestions:
                    print()
                    print_status("Missing environment variables:", "warning")
                    for var, default in suggestions.items():
                        print(f"  • {var}={default or '(needs value)'}")

                    response = input("\n  Add missing vars to .env? (y/N): ").strip().lower()
                    if response == 'y':
                        added = converter.append_missing_env_vars()
                        print_status(f"Added {len(added)} variables to .env", "success")

        print_box_footer()
        return 0

    except Exception as e:
        print_status(f"Apply failed: {e}", "error")
        import traceback
        traceback.print_exc()
        print_box_footer()
        return 1


def cmd_init(args) -> int:
    """Create default manifest."""
    from toolboxv2.utils.manifest import ManifestLoader, TBManifest
    from toolboxv2.utils.manifest.schema import Environment
    from toolboxv2 import tb_root_dir

    loader = ManifestLoader(tb_root_dir)

    print_box_header("Initialize Manifest", "📝")

    if loader.exists() and not args.force:
        print_status("tb-manifest.yaml already exists", "warning")
        print_status("Use --force to overwrite", "info")
        print_box_footer()
        return 1

    try:
        # Create manifest with specified environment
        env_map = {
            "development": Environment.DEVELOPMENT,
            "production": Environment.PRODUCTION,
            "staging": Environment.STAGING,
        }

        manifest = TBManifest(
            app={"environment": env_map[args.env]}
        )

        loader.save(manifest)

        print_status(f"Created: {loader.manifest_path}", "success")
        print_status(f"Environment: {args.env}", "info")
        print()
        print("  Next steps:")
        print("  1. Edit tb-manifest.yaml to customize your config")
        print("  2. Run 'tb manifest validate' to check for errors")
        print("  3. Run 'tb manifest apply' to generate sub-configs")

        print_box_footer()
        return 0

    except Exception as e:
        print_status(f"Failed to create manifest: {e}", "error")
        print_box_footer()
        return 1


def cmd_status(args) -> int:
    """Show service status vs manifest."""
    from toolboxv2.utils.manifest import ManifestServiceManager

    manager = ManifestServiceManager()
    manifest = manager.get_manifest()

    print_box_header("Service Status", "📊")

    if not manifest:
        print_status("No tb-manifest.yaml found", "warning")
        print_status("Showing legacy services.json status", "info")
        print()

    report = manager.get_status_report()

    # Group by status
    running = []
    should_run = []
    extra = []
    stopped = []

    for name, info in report.items():
        status = info.get("status", "unknown")
        if status == "running":
            running.append((name, info))
        elif status == "not_started":
            should_run.append((name, info))
        elif status == "running_extra":
            extra.append((name, info))
        else:
            stopped.append((name, info))

    if running:
        print_status("Running (in manifest):", "success")
        for name, info in running:
            pid = info.get("pid", "?")
            print(f"  ✓ {name} (PID: {pid})")

    if should_run:
        print()
        print_status("Should be running (not started):", "warning")
        for name, info in should_run:
            print(f"  ○ {name}")

    if extra:
        print()
        print_status("Running (not in manifest):", "info")
        for name, info in extra:
            pid = info.get("pid", "?")
            print(f"  ? {name} (PID: {pid})")

    print_box_footer()
    return 0


def cmd_sync(args) -> int:
    """Sync running services with manifest."""
    from toolboxv2.utils.manifest import ManifestServiceManager

    manager = ManifestServiceManager()

    print_box_header("Sync Services", "🔄")

    if not manager.get_manifest():
        print_status("No tb-manifest.yaml found", "error")
        print_box_footer()
        return 1

    if args.restart:
        print_status("Restarting all manifest services...", "progress")
        result = manager.restart_manifest_services()
    else:
        if args.dry_run:
            print_status("Dry run - no services will be started", "info")
        result = manager.sync_services(dry_run=args.dry_run)

    print()

    if result.already_running:
        print_status("Already running:", "info")
        for name in result.already_running:
            print(f"  • {name}")

    if result.started:
        action = "Would start" if args.dry_run else "Started"
        print_status(f"{action}:", "success")
        for name in result.started:
            print(f"  ✓ {name}")

    if result.commands_executed:
        action = "Would execute" if args.dry_run else "Executed"
        print_status(f"{action} commands:", "success")
        for cmd in result.commands_executed:
            print(f"  $ {cmd}")

    if result.failed:
        print_status("Failed:", "error")
        for name, error in result.failed.items():
            print(f"  ✗ {name}: {error}")

    print_box_footer()
    return 0 if not result.failed else 1


def cli_manifest_main():
    """Main entry point for manifest CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "show": cmd_show,
        "validate": cmd_validate,
        "apply": cmd_apply,
        "init": cmd_init,
        "status": cmd_status,
        "sync": cmd_sync,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(cli_manifest_main())
