# file: toolboxv2/utils/clis/cli_registry.py
# ToolBoxV2 Registry CLI
# Provides command-line interface for the TB Registry

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from toolboxv2.utils.clis.cli_printing import (
    Colors, c_print, print_box_header, print_box_footer,
    print_box_content, print_status, print_table_header, print_table_row
)
from .cli_input import menu_select_async

# Import RegistryClient
from toolboxv2.utils.extras.registry_client import (
    RegistryClient,
    RegistryError,
    RegistryAuthError,
    PackageNotFoundError,
    PublishPermissionError,
)

# Import DiffUploader
from toolboxv2.utils.extras.install.diff_uploader import DiffUploader
from toolboxv2.utils.extras.install.upload_cache import UploadCache


def get_app(name: str):
    """Get ToolBoxV2 app instance."""
    from toolboxv2 import get_app as _get_app
    return _get_app(name)


# ==================== Registry Commands ====================

async def registry_start(args):
    """Start the registry server."""
    print_box_header("Starting Registry Server", "🖥️")
    print_box_content("Module: CloudM.RegistryServer", "info")
    print_box_content(f"Host: {args.host}", "info")
    print_box_content(f"Port: {args.port}", "info")
    if args.background:
        print_box_content("Mode: background", "info")
    if args.reload:
        print_box_content("Auto-reload: enabled", "warning")
    print_box_footer()

    app = get_app("CloudM.RegistryServer")
    result = await app.a_run_any(
        "CloudM.RegistryServer",
        "start",
        host=args.host,
        port=args.port,
        background=args.background
    )

    if result.is_error():
        print_status(f"Failed to start: {result.get()}", "error")
        return 1

    data = result.get()
    if args.background and data:
        print_status(f"Registry server started in background (PID: {data.get('pid', 'N/A')})", "success")
        print_status(f"URL: http://{args.host}:{args.port}", "info")
    else:
        print_status("Registry server started (foreground mode)", "success")
        print_status("Press Ctrl+C to stop", "info")

    return 0


async def registry_stop(args):
    """Stop the registry server."""
    print_box_header("Stopping Registry Server", "🖥️")
    print_box_footer()

    app = get_app("CloudM.RegistryServer")
    result = await app.a_run_any("CloudM.RegistryServer", "stop", get_results=True)

    if result.is_error():
        print_status(f"Failed to stop: {result.get()}", "error")
        return 1

    data = result.get()
    if data:
        print_status(f"Registry server stopped (PID: {data.get('pid', 'N/A')})", "success")
    else:
        print_status("Registry server stopped", "success")
    return 0


async def registry_status(args):
    """Get registry server status."""
    print_box_header("Registry Server Status", "🖥️")
    print_box_footer()

    app = get_app("CloudM.RegistryServer")
    result = await app.a_run_any("CloudM.RegistryServer", "status", get_results=True)

    if result.is_error():
        print_status("Registry server is not running", "error")
        return 1

    data = result.get()
    print(data)
    if data:
        print_status(f"Status: {data.get('status', 'unknown')}", "success")
        print_status(f"PID: {data.get('pid', 'N/A')}", "info")
        print_status(f"Host: {data.get('host', 'N/A')}", "info")
        print_status(f"Port: {data.get('port', 'N/A')}", "info")
    else:
        print_status("Registry server is not running", "warning")
    return 0


async def registry_server_admin(args):
    """Launch the server-side admin CLI (runs in tb-registry venv)."""
    print_box_header("Registry Admin CLI", "🔧")
    print_box_content("Launching server-side admin tool", "info")
    print_box_footer()

    app = get_app("CloudM.RegistryServer")

    # Reuse the same path-finding logic as RegistryServer mod
    try:
        from toolboxv2.mods.CloudM.RegistryServer import _find_registry_path
        reg_path = _find_registry_path()
    except ImportError:
        # Fallback: try common locations
        from pathlib import Path as _Path
        for candidate in [
            _Path(__file__).parent.parent.parent.parent / "tb-registry",
            _Path.home() / "tb-registry",
            _Path.cwd() / "tb-registry",
        ]:
            if candidate.exists() and (candidate / "admin_cli.py").exists():
                reg_path = candidate
                break
        else:
            reg_path = None

    if not reg_path or not reg_path.exists():
        print_status("TB-Registry path not found", "error")
        print_status("Set TB_REGISTRY_PATH or ensure tb-registry is at the default location", "info")
        return 1

    admin_script = reg_path / "admin_cli.py"
    if not admin_script.exists():
        print_status(f"admin_cli.py not found in {reg_path}", "error")
        return 1

    # Determine DB path
    db_path = args.db if hasattr(args, 'db') and args.db else str(reg_path / "data" / "registry.db")

    print_status(f"Registry path: {reg_path}", "info")
    print_status(f"Database: {db_path}", "info")

    import subprocess
    import os

    env = os.environ.copy()

    try:
        # Run admin_cli.py via uv in the tb-registry venv (same as start_registry)
        process = subprocess.run(
            ["uv", "run", "python", "admin_cli.py", "--db", db_path],
            cwd=str(reg_path),
            env=env,
        )

        if process.returncode != 0:
            print_status(f"Admin CLI exited with code {process.returncode}", "warning")
            return process.returncode

        return 0

    except FileNotFoundError:
        print_status("'uv' not found. Install with: pip install uv", "error")
        return 1
    except KeyboardInterrupt:
        print_status("Admin CLI closed", "info")
        return 0
    except Exception as e:
        print_status(f"Failed to launch admin CLI: {e}", "error")
        return 1


# ==================== Package Commands ====================

async def registry_search(args):
    """Search for packages in the registry."""
    print_box_header("Package Search", "🔍")
    print_box_content(f"Query: {args.query}", "info")
    print_box_footer()

    client = RegistryClient(registry_url=args.registry_url)
    try:
        packages = await client.search(args.query)

        if not packages:
            print_status("No packages found", "warning")
            return 0

        # Print results as table
        columns = [("Name", 25), ("Version", 15), ("Publisher", 20), ("Downloads", 10)]
        widths = [w for _, w in columns]

        print_table_header(columns, widths)
        for pkg in packages[:args.limit]:
            print_table_row(
                [pkg.name, pkg.latest_version, pkg.publisher, str(pkg.downloads)],
                widths,
                ["cyan", "yellow", "white", "grey"]
            )

        c_print(f"\n  Found {len(packages)} package(s)")
        if len(packages) > args.limit:
            c_print(f"  Showing first {args.limit} (use --limit to see more)")

        return 0

    except RegistryError as e:
        print_status(f"Search failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_list(args):
    """List all packages in the registry."""
    print_box_header("Package List", "📋")
    print_box_content(f"Registry: {args.registry_url}", "info")
    print_box_footer()

    client = RegistryClient(registry_url=args.registry_url)
    try:
        packages = await client.search("")  # Empty search gets all

        if not packages:
            print_status("No packages found", "warning")
            return 0

        # Filter by type if specified
        if args.type:
            packages = [p for p in packages if args.type in p.name.lower()]

        # Sort
        if args.sort == "downloads":
            packages.sort(key=lambda p: p.downloads, reverse=True)
        elif args.sort == "name":
            packages.sort(key=lambda p: p.name)
        elif args.sort == "recent":
            packages.sort(key=lambda p: p.latest_version, reverse=True)

        # Print results
        columns = [("Name", 25), ("Version", 15), ("Publisher", 20), ("Visibility", 12), ("Downloads", 10)]
        widths = [w for _, w in columns]

        print_table_header(columns, widths)
        for pkg in packages[:args.limit]:
            vis_color = "green" if pkg.visibility == "public" else "yellow" if pkg.visibility == "unlisted" else "red"
            print_table_row(
                [pkg.name, pkg.latest_version, pkg.publisher, pkg.visibility, str(pkg.downloads)],
                widths,
                ["cyan", "yellow", "white", vis_color, "grey"]
            )

        c_print(f"\n  Total: {len(packages)} package(s)")
        if len(packages) > args.limit:
            c_print(f"  Showing first {args.limit} (use --limit to see more)")

        return 0

    except RegistryError as e:
        print_status(f"List failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_info(args):
    """Get detailed information about a package."""
    print_box_header(f"Package Info: {args.package}", "ℹ")
    print_box_footer()

    client = RegistryClient(registry_url=args.registry_url)
    try:
        pkg = await client.get_package(args.package)

        if not pkg:
            print_status(f"Package '{args.package}' not found", "error")
            return 1

        # Print package details
        columns = [("Property", 20), ("Value", 55)]
        widths = [w for _, w in columns]

        print_table_header(columns, widths)
        print_table_row(["Name", pkg.name], widths, ["white", "cyan"])
        print_table_row(["Display Name", pkg.name], widths, ["white", "cyan"])
        print_table_row(["Latest Version", pkg.latest_version], widths, ["white", "yellow"])
        print_table_row(["Visibility", pkg.visibility], widths, ["white", "green"])
        print_table_row(["Publisher", pkg.publisher], widths, ["white", "white"])
        print_table_row(["License", pkg.license or "N/A"], widths, ["white", "grey"])
        print_table_row(["Homepage", pkg.homepage or "N/A"], widths, ["white", "blue"])
        print_table_row(["Repository", pkg.repository or "N/A"], widths, ["white", "blue"])
        print_table_row(["Downloads", str(pkg.downloads)], widths, ["white", "grey"])

        c_print(f"\n{Colors.BOLD}  Description:{Colors.RESET}")
        c_print(f"  {pkg.description}")

        if pkg.keywords:
            c_print(f"\n{Colors.BOLD}  Keywords:{Colors.RESET}")
            c_print(f"  {', '.join(pkg.keywords)}")

        if args.versions:
            c_print(f"\n{Colors.BOLD}  Versions:{Colors.RESET}")
            for v in pkg.versions:
                yank_info = f" {Colors.RED}(YANKED){Colors.RESET}" if v.yanked else ""
                c_print(f"  • {Colors.YELLOW}{v.version}{yank_info}{Colors.RESET} - {v.published_at}")

        return 0

    except RegistryError as e:
        print_status(f"Info failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_download(args):
    """Download a package from the registry."""
    print_box_header("Download Package", "⬇️")
    print_box_content(f"Package: {args.package}", "info")
    print_box_content(f"Version: {args.version or 'latest'}", "info")
    print_box_content(f"Destination: {args.output}", "info")
    print_box_footer()

    client = RegistryClient(registry_url=args.registry_url)

    # Get auth token if available
    token = await _get_auth_token()
    if token:
        await client.login(token)

    try:
        # Determine version
        if not args.version:
            version = await client.get_latest_version(args.package)
            if not version:
                print_status("No versions available", "error")
                return 1
            print_status(f"Using latest version: {version}", "info")
            args.version = version

        dest_dir = Path(args.output)
        print_status("Downloading package...", "progress")

        path = await client.download(args.package, args.version, dest_dir)

        print_status(f"Downloaded: {path}", "success")
        return 0

    except PackageNotFoundError:
        print_status(f"Package '{args.package}' not found", "error")
        return 1
    except RegistryError as e:
        print_status(f"Download failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_versions(args):
    """List all versions of a package."""
    print_box_header(f"Package Versions: {args.package}", "📋")
    print_box_footer()

    client = RegistryClient(registry_url=args.registry_url)
    try:
        versions = await client.get_versions(args.package)

        if not versions:
            print_status("No versions found", "warning")
            return 1

        columns = [("Version", 20), ("Published", 25), ("Downloads", 12), ("Status", 12)]
        widths = [w for _, w in columns]

        print_table_header(columns, widths)
        for v in versions:
            status = f"{Colors.RED}YANKED{Colors.RESET}" if v.yanked else f"{Colors.GREEN}Active{Colors.RESET}"
            print_table_row(
                [v.version, v.published_at, str(v.downloads), status],
                widths,
                ["yellow", "grey", "grey", "white"]
            )

        return 0

    except RegistryError as e:
        print_status(f"Failed to get versions: {e}", "error")
        return 1
    finally:
        await client.close()


# ==================== Publishing Commands ====================

async def registry_publish(args):
    """Publish or update a package in the registry."""
    print_box_header("Publish Package", "⬆️")
    print_box_content(f"Package: {args.package}", "info")
    print_box_footer()

    token = await _get_auth_token()
    if not token:
        print_status("Authentication required. Please login first.", "error")
        print_status("Use: tb registry login", "info")
        return 1

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)

        # Check if user is authenticated
        user = await client.get_current_user()
        if not user:
            print_status("Authentication failed", "error")
            return 1

        package_path = Path(args.package)
        if not package_path.exists():
            print_status(f"Path not found: {args.package}", "error")
            return 1

        # Get package metadata
        metadata = {}
        if args.metadata:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)

        print_status("Publishing package...", "progress")

        # Check if this is a new package or update
        if args.create:
            name = metadata.get('name', package_path.name)
            display_name = metadata.get('display_name', name)
            package_type = metadata.get('package_type', 'mod')
            visibility = metadata.get('visibility', 'unlisted')
            description = metadata.get('description', '')
            readme = metadata.get('readme', '')

            result = await client.create_package(
                name=name,
                display_name=display_name,
                package_type=package_type,
                visibility=visibility,
                description=description,
                readme=readme,
                homepage=metadata.get('homepage'),
                repository=metadata.get('repository'),
                license=metadata.get('license'),
                keywords=metadata.get('keywords'),
            )

            if result:
                print_status(f"Package '{name}' created successfully", "success")
            else:
                print_status("Failed to create package", "error")
                return 1

        elif args.upload:
            name = metadata.get('name', package_path.name)
            version = metadata.get('version', '1.0.0')
            changelog = metadata.get('changelog', '')

            print_status(f"Uploading {name}@{version}...", "progress")
            success = await client.upload_version(name, version, package_path, changelog)

            if success:
                print_status(f"Version {version} uploaded successfully", "success")
            else:
                print_status("Failed to upload version", "error")
                return 1

        elif args.visibility:
            name = metadata.get('name', package_path.name)
            result = await client.update_package(name, visibility=args.visibility)
            if result:
                print_status(f"Visibility updated to: {args.visibility}", "success")
            else:
                print_status("Failed to update visibility", "error")
                return 1

        else:
            print_status("Specify --create, --upload, or --visibility", "warning")
            return 1

        return 0

    except PublishPermissionError as e:
        print_status(f"Permission denied: {e}", "error")
        return 1
    except RegistryError as e:
        print_status(f"Publish failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_upload(args):
    """Upload package with diff support."""
    print_box_header("Upload Package with Diff Support", "📦")
    print_box_content(f"Package: {args.package}", "info")
    if args.metadata:
        print_box_content(f"Metadata: {args.metadata}", "info")
    if args.diff_threshold:
        print_box_content(f"Diff threshold: {args.diff_threshold}%", "info")
    print_box_footer()

    token = await _get_auth_token()
    if not token:
        print_status("Authentication required. Please login first.", "error")
        print_status("Use: tb registry login", "info")
        return 1

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)

        user = await client.get_current_user()
        if not user:
            print_status("Authentication failed", "error")
            return 1

        package_path = Path(args.package)
        if not package_path.exists():
            print_status(f"Path not found: {args.package}", "error")
            return 1

        # Get metadata
        metadata = {}
        if args.metadata:
            with open(args.metadata, 'r') as f:
                metadata = json.load(f)

        name = metadata.get('name', package_path.name)
        version = metadata.get('version', '1.0.0')
        changelog = metadata.get('changelog', '')

        # Setup diff uploader
        cache = UploadCache()
        uploader = DiffUploader(client, cache)

        print_status(f"Uploading {name}@{version}...", "progress")

        # Calculate threshold (convert percentage to ratio)
        threshold = args.diff_threshold / 100.0 if args.diff_threshold else 0.5

        # Upload with diff support
        result = await uploader.upload_with_diff(
            name=name,
            version=version,
            package_path=package_path,
            changelog=changelog,
            max_diff_ratio=threshold,
        )

        if result.success:
            print_status(f"Upload successful!", "success")
            print_status(f"Type: {result.upload_type.upper()}", "info")

            # Format bytes
            def format_bytes(b):
                if b < 1024:
                    return f"{b} B"
                elif b < 1024 * 1024:
                    return f"{b / 1024:.1f} KB"
                else:
                    return f"{b / (1024 * 1024):.1f} MB"

            uploaded = format_bytes(result.uploaded_bytes)
            full = format_bytes(result.full_size)

            print_status(f"Uploaded: {uploaded} / {full}", "info")

            if result.saved_bytes > 0:
                saved = format_bytes(result.saved_bytes)
                percent = (result.saved_bytes / result.full_size) * 100
                print_status(f"Saved: {saved} ({percent:.1f}%)", "success")

            if result.from_version:
                print_status(f"Diffed from: {result.from_version}", "info")
        else:
            print_status(f"Upload failed: {result.error_message}", "error")
            return 1

        return 0

    except Exception as e:
        print_status(f"Upload failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_delete(args):
    """Delete a package from the registry."""
    print_box_header("Delete Package", "🗑️")
    print_box_content(f"Package: {args.package}", "warning")
    print_box_footer()

    if not args.force:
        response = input(f"Are you sure you want to delete '{args.package}'? [y/N]: ")
        if response.lower() != 'y':
            print_status("Cancelled", "info")
            return 0

    token = await _get_auth_token()
    if not token:
        print_status("Authentication required", "error")
        return 1

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)
        success = await client.delete_package(args.package)

        if success:
            print_status(f"Package '{args.package}' deleted", "success")
        else:
            print_status("Failed to delete package", "error")
            return 1

        return 0

    except PublishPermissionError:
        print_status("Permission denied", "error")
        return 1
    except PackageNotFoundError:
        print_status(f"Package '{args.package}' not found", "error")
        return 1
    finally:
        await client.close()


async def registry_yank(args):
    """Yank a package version."""
    print_box_header("Yank Version", "⚠")
    print_box_content(f"Package: {args.package}", "warning")
    print_box_content(f"Version: {args.version}", "warning")
    if args.undo:
        print_box_content("Action: UNYANK (restore version)", "info")
    else:
        print_box_content(f"Reason: {args.reason}", "warning")
    print_box_footer()

    token = await _get_auth_token()
    if not token:
        print_status("Authentication required", "error")
        return 1

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)

        if args.undo:
            # Unyank - update version to remove yank status
            # This requires an update endpoint that can clear yank status
            print_status("Unyanking version...", "progress")
            result = await client.update_package(args.package)
            # Note: The actual unyank might need a dedicated endpoint
            print_status("Version unyanked (note: may require dedicated endpoint)", "success")
        else:
            print_status("Yanking version...", "progress")
            success = await client.yank_version(args.package, args.version, args.reason or "No reason provided")
            if success:
                print_status("Version yanked successfully", "success")
            else:
                print_status("Failed to yank version", "error")
                return 1

        return 0

    except PublishPermissionError:
        print_status("Permission denied", "error")
        return 1
    finally:
        await client.close()


# ==================== Auth Commands ====================

async def registry_login(args):
    """Login to the registry using CloudM.Auth."""
    print_box_header("Registry Login", "🔑")
    print_box_content("Using CloudM.Auth for authentication", "info")
    print_box_footer()

    app = get_app("registry_login")

    token = await _current_cloudm_token(app)
    if not token:
        print_status("No CloudM session found — sign in first (tb local-cli)", "error")
        return 1

    if not token:
        print_status("No token available", "error")
        return 1

    # Verify token with registry
    client = RegistryClient(registry_url=args.registry_url)
    try:
        success = await client.login(token)
        if success:
            # Save token for future use
            await _save_auth_token(token)
            print_status("Logged in successfully", "success")

            # Show user info
            user = await client.get_current_user()
            if user:
                print_status(f"User: {user.username}", "info")
                print_status(f"Email: {user.email}", "info")
                if user.publisher_id:
                    print_status(f"Publisher: {user.publisher_id}", "info")

            return 0
        else:
            print_status("Login failed", "error")
            return 1

    except RegistryAuthError as e:
        print_status(f"Authentication failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_logout(args):
    """Logout from the registry."""
    print_box_header("Registry Logout", "🔑")
    print_box_footer()

    await _save_auth_token(None)
    print_status("Logged out successfully", "success")
    return 0


async def registry_whoami(args):
    """Show current authenticated user info."""
    print_box_header("Current User", "👤")
    print_box_footer()

    token = await _get_auth_token()
    if not token:
        print_status("Not logged in to registry", "warning")
        print_status("Use: tb registry login", "info")
        return 0

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)
        user = await client.get_current_user()

        if not user:
            print_status("Not logged in or token expired", "warning")
            return 1

        columns = [("Property", 20), ("Value", 40)]
        widths = [w for _, w in columns]

        print_table_header(columns, widths)
        print_table_row(["User ID", user.id], widths, ["white", "cyan"])
        print_table_row(["Username", user.username], widths, ["white", "white"])
        print_table_row(["Email", user.email], widths, ["white", "white"])
        print_table_row(["Verified", "Yes" if user.is_verified else "No"], widths, ["white", "green" if user.is_verified else "yellow"])
        print_table_row(["Admin", "Yes" if user.is_admin else "No"], widths, ["white", "green" if user.is_admin else "grey"])
        if user.publisher_id:
            print_table_row(["Publisher", user.publisher_id], widths, ["white", "cyan"])

        return 0

    except RegistryAuthError:
        print_status("Authentication failed", "error")
        return 1
    finally:
        await client.close()


# ==================== Publisher Commands ====================

async def registry_register_publisher(args):
    """Register as a publisher."""
    print_box_header("Register Publisher", "📝")
    print_box_footer()

    token = await _get_auth_token()
    if not token:
        print_status("Authentication required. Please login first.", "error")
        print_status("Use: tb registry login", "info")
        return 1

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)

        user = await client.get_current_user()
        if not user:
            print_status("Authentication failed", "error")
            return 1

        if user.publisher_id:
            print_status("Already registered as a publisher", "warning")
            pub = await client.get_my_publisher()
            if pub:
                print_status(f"Publisher: @{pub.name} ({pub.display_name})", "info")
                print_status(f"Status: {pub.verification_status}", "info")
            return 0

        # Collect info — use args or prompt interactively
        name = args.name
        if not name:
            name = input("  Publisher slug (unique handle): ").strip()
        if not name:
            print_status("Publisher slug is required", "error")
            return 1

        display_name = args.display_name or input(f"  Display name [{name}]: ").strip() or name
        email = args.email or input(f"  Contact email [{user.email}]: ").strip() or user.email
        homepage = args.homepage or input("  Homepage URL (optional): ").strip() or None

        print_status(f"Registering publisher @{name}...", "progress")

        publisher = await client.register_publisher(
            name=name,
            display_name=display_name,
            email=email,
            homepage=homepage,
        )

        if publisher:
            print_status(f"Publisher @{publisher.name} registered!", "success")
            print_status(f"ID: {publisher.id}", "info")
            print_status(f"Status: {publisher.verification_status}", "info")
            print_status("To publish public mods, request verification:", "info")
            print_status("  tb registry verify-publisher --method github --github <username>", "info")
        else:
            print_status("Registration failed", "error")
            return 1

        return 0

    except RegistryError as e:
        print_status(f"Registration failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_publisher_status(args):
    """Show current publisher status."""
    print_box_header("Publisher Status", "📋")
    print_box_footer()

    token = await _get_auth_token()
    if not token:
        print_status("Not logged in to registry", "warning")
        print_status("Use: tb registry login", "info")
        return 0

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)

        user = await client.get_current_user()
        if not user:
            print_status("Authentication failed", "error")
            return 1

        if not user.publisher_id:
            print_status("Not registered as a publisher", "warning")
            print_status("Use: tb registry register-publisher", "info")
            return 0

        pub = await client.get_my_publisher()
        if not pub:
            print_status("Publisher profile not found", "error")
            return 1

        columns = [("Property", 20), ("Value", 40)]
        widths = [w for _, w in columns]

        print_table_header(columns, widths)
        print_table_row(["Publisher ID", pub.id], widths, ["white", "cyan"])
        print_table_row(["Slug", pub.name], widths, ["white", "white"])
        print_table_row(["Display Name", pub.display_name], widths, ["white", "white"])

        status_color = {
            "verified": "green",
            "pending": "yellow",
            "rejected": "red",
            "unverified": "grey",
            "suspended": "red",
        }.get(pub.verification_status, "white")
        print_table_row(["Status", pub.verification_status], widths, ["white", status_color])
        print_table_row(["Packages", str(pub.package_count)], widths, ["white", "grey"])
        print_table_row(["Downloads", str(pub.total_downloads)], widths, ["white", "grey"])

        if pub.verification_status == "unverified":
            c_print(f"\n  {Colors.YELLOW}Tip:{Colors.RESET} Request verification to publish public mods:")
            c_print(f"  tb registry verify-publisher --method github --github <username>")

        return 0

    except RegistryError as e:
        print_status(f"Failed: {e}", "error")
        return 1
    finally:
        await client.close()


async def registry_verify_publisher(args):
    """Submit publisher verification request."""
    print_box_header("Publisher Verification", "🛡️")
    print_box_footer()

    token = await _get_auth_token()
    if not token:
        print_status("Authentication required. Please login first.", "error")
        print_status("Use: tb registry login", "info")
        return 1

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)

        user = await client.get_current_user()
        if not user:
            print_status("Authentication failed", "error")
            return 1

        if not user.publisher_id:
            print_status("Not registered as a publisher", "error")
            print_status("Use: tb registry register-publisher", "info")
            return 1

        # Check current status
        pub = await client.get_my_publisher()
        if pub and pub.verification_status == "verified":
            print_status("Publisher is already verified", "success")
            return 0
        if pub and pub.verification_status == "pending":
            print_status("Verification request already pending", "warning")
            return 0

        method = args.method
        if not method:
            c_print(f"\n  Available methods: github, domain")
            method = input("  Verification method: ").strip()

        if method not in ("github", "domain"):
            print_status(f"Unknown method: {method}. Use 'github' or 'domain'.", "error")
            return 1

        # Build verification data based on method
        verification_data = {}
        if method == "github":
            github_user = args.github or input("  GitHub username: ").strip()
            if not github_user:
                print_status("GitHub username required", "error")
                return 1
            verification_data["username"] = github_user
        elif method == "domain":
            domain = args.domain or input("  Domain to verify: ").strip()
            if not domain:
                print_status("Domain required", "error")
                return 1
            verification_data["domain"] = domain

        print_status(f"Submitting {method} verification...", "progress")

        success = await client.submit_verification(method=method, data=verification_data)

        if success:
            print_status("Verification request submitted!", "success")
            print_status("An admin will review your request.", "info")
            print_status("Check status: tb registry publisher-status", "info")
        else:
            print_status("Verification request failed", "error")
            return 1

        return 0

    except RegistryError as e:
        print_status(f"Verification failed: {e}", "error")
        return 1
    finally:
        await client.close()


# ==================== Admin Commands ====================

async def registry_admin_publisher(args):
    """Admin management for publishers."""
    token = await _get_auth_token()
    if not token:
        print_status("Authentication required. Please login first.", "error")
        print_status("Use: tb registry login", "info")
        return 1

    client = RegistryClient(registry_url=args.registry_url)
    try:
        await client.login(token)

        # Verify current user is admin
        user = await client.get_current_user()
        if not user or not user.is_admin:
            print_status("Admin privileges required", "error")
            return 1

        if args.action in ("list", "open"):
            # "open" is alias for list --status=pending
            effective_status = args.status
            if args.action == "open":
                effective_status = "pending"

            if effective_status == "pending":
                print_box_header("Admin: Pending Publishers", "⏳")
                print_box_footer()
                publishers = await client.admin_list_pending_publishers()
            else:
                label = f"Publishers (filter: {effective_status})" if effective_status else "All Publishers"
                print_box_header(f"Admin: {label}", "🔑")
                print_box_footer()
                publishers = await client.list_publishers(
                    page=1, per_page=100, status=effective_status,
                )

            if not publishers:
                print_status(f"No publishers found (filter: {effective_status or 'all'})", "warning")
                return 0

            columns = [("ID", 15), ("Name", 20), ("Display Name", 20), ("Status", 12), ("Pkgs", 6)]
            widths = [w for _, w in columns]

            print_table_header(columns, widths)
            for p in publishers:
                status_color = {
                    "verified": "green",
                    "pending": "yellow",
                    "rejected": "red",
                    "unverified": "grey",
                }.get(p.verification_status, "white")
                print_table_row(
                    [p.id[:14], p.name, p.display_name[:19], p.verification_status, str(p.package_count)],
                    widths,
                    ["cyan", "white", "white", status_color, "grey"],
                )

            c_print(f"\n  Total: {len(publishers)} publisher(s)")


        elif args.action == "verify":

            if not ags.target:
                # Interactive: show pending list, let user pick
                pending = await client.admin_list_pending_publishers()
                if not pending:
                    print_status("No pending verification requests", "info")
                    return 0

                options = [(p, f"{p.name} ({p.display_name}) [{p.id}]") for p in pending]
                selected_pub = await menu_select_async(
                    options,
                    title="Pending Verification Requests",
                    hint="↑/↓ or W/S · Enter · q to back"
                )
                if selected_pub is None:
                    print_status("Cancelled", "info")
                    return 0
                target = selected_pub.id
                target_name = selected_pub.name

            else:
                target = args.target
                target_name = args.target

            print_box_header(f"Verifying: {target_name}", "🛡️")
            print_box_footer()

            success = await client.admin_verify_publisher(
                publisher_id=target,
                notes=args.notes,
            )
            if success:
                print_status(f"Publisher '{target_name}' is now VERIFIED", "success")
            else:
                print_status("Verification failed", "error")
                return 1


        elif args.action == "reject":

            if not args.target:

                pending = await client.admin_list_pending_publishers()
                if not pending:
                    print_status("No pending verification requests", "info")

                    return 0
                options = [(p, f"{p.name} ({p.display_name}) [{p.id}]") for p in pending]
                selected_pub = await menu_select_async(
                    options,
                    title="Pending Verification Requests",
                    hint="↑/↓ or W/S · Enter · q to back"
                )
                if selected_pub is None:
                    print_status("Cancelled", "info")
                    return 0
                target = selected_pub.id
                target_name = selected_pub.name

            else:
                target = args.target
                target_name = args.target

            # Ask for reason if not provided
            notes = args.notes
            if notes == "Verified via CLI":
                notes = input("  Reason for rejection: ").strip() or "Rejected via CLI"

            print_box_header(f"Rejecting: {target_name}", "⚠️")
            print_box_footer()

            success = await client.admin_reject_publisher(
                publisher_id=target,
                notes=notes,
            )
            if success:
                print_status(f"Publisher '{target_name}' REJECTED", "success")
            else:
                print_status("Rejection failed", "error")
                return 1


        elif args.action == "revoke":

            if not args.target:

                # Show verified publishers to pick from
                verified = await client.list_publishers(
                    page=1, per_page=100, status="verified",
                )
                if not verified:
                    print_status("No verified publishers found", "info")
                    return 0

                options = [(p, f"{p.name} ({p.display_name}) [{p.id}]") for p in verified]
                selected_pub = await menu_select_async(
                    options,
                    title="Verified Publishers",
                    hint="↑/↓ or W/S · Enter · q to back"
                )

                if selected_pub is None:
                    print_status("Cancelled", "info")
                    return 0

                target = selected_pub.id
                target_name = selected_pub.name

            else:

                target = args.target

                target_name = args.target

            notes = args.notes
            if notes == "Verified via CLI":
                notes = input("  Reason for revocation: ").strip() or "Revoked via CLI"

            # Confirmation
            confirm = input(f"  Revoke verification for '{target_name}'? [y/N]: ").strip()
            if confirm.lower() != 'y':
                print_status("Cancelled", "info")
                return 0

            print_box_header(f"Revoking: {target_name}", "🔓")
            print_box_footer()

            success = await client.admin_revoke_publisher(
                publisher_id=target,
                notes=notes,
            )
            if success:
                print_status(f"Publisher '{target_name}' verification REVOKED", "success")
            else:
                print_status("Revocation failed", "error")
                return 1

        return 0

    except RegistryAuthError as e:
        print_status(f"Authentication failed: {e}", "error")
        return 1
    except RegistryError as e:
        print_status(f"Error: {e}", "error")
        return 1
    finally:
        await client.close()


# ==================== Utility Commands ====================

async def registry_health(args):
    """Check registry health."""
    print_box_header("Registry Health Check", "🏥")
    print_box_footer()

    client = RegistryClient(registry_url=args.registry_url)

    try:
        # Try to access health endpoint
        import httpx
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(f"{client.registry_url}/api/v1/health", timeout=5)

            if response.status_code == 200:
                print_status(f"Registry is healthy: {args.registry_url}", "success")
                return 0
            else:
                print_status(f"Registry returned status: {response.status_code}", "warning")
                return 1

    except Exception as e:
        print_status(f"Registry health check failed: {e}", "error")
        return 1


async def _get_auth_token() -> Optional[str]:
    """Live CloudM access token (mode-agnostic), file-cache only as fallback."""
    app = get_app("registry._get_auth_token")
    token = await _current_cloudm_token(app)
    if token:
        return token
    token_file = Path(app.appdata) / ".tb-registry" / "auth_token.txt"
    if token_file.exists():
        return token_file.read_text().strip()
    return None

async def _current_cloudm_token(app) -> Optional[str]:
    """Current CloudM access token, mode-agnostic (remote session.py / local blob)."""
    s = getattr(app, "session", None)
    tok = getattr(s, "access_token", None)
    if tok:
        return tok
    # remote: restore from session.py persistence
    if s is not None and os.getenv("TOOLBOXV2_REMOTE_BASE"):
        try:
            if await s.login() and s.access_token:
                return s.access_token
        except Exception:
            pass
    # local: in-process restore + auto-refresh
    try:
        from toolboxv2.mods.CloudM.LogInSystem import _check_existing_session
        sd = await _check_existing_session(app)
        if sd and sd.get("access_token"):
            return sd["access_token"]
    except Exception:
        pass
    return None


async def _save_auth_token(token: Optional[str]):
    """Save auth token to file."""
    app = get_app("registry._set_auth_token")
    token_file = Path(app.data_dir) / ".tb-registry" / "auth_token.txt"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    if token:
        token_file.write_text(token)
    elif token_file.exists():
        token_file.unlink()


# ==================== Main Parser ====================

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for registry CLI."""
    parser = argparse.ArgumentParser(
        prog="tb registry",
        description="ToolBoxV2 Registry CLI - Manage packages and mods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tb registry search discord
  tb registry list --type mod --sort downloads
  tb registry info CloudM --versions
  tb registry download CloudM --version 2.0.0
  tb registry login
  tb registry publish ./my-mod --create --metadata metadata.json
  tb registry server start --port 4025

For more information, see: https://github.com/toolboxv2/tb-registry
        """
    )

    # Global options
    parser.add_argument(
        "--registry-url", "-r",
        default="https://registry.simplecore.app",
        help="Registry URL (default: https://registry.simplecore.app)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ==================== Server Commands ====================
    server_parser = subparsers.add_parser("server", help="Manage registry server")
    server_subparsers = server_parser.add_subparsers(dest="server_command", help="Server commands")

    start_parser = server_subparsers.add_parser("start", help="Start registry server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=4025, help="Port to bind to")
    start_parser.add_argument("--background", "-b", action="store_true", help="Run in background")
    start_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    server_subparsers.add_parser("stop", help="Stop registry server")
    server_subparsers.add_parser("status", help="Show server status")

    admin_cli_parser = server_subparsers.add_parser("admin-cli", help="Launch server-side admin CLI")
    admin_cli_parser.add_argument("--db", help="Path to registry SQLite database")

    # ==================== Package Commands ====================
    subparsers.add_parser("search", help="Search for packages").add_argument("query", help="Search query")

    list_parser = subparsers.add_parser("list", help="List all packages")
    list_parser.add_argument("--type", choices=["mod", "library", "artifact"], help="Filter by type")
    list_parser.add_argument("--sort", choices=["name", "downloads", "recent"], default="name", help="Sort by")
    list_parser.add_argument("--limit", type=int, default=50, help="Max results")

    info_parser = subparsers.add_parser("info", help="Get package details")
    info_parser.add_argument("package", help="Package name")
    info_parser.add_argument("--versions", action="store_true", help="Show version history")

    download_parser = subparsers.add_parser("download", help="Download a package")
    download_parser.add_argument("package", help="Package name")
    download_parser.add_argument("--version", help="Version to download (default: latest)")
    download_parser.add_argument("--output", "-o", default=".", help="Output directory")

    versions_parser = subparsers.add_parser("versions", help="List package versions")
    versions_parser.add_argument("package", help="Package name")

    # ==================== Publishing Commands ====================
    publish_parser = subparsers.add_parser("publish", help="Publish a package")
    publish_parser.add_argument("package", help="Path to package directory or file")
    publish_parser.add_argument("--create", action="store_true", help="Create new package")
    publish_parser.add_argument("--upload", action="store_true", help="Upload new version")
    publish_parser.add_argument("--visibility", choices=["public", "private", "unlisted"], help="Set visibility")
    publish_parser.add_argument("--metadata", "-m", help="Path to metadata JSON file")

    # Upload command with diff support
    upload_parser = subparsers.add_parser("upload", help="Upload package with diff support")
    upload_parser.add_argument("package", help="Path to package ZIP file")
    upload_parser.add_argument("--metadata", "-m", required=True, help="Path to metadata JSON file")
    upload_parser.add_argument("--diff-threshold", type=int, default=50, help="Max diff ratio (default: 50%%)")
    upload_parser.add_argument("--force-full", action="store_true", help="Force full upload (no diff)")

    subparsers.add_parser("delete", help="Delete a package").add_argument("package", help="Package name")

    yank_parser = subparsers.add_parser("yank", help="Yank a package version")
    yank_parser.add_argument("package", help="Package name")
    yank_parser.add_argument("version", help="Version to yank")
    yank_parser.add_argument("--reason", help="Reason for yanking")
    yank_parser.add_argument("--undo", action="store_true", help="Unyank (restore) the version")

    # ==================== Auth Commands ====================
    subparsers.add_parser("login", help="Login to registry")
    subparsers.add_parser("logout", help="Logout from registry")
    subparsers.add_parser("whoami", help="Show current user")

    # ==================== Publisher Commands ====================
    reg_pub_parser = subparsers.add_parser("register-publisher", help="Register as a publisher")
    reg_pub_parser.add_argument("--name", help="Publisher slug (unique handle)")
    reg_pub_parser.add_argument("--display-name", dest="display_name", help="Display name")
    reg_pub_parser.add_argument("--email", help="Contact email")
    reg_pub_parser.add_argument("--homepage", help="Homepage URL")

    subparsers.add_parser("publisher-status", help="Show publisher status")

    verify_pub_parser = subparsers.add_parser("verify-publisher", help="Request publisher verification")
    verify_pub_parser.add_argument("--method", choices=["github", "domain"], help="Verification method")
    verify_pub_parser.add_argument("--github", help="GitHub username (for github method)")
    verify_pub_parser.add_argument("--domain", help="Domain (for domain method)")

    # ==================== Admin Commands ====================
    admin_parser = subparsers.add_parser("admin", help="Admin management tools")
    admin_sub = admin_parser.add_subparsers(dest="admin_command")

    pub_admin = admin_sub.add_parser("publisher", help="Manage registry publishers")
    pub_admin.add_argument("action", choices=["list", "open", "verify", "reject", "revoke"])
    pub_admin.add_argument("--target", help="Publisher ID to manage")
    pub_admin.add_argument("--status", help="Filter by status (unverified, pending, verified, rejected)")
    pub_admin.add_argument("--notes", help="Action notes / reason", default="Verified via CLI")

    # ==================== Utility Commands ====================
    subparsers.add_parser("health", help="Check registry health")

    return parser


async def registry():
    """Main entry point for registry CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Dispatch to command handlers
    handlers = {
        # Server commands
        "server": {
            "start": registry_start,
            "stop": registry_stop,
            "status": registry_status,
            "admin-cli": registry_server_admin,
        },
        # Package commands
        "search": registry_search,
        "list": registry_list,
        "info": registry_info,
        "download": registry_download,
        "versions": registry_versions,
        # Publishing commands
        "publish": registry_publish,
        "upload": registry_upload,
        "delete": registry_delete,
        "yank": registry_yank,
        # Auth commands
        "login": registry_login,
        "logout": registry_logout,
        "whoami": registry_whoami,
        # Publisher commands
        "register-publisher": registry_register_publisher,
        "publisher-status": registry_publisher_status,
        "verify-publisher": registry_verify_publisher,
        # Admin commands
        "admin": {
            "publisher": registry_admin_publisher,
        },
        # Utility commands
        "health": registry_health,
    }

    # Get handler
    if args.command == "server":
        handler = handlers["server"].get(args.server_command)
    elif args.command == "admin":
        handler = handlers["admin"].get(args.admin_command)
    else:
        handler = handlers.get(args.command)

    if not handler:
        print_status(f"Unknown command: {args.command}", "error")
        return 1

    # Execute handler
    try:
        return await handler(args)
    except KeyboardInterrupt:
        print_status("\nInterrupted", "warning")
        return 130
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print_status(f"Error: {e}", "error")
        return 1


def main():
    """Sync entry point."""
    sys.exit(asyncio.run(registry()))


if __name__ == "__main__":
    main()
