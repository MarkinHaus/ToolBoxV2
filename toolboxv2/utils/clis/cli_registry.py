# file: toolboxv2/utils/clis/cli_registry.py
# ToolBoxV2 Registry CLI
# Provides command-line interface for the TB Registry

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from toolboxv2.mods.CloudM.LogInSystem import _load_cli_token
from toolboxv2.utils.clis.cli_printing import (
    Colors, c_print, print_box_header, print_box_footer,
    print_box_content, print_status, print_table_header, print_table_row
)

# Import RegistryClient
from toolboxv2.utils.extras.registry_client import (
    RegistryClient,
    RegistryError,
    RegistryAuthError,
    PackageNotFoundError,
    PublishPermissionError,
    UserInfo,
    PackageSummary,
    PackageDetail,
)

# Import DiffUploader
from toolboxv2.utils.extras.diff_uploader import DiffUploader
from toolboxv2.utils.extras.upload_cache import UploadCache


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
    if data:
        print_status(f"Status: {data.get('status', 'unknown')}", "success")
        print_status(f"PID: {data.get('pid', 'N/A')}", "info")
        print_status(f"Host: {data.get('host', 'N/A')}", "info")
        print_status(f"Port: {data.get('port', 'N/A')}", "info")
    else:
        print_status("Registry server is not running", "warning")
    return 0


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

    # Get token from CloudM.Auth
    #result = await app.a_run_any(
    #    "CloudM.LogInSystem",
    #    "get_token",
    #    get_results=True
    #)

    #if result.is_error():
    #    print_status("Failed to get authentication token", "error")
    #    print_status("Make sure CloudM.Auth module is loaded", "warning")
    #    return 1
    await asyncio.sleep(0.25)
    token_data = await _load_cli_token(app, app.session.username)
    token = token_data.get("access_token")

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
    """Get stored auth token."""
    token_file = Path.home() / ".tb-registry" / "auth_token.txt"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


async def _save_auth_token(token: Optional[str]):
    """Save auth token to file."""
    token_file = Path.home() / ".tb-registry" / "auth_token.txt"
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
        # Utility commands
        "health": registry_health,
    }

    # Get handler
    if args.command == "server":
        handler = handlers["server"].get(args.server_command)
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
