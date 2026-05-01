"""
ToolBoxV2 Docker Image Builder & Compose Manager CLI

Image commands:
    tb docker-image                                    # Build toolboxv2:latest (cli+web)
    tb docker-image --tag dev                          # Build toolboxv2:dev
    tb docker-image --features all                     # Build with ALL features
    tb docker-image --features production              # Build with production preset (cli+web)
    tb docker-image --variant ssh                      # Build SSH variant on top
    tb docker-image --push                            # Build and push to Docker Hub
    tb docker-image --no-cache                        # Force rebuild

Compose commands:
    tb docker-image compose up                         # Start default stack
    tb docker-image compose up --profiles redis,llm    # Start with profiles
    tb docker-image compose down                       # Stop stack
    tb docker-image compose ps                         # List running services
    tb docker-image compose logs [service]             # View logs
    tb docker-image compose scale tb-worker=4          # Scale workers

    Available features: cli, web, desktop, exotic, isaa
    Presets: all, production, dev
"""

import argparse
import os
import subprocess
import sys


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

FEATURE_MAP = {
    'cli': 'FEATURE_CLI',
    'web': 'FEATURE_WEB',
    'desktop': 'FEATURE_DESKTOP',
    'exotic': 'FEATURE_EXOTIC',
    'isaa': 'FEATURE_ISAA',
}

FEATURE_PRESETS = {
    'all': ['cli', 'web', 'desktop', 'exotic', 'isaa'],
    'production': ['cli', 'web'],
    'dev': ['cli', 'web', 'desktop', 'exotic', 'isaa'],
}


# ============================================================================
# IMAGE BUILD
# ============================================================================

def _find_project_root(start=None):
    """Walk up from start until Dockerfile.toolbox is found."""
    current = start or os.getcwd()
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "Dockerfile.toolbox")):
            return current
        current = os.path.dirname(current)
    return None


def _resolve_features(features_str):
    """Parse feature string into dict {feature_name: 1}."""
    if not features_str:
        return {'cli': 1, 'web': 1}

    key = features_str.lower().strip()
    if key in FEATURE_PRESETS:
        return {f: 1 for f in FEATURE_PRESETS[key]}

    active = {}
    for f in key.split(','):
        f = f.strip()
        if f in FEATURE_MAP:
            active[f] = 1
    return active or {'cli': 1, 'web': 1}


def _build_cmd(dockerfile, tag, project_root, active_features, no_cache=False):
    """Construct docker build command list."""
    cmd = ["docker", "build", "-f", dockerfile, "-t", tag]

    for feat, env_var in FEATURE_MAP.items():
        cmd.extend(["--build-arg", f"{env_var}={active_features.get(feat, 0)}"])

    if no_cache:
        cmd.append("--no-cache")

    cmd.append(project_root)
    return cmd


def _run_build(cmd):
    """Execute build command with live output."""
    print(f"  {' '.join(cmd)}\n")
    print("─" * 60)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding="utf-8",
            errors="ignore",
        )
        for line in proc.stdout:
            print(line.rstrip())
        proc.wait()
        return proc.returncode == 0
    except FileNotFoundError:
        print("Error: Docker not found. Install from https://docs.docker.com/get-docker/")
        return False
    except KeyboardInterrupt:
        print("\nBuild cancelled.")
        return False


def build_docker_image(tag="latest", push=False, no_cache=False,
                       project_root=None, features=None, variant=None):
    """Build ToolBoxV2 Docker image(s)."""
    project_root = project_root or _find_project_root()
    if not project_root:
        print("Error: Dockerfile.toolbox not found. Run from ToolBoxV2 root.")
        return False

    active = _resolve_features(features)
    base_tag = f"toolboxv2:{tag}"
    dockerfile = os.path.join(project_root, "Dockerfile.toolbox")

    print(f"Building {base_tag}")
    print(f"  Features: {', '.join(active.keys())}")
    if variant:
        print(f"  Variant:  {variant}")
    print()

    # -- Base image ----------------------------------------------------------
    cmd = _build_cmd(dockerfile, base_tag, project_root, active, no_cache)
    if not _run_build(cmd):
        print(f"\nError: Base build failed")
        return False

    print(f"\n✓ Base image built: {base_tag}")

    # -- SSH variant ---------------------------------------------------------
    if variant == "ssh":
        ssh_dockerfile = os.path.join(project_root, "Dockerfile.ssh")
        if not os.path.exists(ssh_dockerfile):
            print(f"Error: {ssh_dockerfile} not found")
            return False

        ssh_tag = f"toolboxv2:{tag}-ssh"
        ssh_cmd = [
            "docker", "build",
            "-f", ssh_dockerfile,
            "--build-arg", f"BASE_IMAGE={base_tag}",
            "-t", ssh_tag,
        ]
        if no_cache:
            ssh_cmd.append("--no-cache")
        ssh_cmd.append(project_root)

        print(f"\nBuilding SSH variant: {ssh_tag}")
        if not _run_build(ssh_cmd):
            print(f"\nError: SSH variant build failed")
            return False

        print(f"✓ SSH variant built: {ssh_tag}")

    # -- Image info ----------------------------------------------------------
    print()
    subprocess.run(
        ["docker", "images", "toolboxv2", "--format",
         "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"],
        encoding="utf-8", errors="ignore",
    )

    # -- Push ----------------------------------------------------------------
    if push:
        print(f"\nPushing to Docker Hub...")
        hub_image = f"toolboxv2/toolboxv2:{tag}"
        subprocess.run(["docker", "tag", base_tag, hub_image], check=True)
        subprocess.run(["docker", "push", hub_image], check=True)
        print(f"✓ Pushed: {hub_image}")

        if variant == "ssh":
            hub_ssh = f"toolboxv2/toolboxv2:{tag}-ssh"
            subprocess.run(["docker", "tag", f"toolboxv2:{tag}-ssh", hub_ssh], check=True)
            subprocess.run(["docker", "push", hub_ssh], check=True)
            print(f"✓ Pushed: {hub_ssh}")

    return True


# ============================================================================
# COMPOSE MANAGEMENT
# ============================================================================

def _find_compose_file(project_root=None):
    """Find compose.yaml in project root."""
    root = project_root or _find_project_root() or os.getcwd()
    for name in ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            return path
    return None


def compose_command(action, args, project_root=None):
    """Run docker compose commands with profile support."""
    compose_file = _find_compose_file(project_root)
    if not compose_file:
        print("Error: No compose.yaml found in project root.")
        return False

    cmd = ["docker", "compose", "-f", compose_file]

    # Add profiles
    profiles = getattr(args, 'profiles', None)
    if profiles:
        for p in profiles.split(','):
            p = p.strip()
            if p:
                cmd.extend(["--profile", p])

    # Action-specific logic
    if action == "up":
        cmd.extend(["up", "-d"])
        scale = getattr(args, 'scale', None)
        if scale:
            cmd.extend(["--scale", scale])
        build_flag = getattr(args, 'build', False)
        if build_flag:
            cmd.append("--build")

    elif action == "down":
        cmd.extend(["down"])
        volumes_flag = getattr(args, 'volumes', False)
        if volumes_flag:
            cmd.append("-v")

    elif action == "ps":
        cmd.extend(["ps", "-a"])

    elif action == "logs":
        cmd.append("logs")
        follow = getattr(args, 'follow', False)
        if follow:
            cmd.append("-f")
        tail = getattr(args, 'tail', None)
        if tail:
            cmd.extend(["--tail", str(tail)])
        service = getattr(args, 'service', None)
        if service:
            cmd.append(service)

    elif action == "scale":
        scale_spec = getattr(args, 'scale_spec', None)
        if not scale_spec:
            print("Error: scale requires SERVICE=N argument")
            return False
        cmd.extend(["up", "-d", "--scale", scale_spec])

    elif action == "config":
        cmd.append("config")

    else:
        cmd.append(action)

    print(f"  {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, encoding="utf-8", errors="ignore")
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: docker compose not found.")
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="tb docker-image",
        description="Build ToolBoxV2 Docker images & manage Compose stack",
    )
    sub = parser.add_subparsers(dest="subcommand")

    # -- build (default, no subcommand) --------------------------------------
    parser.add_argument("--tag", default="latest", help="Image tag (default: latest)")
    parser.add_argument("--push", action="store_true", help="Push to Docker Hub")
    parser.add_argument("--no-cache", action="store_true", help="Build without cache")
    parser.add_argument("--project-root", help="Project root (auto-detected)")
    parser.add_argument("--features", default=None,
                        help="Features: cli,web,desktop,exotic,isaa or preset (all,production,dev)")
    parser.add_argument("--variant", choices=["ssh"], default=None,
                        help="Build variant on top of base image")

    # -- compose subcommands -------------------------------------------------
    compose_p = sub.add_parser("compose", help="Docker Compose management")
    compose_sub = compose_p.add_subparsers(dest="compose_action")

    up_p = compose_sub.add_parser("up", help="Start stack")
    up_p.add_argument("--profiles", help="Comma-separated profiles (redis,llm,monitoring,...)")
    up_p.add_argument("--scale", help="Scale spec (e.g. tb-worker=4)")
    up_p.add_argument("--build", action="store_true", help="Rebuild images")

    down_p = compose_sub.add_parser("down", help="Stop stack")
    down_p.add_argument("--profiles", help="Profiles")
    down_p.add_argument("-v", "--volumes", action="store_true", help="Remove volumes too")

    ps_p = compose_sub.add_parser("ps", help="List services")
    ps_p.add_argument("--profiles", help="Profiles")

    logs_p = compose_sub.add_parser("logs", help="View logs")
    logs_p.add_argument("service", nargs="?", help="Service name")
    logs_p.add_argument("-f", "--follow", action="store_true")
    logs_p.add_argument("--tail", type=int, default=100)
    logs_p.add_argument("--profiles", help="Profiles")

    scale_p = compose_sub.add_parser("scale", help="Scale a service")
    scale_p.add_argument("scale_spec", help="SERVICE=N (e.g. tb-worker=4)")
    scale_p.add_argument("--profiles", help="Profiles")

    cfg_p = compose_sub.add_parser("config", help="Validate & show resolved config")
    cfg_p.add_argument("--profiles", help="Profiles")

    args = parser.parse_args()

    # -- Dispatch ------------------------------------------------------------
    if args.subcommand == "compose":
        if not args.compose_action:
            compose_p.print_help()
            return 0
        ok = compose_command(args.compose_action, args, args.project_root if hasattr(args, 'project_root') else None)
        return 0 if ok else 1

    # Default: build image
    ok = build_docker_image(
        tag=args.tag,
        push=args.push,
        no_cache=args.no_cache,
        project_root=args.project_root,
        features=args.features,
        variant=args.variant,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
