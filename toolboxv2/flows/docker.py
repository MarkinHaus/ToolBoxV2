import os
import subprocess
import sys

from toolboxv2 import AppArgs, Result

NAME = 'docker'


def run(app, args: AppArgs):
    # Neuer: image build Befehl
    if hasattr(args, 'command') and args.command == 'image':
        return run_image_build(app, args)

    # Legacy: docker compose
    ford_build = ''
    if args.build:
        ford_build = ' --build'
    comm = ''
    if args.modi == 'test':
        comm = 'docker compose up test' + ford_build
    if args.modi == 'live':
        comm = 'docker compose up live' + ford_build
    if args.modi == 'dev':
        comm = 'docker compose up dev --watch' + ford_build
    app.print(f"Running command : {comm}")
    try:
        os.system(comm)
    except KeyboardInterrupt:
        app.print("Exit")


def run_image_build(app, args: AppArgs) -> Result:
    """
    Baue das ToolBoxV2 Haupt-Image.

    Usage:
        tb docker image                    # Baue als toolboxv2:latest
        tb docker image --tag dev          # Baue als toolboxv2:dev
        tb docker image --tag v1.0.0       # Baue als toolboxv2:v1.0.0
        tb docker image --push            # Baue und pushe zu Docker Hub
        tb docker image --no-cache        # Force rebuild ohne cache
    """
    tag = getattr(args, 'tag', 'latest')
    push = getattr(args, 'push', False)
    no_cache = getattr(args, 'no_cache', False)

    image_name = f"toolboxv2:{tag}"

    app.print(f"Building ToolBoxV2 Docker image: {image_name}")
    app.print("")

    # Prüfe ob Dockerfile.toolbox existiert
    dockerfile_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "Dockerfile.toolbox")
    if not os.path.exists(dockerfile_path):
        dockerfile_path = "Dockerfile.toolbox"

    if not os.path.exists(dockerfile_path):
        return Result.error(
            "Dockerfile.toolbox not found. "
            "Please run this command from the ToolBoxV2 root directory."
        )

    # Build Command
    build_cmd = [
        "docker", "build",
        "-f", dockerfile_path,
        "-t", image_name,
        "--build-arg", "BUILDKIT_INLINE_CACHE=1",
    ]

    if no_cache:
        build_cmd.append("--no-cache")

    # Build Context ist das Projektverzeichnis
    build_ctx = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    build_cmd.append(build_ctx)

    app.print(f"Build context: {build_ctx}")
    app.print(f"Command: {' '.join(build_cmd)}")
    app.print("")

    # Build ausführen
    try:
        process = subprocess.Popen(
            build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Output live anzeigen
        for line in process.stdout:
            app.print(line.rstrip())

        process.wait()

        if process.returncode != 0:
            return Result.error(f"Build failed with exit code {process.returncode}")

        app.print("")
        app.print(f"Build successful: {image_name}")
        app.print("")
        app.print("Image size:")
        subprocess.run(["docker", "images", "toolboxv2", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"])

        # Push zu Docker Hub (optional)
        if push:
            app.print("")
            app.print(f"Pushing to Docker Hub...")
            push_cmd = ["docker", "tag", image_name, f"toolboxv2/toolboxv2:{tag}"]
            subprocess.run(push_cmd, check=True)

            push_cmd = ["docker", "push", f"toolboxv2/toolboxv2:{tag}"]
            subprocess.run(push_cmd, check=True)

            app.print(f"Pushed: toolboxv2/toolboxv2:{tag}")

        return Result.ok()

    except subprocess.CalledProcessError as e:
        return Result.error(f"Command failed: {e}")
    except FileNotFoundError:
        return Result.error("Docker not found. Please install Docker first.")
    except KeyboardInterrupt:
        app.print("\nBuild cancelled.")
        return Result.error("Build cancelled")


def get_help():
    """Hilfe-Text für docker Befehle"""
    return """
ToolBoxV2 Docker Commands:

1. Legacy Docker Compose:
   tb docker test              -- Run test environment
   tb docker live              -- Run live environment
   tb docker dev               -- Run dev environment with watch
   tb docker --build           -- Rebuild images before starting

2. Image Build (NEW):
   tb docker image                    -- Build toolboxv2:latest
   tb docker image --tag dev          -- Build toolboxv2:dev
   tb docker image --tag v1.0.0       -- Build toolboxv2:v1.0.0
   tb docker image --push            -- Build and push to Docker Hub
   tb docker image --no-cache        -- Force rebuild without cache

Examples:
   # Lokales Image bauen
   tb docker image

   # Development Image bauen
   tb docker image --tag dev

   # Release Image bauen und pushen
   tb docker image --tag v1.0.0 --push

   # Ohne Cache neu bauen
   tb docker image --no-cache
"""

