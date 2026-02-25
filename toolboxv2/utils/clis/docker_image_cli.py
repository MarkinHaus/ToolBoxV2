"""
ToolBoxV2 Docker Image Builder CLI

Usage:
    tb docker-image                                    # Build toolboxv2:latest (all core features)
    tb docker-image --tag dev                          # Build toolboxv2:dev
    tb docker-image --tag v1.0.0                       # Build toolboxv2:v1.0.0
    tb docker-image --push                            # Build and push to Docker Hub
    tb docker-image --no-cache                        # Force rebuild without cache

    Feature Selection (Manifest-Based Build):
    tb docker-image --features cli,web                # Build with CLI + Web only (minimal)
    tb docker-image --features all                    # Build with ALL features
    tb docker-image --features production             # Build with production preset (cli+web)

    Available Features:
    - cli      CLI Interface (prompt_toolkit, rich, readchar)
    - web      Web Interface (starlette, uvicorn, httpx)
    - desktop  Desktop GUI (PyQt6)
    - exotic   Data Science (scipy, matplotlib, pandas)
    - isaa     AI Agents (litellm, langchain, groq)

    Presets:
    - all          All features enabled
    - production   cli + web (recommended for servers)
    - dev          All features (same as 'all')
"""

import argparse
import os
import subprocess
import sys


def build_docker_image(tag="latest", push=False, no_cache=False, project_root=None, features=None):
    """
    Baue das ToolBoxV2 Docker-Image.

    Args:
        tag: Image tag (default: latest)
        push: Ob das Image nach dem Build gepusht werden soll
        no_cache: Ob ohne Cache gebaut werden soll
        project_root: Projektverzeichnis (wird automatisch erkannt)
        features: Feature selection string (e.g., "cli,web") or preset (e.g., "production")

    Returns:
        bool: True bei Erfolg, False bei Fehler
    """
    # Projektverzeichnis finden
    if project_root is None:
        # Vom aktuellen Verzeichnis aus hochgehen bis wir Dockerfile.toolbox finden
        current = os.getcwd()
        while current != os.path.dirname(current):
            dockerfile = os.path.join(current, "Dockerfile.toolbox")
            if os.path.exists(dockerfile):
                project_root = current
                break
            current = os.path.dirname(current)

    if project_root is None or not os.path.exists(os.path.join(project_root, "Dockerfile.toolbox")):
        print("❌ Dockerfile.toolbox not found!")
        print("   Please run this command from the ToolBoxV2 root directory.")
        return False

    dockerfile_path = os.path.join(project_root, "Dockerfile.toolbox")
    image_name = f"toolboxv2:{tag}"

    # Feature selection
    feature_map = {
        'cli': 'FEATURE_CLI',
        'web': 'FEATURE_WEB',
        'desktop': 'FEATURE_DESKTOP',
        'exotic': 'FEATURE_EXOTIC',
        'isaa': 'FEATURE_ISAA',
    }

    # Feature presets
    feature_presets = {
        'all': ['cli', 'web', 'desktop', 'exotic', 'isaa'],
        'production': ['cli', 'web'],
        'dev': ['cli', 'web', 'desktop', 'exotic', 'isaa'],
    }

    # Parse features
    active_features = {}
    if features:
        if features.lower() in feature_presets:
            # Use preset
            active_features_list = feature_presets[features.lower()]
            print(f"📋 Using preset: {features}")
        else:
            # Parse comma-separated features
            active_features_list = [f.strip().lower() for f in features.split(',') if f.strip()]

        # Set active features
        for feat in active_features_list:
            if feat in feature_map:
                active_features[feat] = 1
    else:
        # Default: all core features enabled (cli is always on)
        active_features = {'cli': 1, 'web': 1}

    print(f"Building ToolBoxV2 Docker image: {image_name}")
    print(f"Dockerfile: {dockerfile_path}")
    print(f"Context: {project_root}")
    if active_features:
        print(f"Features: {', '.join(active_features.keys())}")
    print("")

    # Build Command
    build_cmd = [
        "docker", "build",
        "-f", dockerfile_path,
        "-t", image_name,
    ]

    # Add feature build args
    for feat, env_var in feature_map.items():
        value = active_features.get(feat, 0)
        build_cmd.extend(["--build-arg", f"{env_var}={value}"])

    if no_cache:
        build_cmd.append("--no-cache")

    build_cmd.append(project_root)

    print(f"Command: {' '.join(build_cmd)}")
    print("")
    print("─" * 60)

    # Build ausführen
    try:
        process = subprocess.Popen(
            build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding="utf-8", errors="ignore"
        )

        # Output live anzeigen
        for line in process.stdout:
            print(line.rstrip())

        process.wait()

        if process.returncode != 0:
            print("")
            print(f"❌ Build failed with exit code {process.returncode}")
            return False

        print("─" * 60)
        print("")
        print(f"✅ Build successful: {image_name}")
        print("")

        # Image-Info anzeigen
        print("Image info:")
        info_cmd = ["docker", "images", "toolboxv2", "--format",
                   "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"]
        subprocess.run(info_cmd,
            encoding="utf-8", errors="ignore")

        # Push zu Docker Hub (optional)
        if push:
            print("")
            print(f"📤 Pushing to Docker Hub...")

            # Tag für Docker Hub
            hub_image = f"toolboxv2/toolboxv2:{tag}"
            tag_cmd = ["docker", "tag", image_name, hub_image]
            subprocess.run(tag_cmd, check=True,
            encoding="utf-8", errors="ignore")

            # Push
            push_cmd = ["docker", "push", hub_image]
            subprocess.run(push_cmd, check=True,
            encoding="utf-8", errors="ignore")

            print(f"✅ Pushed: {hub_image}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
        print("   https://docs.docker.com/get-docker/")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Build cancelled.")
        return False


def main():
    """CLI Entry Point"""
    parser = argparse.ArgumentParser(
        prog="tb docker-image",
        description="Build ToolBoxV2 Docker images"
    )

    parser.add_argument(
        "--tag",
        default="latest",
        help="Image tag (default: latest)"
    )

    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to Docker Hub after build"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build without cache"
    )

    parser.add_argument(
        "--project-root",
        help="Project root directory (auto-detected by default)"
    )

    parser.add_argument(
        "--features",
        default=None,
        help="Comma-separated features (cli,web,desktop,exotic,isaa) or preset (all,production,dev). "
             "Default: cli,web"
    )

    args = parser.parse_args()

    success = build_docker_image(
        tag=args.tag,
        push=args.push,
        no_cache=args.no_cache,
        project_root=args.project_root,
        features=args.features
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
