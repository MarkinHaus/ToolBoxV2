import os
import subprocess
import sys
import json
from typing import List, Tuple, Optional
import argparse


def run_command(command: str, live: bool = True) -> Tuple[bool, Optional[str]]:
    print(f"Running command: {command}")
    if live:
        return os.system(command) == 0, None
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}", file=sys.stderr)
        print(e.stdout)
        print("==Error==")
        print(e.stderr, file=sys.stderr)
        return False, None


def update_conda(live=False):
    return run_command("conda update -n base -c defaults conda", live)


def run_script_in_conda_env(script_path: str, conda_env: str, script_args: List[str], live: bool = True,
                            python: bool = True, no_conda: bool = False) -> Tuple[bool, Optional[str]]:
    print(f"Running from {os.path.abspath(os.curdir)}")
    if python:
        command = f"conda run -n {conda_env} python {script_path} {' '.join(script_args)}"
    else:
        command = f"conda run -n {conda_env} {script_path} {' '.join(script_args)}"
    if no_conda:
        command = command.replace(f'conda run -n {conda_env} ', '')
    return run_command(command, live)


def create_conda_env(env_name: str, v='3.10') -> bool:
    command = f"conda create -n {env_name} python={v} -y"
    return run_command(command)[0]


def delete_conda_env(env_name: str) -> bool:
    command = f"conda env remove -n {env_name} -y"
    return run_command(command)[0]


def add_dependency(env_name: str, dependency: str, save: bool = False) -> bool:
    temp_env = f"{env_name}_temp"

    if save:
        command = f"conda install -n {env_name} {dependency} -y"
        success, _ = run_command(command)
        if success:
            update_dependency_registry(env_name, dependency)
        return success
    else:
        # Clone the existing environment
        clone_command = f"conda create --name {temp_env} --clone {env_name} -y"
        clone_success, _ = run_command(clone_command)

        if not clone_success:
            print(f"Failed to clone environment {env_name}")
            return False

        # Try to install the dependency in the cloned environment
        install_command = f"conda install -n {temp_env} {dependency} -y"
        install_success, _ = run_command(install_command)

        if install_success:
            # If installation was successful, update the original environment
            update_command = f"conda install -n {env_name} {dependency} -y"
            update_success, _ = run_command(update_command)

            if update_success:
                update_dependency_registry(env_name, dependency)

            # Clean up the temporary environment
            delete_conda_env(temp_env)

            return update_success
        else:
            # If installation failed, just remove the temporary environment
            delete_conda_env(temp_env)
            print(f"Failed to install {dependency} in {env_name}")
            return False


def update_dependency_registry(env_name: str, dependency: str):
    registry_file = f"{env_name}_dependency_registry.json"
    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    except FileNotFoundError:
        registry = []

    if dependency not in registry:
        registry.append(dependency)

    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)


def update_dependencies(env_name: str) -> bool:
    registry_file = f"{env_name}_dependency_registry.json"
    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    except FileNotFoundError:
        print(f"No dependency registry found for environment {env_name}")
        return False

    command = f"conda update -n {env_name} {' '.join(registry)} -y"
    return run_command(command)[0]


def create_env_registry(env_name: str) -> bool:
    """
    Create a JSON registry of all packages installed in the specified conda environment.

    Args:
    env_name (str): Name of the conda environment

    Returns:
    bool: True if registry creation was successful, False otherwise
    """
    # Get list of installed packages
    command = f"conda list -n {env_name} --json"
    success, output = run_command(command, live=False)

    if not success or output is None:
        print(f"Failed to get package list for environment {env_name}")
        return False

    try:
        # Parse the JSON output
        packages = json.loads(output)

        # Create a simplified registry with package names and versions
        registry = [{"name": pkg["name"], "version": pkg["version"]} for pkg in packages]

        # Write the registry to a JSON file
        registry_file = f"{env_name}_registry.json"
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)

        print(f"Registry created successfully: {registry_file}")
        return True

    except json.JSONDecodeError:
        print(f"Failed to parse package list for environment {env_name}")
        return False
    except IOError:
        print(f"Failed to write registry file for environment {env_name}")
        return False


def conda_runner_main():
    parser = argparse.ArgumentParser(description="Conda environment management script")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run script command
    run_parser = subparsers.add_parser("run", help="Run a script in a conda environment")
    run_parser.add_argument("script", help="Path to the script to run")
    run_parser.add_argument("conda_env", help="Name of the conda environment")
    run_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the script")

    # Create environment command
    create_parser = subparsers.add_parser("create", help="Create a new conda environment")
    create_parser.add_argument("env_name", help="Name of the environment to create")

    # Delete environment command
    delete_parser = subparsers.add_parser("delete", help="Delete a conda environment")
    delete_parser.add_argument("env_name", help="Name of the environment to delete")

    # Add dependency command
    add_parser = subparsers.add_parser("add", help="Add a dependency to a conda environment")
    add_parser.add_argument("env_name", help="Name of the environment")
    add_parser.add_argument("dependency", help="Dependency to add")
    add_parser.add_argument("--save", action="store_true", help="Save the dependency to the registry")

    # Update dependencies command
    update_parser = subparsers.add_parser("update", help="Update dependencies in a conda environment")
    update_parser.add_argument("env_name", help="Name of the environment to update")

    # Create registry command
    registry_parser = subparsers.add_parser("registry", help="Create a JSON registry for a conda environment")
    registry_parser.add_argument("env_name", help="Name of the environment to create registry for")

    args = parser.parse_args()

    if args.command == "run":
        run_script_in_conda_env(args.script, args.conda_env, args.args)
    elif args.command == "create":
        create_conda_env(args.env_name)
    elif args.command == "delete":
        delete_conda_env(args.env_name)
    elif args.command == "add":
        add_dependency(args.env_name, args.dependency, args.save)
    elif args.command == "update":
        if args.env_name == "conda":
            update_conda()
        else:
            update_dependencies(args.env_name)
    elif args.command == "registry":
        create_env_registry(args.env_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    conda_runner_main()
