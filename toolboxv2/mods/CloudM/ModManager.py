"""
CloudM - Advanced Module Manager
Production-ready module management system with multi-platform support
Version: 0.1.0
"""

import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from packaging import version as pv
from packaging.version import Version
from tqdm import tqdm

from toolboxv2 import App, Spinner, __version__, get_app
from toolboxv2.utils.extras.reqbuilder import generate_requirements
from toolboxv2.utils.system.state_system import find_highest_zip_version
from toolboxv2.utils.system.state_system import get_state_from_app
from toolboxv2.utils.system.types import RequestData, Result, ToolBoxInterfaces

# =================== Module Configuration ===================
Name = 'CloudM'
export = get_app(f"{Name}.Export").tb
version = "0.1.0"
default_export = export(mod_name=Name, version=version, interface=ToolBoxInterfaces.native, test=False)


# =================== Enums and Data Classes ===================
class Platform(Enum):
    """Supported platform types for module installation"""
    SERVER = "server"
    CLIENT = "client"
    DESKTOP = "desktop"
    MOBILE = "mobile"
    COMMON = "common"  # Files needed on all platforms
    ALL = "all"


class ModuleType(Enum):
    """Module types for different installation strategies"""
    PACKAGE = "package"  # Full module directory
    SINGLE = "single"  # Single file module
    HYBRID = "hybrid"  # Mix of both


class ConfigVersion(Enum):
    """Configuration file versions"""
    V1 = "1.0"
    V2 = "2.0"


# =================== Configuration Schemas ===================
TB_CONFIG_SCHEMA_V2 = {
    "version": str,
    "config_version": str,
    "module_name": str,
    "module_type": str,
    "description": str,
    "author": str,
    "license": str,
    "homepage": str,
    "dependencies_file": str,
    "platforms": {
        "server": {"files": list, "required": bool},
        "client": {"files": list, "required": bool},
        "desktop": {"files": list, "required": bool},
        "mobile": {"files": list, "required": bool},
        "common": {"files": list, "required": bool}
    },
    "metadata": dict
}

TB_CONFIG_SINGLE_SCHEMA = {
    "version": str,
    "config_version": str,
    "module_name": str,
    "module_type": "single",
    "file_path": str,
    "description": str,
    "author": str,
    "license": str,
    "specification": dict,
    "dependencies": list,
    "platforms": list,
    "metadata": dict
}


# =================== Utility Functions ===================
def increment_version(version_str: str, max_value: int = 99) -> str:
    """
    Increments a version number in the format "vX.Y.Z".

    Args:
        version_str: Current version number (e.g., "v0.0.1")
        max_value: Maximum number per position (default: 99)

    Returns:
        Incremented version number

    Raises:
        ValueError: If version format is invalid
    """
    if not version_str.startswith("v"):
        raise ValueError("Version must start with 'v' (e.g., 'v0.0.1')")

    version_core = version_str[1:]
    try:
        parsed_version = Version(version_core)
    except ValueError as e:
        raise ValueError(f"Invalid version number: {version_core}") from e

    parts = list(parsed_version.release)

    # Increment rightmost position
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] < max_value:
            parts[i] += 1
            break
        else:
            parts[i] = 0
    else:
        # All positions at max_value, add new position
        parts.insert(0, 1)

    return "v" + ".".join(map(str, parts))


def run_command(command: List[str], cwd: Optional[str] = None) -> str:
    """
    Executes a command and returns output.

    Args:
        command: Command and arguments as list
        cwd: Working directory for command execution

    Returns:
        Command stdout output

    Raises:
        subprocess.CalledProcessError: If command fails
    """
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        encoding='utf-8'
    )
    return result.stdout


def download_files(urls: List[str], directory: str, desc: str,
                   print_func: callable, filename: Optional[str] = None) -> str:
    """
    Downloads files from URLs with progress indication.

    Args:
        urls: List of URLs to download
        directory: Target directory
        desc: Progress bar description
        print_func: Function for printing messages
        filename: Optional filename (uses basename if None)

    Returns:
        Path to last downloaded file
    """
    for url in tqdm(urls, desc=desc):
        if filename is None:
            filename = os.path.basename(url)
        print_func(f"Downloading {filename}")
        print_func(f"{url} -> {directory}/{filename}")
        os.makedirs(directory, exist_ok=True)
        urllib.request.urlretrieve(url, f"{directory}/{filename}")
    return f"{directory}/{filename}"


def validate_config(config: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    """
    Validates configuration against schema.

    Args:
        config: Configuration dictionary to validate
        schema: Schema dictionary with expected types

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    def check_type(key: str, value: Any, expected_type: Any, path: str = ""):
        full_path = f"{path}.{key}" if path else key

        if isinstance(expected_type, dict):
            if not isinstance(value, dict):
                errors.append(f"{full_path}: Expected dict, got {type(value).__name__}")
                return
            for sub_key, sub_type in expected_type.items():
                if sub_key in value:
                    check_type(sub_key, value[sub_key], sub_type, full_path)
        elif expected_type == list:
            if not isinstance(value, list):
                errors.append(f"{full_path}: Expected list, got {type(value).__name__}")
        elif expected_type == dict:
            if not isinstance(value, dict):
                errors.append(f"{full_path}: Expected dict, got {type(value).__name__}")
        elif not isinstance(value, expected_type):
            errors.append(f"{full_path}: Expected {expected_type.__name__}, got {type(value).__name__}")

    for key, expected_type in schema.items():
        if key in config:
            check_type(key, config[key], expected_type)

    return len(errors) == 0, errors


# =================== Configuration Management ===================
def create_tb_config_v2(module_name: str, version: str, module_type: ModuleType = ModuleType.PACKAGE,
                        description: str = "", author: str = "", license: str = "MIT",
                        homepage: str = "", platforms: Optional[Dict] = None,
                        metadata: Optional[Dict] = None) -> Dict:
    """
    Creates a v2 tbConfig with platform-specific file management.

    Args:
        module_name: Name of the module
        version: Module version
        module_type: Type of module (package/single/hybrid)
        description: Module description
        author: Module author
        license: Module license
        homepage: Module homepage/repository
        platforms: Platform-specific file configurations
        metadata: Additional metadata

    Returns:
        Configuration dictionary
    """
    if platforms is None:
        platforms = {
            Platform.COMMON.value: {"files": ["*"], "required": True},
            Platform.SERVER.value: {"files": [], "required": False},
            Platform.CLIENT.value: {"files": [], "required": False},
            Platform.DESKTOP.value: {"files": [], "required": False},
            Platform.MOBILE.value: {"files": [], "required": False}
        }

    if metadata is None:
        metadata = {}

    return {
        "version": version,
        "config_version": ConfigVersion.V2.value,
        "module_name": module_name,
        "module_type": module_type.value,
        "description": description,
        "author": author,
        "license": license,
        "homepage": homepage,
        "dependencies_file": f"./mods/{module_name}/requirements.txt",
        "zip": f"RST${module_name}&{__version__}§{version}.zip",
        "platforms": platforms,
        "metadata": metadata
    }


def create_tb_config_single(module_name: str, version: str, file_path: str,
                            description: str = "", author: str = "",
                            specification: Optional[Dict] = None,
                            dependencies: Optional[List] = None,
                            platforms: Optional[List[Platform]] = None,
                            metadata: Optional[Dict] = None) -> Dict:
    """
    Creates configuration for single-file modules.

    Args:
        module_name: Name of the module
        version: Module version
        file_path: Path to the single file
        description: Module description
        author: Module author
        specification: File specifications (exports, functions, etc.)
        dependencies: List of dependencies
        platforms: List of supported platforms
        metadata: Additional metadata

    Returns:
        Configuration dictionary for single file module
    """
    if specification is None:
        specification = {
            "exports": [],
            "functions": [],
            "classes": [],
            "requires": []
        }

    if dependencies is None:
        dependencies = []

    if platforms is None:
        platforms = [Platform.ALL.value]
    else:
        platforms = [p.value if isinstance(p, Platform) else p for p in platforms]

    if metadata is None:
        metadata = {}

    return {
        "version": version,
        "config_version": ConfigVersion.V2.value,
        "module_name": module_name,
        "module_type": ModuleType.SINGLE.value,
        "file_path": file_path,
        "description": description,
        "author": author,
        "license": "MIT",
        "specification": specification,
        "dependencies": dependencies,
        "platforms": platforms,
        "metadata": metadata
    }


def load_and_validate_config(config_path: Path) -> Tuple[Optional[Dict], List[str]]:
    """
    Loads and validates a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (config_dict or None, list_of_errors)
    """
    if not config_path.exists():
        return None, [f"Config file not found: {config_path}"]

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return None, [f"Failed to parse YAML: {str(e)}"]

    # Determine schema based on module_type
    module_type = config.get("module_type", "package")

    if module_type == "single":
        schema = TB_CONFIG_SINGLE_SCHEMA
    else:
        schema = TB_CONFIG_SCHEMA_V2

    is_valid, errors = validate_config(config, schema)

    if not is_valid:
        return config, errors

    return config, []


def get_platform_files(config: Dict, platform: Platform) -> List[str]:
    """
    Extracts file list for specific platform from config.

    Args:
        config: Module configuration dictionary
        platform: Target platform

    Returns:
        List of files for the platform
    """
    platforms = config.get("platforms", {})

    # Get common files (required on all platforms)
    common_files = platforms.get(Platform.COMMON.value, {}).get("files", [])

    # Get platform-specific files
    platform_files = platforms.get(platform.value, {}).get("files", [])

    return common_files + platform_files


# =================== Module Packaging ===================
def create_and_pack_module(path: str, module_name: str = '', version: str = '-.-.-',
                           additional_dirs: Optional[Dict] = None,
                           yaml_data: Optional[Dict] = None,
                           platform_filter: Optional[Platform] = None) -> Optional[str]:
    """
    Creates and packs a module into a ZIP file with platform-specific support.

    Args:
        path: Path to module directory or file
        module_name: Name of the module
        version: Module version
        additional_dirs: Additional directories to include
        yaml_data: Configuration data override
        platform_filter: Optional platform filter for packaging

    Returns:
        Path to created ZIP file or None on failure
    """
    if additional_dirs is None:
        additional_dirs = {}
    if yaml_data is None:
        yaml_data = {}

    os.makedirs("./mods_sto/temp/", exist_ok=True)

    module_path = Path(path) / module_name

    if not module_path.exists():
        module_path = Path(f"{path}/{module_name}.py")

    temp_dir = Path(tempfile.mkdtemp(dir="./mods_sto/temp"))

    platform_suffix = f"_{platform_filter.value}" if platform_filter else ""
    zip_file_name = f"RST${module_name}&{__version__}§{version}{platform_suffix}.zip"
    zip_path = Path(f"./mods_sto/{zip_file_name}")

    if not module_path.exists():
        print(f"Module path does not exist: {module_path}")
        return None

    try:
        if module_path.is_dir():
            # Package module - create v2 config
            config_data = create_tb_config_v2(
                module_name=module_name,
                version=version,
                **yaml_data
            )

            config_path = module_path / "tbConfig.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            # Generate requirements
            req_path = module_path / "requirements.txt"
            generate_requirements(str(module_path), str(req_path))

            # Copy module directory
            shutil.copytree(module_path, temp_dir / module_path.name, dirs_exist_ok=True)

        else:
            # Single file module - create single config
            config_data = create_tb_config_single(
                module_name=module_name,
                version=version,
                file_path=str(module_path),
                **yaml_data
            )

            # Copy file
            shutil.copy2(module_path, temp_dir)

            # Create config
            config_path = temp_dir / f"{module_name}.yaml"
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            # Generate requirements
            req_path = temp_dir / "requirements.txt"
            generate_requirements(str(temp_dir), str(req_path))

        # Add additional directories
        for dir_name, dir_paths in additional_dirs.items():
            if isinstance(dir_paths, str):
                dir_paths = [dir_paths]

            for dir_path in dir_paths:
                dir_path = Path(dir_path)
                full_path = temp_dir / dir_name

                if dir_path.is_dir():
                    shutil.copytree(dir_path, full_path, dirs_exist_ok=True)
                elif dir_path.is_file():
                    full_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dir_path, full_path)
                else:
                    print(f"Path is neither directory nor file: {dir_path}")

        # Create ZIP file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)

        # Cleanup temporary directory
        shutil.rmtree(temp_dir)

        print(f"✓ Successfully created: {zip_path}")
        return str(zip_path)

    except Exception as e:
        print(f"✗ Error creating module package: {str(e)}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None


def unpack_and_move_module(zip_path: str, base_path: str = './mods',
                           module_name: str = '',
                           target_platform: Optional[Platform] = None) -> Optional[str]:
    """
    Unpacks a ZIP file and moves contents with platform filtering.

    Args:
        zip_path: Path to ZIP file
        base_path: Base installation path
        module_name: Module name (extracted from ZIP if not provided)
        target_platform: Optional platform filter for installation

    Returns:
        Name of installed module or None on failure
    """
    zip_path = Path(zip_path)
    base_path = Path(base_path)

    if not module_name:
        module_name = zip_path.name.split('$')[1].split('&')[0]

    module_path = base_path / module_name
    temp_base = Path('./mods_sto/temp')

    try:
        temp_base.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=str(temp_base)) as temp_dir:
            temp_dir = Path(temp_dir)

            with Spinner(f"Extracting {zip_path.name}"):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

            # Load configuration to check for platform-specific installation
            config_path = temp_dir / module_name / "tbConfig.yaml"
            if not config_path.exists():
                config_path = temp_dir / f"{module_name}.yaml"

            config, errors = load_and_validate_config(config_path)

            if errors:
                print(f"⚠ Configuration validation warnings: {', '.join(errors)}")

            # Handle module directory
            source_module = temp_dir / module_name

            if source_module.exists():
                with Spinner(f"Installing module to {module_path}"):
                    if module_path.exists():
                        shutil.rmtree(module_path)

                    # If platform filtering is enabled and config exists
                    if target_platform and config:
                        platform_files = get_platform_files(config, target_platform)

                        # Install only platform-specific files
                        module_path.mkdir(parents=True, exist_ok=True)

                        for pattern in platform_files:
                            if pattern == "*":
                                # Copy all files
                                shutil.copytree(source_module, module_path, dirs_exist_ok=True)
                                break
                            else:
                                # Copy specific files/patterns
                                for file in source_module.glob(pattern):
                                    if file.is_file():
                                        shutil.copy2(file, module_path)
                                    elif file.is_dir():
                                        shutil.copytree(file, module_path / file.name, dirs_exist_ok=True)
                    else:
                        # Install all files
                        shutil.copytree(source_module, module_path, dirs_exist_ok=True)

            # Handle additional files in root
            with Spinner("Installing additional files"):
                for item in temp_dir.iterdir():
                    if item.name == module_name or item.name.endswith('.yaml'):
                        continue

                    target = Path('./') / item.name
                    if item.is_dir():
                        if target.exists():
                            shutil.rmtree(target)
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)

            print(f"✓ Successfully installed/updated module: {module_name}")
            return module_name

    except Exception as e:
        print(f"✗ Error during installation: {str(e)}")
        if module_path.exists():
            shutil.rmtree(module_path)
        raise


def uninstall_module(path: str, module_name: str = '', version: str = '-.-.-',
                     additional_dirs: Optional[Dict] = None,
                     yaml_data: Optional[Dict] = None) -> bool:
    """
    Uninstalls a module by removing its directory and ZIP file.

    Args:
        path: Base path containing module
        module_name: Name of module to uninstall
        version: Module version
        additional_dirs: Additional directories to remove
        yaml_data: Configuration data

    Returns:
        True if successful, False otherwise
    """
    if additional_dirs is None:
        additional_dirs = {}

    base_path = Path(path).parent
    module_path = base_path / module_name
    zip_path = Path(f"./mods_sto/RST${module_name}&{__version__}§{version}.zip")

    if not module_path.exists():
        print(f"⚠ Module {module_name} already uninstalled")
        return False

    try:
        # Remove module directory
        shutil.rmtree(module_path)
        print(f"✓ Removed module directory: {module_path}")

        # Remove additional directories
        for _dir_name, dir_paths in additional_dirs.items():
            if isinstance(dir_paths, str):
                dir_paths = [dir_paths]

            for dir_path in dir_paths:
                dir_path = Path(dir_path)
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    print(f"✓ Removed additional path: {dir_path}")

        # Remove ZIP file
        if zip_path.exists():
            zip_path.unlink()
            print(f"✓ Removed ZIP file: {zip_path}")

        return True

    except Exception as e:
        print(f"✗ Error during uninstallation: {str(e)}")
        return False


# =================== Dependency Management ===================
def install_dependencies(yaml_file: str, auto: bool = False) -> bool:
    """
    Installs dependencies from tbConfig.yaml.

    Args:
        yaml_file: Path to configuration file
        auto: Automatically install without confirmation

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        dependencies_file = config.get("dependencies_file")

        if not dependencies_file:
            print("⚠ No dependencies file specified")
            return True

        dependencies_path = Path(dependencies_file)

        if not dependencies_path.exists():
            print(f"⚠ Dependencies file not found: {dependencies_path}")
            return False

        print(f"Installing dependencies from: {dependencies_path}")

        if not auto:
            response = input("Continue with installation? (y/n): ")
            if response.lower() != 'y':
                print("Installation cancelled")
                return False

        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(dependencies_path)],
            check=True
        )

        print("✓ Dependencies installed successfully")
        return True

    except Exception as e:
        print(f"✗ Error installing dependencies: {str(e)}")
        return False


def uninstall_dependencies(yaml_file: str) -> bool:
    """
    Uninstalls dependencies from tbConfig.yaml.

    Args:
        yaml_file: Path to configuration file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        dependencies = config.get("dependencies", [])

        if not dependencies:
            print("⚠ No dependencies to uninstall")
            return True

        for dependency in dependencies:
            print(f"Uninstalling: {dependency}")
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', dependency],
                check=True
            )

        print("✓ Dependencies uninstalled successfully")
        return True

    except Exception as e:
        print(f"✗ Error uninstalling dependencies: {str(e)}")
        return False


def install_from_zip(app: App, zip_name: str, no_dep: bool = True,
                     auto_dep: bool = False,
                     target_platform: Optional[Platform] = None) -> bool:
    """
    Installs a module from ZIP file with dependency management.

    Args:
        app: Application instance
        zip_name: Name of ZIP file
        no_dep: Skip dependency installation
        auto_dep: Automatically install dependencies
        target_platform: Optional platform filter

    Returns:
        True if successful, False otherwise
    """
    zip_path = Path(app.start_dir) / "mods_sto" / zip_name

    if not zip_path.exists():
        print(f"✗ ZIP file not found: {zip_path}")
        return False

    try:
        with Spinner(f"Unpacking {zip_path.name[-40:]}"):
            module_name = unpack_and_move_module(
                str(zip_path),
                f"{app.start_dir}/mods",
                target_platform=target_platform
            )

        if not module_name:
            return False

        # Install dependencies if requested
        if not no_dep:
            config_path = Path(app.start_dir) / "mods" / module_name / "tbConfig.yaml"

            if config_path.exists():
                with Spinner(f"Installing dependencies for {module_name}"):
                    install_dependencies(str(config_path), auto_dep)

        return True

    except Exception as e:
        print(f"✗ Installation failed: {str(e)}")
        return False


# =================== API Endpoints ===================
@export(mod_name=Name, api=True, interface=ToolBoxInterfaces.remote, test=False)
def list_modules(app: App = None) -> Result:
    """
    Lists all available modules.

    Returns:
        Result with module list
    """
    if app is None:
        app = get_app("cm.list_modules")

    modules = app.get_all_mods()
    return Result.ok({"modules": modules, "count": len(modules)})


@export(mod_name=Name, name="upload_mod", api=True, api_methods=['POST'], test=False)
async def upload_mod(app: App, request: RequestData,
                     form_data: Optional[Dict[str, Any]] = None) -> Result:
    """
    Uploads a module ZIP file to the server.

    Args:
        app: Application instance
        request: Request data
        form_data: Form data containing file

    Returns:
        Result with upload status
    """
    if not isinstance(form_data, dict):
        return Result.default_user_error("No form data provided")

    if form_data is None or 'files' not in form_data:
        return Result.default_user_error("No file provided")

    try:
        uploaded_file = form_data.get('files')[0]
        file_name = uploaded_file.filename
        file_bytes = uploaded_file.file.read()

        # Security validation
        if not file_name.endswith('.zip'):
            return Result.default_user_error("Only ZIP files are allowed")

        if not file_name.startswith('RST$'):
            return Result.default_user_error("Invalid module ZIP format")

        # Save file
        save_path = Path(app.start_dir) / "mods_sto" / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(file_bytes)

        # Extract module info
        module_name = file_name.split('$')[1].split('&')[0]
        module_version = file_name.split('§')[1].replace('.zip', '')

        return Result.ok({
            "message": f"Successfully uploaded {file_name}",
            "module": module_name,
            "version": module_version,
            "size": len(file_bytes)
        })

    except Exception as e:
        return Result.default_internal_error(f"Upload failed: {str(e)}")


@export(mod_name=Name, name="download_mod", api=True, api_methods=['GET'])
async def download_mod(app: App, module_name: str,
                       platform: Optional[str] = None) -> Result:
    """
    Downloads a module ZIP file.

    Args:
        app: Application instance
        module_name: Name of module to download
        platform: Optional platform filter

    Returns:
        Binary result with ZIP file
    """
    try:
        zip_path_str = find_highest_zip_version(module_name)

        if not zip_path_str:
            return Result.default_user_error(
                f"Module '{module_name}' not found",
                exec_code=404
            )

        zip_path = Path(zip_path_str)

        if not zip_path.exists():
            return Result.default_user_error(
                f"Module file not found: {zip_path}",
                exec_code=404
            )

        return Result.binary(
            data=zip_path.read_bytes(),
            content_type="application/zip",
            download_name=zip_path.name
        )

    except Exception as e:
        return Result.default_internal_error(f"Download failed: {str(e)}")


@export(mod_name=Name, name="getModVersion", api=True, api_methods=['GET'])
async def get_mod_version(app: App, module_name: str) -> Result:
    """
    Gets the latest version of a module.

    Args:
        app: Application instance
        module_name: Name of module

    Returns:
        Result with version string
    """
    try:
        version_str = find_highest_zip_version(module_name, version_only=True)

        if version_str:
            return Result.text(version_str)

        return Result.default_user_error(
            f"No build found for module '{module_name}'",
            exec_code=404
        )

    except Exception as e:
        return Result.default_internal_error(f"Failed to get version: {str(e)}")


@export(mod_name=Name, name="getModInfo", api=True, api_methods=['GET'])
async def get_mod_info(app: App, module_name: str) -> Result:
    """
    Gets detailed information about a module.

    Args:
        app: Application instance
        module_name: Name of module

    Returns:
        Result with module information
    """
    try:
        zip_path = find_highest_zip_version(module_name)

        if not zip_path:
            return Result.default_user_error(
                f"Module '{module_name}' not found",
                exec_code=404
            )

        # Extract and read config
        with zipfile.ZipFile(zip_path, 'r') as zf:
            config_files = [f for f in zf.namelist() if f.endswith('tbConfig.yaml') or f.endswith('.yaml')]

            if not config_files:
                return Result.default_user_error("No configuration file found in module")

            config_content = zf.read(config_files[0])
            config = yaml.safe_load(config_content)

        return Result.ok(config)

    except Exception as e:
        return Result.default_internal_error(f"Failed to get module info: {str(e)}")


# =================== CLI Operations ===================
@export(mod_name=Name, name="make_install", test=False)
async def make_installer(app: Optional[App], module_name: str,
                         base: str = "./mods", upload: Optional[bool] = None,
                         platform: Optional[Platform] = None) -> Result:
    """
    Creates an installer package for a module.

    Args:
        app: Application instance
        module_name: Name of module to package
        base: Base directory containing modules
        upload: Whether to upload after creation
        platform: Optional platform filter

    Returns:
        Result with package path or upload status
    """
    if app is None:
        app = get_app(f"{Name}.make_install")

    if module_name not in app.get_all_mods():
        return Result.default_user_error(f"Module '{module_name}' not found")

    try:
        with Spinner("Testing module load"):
            app.save_load(module_name)

        mod = app.get_mod(module_name)
        version_ = getattr(mod, 'version', version)

        with Spinner("Creating and packing module"):
            zip_path = create_and_pack_module(
                base, module_name, version_,
                platform_filter=platform
            )

        if not zip_path:
            return Result.default_internal_error("Failed to create package")

        # Upload if requested
        if upload or (upload is None and 'y' in input("Upload ZIP file? (y/n): ").lower()):
            with Spinner("Uploading file"):
                res = await app.session.upload_file(zip_path, '/installer/upload-file/')

            if isinstance(res, dict):
                if res.get('res', '').startswith('Successfully uploaded'):
                    return Result.ok({
                        "message": "Module packaged and uploaded",
                        "zip_path": zip_path,
                        "upload_response": res
                    })
                return Result.default_user_error(res)

        return Result.ok({
            "message": "Module packaged successfully",
            "zip_path": zip_path
        })

    except Exception as e:
        return Result.default_internal_error(f"Installation creation failed: {str(e)}")


@export(mod_name=Name, name="uninstall", test=False)
def uninstaller(app: Optional[App], module_name: str) -> Result:
    """
    Uninstalls a module.

    Args:
        app: Application instance
        module_name: Name of module to uninstall

    Returns:
        Result with uninstallation status
    """
    if app is None:
        app = get_app(f"{Name}.uninstall")

    if module_name not in app.get_all_mods():
        return Result.default_user_error(f"Module '{module_name}' not found")

    try:
        mod = app.get_mod(module_name)
        version_ = getattr(mod, 'version', version)

        confirm = input(f"Uninstall module '{module_name}' v{version_}? (y/n): ")
        if 'y' not in confirm.lower():
            return Result.ok("Uninstallation cancelled")

        success = uninstall_module(f"./mods/{module_name}", module_name, version_)

        if success:
            return Result.ok(f"Module '{module_name}' uninstalled successfully")
        else:
            return Result.default_internal_error("Uninstallation failed")

    except Exception as e:
        return Result.default_internal_error(f"Uninstallation failed: {str(e)}")


@export(mod_name=Name, name="upload", test=False)
async def upload(app: Optional[App], module_name: str) -> Result:
    """
    Uploads an existing module package to the server.

    Args:
        app: Application instance
        module_name: Name of module to upload

    Returns:
        Result with upload status
    """
    if app is None:
        app = get_app(f"{Name}.upload")

    try:
        zip_path = find_highest_zip_version(module_name)

        if not zip_path:
            return Result.default_user_error(f"No package found for module '{module_name}'")

        confirm = input(f"Upload ZIP file {zip_path}? (y/n): ")
        if 'y' not in confirm.lower():
            return Result.ok("Upload cancelled")

        res = await app.session.upload_file(zip_path, f'/api/{Name}/upload_mod')

        return Result.ok({
            "message": "Upload completed",
            "response": res
        })

    except Exception as e:
        return Result.default_internal_error(f"Upload failed: {str(e)}")


@export(mod_name=Name, name="install", test=False)
async def installer(app: Optional[App], module_name: str,
                    build_state: bool = True,
                    platform: Optional[Platform] = None) -> Result:
    """
    Installs or updates a module from the server.

    Args:
        app: Application instance
        module_name: Name of module to install
        build_state: Whether to rebuild state after installation
        platform: Optional platform filter for installation

    Returns:
        Result with installation status
    """
    if app is None:
        app = get_app(f"{Name}.install")

    if not app.session.valid and not await app.session.login():
        return Result.default_user_error("Please login with CloudM")

    try:
        # Get remote version
        response = await app.session.fetch(
            f"/api/{Name}/getModVersion?module_name={module_name}",
            method="GET"
        )
        remote_version = await response.text()
        remote_version = None if remote_version == "None" else remote_version.strip('"')

        # Get local version
        local_version = find_highest_zip_version(module_name, version_only=True)

        if not local_version and not remote_version:
            return Result.default_user_error(f"Module '{module_name}' not found (404)")

        # Compare versions
        local_ver = pv.parse(local_version) if local_version else pv.parse("0.0.0")
        remote_ver = pv.parse(remote_version) if remote_version else pv.parse("0.0.0")

        app.print(f"Module versions - Local: {local_ver}, Remote: {remote_ver}")

        if remote_ver > local_ver:
            download_path = Path(app.start_dir) / 'mods_sto'
            download_url = f"/api/{Name}/download_mod?module_name={module_name}"

            if platform:
                download_url += f"&platform={platform.value}"

            app.print(f"Downloading from {app.session.base}{download_url}")

            if not await app.session.download_file(download_url, str(download_path)):
                app.print("⚠ Automatic download failed")
                manual = input("Download manually and place in mods_sto folder. Done? (y/n): ")
                if 'y' not in manual.lower():
                    return Result.default_user_error("Installation cancelled")

            zip_name = f"RST${module_name}&{app.version}§{remote_version}.zip"

            with Spinner("Installing from ZIP"):
                success = install_from_zip(app, zip_name, target_platform=platform)

            if not success:
                return Result.default_internal_error("Installation failed")

            if build_state:
                with Spinner("Rebuilding state"):
                    get_state_from_app(app)

            return Result.ok({
                "message": f"Module '{module_name}' installed successfully",
                "version": remote_version
            })

        app.print("✓ Module is already up to date")
        return Result.ok("Module is up to date")

    except Exception as e:
        return Result.default_internal_error(f"Installation failed: {str(e)}")


@export(mod_name=Name, name="update_all", test=False)
async def update_all_mods(app: Optional[App]) -> Result:
    """
    Updates all installed modules.

    Args:
        app: Application instance

    Returns:
        Result with update summary
    """
    if app is None:
        app = get_app(f"{Name}.update_all")

    all_mods = app.get_all_mods()
    results = {"updated": [], "failed": [], "up_to_date": []}

    async def check_and_update(mod_name: str):
        try:
            # Get remote version
            response = await app.session.fetch(
                f"/api/{Name}/getModVersion?module_name={mod_name}"
            )
            remote_version = await response.text()
            remote_version = remote_version.strip('"') if remote_version != "None" else None

            if not remote_version:
                results["failed"].append({"module": mod_name, "reason": "Version not found"})
                return

            local_mod = app.get_mod(mod_name)
            if not local_mod:
                results["failed"].append({"module": mod_name, "reason": "Local module not found"})
                return

            local_version = getattr(local_mod, 'version', '0.0.0')

            if pv.parse(remote_version) > pv.parse(local_version):
                result = await installer(app, mod_name, build_state=False)
                if result.is_error:
                    results["failed"].append({"module": mod_name, "reason": str(result)})
                else:
                    results["updated"].append({"module": mod_name, "version": remote_version})
            else:
                results["up_to_date"].append(mod_name)

        except Exception as e:
            results["failed"].append({"module": mod_name, "reason": str(e)})

    # Run updates in parallel
    await asyncio.gather(*[check_and_update(mod) for mod in all_mods])

    # Rebuild state once at the end
    with Spinner("Rebuilding application state"):
        get_state_from_app(app)

    return Result.ok({
        "summary": {
            "total": len(all_mods),
            "updated": len(results["updated"]),
            "up_to_date": len(results["up_to_date"]),
            "failed": len(results["failed"])
        },
        "details": results
    })


@export(mod_name=Name, name="build_all", test=False)
async def build_all_mods(app: Optional[App], base: str = "mods",
                         upload: bool = True) -> Result:
    """
    Builds installer packages for all modules.

    Args:
        app: Application instance
        base: Base directory containing modules
        upload: Whether to upload packages after building

    Returns:
        Result with build summary
    """
    if app is None:
        app = get_app(f"{Name}.build_all")

    all_mods = app.get_all_mods()
    results = {"success": [], "failed": []}

    async def build_pipeline(mod_name: str):
        try:
            result = await make_installer(app, mod_name, os.path.join('.', base), upload)
            if result.is_error:
                results["failed"].append({"module": mod_name, "reason": str(result)})
            else:
                results["success"].append(mod_name)
            return result
        except Exception as e:
            results["failed"].append({"module": mod_name, "reason": str(e)})
            return Result.default_internal_error(str(e))

    # Build all modules
    build_results = [await build_pipeline(mod) for mod in all_mods]

    return Result.ok({
        "summary": {
            "total": len(all_mods),
            "success": len(results["success"]),
            "failed": len(results["failed"])
        },
        "details": results
    })


# =================== Interactive CLI Manager ===================
import asyncio
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass

from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window, FormattedTextControl
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import input_dialog, yes_no_dialog, message_dialog, radiolist_dialog
from prompt_toolkit import print_formatted_text
import yaml

# =================== Styles ===================

MODERN_STYLE = Style.from_dict({
    # Menu styles
    'menu-border': '#00d9ff bold',
    'menu-title': '#00d9ff bold',
    'menu-item': '#e0e0e0',
    'menu-item-selected': '#000000 bg:#00d9ff bold',
    'menu-key': '#ff79c6 bold',
    'menu-separator': '#6272a4',
    'menu-category': '#bd93f9 bold',

    # Status styles
    'success': '#50fa7b bold',
    'error': '#ff5555 bold',
    'warning': '#ffb86c bold',
    'info': '#8be9fd',

    # UI elements
    'border': '#6272a4',
    'header': '#ff79c6 bold',
    'footer': '#6272a4 italic',
    'prompt': '#00d9ff bold',

    # Dialog
    'dialog': 'bg:#282a36',
    'dialog.body': 'bg:#282a36 #f8f8f2',
    'dialog frame.label': '#ff79c6 bold',
    'button': 'bg:#44475a #f8f8f2',
    'button.focused': 'bg:#00d9ff #000000 bold',
})


# =================== Menu Items ===================

@dataclass
class MenuItem:
    """Menu item with action."""
    key: str
    label: str
    action: Callable
    category: str = ""
    icon: str = "•"
    description: str = ""


@dataclass
class MenuCategory:
    """Menu category."""
    name: str
    icon: str
    items: List[MenuItem]


# =================== Helper Functions ===================

def format_status(status: str, message: str) -> HTML:
    """Format status message with icon."""
    icons = {
        'success': '✓',
        'error': '✗',
        'warning': '⚠',
        'info': 'ℹ'
    }
    icon = icons.get(status, '•')
    return HTML(f'<{status}>{icon} {message}</{status}>')


async def show_message(title: str, text: str, style: str = "info"):
    """Show a message dialog."""
    icons = {
        'success': '✓',
        'error': '✗',
        'warning': '⚠',
        'info': 'ℹ'
    }
    icon = icons.get(style, 'ℹ')

    await message_dialog(
        title=f"{icon} {title}",
        text=text,
        style=MODERN_STYLE
    ).run_async()


async def show_confirm(title: str, text: str) -> bool:
    """Show confirmation dialog."""
    result = await yes_no_dialog(
        title=f"⚠ {title}",
        text=text,
        style=MODERN_STYLE
    ).run_async()
    return result if result is not None else False


async def show_input(title: str, label: str, default: str = "") -> Optional[str]:
    """Show input dialog."""
    result = await input_dialog(
        title=f"✎ {title}",
        text=label,
        default=default,
        style=MODERN_STYLE
    ).run_async()
    return result


async def show_choice(title: str, text: str, choices: List[tuple]) -> Optional[Any]:
    """Show radio list dialog."""
    result = await radiolist_dialog(
        title=f"◉ {title}",
        text=text,
        values=choices,
        style=MODERN_STYLE
    ).run_async()
    return result


async def show_progress(title: str, message: str):
    """Show a simple progress message."""
    print_formatted_text(HTML(f'\n<info>⟳ {title}</info>'))
    print_formatted_text(HTML(f'<menu-item>  {message}</menu-item>\n'))


# =================== Menu Manager ===================

class ModernMenuManager:
    """Modern menu manager with arrow key navigation."""

    def __init__(self, app_instance: Optional[Any] = None):
        self.app_instance = app_instance
        self.selected_index = 0
        self.categories: List[MenuCategory] = []
        self.flat_items: List[MenuItem] = []
        self.running = True

    def add_category(self, category: MenuCategory):
        """Add a menu category."""
        self.categories.append(category)
        self.flat_items.extend(category.items)

    def get_menu_text(self) -> List[tuple]:
        """Generate formatted menu text."""
        lines = []

        # Header
        lines.append(('class:menu-border', '╔' + '═' * 68 + '╗\n'))
        lines.append(('class:menu-border', '║'))
        lines.append(('class:menu-title', '  🌩️  CloudM - Module Manager'.center(68)))
        lines.append(('class:menu-border', '║\n'))
        lines.append(('class:menu-border', '╠' + '═' * 68 + '╣\n'))

        # Menu items by category
        current_flat_index = 0

        for cat_idx, category in enumerate(self.categories):
            # Category header
            if cat_idx > 0:
                lines.append(('class:menu-border', '║' + '─' * 68 + '║\n'))

            lines.append(('class:menu-border', '║ '))
            lines.append(('class:menu-category', f'{category.icon} {category.name}'))
            lines.append(('', ' ' * (67 - len(category.name) - len(category.icon)- (2 if len(category.icon) == 1 else 1))))
            lines.append(('class:menu-border', '║\n'))

            # Category items
            for item in category.items:
                is_selected = current_flat_index == self.selected_index

                lines.append(('class:menu-border', '║ '))

                if is_selected:
                    lines.append(('class:menu-item-selected', f' ▶ '))
                else:
                    lines.append(('', '   '))

                # Key
                if is_selected:
                    lines.append(('class:menu-item-selected', f'{item.key:>3}'))
                else:
                    lines.append(('class:menu-key', f'{item.key:>3}'))

                # Label

                if is_selected:
                    lines.append(('class:menu-item-selected', f' {item.icon} {item.label}'))
                    remaining = 60 - len(item.label) - len(item.icon) - (2 if len(item.icon) == 1 else 1)
                    lines.append(('class:menu-item-selected', ' ' * remaining))
                else:
                    lines.append(('class:menu-item', f' {item.icon} {item.label}'))
                    remaining = 60 - len(item.label) - len(item.icon) - (2 if len(item.icon) == 1 else 1)
                    lines.append(('', ' ' * remaining))

                lines.append(('class:menu-border', '║\n'))
                current_flat_index += 1

        # Footer
        lines.append(('class:menu-border', '╚' + '═' * 68 + '╝\n'))
        lines.append(('class:footer', '\n  ↑↓ or w/s: Navigate  │  Enter: Select  │  q: Quit\n'))

        return lines

    def move_up(self):
        """Move selection up."""
        if self.selected_index > 0:
            self.selected_index -= 1

    def move_down(self):
        """Move selection down."""
        if self.selected_index < len(self.flat_items) - 1:
            self.selected_index += 1

    def get_selected_item(self) -> Optional[MenuItem]:
        """Get currently selected menu item."""
        if 0 <= self.selected_index < len(self.flat_items):
            return self.flat_items[self.selected_index]
        return None

    async def run(self):
        """Run the menu manager."""
        # Build menu structure
        self._build_menu()

        while self.running:
            # Clear screen
            print('\033[2J\033[H')

            # Display menu
            menu_text = self.get_menu_text()
            print_formatted_text(FormattedText(menu_text), style=MODERN_STYLE)

            # Key bindings
            kb = KeyBindings()

            @kb.add('up')
            @kb.add('w')
            def move_up_handler(event):
                self.move_up()
                event.app.exit()

            @kb.add('down')
            @kb.add('s')
            def move_down_handler(event):
                self.move_down()
                event.app.exit()

            @kb.add('enter')
            def select_handler(event):
                event.app.exit(result='select')

            @kb.add('q')
            @kb.add('escape')
            def quit_handler(event):
                event.app.exit(result='quit')

            # Wait for input
            dummy_app = Application(
                layout=Layout(Window(FormattedTextControl(''))),
                key_bindings=kb,
                full_screen=False
            )

            result = await dummy_app.run_async()

            if result == 'quit':
                if await show_confirm('Exit Manager', 'Are you sure you want to exit?'):
                    self.running = False
                    break
            elif result == 'select':
                selected = self.get_selected_item()
                if selected:
                    try:
                        await selected.action()
                    except KeyboardInterrupt:
                        continue
                    except Exception as e:
                        await show_message('Error', f'An error occurred:\n\n{str(e)}', 'error')

    def _build_menu(self):
        """Build menu structure with all operations."""

        # =================== MODULE OPERATIONS ===================
        module_ops = MenuCategory(
            name="MODULE OPERATIONS",
            icon="📦",
            items=[
                MenuItem("1", "List all modules", self._list_modules, icon="📋"),
                MenuItem("2", "Install/Update module", self._install_module, icon="📥"),
                MenuItem("3", "Uninstall module", self._uninstall_module, icon="🗑️"),
                MenuItem("4", "Build installer", self._build_installer, icon="🔨"),
                MenuItem("5", "Upload module", self._upload_module, icon="☁️"),
                MenuItem("6", "Update ALL modules", self._update_all, icon="🔄"),
                MenuItem("7", "Build ALL modules", self._build_all, icon="🏗️"),
            ]
        )

        # =================== CONFIGURATION ===================
        config_ops = MenuCategory(
            name="CONFIGURATION",
            icon="⚙️",
            items=[
                MenuItem("8", "View module info", self._view_info, icon="ℹ️"),
                MenuItem("9", "Validate config", self._validate_config, icon="✓ "),
                MenuItem("10", "Create new config", self._create_config, icon="✎ "),
                MenuItem("11", "Generate ALL configs", self._generate_all_configs, icon="⚡"),
                MenuItem("12", "Generate config for module", self._generate_single_config, icon="⚙️"),
            ]
        )

        # =================== PLATFORM & TEMPLATES ===================
        platform_ops = MenuCategory(
            name="PLATFORM & TEMPLATES",
            icon="🌐",
            items=[
                MenuItem("13", "Build platform installer", self._build_platform, icon="🖥️"),
                MenuItem("14", "Install for platform", self._install_platform, icon="💾"),
                MenuItem("15", "Create from template", self._create_from_template, icon="🎨"),
                MenuItem("16", "List templates", self._list_templates, icon="📚"),
            ]
        )

        self.add_category(module_ops)
        self.add_category(config_ops)
        self.add_category(platform_ops)

    # =================== Action Handlers ===================

    async def _list_modules(self):
        """List all modules."""
        await show_progress("Loading Modules", "Scanning module directory...")

        mods = self.app_instance.get_all_mods()

        if not mods:
            await show_message("No Modules", "No modules found in the directory.", "warning")
            return

        # Build module list
        lines = [f"\n{'#':<4} {'Status':<8} {'Module Name':<35} {'Version':<10}"]
        lines.append('─' * 75)

        for i, mod in enumerate(mods, 1):
            mod_obj = self.app_instance.get_mod(mod)
            ver = getattr(mod_obj, 'version', '?.?.?') if mod_obj else '?.?.?'

            # Check config
            config_path = Path('./mods') / mod / 'tbConfig.yaml'
            single_config = Path('./mods') / f'{mod}.yaml'
            status = "✓ OK" if (config_path.exists() or single_config.exists()) else "✗ No cfg"

            lines.append(f"{i:<4} {status:<8} {mod:<35} {ver:<10}")

        lines.append('─' * 75)
        lines.append(f"\nTotal: {len(mods)} modules")

        await show_message(f"📦 Available Modules ({len(mods)})", '\n'.join(lines), "info")

    async def _install_module(self):
        """Install or update a module."""
        module_name = await show_input("Install Module", "Enter module name:")

        if not module_name:
            return

        await show_progress("Installing", f"Installing module '{module_name}'...")

        result = await installer(self.app_instance, module_name)

        if result.is_error:
            await show_message("Installation Failed", f"Error: {result}", "error")
        else:
            await show_message("Success", f"Module '{module_name}' installed successfully!", "success")

    async def _uninstall_module(self):
        """Uninstall a module."""
        module_name = await show_input("Uninstall Module", "Enter module name:")

        if not module_name:
            return

        if not await show_confirm("Confirm Uninstall", f"Really uninstall '{module_name}'?"):
            return

        await show_progress("Uninstalling", f"Removing module '{module_name}'...")

        result = uninstaller(self.app_instance, module_name)

        if result.is_error:
            await show_message("Uninstall Failed", f"Error: {result}", "error")
        else:
            await show_message("Success", f"Module '{module_name}' uninstalled successfully!", "success")

    async def _build_installer(self):
        """Build module installer."""
        module_name = await show_input("Build Installer", "Enter module name:")

        if not module_name:
            return

        upload = await show_confirm("Upload", "Upload after building?")

        await show_progress("Building", f"Building installer for '{module_name}'...")

        result = await make_installer(self.app_instance, module_name, upload=upload)

        if result.is_error:
            await show_message("Build Failed", f"Error: {result}", "error")
        else:
            msg = f"Installer built successfully!"
            if upload:
                msg += "\n\nModule uploaded to cloud!"
            await show_message("Success", msg, "success")

    async def _upload_module(self):
        """Upload module to cloud."""
        module_name = await show_input("Upload Module", "Enter module name:")

        if not module_name:
            return

        await show_progress("Uploading", f"Uploading '{module_name}' to cloud...")

        result = await upload(self.app_instance, module_name)

        if result.is_error:
            await show_message("Upload Failed", f"Error: {result}", "error")
        else:
            await show_message("Success", f"Module '{module_name}' uploaded successfully!", "success")

    async def _update_all(self):
        """Update all modules."""
        if not await show_confirm(
            "Batch Update",
            "This will update ALL modules.\nThis may take several minutes.\n\nContinue?"
        ):
            return

        await show_progress("Batch Update", "Updating all modules... Please wait.")

        result = await update_all_mods(self.app_instance)

        if result.is_error:
            await show_message("Update Completed", f"Completed with errors:\n\n{result}", "warning")
        else:
            await show_message("Success", "All modules updated successfully!", "success")

    async def _build_all(self):
        """Build all modules."""
        upload = await show_confirm("Upload", "Upload after building?")

        if not await show_confirm(
            "Batch Build",
            "This will build ALL modules.\nThis may take several minutes.\n\nContinue?"
        ):
            return

        await show_progress("Batch Build", "Building all modules... Please wait.")

        result = await build_all_mods(self.app_instance, upload=upload)

        if result.is_error:
            await show_message("Build Completed", f"Completed with errors:\n\n{result}", "warning")
        else:
            msg = "All modules built successfully!"
            if upload:
                msg += "\n\nAll modules uploaded to cloud!"
            await show_message("Success", msg, "success")

    async def _view_info(self):
        """View module information."""
        module_name = await show_input("Module Info", "Enter module name:")

        if not module_name:
            return

        await show_progress("Loading", f"Fetching info for '{module_name}'...")

        result = await get_mod_info(self.app_instance, module_name)

        if result.is_error:
            await show_message("Error", f"Could not get module info:\n\n{result}", "error")
        else:
            info_text = yaml.dump(result.get(), default_flow_style=False, allow_unicode=True)
            await show_message(f"Module Info: {module_name}", info_text, "info")

    async def _validate_config(self):
        """Validate module configuration."""
        module_name = await show_input("Validate Config", "Enter module name:")

        if not module_name:
            return

        config_path = Path('./mods') / module_name / 'tbConfig.yaml'
        if not config_path.exists():
            config_path = Path('./mods') / f'{module_name}.yaml'

        if not config_path.exists():
            await show_message("Error", f"Config file not found for '{module_name}'", "error")
            return

        await show_progress("Validating", f"Checking configuration...")

        config, errors = load_and_validate_config(config_path)

        if errors:
            error_text = '\n'.join([f"  {i}. {err}" for i, err in enumerate(errors, 1)])
            await show_message("Validation Failed", f"Errors found:\n\n{error_text}", "error")
        else:
            await show_message("Success", f"Configuration is valid! ✓", "success")

    async def _create_config(self):
        """Create new module configuration."""
        module_name = await show_input("Create Config", "Module name:")
        if not module_name:
            return

        version = await show_input("Version", "Version:", "0.0.1")
        description = await show_input("Description", "Description (optional):")
        author = await show_input("Author", "Author (optional):")

        # Module type selection
        module_type_choice = await show_choice(
            "Module Type",
            "Select module type:",
            [
                ("package", "📦 Package (directory with multiple files)"),
                ("single", "📄 Single (single file module)")
            ]
        )

        if not module_type_choice:
            return

        module_type = ModuleType.SINGLE if module_type_choice == "single" else ModuleType.PACKAGE

        # Create config
        if module_type == ModuleType.PACKAGE:
            config = create_tb_config_v2(
                module_name=module_name,
                version=version,
                module_type=module_type,
                description=description,
                author=author
            )
        else:
            file_path = await show_input("File Path", "Enter file path:")
            if not file_path:
                return

            config = create_tb_config_single(
                module_name=module_name,
                version=version,
                file_path=file_path,
                description=description,
                author=author
            )

        # Save config
        default_path = f"./mods/{module_name}/tbConfig.yaml"
        save_path = await show_input("Save Location", "Save to:", default_path)

        if not save_path:
            return

        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            await show_message("Success", f"Configuration saved to:\n{save_path}", "success")
        except Exception as e:
            await show_message("Error", f"Could not save config:\n\n{str(e)}", "error")

    async def _generate_all_configs(self):
        """Generate configs for all modules."""
        root_dir = await show_input("Root Directory", "Enter root directory:", "./mods")

        if not root_dir:
            return

        # Generation mode
        mode_choice = await show_choice(
            "Generation Mode",
            "Select generation mode:",
            [
                ("interactive", "💬 Interactive (ask for each module)"),
                ("auto", "🤖 Auto (skip existing configs)"),
                ("force", "⚡ Force (overwrite all configs)")
            ]
        )

        if not mode_choice:
            return

        backup = await show_confirm("Backup", "Create backups of existing configs?")

        interactive_mode = mode_choice == "interactive"
        overwrite_mode = mode_choice == "force"

        if not await show_confirm(
            "Confirm Generation",
            f"Mode: {mode_choice.title()}\n"
            f"Backup: {'Yes' if backup else 'No'}\n"
            f"Root: {root_dir}\n\n"
            "Start generation?"
        ):
            return

        await show_progress("Generating", "Generating configs for all modules...")

        result = await generate_configs_for_existing_mods(
            app=self.app_instance,
            root_dir=root_dir,
            backup=backup,
            interactive=interactive_mode,
            overwrite=overwrite_mode
        )

        if result.is_error:
            await show_message("Completed", f"Generation completed with errors:\n\n{result}", "warning")
        else:
            await show_message("Success", "Config generation completed successfully!", "success")

    async def _generate_single_config(self):
        """Generate config for specific module."""
        # Get module list
        mods = self.app_instance.get_all_mods()

        if not mods:
            await show_message("No Modules", "No modules found.", "warning")
            return

        # Build choices
        choices = []
        for mod in mods:
            config_path = Path('./mods') / mod / 'tbConfig.yaml'
            single_config = Path('./mods') / f'{mod}.yaml'
            status = "✓" if (config_path.exists() or single_config.exists()) else "✗"
            choices.append((mod, f"[{status}] {mod}"))

        module_name = await show_choice(
            "Select Module",
            "Choose module to generate config for:",
            choices
        )

        if not module_name:
            return

        # Check if config exists
        module_path = Path('./mods') / module_name
        config_exists = False

        if module_path.is_dir():
            config_exists = (module_path / 'tbConfig.yaml').exists()
        else:
            config_exists = (Path('./mods') / f'{module_name}.yaml').exists()

        force = False
        if config_exists:
            if not await show_confirm(
                "Config Exists",
                f"Config already exists for '{module_name}'.\n\nOverwrite?"
            ):
                return
            force = True

        await show_progress("Generating", f"Generating config for '{module_name}'...")

        result = await generate_single_module_config(
            app=self.app_instance,
            module_name=module_name,
            force=force
        )

        if result.is_error:
            await show_message("Error", f"Generation failed:\n\n{result}", "error")
        else:
            await show_message("Success", f"Config generated for '{module_name}'!", "success")

    async def _build_platform(self):
        """Build platform-specific installer."""
        module_name = await show_input("Platform Build", "Enter module name:")

        if not module_name:
            return

        # Platform selection
        platform_choices = [(p, f"{p.value}") for p in Platform]
        platform = await show_choice(
            "Select Platform",
            "Choose target platform:",
            platform_choices
        )

        if not platform:
            return

        upload = await show_confirm("Upload", "Upload after building?")

        await show_progress("Building", f"Building for {platform.value}...")

        result = await make_installer(
            self.app_instance, module_name,
            upload=upload,
            platform=platform
        )

        if result.is_error:
            await show_message("Error", f"Build failed:\n\n{result}", "error")
        else:
            await show_message("Success", f"Platform-specific installer built!", "success")

    async def _install_platform(self):
        """Install for specific platform."""
        module_name = await show_input("Platform Install", "Enter module name:")

        if not module_name:
            return

        # Platform selection
        platform_choices = [(p, f"{p.value}") for p in Platform]
        platform = await show_choice(
            "Select Platform",
            "Choose target platform:",
            platform_choices
        )

        if not platform:
            return

        await show_progress("Installing", f"Installing for {platform.value}...")

        result = await installer(self.app_instance, module_name, platform=platform)

        if result.is_error:
            await show_message("Error", f"Installation failed:\n\n{result}", "error")
        else:
            await show_message("Success", "Module installed successfully!", "success")

    async def _create_from_template(self):
        """Create module from template."""
        # Get templates
        result = await list_module_templates(self.app_instance)
        templates = result.get()['templates']

        # Build choices
        template_choices = [
            (t['name'], f"{t['name']:<25} - {t['description']}")
            for t in templates
        ]

        selected_template = await show_choice(
            "Select Template",
            "Choose module template:",
            template_choices
        )

        if not selected_template:
            return

        # Collect information
        module_name = await show_input("Module Name", "Enter module name:")
        if not module_name:
            return

        description = await show_input("Description", "Description (optional):")
        version = await show_input("Version", "Version:", "0.0.1")
        author = await show_input("Author", "Author (optional):")
        location = await show_input("Location", "Location:", "./mods")

        external = await show_confirm("External", "Create external to toolbox?")
        create_config = await show_confirm("Config", "Create tbConfig.yaml?")

        await show_progress("Creating", f"Creating {selected_template} module '{module_name}'...")

        result = await create_module_from_blueprint(
            app=self.app_instance,
            module_name=module_name,
            module_type=selected_template,
            description=description,
            version=version,
            location=location,
            author=author,
            create_config=create_config,
            external=external
        )

        if result.is_error:
            await show_message("Error", f"Module creation failed:\n\n{result}", "error")
        else:
            await show_message(
                "Success",
                f"Module '{module_name}' created successfully!\n\nLocation: {location}/{module_name}",
                "success"
            )

    async def _list_templates(self):
        """List available templates."""
        result = await list_module_templates(self.app_instance)
        templates = result.get()['templates']

        lines = []
        for t in templates:
            lines.append(f"\n┌─ {t['name']}")
            lines.append(f"│  Description: {t['description']}")
            lines.append(f"│  Type: {t['type']}")
            lines.append(f"│  Requires: {', '.join(t['requires']) if t['requires'] else 'None'}")
            lines.append("└" + "─" * 60)

        await show_message("📚 Available Templates", '\n'.join(lines), "info")


# =================== Main Export ===================

@export(mod_name=Name, name="manager", test=False)
async def interactive_manager(app: Optional[App] = None):
    """
    Modern interactive CLI manager for module operations.

    Features:
    - Arrow key navigation (↑↓ or w/s)
    - Modern, minimalistic UI
    - All original functionality preserved
    - Better visual feedback
    - Elegant dialogs and prompts
    """
    if app is None:
        app = get_app(f"{Name}.manager")

    # Clear screen
    print('\033[2J\033[H')

    # Welcome message
    print_formatted_text(HTML(
        '\n<menu-title>╔════════════════════════════════════════════════════════════════════╗</menu-title>\n'
        '<menu-title>║          Welcome to CloudM Interactive Module Manager              ║</menu-title>\n'
        '<menu-title>╚════════════════════════════════════════════════════════════════════╝</menu-title>\n'
    ), style=MODERN_STYLE)

    await asyncio.sleep(1)

    # Create and run manager
    manager = ModernMenuManager(app)

    try:
        await manager.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Goodbye message
        print('\033[2J\033[H')
        print_formatted_text(HTML(
            '\n<success>╔════════════════════════════════════════════════════════════════════╗</success>\n'
            '<success>║          Thank you for using CloudM Module Manager! 👋             ║</success>\n'
            '<success>╚════════════════════════════════════════════════════════════════════╝</success>\n'
        ), style=MODERN_STYLE)



# =================== Web UI ===================
@export(mod_name=Name, name="ui", api=True, api_methods=['GET'])
def mod_manager_ui(app: App) -> Result:
    """
    Serves the module manager web interface.

    Returns:
        HTML result with UI
    """
    ui_path = Path(__file__).parent / "mod_manager.html"

    if not ui_path.exists():
        # Generate default UI if file doesn't exist
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CloudM - Module Manager</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .content {
            padding: 30px;
        }
        .module-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .module-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .module-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            transition: background 0.3s;
        }
        .btn:hover { background: #5568d3; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 CloudM Module Manager</h1>
            <p>Manage your modules with ease</p>
        </div>
        <div class="content">
            <div class="controls">
                <button class="btn" onclick="loadModules()">🔄 Refresh</button>
                <button class="btn" onclick="updateAll()">⬆️ Update All</button>
            </div>
            <div id="modules" class="module-list"></div>
        </div>
    </div>
    <script>
        async function loadModules() {
            const response = await fetch('/api/CloudM/list_modules');
            const data = await response.json();
            const container = document.getElementById('modules');
            container.innerHTML = data.modules.map(mod => `
                <div class="module-card">
                    <h3>📦 ${mod}</h3>
                    <button class="btn" onclick="installModule('${mod}')">Install</button>
                    <button class="btn btn-danger" onclick="uninstallModule('${mod}')">Uninstall</button>
                </div>
            `).join('');
        }
        async function installModule(name) {
            alert(`Installing ${name}...`);
        }
        async function uninstallModule(name) {
            if (confirm(`Uninstall ${name}?`)) {
                alert(`Uninstalling ${name}...`);
            }
        }
        async function updateAll() {
            alert('Updating all modules...');
        }
        loadModules();
    </script>
</body>
</html>
        """
        return Result.html(html_content)

    return Result.html(ui_path.read_text(encoding='utf-8'))


# =================== Auto-Config Generation ===================

@export(mod_name=Name, name="generate_configs", test=False)
async def generate_configs_for_existing_mods(
    app: Optional[App] = None,
    root_dir: str = './mods',
    backup: bool = True,
    interactive: bool = True,
    overwrite: bool = False
) -> Result:
    """
    Generates tbConfig.yaml files for all existing modules in the mods directory.

    Supports:
    - Package modules (directories) -> tbConfig.yaml (v2)
    - Single file modules (.py files) -> {module_name}.yaml (single)

    Args:
        app: Application instance
        root_dir: Root directory containing modules
        backup: Create backups of existing configs
        interactive: Ask for confirmation before each operation
        overwrite: Overwrite existing configs without asking

    Returns:
        Result with generation summary
    """
    if app is None:
        app = get_app(f"{Name}.generate_configs")

    root_path = Path(root_dir)
    if not root_path.exists():
        return Result.default_user_error(f"Directory not found: {root_dir}")

    results = {
        "generated": [],
        "skipped": [],
        "failed": [],
        "backed_up": []
    }

    def create_backup(config_path: Path) -> bool:
        """Creates a backup of existing config file"""
        if not config_path.exists():
            return False

        backup_path = config_path.with_suffix('.yaml.backup')
        counter = 1
        while backup_path.exists():
            backup_path = config_path.with_suffix(f'.yaml.backup{counter}')
            counter += 1

        shutil.copy2(config_path, backup_path)
        results["backed_up"].append(str(backup_path))
        print(f"  📦 Backup created: {backup_path.name}")
        return True

    def read_requirements(module_path: Path) -> List[str]:
        """Reads dependencies from requirements.txt"""
        req_file = module_path / 'requirements.txt' if module_path.is_dir() else module_path.parent / 'requirements.txt'

        if not req_file.exists():
            return []

        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            print(f"  ⚠ Error reading requirements: {e}")
            return []

    def extract_module_info(module_path: Path, module_name: str) -> Dict[str, Any]:
        """Extracts metadata from module by analyzing the code"""
        info = {
            "version": "0.0.1",
            "description": f"Module {module_name}",
            "author": "",
            "exports": [],
            "dependencies": []
        }

        try:
            # Try to load module to get version
            if module_name in app.get_all_mods():
                mod = app.get_mod(module_name)
                if mod:
                    info["version"] = getattr(mod, 'version', '0.0.1')

            # Analyze Python file for exports
            py_file = module_path if module_path.is_file() else module_path / '__init__.py'
            if not py_file.exists() and module_path.is_dir():
                py_file = module_path / f"{module_name}.py"

            if py_file.exists():
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Extract version
                    import re
                    version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                    if version_match:
                        info["version"] = version_match.group(1)

                    # Extract exports
                    export_matches = re.findall(r'@export\([^)]*name=["\']([^"\']+)["\']', content)
                    info["exports"] = export_matches

                    # Extract docstring
                    docstring_match = re.search(r'"""([^"]+)"""', content)
                    if docstring_match:
                        info["description"] = docstring_match.group(1).strip().split('\n')[0]

            # Get dependencies
            info["dependencies"] = read_requirements(module_path)

        except Exception as e:
            print(f"  ⚠ Error extracting info: {e}")

        return info

    def generate_package_config(module_path: Path, module_name: str) -> bool:
        """Generates tbConfig.yaml for package modules"""
        config_path = module_path / "tbConfig.yaml"

        # Check if config exists
        if config_path.exists() and not overwrite:
            if interactive:
                response = input(f"  Config exists for {module_name}. Overwrite? (y/n/b=backup): ").lower()
                if response == 'n':
                    results["skipped"].append(module_name)
                    print(f"  ⏭  Skipped: {module_name}")
                    return False
                elif response == 'b':
                    create_backup(config_path)
            else:
                results["skipped"].append(module_name)
                print(f"  ⏭  Skipped (exists): {module_name}")
                return False
        elif config_path.exists() and backup:
            create_backup(config_path)

        # Extract module information
        info = extract_module_info(module_path, module_name)

        # Determine platform files
        platform_config = {
            Platform.COMMON.value: {
                "files": ["*"],
                "required": True
            },
            Platform.SERVER.value: {
                "files": [],
                "required": False
            },
            Platform.CLIENT.value: {
                "files": [],
                "required": False
            },
            Platform.DESKTOP.value: {
                "files": [],
                "required": False
            },
            Platform.MOBILE.value: {
                "files": [],
                "required": False
            }
        }

        # Create config
        config = create_tb_config_v2(
            module_name=module_name,
            version=info["version"],
            module_type=ModuleType.PACKAGE,
            description=info["description"],
            author=info["author"],
            platforms=platform_config,
            metadata={
                "exports": info["exports"],
                "auto_generated": True,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

        # Write config
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            # Generate requirements.txt if not exists
            req_path = module_path / "requirements.txt"
            if not req_path.exists():
                generate_requirements(str(module_path), str(req_path))

            results["generated"].append(module_name)
            print(f"  ✓ Generated: {config_path}")
            return True

        except Exception as e:
            results["failed"].append({"module": module_name, "error": str(e)})
            print(f"  ✗ Failed: {module_name} - {e}")
            return False

    def generate_single_config(file_path: Path, module_name: str) -> bool:
        """Generates {module_name}.yaml for single file modules"""
        config_path = file_path.parent / f"{module_name}.yaml"

        # Check if config exists
        if config_path.exists() and not overwrite:
            if interactive:
                response = input(f"  Config exists for {module_name}. Overwrite? (y/n/b=backup): ").lower()
                if response == 'n':
                    results["skipped"].append(module_name)
                    print(f"  ⏭  Skipped: {module_name}")
                    return False
                elif response == 'b':
                    create_backup(config_path)
            else:
                results["skipped"].append(module_name)
                print(f"  ⏭  Skipped (exists): {module_name}")
                return False
        elif config_path.exists() and backup:
            create_backup(config_path)

        # Extract module information
        info = extract_module_info(file_path, module_name)

        # Create single config
        config = create_tb_config_single(
            module_name=module_name,
            version=info["version"],
            file_path=str(file_path.relative_to(root_path.parent)),
            description=info["description"],
            author=info["author"],
            specification={
                "exports": info["exports"],
                "functions": [],
                "classes": [],
                "requires": info["dependencies"]
            },
            dependencies=info["dependencies"],
            platforms=[Platform.ALL.value],
            metadata={
                "auto_generated": True,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

        # Write config
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            results["generated"].append(module_name)
            print(f"  ✓ Generated: {config_path}")
            return True

        except Exception as e:
            results["failed"].append({"module": module_name, "error": str(e)})
            print(f"  ✗ Failed: {module_name} - {e}")
            return False

    # Main processing loop
    print(f"\n🔍 Scanning directory: {root_path}")
    print("=" * 60)

    items = sorted(root_path.iterdir())
    total_items = len(items)

    for idx, item in enumerate(items, 1):
        # Skip hidden files/folders and __pycache__
        if item.name.startswith('.') or item.name == '__pycache__':
            continue

        print(f"\n[{idx}/{total_items}] Processing: {item.name}")

        if item.is_dir():
            # Package module
            module_name = item.name
            generate_package_config(item, module_name)

        elif item.is_file() and item.suffix == '.py':
            # Single file module
            module_name = item.stem
            generate_single_config(item, module_name)

    # Summary
    print("\n" + "=" * 60)
    print("📊 Generation Summary:")
    print(f"  ✓ Generated: {len(results['generated'])}")
    print(f"  ⏭  Skipped:   {len(results['skipped'])}")
    print(f"  ✗ Failed:    {len(results['failed'])}")
    print(f"  📦 Backed up: {len(results['backed_up'])}")

    if results['generated']:
        print("\n✓ Generated configs for:")
        for mod in results['generated']:
            print(f"  - {mod}")

    if results['failed']:
        print("\n✗ Failed to generate configs for:")
        for fail in results['failed']:
            print(f"  - {fail['module']}: {fail['error']}")

    return Result.ok({
        "summary": {
            "total_processed": total_items,
            "generated": len(results['generated']),
            "skipped": len(results['skipped']),
            "failed": len(results['failed']),
            "backed_up": len(results['backed_up'])
        },
        "details": results
    })


@export(mod_name=Name, name="generate_single_config", test=False)
async def generate_single_module_config(
    app: Optional[App] = None,
    module_name: str = "",
    force: bool = False
) -> Result:
    """
    Generates config for a single specific module.

    Args:
        app: Application instance
        module_name: Name of module to generate config for
        force: Force overwrite without asking

    Returns:
        Result with generation status
    """
    if app is None:
        app = get_app(f"{Name}.generate_single_config")

    if not module_name:
        return Result.default_user_error("Module name is required")

    # Find module path
    module_path = Path('./mods') / module_name

    if not module_path.exists():
        # Try as single file
        module_path = Path(f'./mods/{module_name}.py')
        if not module_path.exists():
            return Result.default_user_error(f"Module not found: {module_name}")

    print(f"\n🔧 Generating config for: {module_name}")

    # Use the main function with specific parameters
    result = await generate_configs_for_existing_mods(
        app=app,
        root_dir=str(module_path.parent),
        backup=True,
        interactive=not force,
        overwrite=force
    )

    return result


# =================== Blueprint Templates ===================

MODULE_TEMPLATES = {
    "basic": {
        "description": "Basic Function Module - Simple functions with exports",
        "type": "file",
        "requires": ["core"],
        "content": '''"""
{MODULE_NAME} - {DESCRIPTION}
Version: {VERSION}
"""
from toolboxv2 import App, Result, get_app

Name = '{MODULE_NAME}'
export = get_app(f"{Name}.Export").tb
version = '{VERSION}'


@export(mod_name=Name, version=version)
def example_function(app: App, param: str) -> Result:
    """
    Example function demonstrating basic usage.

    Args:
        app: Application instance
        param: Example parameter

    Returns:
        Result with processed data
    """
    app.print(f"Processing: {param}")
    return Result.ok(data={{"processed": param}})


@export(mod_name=Name, version=version, row=True)
def get_version() -> str:
    """Returns module version."""
    return version


@export(mod_name=Name, version=version, initial=True)
def initialize(app: App) -> Result:
    """
    Module initialization function.
    Called when module is loaded.
    """
    app.print(f"Initializing {Name} v{version}")
    return Result.ok(info=f"{Name} initialized")
'''
    },

    "async_service": {
        "description": "Async Service Module - Network/IO intensive operations",
        "type": "package",
        "requires": ["core", "async"],
        "content": '''"""
{MODULE_NAME} - {DESCRIPTION}
Async service module for network/IO operations
Version: {VERSION}
"""
import asyncio
from typing import Optional, Dict, Any
from toolboxv2 import App, Result, get_app

Name = '{MODULE_NAME}'
export = get_app(f"{Name}.Export").tb
version = '{VERSION}'


class Tool:
    """Core service class for {MODULE_NAME}"""

    def __init__(self, app: App):
        self.app = app
        self.logger = app.logger
        self.connections: Dict[str, Any] = {{}}

    async def connect(self, endpoint: str) -> bool:
        """Establish connection to endpoint"""
        try:
            self.app.print(f"Connecting to {endpoint}...")
            # Your connection logic here
            self.connections[endpoint] = {{"status": "connected"}}
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self, endpoint: str) -> bool:
        """Disconnect from endpoint"""
        if endpoint in self.connections:
            del self.connections[endpoint]
            return True
        return False


@export(mod_name=Name, version=version, initial=True)
async def initialize(app: App) -> Result:
    """Initialize service core"""
    core = {MODULE_NAME}Core(app)
    app.{MODULE_NAME_LOWER}_core = core
    app.print(f"{Name} service initialized")
    return Result.ok(info="Service ready")


@export(mod_name=Name, version=version)
async def connect_endpoint(app: App, endpoint: str) -> Result:
    """
    Connect to a service endpoint.

    Args:
        app: Application instance
        endpoint: Endpoint URL/address

    Returns:
        Result with connection status
    """
    core = getattr(app, '{MODULE_NAME_LOWER}_core', None)
    if not core:
        return Result.default_internal_error("Service not initialized")

    success = await core.connect(endpoint)
    if success:
        return Result.ok(data={{"endpoint": endpoint, "status": "connected"}})
    return Result.default_user_error("Connection failed")


@export(mod_name=Name, version=version, api=True, api_methods=['GET'])
async def get_status(app: App) -> Result:
    """
    Get service status.

    Returns:
        Result with current status
    """
    core = getattr(app, '{MODULE_NAME_LOWER}_core', None)
    if not core:
        return Result.default_internal_error("Service not initialized")

    return Result.json(data={{
        "service": Name,
        "version": version,
        "connections": len(core.connections),
        "status": "running"
    }})


@export(mod_name=Name, version=version, exit_f=True)
async def cleanup(app: App) -> Result:
    """Cleanup service resources"""
    core = getattr(app, '{MODULE_NAME_LOWER}_core', None)
    if core:
        for endpoint in list(core.connections.keys()):
            await core.disconnect(endpoint)
    app.print(f"{Name} service stopped")
    return Result.ok(info="Cleanup complete")
'''
    },

    "workflow": {
        "description": "Workflow Module - Multi-step processes",
        "type": "file",
        "requires": ["core", "async"],
        "content": '''"""
{MODULE_NAME} - {DESCRIPTION}
Workflow orchestration module
Version: {VERSION}
"""
import asyncio
from typing import List, Dict, Any, Optional
from toolboxv2 import App, Result, get_app, AppArgs

Name = '{MODULE_NAME}'
version = '{VERSION}'


async def run(app: App, app_args: AppArgs):
    """
    Main workflow execution function.

    This is called when the module is run as a workflow.

    Args:
        app: Application instance
        app_args: Application arguments
    """
    app.print(f"Starting {Name} workflow v{version}")

    # Step 1: Initialize required modules
    app.print("Step 1: Initializing modules...")
    # await app.load_mod("RequiredModule")

    # Step 2: Execute parallel tasks
    app.print("Step 2: Executing tasks...")
    results = await asyncio.gather(
        task_one(app),
        task_two(app),
        task_three(app),
        return_exceptions=True
    )

    # Step 3: Process results
    app.print("Step 3: Processing results...")
    for idx, result in enumerate(results, 1):
        if isinstance(result, Exception):
            app.logger.error(f"Task {idx} failed: {result}")
        else:
            app.print(f"Task {idx} completed: {result}")

    app.print(f"{Name} workflow completed")


async def task_one(app: App) -> str:
    """First workflow task"""
    app.print("Executing task 1...")
    await asyncio.sleep(1)
    return "Task 1 completed"


async def task_two(app: App) -> str:
    """Second workflow task"""
    app.print("Executing task 2...")
    await asyncio.sleep(1)
    return "Task 2 completed"


async def task_three(app: App) -> str:
    """Third workflow task"""
    app.print("Executing task 3...")
    await asyncio.sleep(1)
    return "Task 3 completed"


if __name__ == "__main__":
    from toolboxv2 import get_app
    app = get_app('{MODULE_NAME}.Main')
    asyncio.run(run(app, None))
'''
    },

    "minimal_flow": {
        "description": "Minimal Flow Module - Simple background process",
        "type": "file",
        "requires": ["core", "async"],
        "content": '''"""
{MODULE_NAME} - {DESCRIPTION}
Minimal flow module
Version: {VERSION}
"""
NAME = '{MODULE_NAME}'
version = '{VERSION}'


async def run(app, _):
    """
    Minimal flow entry point.

    Args:
        app: Application instance
        _: App arguments (unused)
    """
    app.print(f"Running {NAME} v{version}...")

    # Connect to daemon if available
    if hasattr(app, 'daemon_app'):
        await app.daemon_app.connect(app)

    # Your flow logic here
    app.print(f"{NAME} flow completed")
'''
    },

    "api_endpoint": {
        "description": "API Endpoint Module - REST API endpoints",
        "type": "package",
        "requires": ["core", "async"],
        "content": '''"""
{MODULE_NAME} - {DESCRIPTION}
REST API endpoints module
Version: {VERSION}
"""
from typing import Optional, Dict, Any
from toolboxv2 import App, Result, get_app, RequestData

Name = '{MODULE_NAME}'
export = get_app(f"{Name}.Export").tb
version = '{VERSION}'


@export(mod_name=Name, api=True, api_methods=['GET'], request_as_kwarg=True)
async def get_items(app: App, request: Optional[RequestData] = None) -> Result:
    """
    Get list of items.

    Query parameters:
        - limit: Number of items to return (default: 10)
        - offset: Pagination offset (default: 0)

    Returns:
        Result with items list
    """
    limit = 10
    offset = 0

    if request and request.query_params:
        limit = int(request.query_params.get('limit', 10))
        offset = int(request.query_params.get('offset', 0))

    # Fetch items (mock data)
    items = [
        {{"id": i, "name": f"Item {i}"}}
        for i in range(offset, offset + limit)
    ]

    return Result.json(data={{
        "items": items,
        "total": 100,
        "limit": limit,
        "offset": offset
    }})


@export(mod_name=Name, api=True, api_methods=['POST'])
async def create_item(app: App, data: Dict[str, Any]) -> Result:
    """
    Create a new item.

    Request body:
        {{"name": "Item name", "description": "Item description"}}

    Returns:
        Result with created item
    """
    if not data or 'name' not in data:
        return Result.default_user_error("Missing required field: name")

    # Create item (mock)
    new_item = {{
        "id": 123,
        "name": data['name'],
        "description": data.get('description', ''),
        "created_at": "2024-01-01T00:00:00Z"
    }}

    return Result.json(data=new_item, info="Item created successfully")


@export(mod_name=Name, api=True, api_methods=['PUT'])
async def update_item(app: App, item_id: int, data: Dict[str, Any]) -> Result:
    """
    Update an existing item.

    Args:
        item_id: ID of item to update
        data: Update data

    Returns:
        Result with updated item
    """
    # Update item (mock)
    updated_item = {{
        "id": item_id,
        "name": data.get('name', f'Item {item_id}'),
        "description": data.get('description', ''),
        "updated_at": "2024-01-01T00:00:00Z"
    }}

    return Result.json(data=updated_item, info="Item updated successfully")


@export(mod_name=Name, api=True, api_methods=['DELETE'])
async def delete_item(app: App, item_id: int) -> Result:
    """
    Delete an item.

    Args:
        item_id: ID of item to delete

    Returns:
        Result with deletion status
    """
    # Delete item (mock)
    return Result.ok(info=f"Item {item_id} deleted successfully")


@export(mod_name=Name, api=True, api_methods=['GET'])
async def health_check(app: App) -> Result:
    """
    Health check endpoint.

    Returns:
        Result with service status
    """
    return Result.json(data={{
        "service": Name,
        "version": version,
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }})
'''
    },

    "websocket": {
        "description": "WebSocket Module - Real-time bidirectional communication",
        "type": "package",
        "requires": ["core", "async"],
        "content": '''"""
{MODULE_NAME} - {DESCRIPTION}
WebSocket real-time communication module
Version: {VERSION}
"""
from toolboxv2 import App, get_app

Name = '{MODULE_NAME}'
export = get_app(f"{Name}.Export").tb
version = '{VERSION}'


async def on_connect(app: App, conn_id: str, session: dict):
    """
    Called when a client connects.

    Args:
        app: Application instance
        conn_id: Unique connection ID
        session: Client session data
    """
    username = session.get("user_name", "Anonymous")
    app.print(f"WS CONNECT: User '{username}' connected (conn_id: {conn_id})")

    # Send welcome message to the new client
    await app.ws_send(conn_id, {{
        "event": "welcome",
        "data": f"Welcome to {Name}, {username}!"
    }})

    # Broadcast to all others
    await app.ws_broadcast(
        channel_id=f"{Name}/main",
        payload={{
            "event": "user_joined",
            "data": f"{username} joined the room"
        }},
        source_conn_id=conn_id
    )


async def on_message(app: App, conn_id: str, session: dict, payload: dict):
    """
    Called when a message is received from a client.

    Args:
        app: Application instance
        conn_id: Unique connection ID
        session: Client session data
        payload: Message payload from client
    """
    username = session.get("user_name", "Anonymous")
    event_type = payload.get("event", "unknown")
    data = payload.get("data", {{}})

    app.print(f"WS MESSAGE from {username} ({conn_id}): {event_type}")

    if event_type == "message":
        # Broadcast message to all clients
        await app.ws_broadcast(
            channel_id=f"{Name}/main",
            payload={{
                "event": "new_message",
                "data": {{
                    "user": username,
                    "text": data.get("text", ""),
                    "timestamp": data.get("timestamp", "")
                }}
            }}
        )

    elif event_type == "ping":
        # Send pong only to requesting client
        await app.ws_send(conn_id, {{
            "event": "pong",
            "data": {{"timestamp": data.get("timestamp", "")}}
        }})


async def on_disconnect(app: App, conn_id: str, session: dict):
    """
    Called when a client disconnects.

    Args:
        app: Application instance
        conn_id: Unique connection ID
        session: Client session data
    """
    username = session.get("user_name", "Anonymous")
    app.print(f"WS DISCONNECT: User '{username}' disconnected (conn_id: {conn_id})")

    # Notify all remaining clients
    await app.ws_broadcast(
        channel_id=f"{Name}/main",
        payload={{
            "event": "user_left",
            "data": f"{username} left the room"
        }}
    )


@export(mod_name=Name, websocket_handler="main")
def register_websocket_handlers(app: App):
    """
    Register WebSocket event handlers.

    Returns:
        Dictionary mapping events to handler functions
    """
    return {{
        "on_connect": on_connect,
        "on_message": on_message,
        "on_disconnect": on_disconnect,
    }}
'''
    },

    "hybrid": {
        "description": "Hybrid Module - Mix of sync/async, with/without app",
        "type": "package",
        "requires": ["core", "async"],
        "content": '''"""
{MODULE_NAME} - {DESCRIPTION}
Hybrid module with various function types
Version: {VERSION}
"""
from typing import Optional
from toolboxv2 import App, Result, get_app

Name = '{MODULE_NAME}'
export = get_app(f"{Name}.Export").tb
version = '{VERSION}'


@export(mod_name=Name, version=version, row=True)
def get_version() -> str:
    """
    Simple function without App parameter, returns raw value.
    """
    return version


@export(mod_name=Name, version=version)
def process_data(data: str) -> Result:
    """
    Function without App parameter, returns Result.

    Args:
        data: Data to process

    Returns:
        Result with processed data
    """
    processed = data.upper()
    return Result.ok(data={{"original": data, "processed": processed}})


@export(mod_name=Name, version=version)
def with_app_call(app: App, value: int) -> Result:
    """
    Synchronous function with App parameter.

    Args:
        app: Application instance
        value: Value to process

    Returns:
        Result with calculated value
    """
    app.print(f"Processing value: {value}")
    result = value * 2
    return Result.ok(data={{"input": value, "output": result}})


@export(mod_name=Name, version=version)
async def async_with_app(app: App, endpoint: str) -> Result:
    """
    Asynchronous function with App parameter.

    Args:
        app: Application instance
        endpoint: Endpoint to connect to

    Returns:
        Result with connection status
    """
    import asyncio

    app.print(f"Connecting to {endpoint}...")
    await asyncio.sleep(1)  # Simulate async operation

    return Result.ok(data={{
        "endpoint": endpoint,
        "status": "connected",
        "latency_ms": 42
    }})


@export(mod_name=Name, version=version, api=True, api_methods=['GET'])
async def api_endpoint(app: App, param: Optional[str] = None) -> Result:
    """
    API endpoint function.

    Query parameters:
        param: Optional parameter

    Returns:
        JSON result
    """
    return Result.json(data={{
        "module": Name,
        "version": version,
        "param": param,
        "status": "ok"
    }})


@export(mod_name=Name, version=version, memory_cache=True, memory_cache_ttl=300)
def cached_function(key: str) -> Result:
    """
    Function with memory caching enabled.
    Cache TTL: 300 seconds

    Args:
        key: Cache key

    Returns:
        Result with cached data
    """
    import time

    # Expensive operation
    result = f"Expensive calculation for {key}"
    timestamp = time.time()

    return Result.ok(data={{
        "key": key,
        "result": result,
        "timestamp": timestamp
    }})


@export(mod_name=Name, version=version, initial=True)
def initialize(app: App) -> Result:
    """
    Initialization function called when module is loaded.
    """
    app.print(f"Initializing {Name} v{version}")
    # Setup resources, connections, etc.
    return Result.ok(info=f"{Name} initialized")


@export(mod_name=Name, version=version, exit_f=True)
def cleanup(app: App) -> Result:
    """
    Cleanup function called when application shuts down.
    """
    app.print(f"Cleaning up {Name}")
    # Release resources, close connections, etc.
    return Result.ok(info=f"{Name} cleanup complete")
'''
    }
}


# =================== Blueprint Generator Function ===================

@export(mod_name=Name, name="create_module", test=False)
async def create_module_from_blueprint(
    app: Optional[App] = None,
    module_name: str = "",
    module_type: str = "basic",
    description: str = "",
    version: str = "0.0.1",
    location: str = "./mods",
    author: str = "",
    create_config: bool = True,
    external: bool = False
) -> Result:
    """
    Creates a new module from blueprint template.

    Args:
        app: Application instance
        module_name: Name of the new module
        module_type: Type of module (basic, async_service, workflow, etc.)
        description: Module description
        version: Initial version
        location: Where to create the module
        author: Module author
        create_config: Whether to create tbConfig.yaml
        external: If True, create external to toolbox structure

    Returns:
        Result with creation status
    """
    if app is None:
        app = get_app(f"{Name}.create_module")

    if not module_name:
        return Result.default_user_error("Module name is required")

    if module_type not in MODULE_TEMPLATES:
        return Result.default_user_error(
            f"Invalid module type. Available: {', '.join(MODULE_TEMPLATES.keys())}"
        )

    template = MODULE_TEMPLATES[module_type]

    # Prepare paths
    location_path = Path(location)

    if template["type"] == "package":
        module_path = location_path / module_name
        module_file = module_path / "__init__.py"
    else:
        module_path = location_path
        module_file = module_path / f"{module_name}.py"

    # Check if module already exists
    if module_file.exists():
        return Result.default_user_error(f"Module already exists: {module_file}")

    try:
        # Create directory structure
        if template["type"] == "package":
            module_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created package directory: {module_path}")
        else:
            module_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Using directory: {module_path}")

        # Generate module content
        content = template["content"].format(
            MODULE_NAME=module_name,
            MODULE_NAME_LOWER=module_name.lower(),
            VERSION=version,
            DESCRIPTION=description or template["description"]
        )

        # Write module file
        with open(module_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created module file: {module_file}")

        # Create requirements.txt
        req_path = module_path if template["type"] == "package" else module_path
        requirements = []

        if "async" in template["requires"]:
            requirements.append("aiohttp>=3.8.0")

        if requirements:
            req_file = (module_path if template["type"] == "package" else module_path) / "requirements.txt"
            with open(req_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(requirements))
            print(f"✓ Created requirements.txt")

        # Create tbConfig.yaml
        if create_config and not external:
            if template["type"] == "package":
                config = create_tb_config_v2(
                    module_name=module_name,
                    version=version,
                    module_type=ModuleType.PACKAGE,
                    description=description or template["description"],
                    author=author,
                    metadata={
                        "template": module_type,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                config_path = module_path / "tbConfig.yaml"
            else:
                config = create_tb_config_single(
                    module_name=module_name,
                    version=version,
                    file_path=str(module_file.relative_to(location_path.parent)),
                    description=description or template["description"],
                    author=author,
                    metadata={
                        "template": module_type,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                config_path = module_path / f"{module_name}.yaml"

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"✓ Created config: {config_path}")

        # Create additional files based on type
        if module_type == "api_endpoint":
            # Create example API documentation
            api_doc = module_path / "API.md" if template["type"] == "package" else module_path / f"{module_name}_API.md"
            with open(api_doc, 'w', encoding='utf-8') as f:
                f.write(f"""# {module_name} API Documentation

## Endpoints

### GET /api/{module_name}/get_items
Get list of items with pagination.

**Query Parameters:**
- `limit` (int, optional): Number of items (default: 10)
- `offset` (int, optional): Pagination offset (default: 0)

**Response:**
```json
{{
  "items": [...],
  "total": 100,
  "limit": 10,
  "offset": 0
}}
```

### POST /api/{module_name}/create_item
Create a new item.

**Request Body:**
```json
{{
  "name": "Item name",
  "description": "Item description"
}}
```

### GET /api/{module_name}/health_check
Health check endpoint.
""")
            print(f"✓ Created API documentation")

        elif module_type == "websocket":
            # Create WebSocket client example
            ws_example = module_path / "client_example.html" if template[
                                                                    "type"] == "package" else module_path / f"{module_name}_client.html"
            with open(ws_example, 'w', encoding='utf-8') as f:
                f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>{module_name} WebSocket Client</title>
    <script src="/static/tbjs/tb.js"></script>
</head>
<body>
    <h1>{module_name} WebSocket Demo</h1>
    <div id="messages"></div>
    <input type="text" id="messageInput" placeholder="Type a message...">
    <button onclick="sendMessage()">Send</button>

    <script>
        // Connect to WebSocket
        TB.ws.connect('/ws/{module_name}/main', {{
            onOpen: () => {{
                console.log('Connected to {module_name}');
            }},
            onMessage: (data) => {{
                console.log('Message:', data);
                displayMessage(data);
            }}
        }});

        // Listen for specific events
        TB.events.on('ws:event:new_message', ({{ data }}) => {{
            displayMessage(data.data);
        }});

        function sendMessage() {{
            const input = document.getElementById('messageInput');
            TB.ws.send({{
                event: 'message',
                data: {{
                    text: input.value,
                    timestamp: new Date().toISOString()
                }}
            }});
            input.value = '';
        }}

        function displayMessage(msg) {{
            const div = document.getElementById('messages');
            div.innerHTML += `<div>${{JSON.stringify(msg)}}</div>`;
        }}
    </script>
</body>
</html>
""")
            print(f"✓ Created WebSocket client example")

        # Create README
        readme_path = module_path / "README.md" if template[
                                                       "type"] == "package" else module_path / f"{module_name}_README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"""# {module_name}

{description or template['description']}

## Version
{version}

## Type
{template['description']}

## Installation

```bash
# Install module
python CloudM.py install {module_name}
```

## Usage

```python
from toolboxv2 import get_app

app = get_app("{module_name}.Example")

# Use module functions
# Example code here
```

## Author
{author or 'ToolBoxV2'}

## Created
{time.strftime("%Y-%m-%d %H:%M:%S")}

## Template
{module_type}
""")
        print(f"✓ Created README.md")

        print(f"\n{'=' * 60}")
        print(f"✓ Module '{module_name}' created successfully!")
        print(f"{'=' * 60}")
        print(f"\nLocation: {module_file}")
        print(f"Type: {template['description']}")
        print(f"Version: {version}")

        if not external:
            print(f"\nNext steps:")
            print(f"1. Review and customize the generated code")
            print(f"2. Install dependencies: pip install -r requirements.txt")
            print(
                f"3. Test the module: python -c 'from toolboxv2 import get_app; app = get_app(\"{module_name}.Test\")'")
            print(f"4. Build installer: python CloudM.py build {module_name}")

        return Result.ok(data={
            "module_name": module_name,
            "type": module_type,
            "location": str(module_file),
            "config_created": create_config,
            "files_created": [
                str(module_file),
                str(readme_path)
            ]
        })

    except Exception as e:
        return Result.default_internal_error(f"Failed to create module: {str(e)}")


@export(mod_name=Name, name="list_templates", test=False)
def list_module_templates(app: Optional[App] = None) -> Result:
    """
    Lists all available module templates.

    Returns:
        Result with template information
    """
    templates = []
    for template_name, template_info in MODULE_TEMPLATES.items():
        templates.append({
            "name": template_name,
            "description": template_info["description"],
            "type": template_info["type"],
            "requires": template_info["requires"]
        })

    return Result.ok(data={"templates": templates, "count": len(templates)})


@export(mod_name=Name, name="mods")
async def main(app, command="", module_name="", **kwargs):

    if command:
        if command == "list":
            mods = app.get_all_mods()
            print(f"Available modules ({len(mods)}):")
            for mod in mods:
                print(f"  - {mod}")

        elif command == "manager":
            await interactive_manager(app)


        elif command == "build" and module_name:
            module_name = module_name
            await make_installer(app, module_name, upload=False)

        elif command == "install" and module_name:
            module_name = module_name
            await installer(app, module_name)

        elif command == "gen-configs":
            # Generate configs for all modules
            interactive_mode = "--non-interactive" not in kwargs
            overwrite_mode = "--force" in kwargs
            no_backup = "--no-backup" in kwargs

            await generate_configs_for_existing_mods(
                app=app,
                root_dir='./mods',
                backup=not no_backup,
                interactive=interactive_mode,
                overwrite=overwrite_mode
            )

        elif command == "gen-config" and module_name:
            # Generate config for specific module
            force_mode = "--force" in kwargs

            await generate_single_module_config(
                app=app,
                module_name=module_name,
                force=force_mode
            )

            if command == "create":
                if len(sys.argv) < 3:
                    print("Usage: python CloudM.py create <module_name> [options]")
                    print("\nOptions:")
                    print("  --type=<type>          Module type (default: basic)")
                    print("  --desc=<description>   Module description")
                    print("  --version=<version>    Initial version (default: 0.0.1)")
                    print("  --location=<path>      Where to create (default: ./mods)")
                    print("  --author=<author>      Module author")
                    print("  --external             Create external to toolbox")
                    print("  --no-config            Don't create tbConfig.yaml")
                    print("\nAvailable types:")
                    result = list_module_templates(app)
                    for t in result.get('templates'):
                        print(f"  - {t['name']:<20} {t['description']}")
                    sys.exit(1)

                module_name = sys.argv[2]

                # Parse options
                options = {
                    **kwargs,
                    "module_type": "basic",
                    "description": "",
                    "version": "0.0.1",
                    "location": "./mods",
                    "author": "",
                    "external": False,
                    "create_config": True
                }

                for arg in kwargs:
                    if arg == "--external":
                        options["external"] = True
                    elif arg == "--no-config":
                        options["create_config"] = False

                result = await create_module_from_blueprint(
                    app=app,
                    module_name=module_name,
                    **options
                )

                print(result)

        elif command == "templates":
            # List available templates
            result = list_module_templates(app)
            templates = result.get('templates')

            print("\n📦 Available Module Templates:")
            print("=" * 80)
            for t in templates:
                print(f"\n{t['name']}")
                print(f"  Description: {t['description']}")
                print(f"  Type: {t['type']}")
                print(f"  Requires: {', '.join(t['requires'])}")
            print("\n" + "=" * 80)

        else:
            print("Usage:")
            print("  tb -c CloudM mods list              - List all modules")
            print("  tb -c CloudM mods manager           - Interactive manager")
            print("  tb -c CloudM mods build <module>    - Build module installer")
            print("  tb -c CloudM mods install <module>  - Install module")
            print("\n  Config Generation:")
            print("  tb -c CloudM mods gen-configs --kwargs         - Generate configs for all modules")
            print("                   [--force=true]                     - Overwrite without asking")
            print("                   [--non-interactive=true]           - Don't ask for confirmation")
            print("                   [--no-backup=true]                 - Don't create backups")
            print("  tb -c CloudM mods gen-config <module> --kwargs [--force=true] - Generate config for specific module")
            print("\n  Module Creation:")
            print("  tb -c CloudM mods create <module_name> --kwargs [options] - Create new module from template")
            print("  tb -c CloudM mods templates                      - List available templates")

    else:
        # Run interactive manager by default
        await interactive_manager(app)


# =================== Main Entry Point ===================
if __name__ == "__main__":
    """
    Main entry point for CLI usage
    """
    import sys

    asyncio.run(main(get_app('CloudM.Main'), *sys.argv[1:]))
