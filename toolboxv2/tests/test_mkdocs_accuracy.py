"""Hard accuracy benchmark for mkdocs system.

Tests that the system finds the CORRECT source definitions,
not just any file that mentions the name.
"""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from toolboxv2.utils.extras.mkdocs import DocsSystem


# Hard-coded expected locations - these are the ACTUAL source definitions
EXPECTED_CLASSES = {
    # Core App classes - MUST be from toolbox.py or types.py
    "App": "toolboxv2/utils/toolbox.py",
    "Result": "toolboxv2/utils/system/types.py",
    "AppType": "toolboxv2/utils/system/types.py",
    "AppArgs": "toolboxv2/utils/system/types.py",
    "ToolBoxError": "toolboxv2/utils/system/types.py",
    "ToolBoxResult": "toolboxv2/utils/system/types.py",
    "ToolBoxInfo": "toolboxv2/utils/system/types.py",
    "RequestData": "toolboxv2/utils/system/types.py",
    # mkdocs classes
    "DocsSystem": "toolboxv2/utils/extras/mkdocs.py",
    "CodeAnalyzer": "toolboxv2/utils/extras/mkdocs.py",
    "DocParser": "toolboxv2/utils/extras/mkdocs.py",
    "IndexManager": "toolboxv2/utils/extras/mkdocs.py",
    "ContextEngine": "toolboxv2/utils/extras/mkdocs.py",
    "FileScanner": "toolboxv2/utils/extras/mkdocs.py",
    "JSTSAnalyzer": "toolboxv2/utils/extras/mkdocs.py",
    "InvertedIndex": "toolboxv2/utils/extras/mkdocs.py",
    # System classes - MUST be from utils/system
    "MainTool": "toolboxv2/utils/system/main_tool.py",
    "FileHandler": "toolboxv2/utils/system/file_handler.py",
    # Extras classes - MUST be from utils/extras
    "BlobStorage": "toolboxv2/utils/extras/blobs.py",
    "Style": "toolboxv2/utils/extras/Style.py",
    "Spinner": "toolboxv2/utils/extras/Style.py",
    # Mod-specific classes - MUST be from mods/
    # Note: User exists in both types.py (dataclass) and models.py (pydantic)
    "User": ["toolboxv2/mods/CloudM/types.py", "toolboxv2/mods/CloudM/models.py"],
    "LocalUserData": "toolboxv2/mods/CloudM/AuthClerk.py",
    "StorageProvider": "toolboxv2/mods/CloudM/UserDataAPI.py",
    "ModDataClient": "toolboxv2/mods/CloudM/UserDataAPI.py",
}

EXPECTED_FUNCTIONS = {
    # Core functions - MUST be from correct source files
    "get_app": "toolboxv2/utils/system/getting_and_closing_app.py",
    "save_closing_app": "toolboxv2/utils/system/getting_and_closing_app.py",
    "override_main_app": "toolboxv2/utils/system/getting_and_closing_app.py",
    # mkdocs functions
    "create_docs_system": "toolboxv2/utils/extras/mkdocs.py",
    "add_to_app": "toolboxv2/utils/extras/mkdocs.py",
    # Blob storage functions
    "create_server_storage": "toolboxv2/utils/extras/blobs.py",
    "create_offline_storage": "toolboxv2/utils/extras/blobs.py",
}


def normalize_path(path: str) -> str:
    """Normalize path for comparison."""
    return path.replace("\\", "/").lower()


def path_matches(actual: str, expected: str) -> bool:
    """Check if actual path ends with expected path."""
    actual_norm = normalize_path(actual)
    expected_norm = normalize_path(expected)
    return actual_norm.endswith(expected_norm)


async def run_accuracy_test():
    project_root = Path(__file__).parent.parent.parent
    docs_root = project_root / "toolboxv2" / "docs"

    system = DocsSystem(
        project_root=project_root,
        docs_root=docs_root,
        include_dirs=["toolboxv2"],
    )

    print("=== Initializing DocsSystem ===")
    result = await system.initialize(force_rebuild=True)
    print(f"Status: {result['status']}")
    print(f"Code elements: {result['elements']}")
    print()

    # Test classes
    print("=== CLASS ACCURACY TEST ===")
    class_passed = 0
    class_failed = 0

    for class_name, expected_file in EXPECTED_CLASSES.items():
        lookup = await system.lookup_code(name=class_name, element_type="class")
        if lookup["results"]:
            actual_file = lookup["results"][0]["file"]
            # Handle multiple valid expected files
            expected_files = expected_file if isinstance(expected_file, list) else [expected_file]
            if any(path_matches(actual_file, ef) for ef in expected_files):
                print(f"  ✓ {class_name}: {normalize_path(actual_file).split('toolboxv2/')[-1]}")
                class_passed += 1
            else:
                print(f"  ✗ {class_name}: WRONG!")
                print(f"      Expected: {expected_files}")
                print(f"      Got:      {actual_file}")
                class_failed += 1
        else:
            print(f"  ✗ {class_name}: NOT FOUND!")
            class_failed += 1

    print(f"\nClasses: {class_passed}/{len(EXPECTED_CLASSES)} passed")
    print()

    # Test functions
    print("=== FUNCTION ACCURACY TEST ===")
    func_passed = 0
    func_failed = 0

    for func_name, expected_file in EXPECTED_FUNCTIONS.items():
        lookup = await system.lookup_code(name=func_name, element_type="function")
        if lookup["results"]:
            actual_file = lookup["results"][0]["file"]
            if path_matches(actual_file, expected_file):
                print(f"  ✓ {func_name}: {expected_file}")
                func_passed += 1
            else:
                print(f"  ✗ {func_name}: WRONG!")
                print(f"      Expected: {expected_file}")
                print(f"      Got:      {actual_file}")
                func_failed += 1
        else:
            print(f"  ✗ {func_name}: NOT FOUND!")
            func_failed += 1

    print(f"\nFunctions: {func_passed}/{len(EXPECTED_FUNCTIONS)} passed")
    print()

    # Summary
    total_passed = class_passed + func_passed
    total_tests = len(EXPECTED_CLASSES) + len(EXPECTED_FUNCTIONS)
    accuracy = (total_passed / total_tests) * 100

    print("=" * 50)
    print(f"TOTAL ACCURACY: {total_passed}/{total_tests} ({accuracy:.1f}%)")
    print("=" * 50)

    return accuracy == 100.0


if __name__ == "__main__":
    success = asyncio.run(run_accuracy_test())
    sys.exit(0 if success else 1)

