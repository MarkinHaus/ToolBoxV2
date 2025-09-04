import os
import subprocess
import sys


# Function 1: Scan a folder for Python files and generate requirements using pipreqs
def generate_requirements(folder: str, output_file: str):
    """Generates requirements.txt for the specified folder using pipreqs."""
    print(folder, output_file, os.path.abspath(os.curdir))
    print("Not Implemented ")
    """try:
        from pipreqs.pipreqs import get_all_imports
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "pipreqs"], check=True)
    from pipreqs.pipreqs import get_all_imports
    imports = set(get_all_imports(os.path.abspath(folder)))
    imports.remove('toolboxv2') if 'toolboxv2' in imports else None
    with open(os.path.abspath(output_file), "w") as f:
        f.write("\n".join(imports))"""


def run_pipeline(base_dir: str):
    """Runs the entire pipeline to generate requirements files."""
    toolbox_path = os.path.join(base_dir, "toolboxv2")
    utils_path = os.path.join(toolbox_path, "utils")
    mini_req_file = os.path.join(base_dir, "requirements_mini.txt")
    extras_req_file = os.path.join(base_dir, "requirements_tests.txt")

    # Step 1: Generate minimal requirements
    print("Step 1/2: ")
    generate_requirements(utils_path, mini_req_file)

    # Step 2: Generate extended requirements
    print("Step 2/2: ")
    extras_path = os.path.join(toolbox_path, "tests")
    generate_requirements(extras_path, extras_req_file)
