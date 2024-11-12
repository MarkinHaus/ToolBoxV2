import os
import re
import zipfile
import git
import shutil
from datetime import datetime
from typing import Optional, List, Tuple

from tqdm import tqdm

from ..singelton_class import Singleton


class GitZipManager(metaclass=Singleton):
    def __init__(self, working_dir: str):
        """
        Initialize the GitZipManager with a working directory.

        Args:
            working_dir (str): Path to the working directory
        """
        self.working_dir = os.path.abspath(working_dir)
        self.repo, f = self._init_git_repo()
        if f:
            self.compress_all()

    def _init_git_repo(self) -> (git.Repo,bool):
        """Initialize Git repository if it doesn't exist."""
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        f = False
        try:
            repo = git.Repo(self.working_dir)
        except git.exc.InvalidGitRepositoryError:
            repo = git.Repo.init(self.working_dir)
            # Initial commit
            repo.index.add([])
            repo.index.commit("Initial commit")
            f = True
        return repo, f

    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Parse filename into components: name, app_version, file_version
        Format: RST$name&vapp_version§file_version
        """
        pattern = r"RST\$(.+)&v(.+)§(.+)"
        match = re.match(pattern, filename)
        if match:
            return match.groups()
        raise ValueError(f"Invalid filename format: {filename}")

    def _create_base_filename(self, name: str, app_version: str) -> str:
        """Create base filename without file version."""
        return f"RST${name}&v{app_version}.zip"

    def compress_all(self):
        """
        Compress all files in the working directory with the same name and app version
        into single ZIP files.
        """
        # Group files by name and app version
        files_dict = {}
        for filename in tqdm(os.listdir(self.working_dir), desc="Grouping"):
            if not filename.startswith("RST$"):
                continue

            try:
                name, app_version, file_version = self._parse_filename(
                    os.path.splitext(filename)[0]
                )
                key = (name, app_version)
                if key not in files_dict:
                    files_dict[key] = []
                files_dict[key].append((filename, file_version))
            except ValueError:
                continue

        # Create consolidated ZIP files
        for (name, app_version), files in tqdm(files_dict.items(), desc="Create consolidated ZIP files"):
            base_filename = self._create_base_filename(name, app_version)
            base_path = os.path.join(self.working_dir, base_filename)

            with zipfile.ZipFile(base_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for filename, file_version in files:
                    file_path = os.path.join(self.working_dir, filename)
                    zf.write(file_path, os.path.basename(filename))
                    os.remove(file_path)  # Remove original file

            # Commit changes to Git
            self.repo.index.add([base_filename])
            self.repo.index.commit(
                f"Compressed files for {name} v{app_version}"
            )

    def add_file_version(self, filename: str):
        """
        Add a specific file version to its corresponding base ZIP file.

        Args:
            filename (str): Filename in format RST$name&vapp_version§file_version
        """
        name, app_version, file_version = self._parse_filename(
            os.path.splitext(filename)[0]
        )
        base_filename = self._create_base_filename(name, app_version)
        source_path = os.path.join(self.working_dir, filename)
        base_path = os.path.join(self.working_dir, base_filename)

        # Create or update base ZIP file
        if os.path.exists(base_path):
            # Create temporary ZIP with updated content
            temp_path = base_path + '.temp'
            shutil.copy2(base_path, temp_path)

            with zipfile.ZipFile(temp_path, 'a', zipfile.ZIP_DEFLATED) as zf:
                zf.write(source_path, os.path.basename(filename))

            os.remove(base_path)
            os.rename(temp_path, base_path)
        else:
            with zipfile.ZipFile(base_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(source_path, os.path.basename(filename))

        # Remove original file and commit changes
        os.remove(source_path)
        self.repo.index.add([base_filename])
        self.repo.index.commit(
            f"Added {filename} to {base_filename}"
        )

    def extract_version(self, name: str, app_version: str,
                        file_version: Optional[str] = None) -> str:
        """
        Extract a specific version of a file from Git history.

        Args:
            name (str): Base name of the file
            app_version (str): Application version
            file_version (str, optional): Specific file version to extract
                                        If None, extracts the latest version

        Returns:
            str: Path to the extracted file
        """
        base_filename = self._create_base_filename(name, app_version)

        # Get the latest commit containing the file
        commits = list(self.repo.iter_commits(paths=base_filename))
        if not commits:
            raise ValueError(f"No history found for {base_filename}")

        latest_commit = commits[0]

        # Create temporary directory for extraction
        temp_dir = os.path.join(self.working_dir, "temp_extract")
        os.makedirs(temp_dir, exist_ok=True)

        # Get file from Git history
        base_blob = latest_commit.tree / base_filename
        with open(os.path.join(temp_dir, base_filename), 'wb') as f:
            f.write(base_blob.data_stream.read())

        # Extract specific version or latest
        target_filename = None
        with zipfile.ZipFile(os.path.join(temp_dir, base_filename), 'r') as zf:
            if file_version:
                target_filename = f"RST${name}&v{app_version}§{file_version}.zip"
                if target_filename not in zf.namelist():
                    raise ValueError(f"Version {file_version} not found")
            else:
                # Get the latest version by sorting file versions
                files = [f for f in zf.namelist() if f.startswith(f"RST${name}&v{app_version}§")]
                if not files:
                    raise ValueError("No versions found")
                target_filename = sorted(files)[-1]

            # Extract the target file
            zf.extract(target_filename, temp_dir)

        # Cleanup and return path to extracted file
        os.remove(os.path.join(temp_dir, base_filename))
        return os.path.join(temp_dir, target_filename)


# Example usage
if __name__ == "__main__":
    # Initialize manager
    # = GitZipManager("./zip_versions")

    # Example: Add a new file version
    # manager.add_file_version("RST$talk&v0.1.20§0.0.1.zip")

    # Example: Compress all files
    #manager.compress_all()

    # Example: Extract specific version
    # extracted_path = manager.extract_version("talk", "0.1.20", "0.0.1")
    # print(f"Extracted file path: {extracted_path}")
    pass
