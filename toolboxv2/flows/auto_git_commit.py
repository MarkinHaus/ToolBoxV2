\"\"\"Flow for automatically creating git commit messages based on file changes using ISAA.\"\"\"

import os
import subprocess
from isaa import ISAA

NAME = \"AutoGitCommit\"

async def run(app, *args, **kwargs):
    \"\"\"Automatically create a git commit message based on file changes.\"\"\"
    try:
        # Initialize ISAA
        isaa = ISAA()
        
        # Get the current working directory
        cwd = os.getcwd()
        
        # Get the list of changed files
        result = subprocess.run(['git', 'diff', '--name-only'], cwd=cwd, capture_output=True, text=True)
        changed_files = result.stdout.strip().split('\n') if result.stdout else []
        
        if not changed_files or changed_files == ['']:
            return {\"success\": True, \"message\": \"No files changed.\"}
        
        # Use ISAA to generate a commit message based on file changes
        prompt = f\"Generate a concise git commit message based on these changed files: {', '.join(changed_files)}\"
        commit_message = await isaa.generate(prompt)
        
        # Commit the changes
        subprocess.run(['git', 'add', '.'], cwd=cwd)
        subprocess.run(['git', 'commit', '-m', commit_message], cwd=cwd)
        
        return {\"success\": True, \"message\": f\"Committed with message: {commit_message}\"}
    except Exception as e:
        return {\"success\": False, \"error\": str(e)}