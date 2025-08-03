"""Flow for automatically creating git commit messages based on file changes using ISAA."""

import locale
import os
import subprocess
import sys
import argparse

from toolboxv2 import Spinner, remove_styles
from toolboxv2.mods.isaa.base.Agent.agent import EnhancedAgent

NAME = "AutoGitCommit"

def safe_decode(data: bytes) -> str:
    encodings = [sys.stdout.encoding, locale.getpreferredencoding(), 'utf-8', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')

async def run(app, *args, **kwargs):
    """Automatically create a git commit message based on file changes."""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--tags', nargs='*', default=[], help='Additional tags to add to the commit')
        parsed_args = parser.parse_args(args)
        
        # Initialize ISAA
        isaa = app.get_mod("isaa")

        # Get the current working directory
        cwd = os.getcwd()

        # Get the list of changed files with their status
        result = subprocess.run(['git', 'diff', '--name-status'], cwd=cwd, capture_output=True, text=True)
        changed_files_info = remove_styles(safe_decode(result.stdout.strip().encode())).split('\n') if result.stdout else []

        if not changed_files_info or changed_files_info == ['']:
            print({"success": True, "message": "No files changed."})
            return

        # Parse file changes with their status (A for added, M for modified, D for deleted)
        file_changes = []
        for line in changed_files_info:
            if line:
                status, file_path = line.split('\t', 1)
                if status.startswith('A'):
                    file_changes.append(f"Added: {file_path}")
                elif status.startswith('M'):
                    file_changes.append(f"Modified: {file_path}")
                elif status.startswith('D'):
                    file_changes.append(f"Deleted: {file_path}")

        if not file_changes:
            print({"success": True, "message": "No files changed."})
            return

        str_file_changes = "\n".join(file_changes)
        if len(str_file_changes) > 3700:
            str_file_changes = await isaa.mas_text_summaries(str_file_changes, ref="file changes")

        # Create detailed prompt for ISAA with context about changes
        agent: EnhancedAgent = await isaa.get_agent("GitCommitMessageGenerator")
        agent.amd.system_message = "You are a git commit message generator. Return only the commit message without any other text. Based on these file changes, generate a concise and descriptive git commit message"

        with Spinner("Generating commit message..."):
            commit_message = await isaa.init_task_completion(
                mini_task=str_file_changes,
                user_task="Generate a git commit message based on the file changes",
                agent_name="GitCommitMessageGenerator"
            )

        # Clean up the commit message (in case of extra text)
        commit_message = commit_message.strip().split('\n')[0]

        # Add tags to commit message if provided
        if parsed_args.tags:
            tags_str = ' '.join([f'#{tag}' for tag in parsed_args.tags])
            commit_message = f"{commit_message} {tags_str}"

        print("="*20)
        print(commit_message)
        print("=" * 20)

        # Create a local commit with the generated message
        subprocess.run(['git', 'add', '.'], cwd=cwd)
        subprocess.run(['git', 'commit', '-m', commit_message], cwd=cwd)

        return {"success": True, "message": commit_message}
    except Exception as e:
        print({"success": False, "error": str(e)})
        return {"success": False, "error": str(e)}