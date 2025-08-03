import locale
import os
import subprocess
import sys
from typing import Optional

from toolboxv2 import Spinner, remove_styles
from toolboxv2.mods.isaa.base.Agent.agent import EnhancedAgent

NAME = "AutoGitCommit"

def safe_decode(data: bytes) -> str:
    """Decodes bytes to a string using a list of common encodings."""
    encodings = [sys.stdout.encoding, locale.getpreferredencoding(), 'utf-8', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')

async def run(app, args_sto, tags: Optional[str] = None, summarize: bool = False, **kwargs):
    """
    Automatically create a git commit message based on file changes.

    Args:
        app: The application instance.
        tags (str, optional): A list of tags to add to the commit message. Defaults to None.
        summarize (bool, optional): Force summarization of file changes. Defaults to False.
    """
    try:
        # Initialize ISAA
        isaa = app.get_mod("isaa")

        # Get the current working directory
        from toolboxv2 import __init_cwd__
        cwd = __init_cwd__

        # Get the list of changed files with their status
        result = subprocess.run(['git', 'diff', '--name-status'], cwd=cwd, capture_output=True)
        changed_files_info = remove_styles(safe_decode(result.stdout).strip()).split('\n') if result.stdout else []

        if not changed_files_info or changed_files_info == ['']:
            print({"success": True, "message": "No modified files to commit."})
            return

        # Parse file changes with their status and prepare for staging
        file_changes_for_prompt = []
        files_to_stage = []
        for line in changed_files_info:
            if line and line.startswith('M'):  # Only process modified files
                status, file_path = line.split('\t', 1)
                files_to_stage.append(file_path)

                # Get the contextual diff for the modified file.
                # The -U3 option provides 3 lines of context before and after each change.
                diff_result = subprocess.run(['git', 'diff', '-U3', file_path], cwd=cwd, capture_output=True)
                diff_content = safe_decode(diff_result.stdout)

                # Add the file path and its diff content to the list for the prompt
                prompt_entry = f"Changes for file: {file_path}\n---\n```diff\n{diff_content}\n```\n---"
                file_changes_for_prompt.append(prompt_entry)


        if not file_changes_for_prompt:
            print({"success": True, "message": "No modified files to commit."})
            return

        # Stage only the detected modified files
        for file_path in files_to_stage:
            subprocess.run(['git', 'add', file_path], cwd=cwd)

        str_file_changes = "\n\n".join(file_changes_for_prompt)

        # Summarize if the text is too long or if summarization is forced
        if summarize or len(str_file_changes) > 3700:
            str_file_changes = await isaa.mas_text_summaries(str_file_changes, ref="file changes with context")

        # Create detailed prompt for ISAA with context about changes
        agent: EnhancedAgent = await isaa.get_agent("GitCommitMessageGenerator")
        agent.amd.system_message = (
            "You are a git commit message generator. Return only the commit message without any other text. "
            "Based on the following file changes, which include a diff with context, "
            "generate a concise and descriptive git commit message."
        )

        with Spinner("Generating commit message..."):
            commit_message = await isaa.mini_task_completion(
                mini_task=str_file_changes,
                user_task="Generate a git commit message based on the following file content changes.",
                agent_name="GitCommitMessageGenerator"
            )

        # Clean up the commit message
        commit_message = commit_message.strip().split('\n')[0]

        # Add tags to commit message if provided
        print(tags)
        if tags is not None:
            tags_str = tags
            commit_message = f"{commit_message} {tags_str}"

        print("="*20)
        print(commit_message)
        print("=" * 20)

        # Create a local commit with the generated message
        subprocess.run(['git', 'commit', '-m', commit_message], cwd=cwd)

        return {"success": True, "message": commit_message}
    except Exception as e:
        print({"success": False, "error": str(e)})
        app.debug_rains(e)
        return {"success": False, "error": str(e)}
