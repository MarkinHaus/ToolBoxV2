import locale
import os
import subprocess
import sys

from toolboxv2 import Spinner, remove_styles, ApiResult
from toolboxv2.mods.isaa.base.Agent.agent import FlowAgent

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

async def run(app, args_sto, tags: str | None = None, summarize: bool = False, **kwargs):
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
            if not line:
                continue

            status, file_path = line.split('\t', 1)

            if status == 'M':  # Modified files
                files_to_stage.append(file_path)

                diff_result = subprocess.run(['git', 'diff', '-U3', file_path], cwd=cwd, capture_output=True)
                diff_content = safe_decode(diff_result.stdout)

                prompt_entry = f"Changes for file: {file_path}\n---\n```diff\n{diff_content}\n```\n---"
                file_changes_for_prompt.append(prompt_entry)

            elif status == 'D':  # Deleted files
                prompt_entry = f"Deleted file: {file_path}\n---\n(This file was deleted and has no content.)\n---"
                file_changes_for_prompt.append(prompt_entry)

            elif status == 'A':  # Newly added files
                files_to_stage.append(file_path)

                try:
                    with open(os.path.join(cwd, file_path), encoding='utf-8') as f:
                        file_content = f.read()
                except Exception as e:
                    file_content = f"(Could not read file: {e})"

                prompt_entry = f"New file added: {file_path}\n---\n```{file_path.split('.')[-1]}\n{file_content}\n```\n---"
                file_changes_for_prompt.append(prompt_entry)

        if not file_changes_for_prompt:
            print({"success": True, "message": "No modified files to commit."})
            return

        # Stage only the detected modified files
        for file_path in files_to_stage:
            subprocess.run(['git', 'add', file_path], cwd=cwd)

        str_file_changes = "\n\n".join(file_changes_for_prompt)

        # Summarize if the text is too long or if summarization is forced
        if summarize or len(str_file_changes) > 370000:
            str_file_changes = await isaa.mas_text_summaries(str_file_changes, ref="file changes with context")

        # Create detailed prompt for ISAA with context about changes
        agent: FlowAgent = await isaa.get_agent("GitCommitMessageGenerator")
        agent.amd.system_message = (
            "You are a git commit message generator. Return only the commit message without any other text. "
            "Based on the following file changes, which include a diff with context, "
            "generate a concise and descriptive git commit message."
        )

        with Spinner("Generating commit message..."):
            commit_message = await isaa.mini_task_completion(
                mini_task=str_file_changes,
                user_task="Generate a git commit message based on the following file content changes. with key details!",
                agent_name="GitCommitMessageGenerator"
            )
        if isinstance(commit_message, ApiResult):
            commit_message = commit_message.as_result().get()

        # Clean up the commit message
        commit_message = commit_message.strip()

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
