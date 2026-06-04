#!/usr/bin/env python3
import locale
import os
import subprocess
import sys
import json

from toolboxv2 import Spinner, remove_styles, ApiResult
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent

NAME = "auto_git_commit"


def safe_decode(data: bytes) -> str:
    """Decodes bytes to a string using a list of common encodings."""
    encodings = [sys.stdout.encoding, locale.getpreferredencoding(), 'utf-8', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


def token_estimate(text: str) -> int:
    """
    Rough token estimate for text.
    Uses OpenAI heuristic: ~4 characters ≈ 1 token.
    """
    return max(1, len(text) // 4)


def chunk_list(items: list, chunk_size: int) -> list:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def run(app, _, tags: str | None = None, summarize: bool = False, **kwargs):
    """
    Automatically create a git commit message based on file changes.
    Improved to handle arbitrarily large change sets via token-based size management,
    chunking, and progressive summarization.

    Args:
        app: The application instance.
        tags (str, optional): A list of tags to add to the commit message. Defaults to None.
        summarize (bool, optional): Force summarization of file changes. Defaults to False.
        max_tokens (int, optional): Maximum token budget for LLM context. Defaults to 8000.
        chunk_size (int, optional): Number of files to process per chunk. Defaults to 20.
        push (bool, optional): Automatically push after commit. Defaults to True.
    """
    try:
        # Initialize ISAA
        isaa = app.get_mod("isaa")

        # Get the current working directory
        from toolboxv2 import tb_root_dir
        cwd = tb_root_dir.parent

        # Configuration
        max_tokens = kwargs.get('max_tokens', 8000)
        chunk_size = kwargs.get('chunk_size', 20)
        max_summarization_iterations = kwargs.get('max_summarization_iterations', 3)
        auto_push = kwargs.get('push', True)

        # Get the list of changed files with their status
        result = subprocess.run(['git', 'diff', '--name-status'], cwd=cwd, capture_output=True)
        changed_files_info = remove_styles(safe_decode(result.stdout).strip()).split('\n') if result.stdout else []

        if not changed_files_info or changed_files_info == ['']:
            print(json.dumps({"success": True, "message": "No modified files to commit."}))
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
            print(json.dumps({"success": True, "message": "No modified files to commit."}))
            return

        # Stage only the detected modified files
        for file_path in files_to_stage:
            subprocess.run(['git', 'add', file_path], cwd=cwd)

        str_file_changes = "\n\n".join(file_changes_for_prompt)

        # STRATEGY 1: Chunking for very large change sets
        # If we have many files, process them in chunks
        if len(file_changes_for_prompt) > chunk_size:
            print(f"Processing {len(file_changes_for_prompt)} files in chunks...")
            chunks = chunk_list(file_changes_for_prompt, chunk_size)
            chunk_summaries = []

            for i, chunk in enumerate(chunks):
                chunk_text = "\n\n".join(chunk)

                # Summarize each chunk
                chunk_summary = await isaa.mas_text_summaries(
                    chunk_text,
                    ref=f"file changes chunk {i + 1}/{len(chunks)}"
                )
                chunk_summaries.append(chunk_summary)

            # Combine summaries
            str_file_changes = "\n\n".join(chunk_summaries)

        # STRATEGY 2: Progressive summarization for token budget
        # Keep summarizing until we fit within the token budget
        current_tokens = token_estimate(str_file_changes)
        iteration = 0

        while current_tokens > max_tokens and iteration < max_summarization_iterations:
            print(
                f"Token count ({current_tokens}) exceeds budget ({max_tokens}), summarizing... (iteration {iteration + 1}/{max_summarization_iterations})")

            str_file_changes = await isaa.mas_text_summaries(
                str_file_changes,
                ref="file changes - progressive summarization"
            )

            current_tokens = token_estimate(str_file_changes)
            iteration += 1

            # Safety check: if summarization isn't helping, break
            if iteration > 0 and current_tokens > max_tokens * 1.5:
                print("Warning: Summarization not reducing size effectively, using fallback strategy")
                break

        # STRATEGY 3: Fallback to simple file list if still too large
        if current_tokens > max_tokens:
            print("Changes still too large after summarization, using simple file list...")
            simple_list = []
            for entry in file_changes_for_prompt:
                # Extract just the filename from each entry
                lines = entry.split('\n')
                if lines:
                    filename = lines[0].replace('Changes for file: ', '').replace('Deleted file: ', '').replace(
                        'New file added: ', '')
                    simple_list.append(f"- {filename}")
            str_file_changes = "Files changed:\n" + "\n".join(simple_list)

        # Final forced summarization if requested
        if summarize:
            str_file_changes = await isaa.mas_text_summaries(str_file_changes, ref="forced summarization")

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
                user_task="Generate a git commit message based on the following file content changes. Include key details!",
                agent_name="GitCommitMessageGenerator",
                use_blitz=True,
            )

        if isinstance(commit_message, ApiResult):
            commit_message = commit_message.as_result().get()

        # Clean up the commit message
        commit_message = commit_message.strip()

        # Add tags to commit message if provided
        if tags is not None:
            tags_str = tags
            commit_message = f"{commit_message} {tags_str}"

        print("=" * 20)
        print(commit_message)
        print("=" * 20)

        # Create a local commit with the generated message
        subprocess.run(['git', 'commit', '-m', commit_message], cwd=cwd)

        # Automatic push (if enabled)
        if auto_push:
            print("Pushing to remote...")
            push_result = subprocess.run(
                ['git', 'push'],
                cwd=cwd,
                capture_output=True,
                text=True
            )

            if push_result.returncode != 0:
                error_msg = push_result.stderr.strip()
                print(json.dumps({
                    "success": False,
                    "error": f"Commit succeeded but push failed: {error_msg}",
                    "commit_message": commit_message
                }))
                return {
                    "success": False,
                    "error": error_msg,
                    "commit_message": commit_message
                }

            print("Successfully pushed to remote.")
            return {"success": True, "message": commit_message, "pushed": True}

        return {"success": True, "message": commit_message, "pushed": False}

    except Exception as e:
        error_msg = str(e)
        print(json.dumps({"success": False, "error": error_msg}))
        app.debug_rains(e)
        return {"success": False, "error": error_msg}


if __name__ == "__main__":
    from toolboxv2 import get_app
    import asyncio

    asyncio.run(run(get_app(), None))
