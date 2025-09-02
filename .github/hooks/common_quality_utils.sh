#!/bin/sh

# common_quality_utils.sh
# Contains shared functions for Git hooks to run quality checks.

# === Konfiguration ===
# Tools definition: "COMMAND_TO_RUN|Friendly Name|IS_CRITICAL (true/false)"
# IS_CRITICAL: if true and check fails, prepare-commit-msg will abort commit.
#
# Example:
# CHECKS_DEFINITIONS="
# ruff check .|Ruff Linter|true
# bandit -r .|Bandit Security|true
# safety check --full-report|Safety Dependencies|false
# "
# For your tools:
CHECKS_DEFINITIONS="
ruff check .\toolboxv2\|Ruff|true
safety scan --full-report|Safety|true
deptry .\toolboxv2\|Deptry|true
" # Assuming both are critical for commit

VERSIONS_CMD="tb -v"
AUTO_SUMMARY_CMD="tb -c isaa auto_commit_msg"

# Temporary directory for check outputs, typically within .git
# This allows prepare-commit-msg to save results for post-commit to use.
GIT_DIR_TOPLEVEL=$(git rev-parse --show-toplevel)/.git
TEMP_CHECK_OUTPUT_DIR="$GIT_DIR_TOPLEVEL/tmp_check_outputs"

# === Helper Functions ===

# Function to run a single check and store its output and exit code
# Args: $1: Command string, $2: Check Name, $3: Temp output file base
run_single_check_and_store() {
  _cmd="$1"
  _name="$2" # Check name (e.g., "Ruff Linter & Security")
  _output_base_path="$3" # FULL base path (e.g., $TEMP_CHECK_OUTPUT_DIR/Ruff Linter & Security)

  echo "[check-runner] Running $_name..."
  echo "[check-runner] CMD: [$_cmd]"
  echo "[check-runner] Output base path: [$_output_base_path]" # DEBUG

  # Create directory for the output files if _name contains slashes (though unlikely here)
  # dirname_output_base_path=$(dirname "$_output_base_path")
  # mkdir -p "$dirname_output_base_path" # Usually $TEMP_CHECK_OUTPUT_DIR is already created

  output_content=$(sh -c "$_cmd" 2>&1)
  _exit_code=$?

  echo "$output_content" > "${_output_base_path}.output"
  if [ $? -ne 0 ]; then echo "ERROR writing to ${_output_base_path}.output" >&2; fi

  echo "$_exit_code" > "${_output_base_path}.exitcode"
  if [ $? -ne 0 ]; then echo "ERROR writing to ${_output_base_path}.exitcode" >&2; fi

  return $_exit_code
}

# Function to extract the version summary part from tb -v output
# Args: $1: Raw output from tb -v
get_versions_summary_part() {
  _raw_output="$1"

  # Pre-process to remove ANSI escape codes like [K
  # \x1B is ESC. Escape sequences are typically ESC [ ...
  # This sed command removes ESC [ sequences followed by any characters until m, K, J, etc.
  _processed_output=$(printf '%s\n' "$_raw_output" | sed 's/\x1B\[[0-9;]*[mKJHfABCDsu]//g')

  # Debug: See the processed output
  # printf "DEBUG get_versions_summary_part: Processed input (ANSI stripped):\n%s\nEND DEBUG PROCESSED INPUT\n" "$_processed_output" >&2

  echo "$_processed_output" | awk '
    BEGIN {
      in_version_block = 0
      blank_line_count = 0
      # FS = "[ \t]*:[ \t]*" # Optional: If you only want to match "Key : Value" lines
    }

    # Start pattern
    /^-+ Version -+$/ {
      in_version_block = 1
      blank_line_count = 0 # Reset blank line counter
      next # Skip printing the "--- Version ---" line itself
    }

    # Stop conditions if already in the version block
    in_version_block {
      # Stop if we see "Building State data" or "working on"
      if (/^Building State data:/ || /^working on/) {
        in_version_block = 0
        exit
      }

      # Check for blank lines
      if (/^[[:space:]]*$/) {
        blank_line_count++
        if (blank_line_count >= 2) { # Stop after 2 or more consecutive blank lines
          in_version_block = 0
          exit
        }
        next # Skip printing this blank line, but continue checking
      } else {
        blank_line_count = 0 # Reset if not a blank line
      }

      # If we are here, it means we are in the block, it is not a stop pattern,
      # and it is not an excessive blank line. So, print it.
      # You could add more specific matching here if you only want lines with " : "
      # For example: if ($0 ~ /[[:space:]]:[[:space:]]/) print
      print
    }
  '
}
# Function to escape text for sed replacement
escape_for_sed() {
  # Handles common sed metacharacters: \, &, /, the chosen delimiter |, and newlines
  # The delimiter used in the main sed command is '|'.
  printf '%s\n' "$1" | sed \
    -e 's/\\/\\\\/g' \
    -e 's/&/\\&/g' \
    -e 's/\//\\\//g' \
    -e 's/|/\\|/g' \
    -e ':a;N;$!ba;s/\n/\\n/g'
}

# Function to clean up temporary check outputs
cleanup_temp_check_outputs() {
  if [ -d "$TEMP_CHECK_OUTPUT_DIR" ]; then
    echo "[check-runner] Cleaning up temporary check files from $TEMP_CHECK_OUTPUT_DIR..."
    rm -rf "$TEMP_CHECK_OUTPUT_DIR"
  fi
}
