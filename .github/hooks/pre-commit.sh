#!/bin/sh

# GIT HOOK: post-commit
#
# Runs after a successful commit.
# Generates a detailed report from check outputs saved by prepare-commit-msg.

# Source the common utility functions
HOOK_DIR=$(dirname "$0")
# shellcheck source=./common_quality_utils.sh
. "$HOOK_DIR/common_quality_utils.sh" || {
  echo "Error: common_quality_utils.sh not found or failed to source." >&2
  exit 0 # Don't fail the post-commit for this, just log
}

# === Konfiguration ===
REPORT_DIR_NAME="local-reports" # Relative to project root
LATEST_REPORT_NAME="latest-report.txt"

# Determine project root and report directory
PROJECT_ROOT=$(git rev-parse --show-toplevel)
REPORT_DIR="$PROJECT_ROOT/$REPORT_DIR_NAME"
mkdir -p "$REPORT_DIR" # Ensure report directory exists

TIMESTAMP=$(date +"%Y-%m-%d-%H%M%S")
CURRENT_REPORT_FILE="$REPORT_DIR/$TIMESTAMP-report.txt"
LATEST_REPORT_LINK="$REPORT_DIR/$LATEST_REPORT_NAME"

echo "[post-commit] Generating quality report..."

if [ ! -d "$TEMP_CHECK_OUTPUT_DIR" ] || [ -z "$(ls -A "$TEMP_CHECK_OUTPUT_DIR")" ]; then
  echo "[post-commit] No temporary check outputs found from prepare-commit-msg. Skipping detailed report."
  # Optionally, you could re-run non-critical checks here if needed.
  cleanup_temp_check_outputs # Clean just in case
  exit 0
fi

# === Start Report File ===
echo "Quality Check Report - $TIMESTAMP" > "$CURRENT_REPORT_FILE"
echo "Commit: $(git rev-parse HEAD)" >> "$CURRENT_REPORT_FILE"
echo "========================================" >> "$CURRENT_REPORT_FILE"
echo "" >> "$CURRENT_REPORT_FILE"

ALL_CHECKS_PASSED_IN_PREPARE=true

# Process saved check outputs
# We need to iterate through defined checks to maintain order and friendly names
echo "$CHECKS_DEFINITIONS" | while IFS='|' read -r CMD_STR NAME_STR IS_CRITICAL_STR; do
  _name_trimmed=$(echo "$NAME_STR" | awk '{$1=$1};1')
  if [ -n "$_name_trimmed" ]; then
    _output_file="$TEMP_CHECK_OUTPUT_DIR/${_name_trimmed}.output"
    _exitcode_file="$TEMP_CHECK_OUTPUT_DIR/${_name_trimmed}.exitcode"

    if [ -f "$_output_file" ] && [ -f "$_exitcode_file" ]; then
      _output_content=$(cat "$_output_file")
      _exit_code=$(cat "$_exitcode_file")

      echo "[$_name_trimmed] Exit Code: $_exit_code" >> "$CURRENT_REPORT_FILE"
      echo "[$_name_trimmed] Output:" >> "$CURRENT_REPORT_FILE"
      echo "$_output_content" >> "$CURRENT_REPORT_FILE"
      echo "" >> "$CURRENT_REPORT_FILE"
      if [ "$_exit_code" -ne 0 ]; then
        ALL_CHECKS_PASSED_IN_PREPARE=false
      fi
    else
      echo "[$_name_trimmed] Data not found in temp storage." >> "$CURRENT_REPORT_FILE"
      echo "" >> "$CURRENT_REPORT_FILE"
      ALL_CHECKS_PASSED_IN_PREPARE=false # Assume failure if data is missing
    fi
  fi
done

# Process Versions output
_versions_output_file="$TEMP_CHECK_OUTPUT_DIR/Versions.output"
_versions_exitcode_file="$TEMP_CHECK_OUTPUT_DIR/Versions.exitcode"
if [ -f "$_versions_output_file" ] && [ -f "$_versions_exitcode_file" ]; then
  _versions_raw_output=$(cat "$_versions_output_file")
  _versions_exit_code=$(cat "$_versions_exitcode_file")
  echo "[Versions] Exit Code: $_versions_exit_code" >> "$CURRENT_REPORT_FILE"
  echo "[Versions] Output:" >> "$CURRENT_REPORT_FILE"
  echo "$_versions_raw_output" >> "$CURRENT_REPORT_FILE"
  echo "" >> "$CURRENT_REPORT_FILE"
else
  echo "[Versions] Data not found in temp storage." >> "$CURRENT_REPORT_FILE"
  echo "" >> "$CURRENT_REPORT_FILE"
fi

# === Erstelle/Aktualisiere "latest" Report Symlink ===
# Use relative path for symlink for better portability if project moves
cd "$REPORT_DIR" || exit 1 # Enter report dir to make relative symlink
ln -snf "$(basename "$CURRENT_REPORT_FILE")" "$LATEST_REPORT_NAME"
cd "$PROJECT_ROOT" || exit 1 # Go back to project root

echo "[post-commit] Report created: $CURRENT_REPORT_FILE"
echo "[post-commit] Latest report link: $LATEST_REPORT_LINK"

# === Clean up temporary files ===
cleanup_temp_check_outputs

# === Abschlussmeldung ===
if [ "$ALL_CHECKS_PASSED_IN_PREPARE" = false ]; then
  # Note: This reflects status from prepare-commit-msg. Commit itself succeeded.
  echo "[post-commit] ⚠️ Note: One or more checks had issues during pre-commit phase. Review report $LATEST_REPORT_LINK"
else
  echo "[post-commit] ✅ All checks noted in pre-commit phase passed. Report generated."
fi

exit 0
