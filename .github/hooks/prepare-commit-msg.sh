#!/bin/sh

# GIT HOOK: prepare-commit-msg
#
# Modifies the commit message file to include check statuses and auto-summary.
# Aborts commit if critical checks fail.
#
# Args:
#   $1: Name of the file that contains the commit log message.
#   $2: Source of the commit message (e.g., "message", "template", "merge", "squash", "commit").
#   $3: SHA-1 of the commit, if amending an existing commit (optional).

COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"
# SHA1="$3" # Unused for now, but available

# Source the common utility functions
HOOK_DIR=$(dirname "$0")
# shellcheck source=./common_quality_utils.sh
. "$HOOK_DIR/common_quality_utils.sh" || {
  echo "Error: common_quality_utils.sh not found or failed to source." >&2
  exit 1
}

# --- Configuration ---
PLACEHOLDER_CHECKS_SUMMARY="<#>"      # Placeholder for check statuses
PLACEHOLDER_AUTO_SUMMARY="<sum>"    # Placeholder for auto-generated summary
APPEND_IF_NO_PLACEHOLDER_CHECKS=true # If true, append check summary if <#> not found
APPEND_IF_NO_PLACEHOLDER_SUM=false   # If true, append auto-summary if <sum> not found (usually not desired)


# --- Script Logic ---
echo "[DEBUG prepare-commit-msg] Starting hook for source: $COMMIT_SOURCE, file: $COMMIT_MSG_FILE"

case "$COMMIT_SOURCE" in
  message|template|"")
    echo "[DEBUG prepare-commit-msg] Proceeding for commit source: $COMMIT_SOURCE"
    ;;
  *)
    echo "[DEBUG prepare-commit-msg] Skipping for commit source: $COMMIT_SOURCE"
    cleanup_temp_check_outputs
    exit 0
    ;;
esac

# ... (mkdir $TEMP_CHECK_OUTPUT_DIR) ...

echo "$CHECKS_DEFINITIONS" | while IFS='|' read -r CMD_STR NAME_STR IS_CRITICAL_STR; do
  # ... (inside the loop)
  run_single_check_and_store "$CMD_STR_TRIMMED" "$NAME_STR_TRIMMED" "$TEMP_CHECK_OUTPUT_DIR/$NAME_STR_TRIMMED"
  check_exit_code=$?
  echo "[DEBUG prepare-commit-msg] Check: $NAME_STR_TRIMMED, Exit: $check_exit_code"

  if [ "$check_exit_code" -ne 0 ]; then
    ALL_CHECKS_OK=false
    SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG $NAME_STR_TRIMMED❌"
    # ...
  else
    SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG $NAME_STR_TRIMMED✅"
  fi
  echo "[DEBUG prepare-commit-msg] Intermediate SUMMARY_LOG_MSG: [$SUMMARY_LOG_MSG]"
done

# ... (Versions Check) ...
echo "[DEBUG prepare-commit-msg] Raw Versions Output: [$VERSIONS_RAW_OUTPUT]"
VERSIONS_SUMMARY_PART_TEXT=$(get_versions_summary_part "$VERSIONS_RAW_OUTPUT")
echo "[DEBUG prepare-commit-msg] Parsed Versions Summary: [$VERSIONS_SUMMARY_PART_TEXT]"

if [ -n "$VERSIONS_SUMMARY_PART_TEXT" ]; then
  SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG
Version Info:
$VERSIONS_SUMMARY_PART_TEXT"
fi
echo "[DEBUG prepare-commit-msg] Final SUMMARY_LOG_MSG before cleaning: [$SUMMARY_LOG_MSG]"

# --- Update Commit Message File ---
ORIGINAL_MSG_CONTENT=$(cat "$COMMIT_MSG_FILE")
NEW_MSG_CONTENT="$ORIGINAL_MSG_CONTENT"
echo "[DEBUG prepare-commit-msg] Original commit message content: [$ORIGINAL_MSG_CONTENT]"


CLEANED_SUMMARY_LOG_MSG=$(echo "$SUMMARY_LOG_MSG" | awk '{$1=$1};1' | sed '/^$/d')
echo "[DEBUG prepare-commit-msg] CLEANED_SUMMARY_LOG_MSG: [$CLEANED_SUMMARY_LOG_MSG]"

ESCAPED_SUMMARY_FOR_SED=$(escape_for_sed "$CLEANED_SUMMARY_LOG_MSG")
echo "[DEBUG prepare-commit-msg] ESCAPED_SUMMARY_FOR_SED: [$ESCAPED_SUMMARY_FOR_SED]"


if grep -qF "$PLACEHOLDER_CHECKS_SUMMARY" "$COMMIT_MSG_FILE"; then # -F for fixed string grep
  echo "[DEBUG prepare-commit-msg] Found placeholder '$PLACEHOLDER_CHECKS_SUMMARY'. Replacing."
  NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_CHECKS_SUMMARY|$ESCAPED_SUMMARY_FOR_SED|g")
elif [ "$APPEND_IF_NO_PLACEHOLDER_CHECKS" = true ] && [ -n "$CLEANED_SUMMARY_LOG_MSG" ]; then
  echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_CHECKS_SUMMARY' not found. Appending check status."
  NEW_MSG_CONTENT=$(printf '%s\n\nCheck Status:\n%s' "$NEW_MSG_CONTENT" "$CLEANED_SUMMARY_LOG_MSG")
else
  echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_CHECKS_SUMMARY' not found and not appending."
fi
echo "[DEBUG prepare-commit-msg] NEW_MSG_CONTENT after <#> processing: [$NEW_MSG_CONTENT]"

# ... (rest of <sum> processing and final decision) ...
# Before writing to file:
echo "[DEBUG prepare-commit-msg] Attempting to write NEW_MSG_CONTENT to $COMMIT_MSG_FILE: [$NEW_MSG_CONTENT]"
printf '%s\n' "$NEW_MSG_CONTENT" > "$COMMIT_MSG_FILE"
echo "[DEBUG prepare-commit-msg] Write complete. Content of $COMMIT_MSG_FILE is now:"
cat "$COMMIT_MSG_FILE"
echo "[DEBUG prepare-commit-msg] --------------------------------------------------"
# --- Final Decision: Abort or Proceed ---
if [ "$ANY_CRITICAL_FAILED" = true ]; then
  echo "[prepare-commit-msg] ❌ Critical checks failed. Aborting commit." >&2
  echo "Please fix the issues and try committing again." >&2
  echo "Detailed check outputs are temporarily stored in $TEMP_CHECK_OUTPUT_DIR (will be cleaned on next attempt or success)." >&2
  exit 1 # ABORT COMMIT
else
  echo "[prepare-commit-msg] ✅ Checks completed. Proceeding with commit."
  # Temp files will be cleaned by post-commit or next prepare-commit-msg run
  exit 0 # PROCEED WITH COMMIT
fi
