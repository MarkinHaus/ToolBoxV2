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

# Only run for "message" (new commits from -m or editor), "template"
# Add other sources if needed, e.g., "commit" when amending.
# For amending, you might want different logic or to skip if $3 (SHA1) is present.
case "$COMMIT_SOURCE" in
  message|template)
    # Proceed
    ;;
  "") # When -F is used and source is empty
    # Proceed
    ;;
  *)
    # echo "[prepare-commit-msg] Skipping for commit source: $COMMIT_SOURCE"
    cleanup_temp_check_outputs # Clean up if we skip, to prevent stale data
    exit 0
    ;;
esac

echo "[prepare-commit-msg] Running pre-commit quality checks..."

mkdir -p "$TEMP_CHECK_OUTPUT_DIR" # Ensure temp dir exists

ALL_CHECKS_OK=true
ANY_CRITICAL_FAILED=false
SUMMARY_LOG_MSG="" # For building the summary text for the commit message

# Run defined CHECKS
echo "$CHECKS_DEFINITIONS" | while IFS='|' read -r CMD_STR NAME_STR IS_CRITICAL_STR; do
  CMD_STR_TRIMMED=$(echo "$CMD_STR" | awk '{$1=$1};1')
  NAME_STR_TRIMMED=$(echo "$NAME_STR" | awk '{$1=$1};1')
  IS_CRITICAL_TRIMMED=$(echo "$IS_CRITICAL_STR" | awk '{$1=$1};1')

  if [ -n "$CMD_STR_TRIMMED" ] && [ -n "$NAME_STR_TRIMMED" ]; then
    run_single_check_and_store "$CMD_STR_TRIMMED" "$NAME_STR_TRIMMED" "$TEMP_CHECK_OUTPUT_DIR/$NAME_STR_TRIMMED"
    check_exit_code=$?

    if [ "$check_exit_code" -ne 0 ]; then
      ALL_CHECKS_OK=false
      SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG $NAME_STR_TRIMMED❌"
      if [ "$IS_CRITICAL_TRIMMED" = "true" ]; then
        ANY_CRITICAL_FAILED=true
        echo "[prepare-commit-msg] CRITICAL CHECK FAILED: $NAME_STR_TRIMMED" >&2
      fi
    else
      SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG $NAME_STR_TRIMMED✅"
    fi
  fi
done

# Run Versions Check and store its output
echo "[prepare-commit-msg] Running Versions Check ($VERSIONS_CMD)..."
VERSIONS_RAW_OUTPUT=$($VERSIONS_CMD 2>&1)
VERSIONS_EXIT_CODE=$?
echo "$VERSIONS_RAW_OUTPUT" > "$TEMP_CHECK_OUTPUT_DIR/Versions.output"
echo "$VERSIONS_EXIT_CODE" > "$TEMP_CHECK_OUTPUT_DIR/Versions.exitcode"

VERSIONS_SUMMARY_PART_TEXT=$(get_versions_summary_part "$VERSIONS_RAW_OUTPUT")
if [ -n "$VERSIONS_SUMMARY_PART_TEXT" ]; then
  SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG
Version Info:
$VERSIONS_SUMMARY_PART_TEXT"
fi

# --- Update Commit Message File ---
ORIGINAL_MSG_CONTENT=$(cat "$COMMIT_MSG_FILE")
NEW_MSG_CONTENT="$ORIGINAL_MSG_CONTENT"

# 1. Replace <#> with check summary
CLEANED_SUMMARY_LOG_MSG=$(echo "$SUMMARY_LOG_MSG" | awk '{$1=$1};1' | sed '/^$/d') # Trim and remove blank lines
ESCAPED_SUMMARY_FOR_SED=$(escape_for_sed "$CLEANED_SUMMARY_LOG_MSG")

if grep -q "$PLACEHOLDER_CHECKS_SUMMARY" "$COMMIT_MSG_FILE"; then
  NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_CHECKS_SUMMARY|$ESCAPED_SUMMARY_FOR_SED|g")
elif [ "$APPEND_IF_NO_PLACEHOLDER_CHECKS" = true ] && [ -n "$CLEANED_SUMMARY_LOG_MSG" ]; then
  NEW_MSG_CONTENT=$(printf '%s\n\nCheck Status:\n%s' "$NEW_MSG_CONTENT" "$CLEANED_SUMMARY_LOG_MSG")
fi

# 2. Replace <sum> with auto-summary if all checks OK
if grep -q "$PLACEHOLDER_AUTO_SUMMARY" "$COMMIT_MSG_FILE"; then
  if [ "$ALL_CHECKS_OK" = true ]; then
    echo "[prepare-commit-msg] All checks passed. Generating auto-summary using '$AUTO_SUMMARY_CMD'..."
    AUTO_GENERATED_SUMMARY_TEXT=$($AUTO_SUMMARY_CMD 2>/dev/null)

    if [ -n "$AUTO_GENERATED_SUMMARY_TEXT" ]; then
      ESCAPED_AUTO_SUMMARY=$(escape_for_sed "$AUTO_GENERATED_SUMMARY_TEXT")
      NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_AUTO_SUMMARY|$ESCAPED_AUTO_SUMMARY|g")
    else
      NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_AUTO_SUMMARY|(Auto-summary not generated)|g")
      echo "[prepare-commit-msg] Auto-summary command did not produce output."
    fi
  else
    NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_AUTO_SUMMARY|(Auto-summary skipped due to check failures)|g")
    echo "[prepare-commit-msg] Auto-summary skipped because one or more checks failed."
  fi
elif [ "$APPEND_IF_NO_PLACEHOLDER_SUM" = true ] && [ "$ALL_CHECKS_OK" = true ]; then
    AUTO_GENERATED_SUMMARY_TEXT=$($AUTO_SUMMARY_CMD 2>/dev/null)
    if [ -n "$AUTO_GENERATED_SUMMARY_TEXT" ]; then
        NEW_MSG_CONTENT=$(printf '%s\n\nAuto Summary:\n%s' "$NEW_MSG_CONTENT" "$AUTO_GENERATED_SUMMARY_TEXT")
    fi
fi

# Write changes back to the commit message file
if [ "$NEW_MSG_CONTENT" != "$ORIGINAL_MSG_CONTENT" ]; then
  echo "[prepare-commit-msg] Modifying commit message in $COMMIT_MSG_FILE."
  printf '%s\n' "$NEW_MSG_CONTENT" > "$COMMIT_MSG_FILE"
fi

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
