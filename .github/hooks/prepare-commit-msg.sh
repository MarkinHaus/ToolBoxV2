#!/bin/sh

echo "--------------------------------------------------" >&2
echo "prepare-commit-msg ARGS:" >&2
echo "  \$1 (COMMIT_MSG_FILE): [$1]" >&2
echo "  \$2 (COMMIT_SOURCE):   [$2]" >&2
echo "  \$3 (SHA1):            [$3]" >&2
echo "--------------------------------------------------" >&2

COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="$2"

if [ -z "$COMMIT_MSG_FILE" ]; then
    echo "CRITICAL ERROR in prepare-commit-msg: COMMIT_MSG_FILE (\$1) is empty. Aborting hook (exit 1)." >&2
    exit 1 # Still exit 1 here, as the hook cannot function without the file.
fi

HOOK_DIR=$(dirname "$0")
if [ -f "$HOOK_DIR/common_quality_utils.sh" ]; then
  . "$HOOK_DIR/common_quality_utils.sh"
else
  echo "Error: common_quality_utils.sh not found in $HOOK_DIR. Exiting (1)." >&2
  exit 1
fi

PLACEHOLDER_CHECKS_SUMMARY="<#>"
PLACEHOLDER_AUTO_SUMMARY="<sum>"
APPEND_IF_NO_PLACEHOLDER_CHECKS=true
APPEND_IF_NO_PLACEHOLDER_SUM=false

ALL_CHECKS_OK=true          # This will be set by the checks loop
ANY_CRITICAL_FAILED=false   # This will be set by the checks loop
SUMMARY_LOG_MSG=""          # This will be built by checks and version info
VERSIONS_RAW_OUTPUT=""
VERSIONS_EXIT_CODE=0

echo "[DEBUG prepare-commit-msg] Starting hook for source: '$COMMIT_SOURCE', file: '$COMMIT_MSG_FILE'"

case "$COMMIT_SOURCE" in
  message|template|"")
    echo "[DEBUG prepare-commit-msg] Proceeding for commit source: '$COMMIT_SOURCE'"
    ;;
  *)
    echo "[DEBUG prepare-commit-msg] Skipping modifications for commit source: '$COMMIT_SOURCE'"
    cleanup_temp_check_outputs
    exit 0 # Exit 0 as we are just skipping, not an error.
    ;;
esac

mkdir -p "$TEMP_CHECK_OUTPUT_DIR"

# --- Run Defined CHECKS from CHECKS_DEFINITIONS ---
# IMPORTANT: Ensure this loop uses Process Substitution or a temp file
# to make SUMMARY_LOG_MSG, ALL_CHECKS_OK, ANY_CRITICAL_FAILED persistent.
# Example with Process Substitution (assuming bash-like shell):
if [ -z "$CHECKS_DEFINITIONS" ]; then
    echo "[DEBUG prepare-commit-msg] CHECKS_DEFINITIONS is empty." >&2
else
    echo "[DEBUG prepare-commit-msg] CHECKS_DEFINITIONS IS: [$CHECKS_DEFINITIONS]"
    INPUT_FOR_LOOP=$(printf '%s\n' "$CHECKS_DEFINITIONS" | sed '/^[[:space:]]*$/d')
    echo "[DEBUG prepare-commit-msg] INPUT_FOR_LOOP for checks: [$INPUT_FOR_LOOP]"
    if [ -n "$INPUT_FOR_LOOP" ]; then
        while IFS='|' read -r CMD_STR NAME_STR IS_CRITICAL_STR || [ -n "$CMD_STR" ]; do
          CMD_STR_TRIMMED=$(echo "$CMD_STR" | awk '{$1=$1};1')
          NAME_STR_TRIMMED=$(echo "$NAME_STR" | awk '{$1=$1};1')
          IS_CRITICAL_TRIMMED=$(echo "$IS_CRITICAL_STR" | awk '{$1=$1};1')

          echo "[DEBUG prepare-commit-msg] Read Loop - CMD: [$CMD_STR_TRIMMED], NAME: [$NAME_STR_TRIMMED], CRITICAL: [$IS_CRITICAL_TRIMMED]"
          if [ -n "$CMD_STR_TRIMMED" ] && [ -n "$NAME_STR_TRIMMED" ]; then
            run_single_check_and_store "$CMD_STR_TRIMMED" "$NAME_STR_TRIMMED" "$TEMP_CHECK_OUTPUT_DIR/$NAME_STR_TRIMMED"
            check_exit_code=$?
            echo "[DEBUG prepare-commit-msg] Check Ran: '$NAME_STR_TRIMMED', Exit: $check_exit_code"
            if [ "$check_exit_code" -ne 0 ]; then
              ALL_CHECKS_OK=false
              SUMMARY_LOG_MSG="${SUMMARY_LOG_MSG}${SUMMARY_LOG_MSG:+, }$NAME_STR_TRIMMED❌"
              if [ "$IS_CRITICAL_TRIMMED" = "true" ]; then
                ANY_CRITICAL_FAILED=true # Still track this for <sum> logic and console message
                echo "[prepare-commit-msg] WARNING: CRITICAL CHECK FAILED: $NAME_STR_TRIMMED (Exit: $check_exit_code)" >&2
              fi
            else
              SUMMARY_LOG_MSG="${SUMMARY_LOG_MSG}${SUMMARY_LOG_MSG:+, }$NAME_STR_TRIMMED✅"
            fi
          fi
          echo "[DEBUG prepare-commit-msg] Intermediate SUMMARY_LOG_MSG (in loop): [$SUMMARY_LOG_MSG]"
        done < <(echo "$INPUT_FOR_LOOP") # Process substitution
    else
        echo "[DEBUG prepare-commit-msg] INPUT_FOR_LOOP for checks is empty."
    fi
fi
echo "[DEBUG prepare-commit-msg] SUMMARY_LOG_MSG *after* checks loop: [$SUMMARY_LOG_MSG]"
echo "[DEBUG prepare-commit-msg] ALL_CHECKS_OK *after* checks loop: [$ALL_CHECKS_OK]"
echo "[DEBUG prepare-commit-msg] ANY_CRITICAL_FAILED *after* checks loop: [$ANY_CRITICAL_FAILED]"

# --- Run Versions Check ---
if [ -n "$VERSIONS_CMD" ]; then
    echo "[DEBUG prepare-commit-msg] Running Versions Check ($VERSIONS_CMD)..."
    VERSIONS_RAW_OUTPUT=$($VERSIONS_CMD 2>&1)
    VERSIONS_EXIT_CODE=$?
    echo "[DEBUG prepare-commit-msg] VERSIONS_RAW_OUTPUT variable content: [$VERSIONS_RAW_OUTPUT]"
    echo "[DEBUG prepare-commit-msg] Versions Exit Code: $VERSIONS_EXIT_CODE"

    echo "$VERSIONS_RAW_OUTPUT" > "$TEMP_CHECK_OUTPUT_DIR/Versions.output"
    echo "$VERSIONS_EXIT_CODE" > "$TEMP_CHECK_OUTPUT_DIR/Versions.exitcode"

    VERSIONS_SUMMARY_PART_TEXT=$(get_versions_summary_part "$VERSIONS_RAW_OUTPUT")
    echo "[DEBUG prepare-commit-msg] Parsed Versions Summary: [$VERSIONS_SUMMARY_PART_TEXT]"

    if [ -n "$VERSIONS_SUMMARY_PART_TEXT" ]; then
      if [ -n "$SUMMARY_LOG_MSG" ]; then
        # If there are check results, add version info on a new line, prefixed by "Version Info:"
        SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG
Version Info:
$VERSIONS_SUMMARY_PART_TEXT"
      else
        # If no check results, Version Info is the only thing
        SUMMARY_LOG_MSG="Version Info:
$VERSIONS_SUMMARY_PART_TEXT"
      fi
    else
        echo "[DEBUG prepare-commit-msg] No version summary extracted."
    fi
else
    echo "[DEBUG prepare-commit-msg] VERSIONS_CMD is not defined. Skipping version check."
fi
echo "[DEBUG prepare-commit-msg] Final SUMMARY_LOG_MSG before cleaning: [$SUMMARY_LOG_MSG]"

# --- Update Commit Message File ---
ORIGINAL_MSG_CONTENT=$(cat "$COMMIT_MSG_FILE")
NEW_MSG_CONTENT="$ORIGINAL_MSG_CONTENT"
echo "[DEBUG prepare-commit-msg] Original commit message content: [$ORIGINAL_MSG_CONTENT]"

CLEANED_SUMMARY_LOG_MSG=$(printf '%s\n' "$SUMMARY_LOG_MSG" | awk 'NF > 0 {$1=$1};1' | sed '/^[[:space:]]*$/d')
echo "[DEBUG prepare-commit-msg] CLEANED_SUMMARY_LOG_MSG for <#>: [$CLEANED_SUMMARY_LOG_MSG]"

# If CLEANED_SUMMARY_LOG_MSG is empty, ESCAPED_SUMMARY_FOR_SED will also be empty (or just \n)
# This means <#> will be replaced by "nothing" or a newline, effectively removing it.
ESCAPED_SUMMARY_FOR_SED=$(escape_for_sed "$CLEANED_SUMMARY_LOG_MSG")
echo "[DEBUG prepare-commit-msg] ESCAPED_SUMMARY_FOR_SED: [$ESCAPED_SUMMARY_FOR_SED]"

if echo "$NEW_MSG_CONTENT" | grep -qF "$PLACEHOLDER_CHECKS_SUMMARY"; then
  echo "[DEBUG prepare-commit-msg] Found placeholder '$PLACEHOLDER_CHECKS_SUMMARY'. Replacing."
  NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_CHECKS_SUMMARY|$ESCAPED_SUMMARY_FOR_SED|g")
elif [ "$APPEND_IF_NO_PLACEHOLDER_CHECKS" = true ] && [ -n "$CLEANED_SUMMARY_LOG_MSG" ]; then
  echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_CHECKS_SUMMARY' not found. Appending."
  NEW_MSG_CONTENT=$(printf '%s\n\nCheck Status:\n%s' "$NEW_MSG_CONTENT" "$CLEANED_SUMMARY_LOG_MSG")
else
  echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_CHECKS_SUMMARY' not found and not appending, or CLEANED_SUMMARY_LOG_MSG is empty (so <#> effectively removed if present)."
fi
echo "[DEBUG prepare-commit-msg] NEW_MSG_CONTENT after <#> processing: [$NEW_MSG_CONTENT]"

if [ "$NEW_MSG_CONTENT" != "$ORIGINAL_MSG_CONTENT" ]; then
  echo "[DEBUG prepare-commit-msg] Modifying commit message in $COMMIT_MSG_FILE."
  printf '%s\n' "$NEW_MSG_CONTENT" > "$COMMIT_MSG_FILE"
  echo "[DEBUG prepare-commit-msg] Write complete. Content of $COMMIT_MSG_FILE is now:"
  cat "$COMMIT_MSG_FILE"
  echo "[DEBUG prepare-commit-msg] --- (end of COMMIT_EDITMSG content) ---"
else
  echo "[DEBUG prepare-commit-msg] No changes made to commit message content."
fi

# --- Final Decision: ALWAYS PROCEED WITH COMMIT ---
if [ "$ANY_CRITICAL_FAILED" = true ]; then # Still inform user about critical failures
  echo "" >&2
  echo "[prepare-commit-msg] ---------------------------------------------------------------------" >&2
  echo "[prepare-commit-msg] ⚠️ WARNING: CRITICAL CHECKS FAILED. (Details above/in report)" >&2
  echo "[prepare-commit-msg] Proceeding with commit as per configuration." >&2
  echo "[prepare-commit-msg] ---------------------------------------------------------------------" >&2
elif [ "$ALL_CHECKS_OK" = false ]; then # Non-critical checks failed
  echo "" >&2
  echo "[prepare-commit-msg] ---------------------------------------------------------------------" >&2
  echo "[prepare-commit-msg] ℹ️ Note: Some non-critical checks failed. (Details above/in report)" >&2
  echo "[prepare-commit-msg] Proceeding with commit." >&2
  echo "[prepare-commit-msg] ---------------------------------------------------------------------" >&2
else
  echo "[prepare-commit-msg] ✅ All checks passed. Proceeding with commit."
fi

exit 0 # ALWAYS exit 0 to allow the commit to proceed.
