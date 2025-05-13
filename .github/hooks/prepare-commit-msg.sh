#!/bin/sh

# GIT HOOK: prepare-commit-msg
# ... (initial parts of the script up to sourcing common_quality_utils.sh) ...
# Source the common utility functions
HOOK_DIR=$(dirname "$0")
# shellcheck source=./common_quality_utils.sh
if [ -f "$HOOK_DIR/common_quality_utils.sh" ]; then
  . "$HOOK_DIR/common_quality_utils.sh"
else
  echo "Error: common_quality_utils.sh not found in $HOOK_DIR." >&2
  exit 1
fi

# --- Configuration (mostly from common_quality_utils.sh now, but placeholders are hook-specific) ---
PLACEHOLDER_CHECKS_SUMMARY="<#>"      # Placeholder for check statuses
PLACEHOLDER_AUTO_SUMMARY="<sum>"    # Placeholder for auto-generated summary
APPEND_IF_NO_PLACEHOLDER_CHECKS=true # If true, append check summary if <#> not found
APPEND_IF_NO_PLACEHOLDER_SUM=false   # If true, append auto-summary if <sum> not found

# --- Script State Variables ---
ALL_CHECKS_OK=true          # Assume true initially
ANY_CRITICAL_FAILED=false   # Assume false initially
SUMMARY_LOG_MSG=""          # For building the summary text for the commit message
VERSIONS_RAW_OUTPUT=""      # Initialize to avoid unbound variable errors later
VERSIONS_EXIT_CODE=0        # Initialize

# --- Script Logic ---
echo "[DEBUG prepare-commit-msg] Starting hook for source: '$COMMIT_SOURCE', file: '$COMMIT_MSG_FILE'"

case "$COMMIT_SOURCE" in
  message|template|"") # "" for -F option
    echo "[DEBUG prepare-commit-msg] Proceeding for commit source: '$COMMIT_SOURCE'"
    ;;
  merge|squash|commit)
    echo "[DEBUG prepare-commit-msg] Skipping modifications for commit source: '$COMMIT_SOURCE' (e.g. merge, squash, amend)"
    cleanup_temp_check_outputs
    exit 0
    ;;
  *)
    echo "[DEBUG prepare-commit-msg] Unknown commit source: '$COMMIT_SOURCE'. Skipping."
    cleanup_temp_check_outputs
    exit 0
    ;;
esac

mkdir -p "$TEMP_CHECK_OUTPUT_DIR"

# --- Run Defined CHECKS from CHECKS_DEFINITIONS ---
if [ -z "$CHECKS_DEFINITIONS" ]; then
    echo "[DEBUG prepare-commit-msg] CHECKS_DEFINITIONS is empty. No checks to run." >&2
else
    echo "[DEBUG prepare-commit-msg] CHECKS_DEFINITIONS: [$CHECKS_DEFINITIONS]"
    printf '%s\n' "$CHECKS_DEFINITIONS" | sed '/^[[:space:]]*$/d' | while IFS='|' read -r CMD_STR NAME_STR IS_CRITICAL_STR || [ -n "$CMD_STR" ]; do
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
            ANY_CRITICAL_FAILED=true
            echo "[prepare-commit-msg] CRITICAL CHECK FAILED: $NAME_STR_TRIMMED" >&2
          fi
        else
          SUMMARY_LOG_MSG="${SUMMARY_LOG_MSG}${SUMMARY_LOG_MSG:+, }$NAME_STR_TRIMMED✅"
        fi
      else
          echo "[DEBUG prepare-commit-msg] Skipped check processing due to empty CMD_STR_TRIMMED or NAME_STR_TRIMMED."
      fi
      echo "[DEBUG prepare-commit-msg] Intermediate SUMMARY_LOG_MSG: [$SUMMARY_LOG_MSG]"
    done
fi

# --- Run Versions Check ---
if [ -n "$VERSIONS_CMD" ]; then
    echo "[DEBUG prepare-commit-msg] Running Versions Check ($VERSIONS_CMD)..."
    VERSIONS_RAW_OUTPUT=$($VERSIONS_CMD 2>&1)
    VERSIONS_EXIT_CODE=$?
    echo "[DEBUG prepare-commit-msg] VERSIONS_RAW_OUTPUT variable content: [$VERSIONS_RAW_OUTPUT]" # Added this
    echo "[DEBUG prepare-commit-msg] Versions Exit Code: $VERSIONS_EXIT_CODE"

    echo "$VERSIONS_RAW_OUTPUT" > "$TEMP_CHECK_OUTPUT_DIR/Versions.output"
    echo "$VERSIONS_EXIT_CODE" > "$TEMP_CHECK_OUTPUT_DIR/Versions.exitcode"

    echo "[DEBUG prepare-commit-msg] Content of $TEMP_CHECK_OUTPUT_DIR/Versions.output IS:" # Added this block
    cat "$TEMP_CHECK_OUTPUT_DIR/Versions.output"
    echo "[DEBUG prepare-commit-msg] --- END OF Versions.output ---"

    # IMPORTANT: Use the content from the file for parsing if variable might be incomplete
    # However, for consistency, let's first ensure VERSIONS_RAW_OUTPUT is correctly populated.
    # If it's confirmed empty while file has content, then we might need to read from file here.
    # For now, assume $VERSIONS_RAW_OUTPUT *should* be the source of truth.
    VERSIONS_SUMMARY_PART_TEXT=$(get_versions_summary_part "$VERSIONS_RAW_OUTPUT")
    echo "[DEBUG prepare-commit-msg] Parsed Versions Summary (from VERSIONS_RAW_OUTPUT var): [$VERSIONS_SUMMARY_PART_TEXT]"

    if [ -n "$VERSIONS_SUMMARY_PART_TEXT" ]; then
      if [ -n "$SUMMARY_LOG_MSG" ]; then
        SUMMARY_LOG_MSG="$SUMMARY_LOG_MSG
Version Info:
$VERSIONS_SUMMARY_PART_TEXT"
      else
        SUMMARY_LOG_MSG="Version Info:
$VERSIONS_SUMMARY_PART_TEXT"
      fi
    else
        echo "[DEBUG prepare-commit-msg] No version summary extracted from VERSIONS_RAW_OUTPUT variable."
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
echo "[DEBUG prepare-commit-msg] CLEANED_SUMMARY_LOG_MSG: [$CLEANED_SUMMARY_LOG_MSG]"

ESCAPED_SUMMARY_FOR_SED=$(escape_for_sed "$CLEANED_SUMMARY_LOG_MSG")
echo "[DEBUG prepare-commit-msg] ESCAPED_SUMMARY_FOR_SED: [$ESCAPED_SUMMARY_FOR_SED]"

# Check the variable $NEW_MSG_CONTENT for the placeholder
if echo "$NEW_MSG_CONTENT" | grep -qF "$PLACEHOLDER_CHECKS_SUMMARY"; then
  echo "[DEBUG prepare-commit-msg] Found placeholder '$PLACEHOLDER_CHECKS_SUMMARY' in NEW_MSG_CONTENT. Replacing."
  NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_CHECKS_SUMMARY|$ESCAPED_SUMMARY_FOR_SED|g")
elif [ "$APPEND_IF_NO_PLACEHOLDER_CHECKS" = true ] && [ -n "$CLEANED_SUMMARY_LOG_MSG" ]; then
  echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_CHECKS_SUMMARY' not found in NEW_MSG_CONTENT. Appending check status."
  NEW_MSG_CONTENT=$(printf '%s\n\nCheck Status:\n%s' "$NEW_MSG_CONTENT" "$CLEANED_SUMMARY_LOG_MSG")
else
  echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_CHECKS_SUMMARY' not found and not appending, or CLEANED_SUMMARY_LOG_MSG is empty."
fi
echo "[DEBUG prepare-commit-msg] NEW_MSG_CONTENT after <#> processing: [$NEW_MSG_CONTENT]"

# Check the variable $NEW_MSG_CONTENT for the placeholder
if [ -n "$AUTO_SUMMARY_CMD" ] && (echo "$NEW_MSG_CONTENT" | grep -qF "$PLACEHOLDER_AUTO_SUMMARY"); then
  if [ "$ALL_CHECKS_OK" = true ]; then
    echo "[DEBUG prepare-commit-msg] All checks passed. Generating auto-summary using '$AUTO_SUMMARY_CMD'..."
    AUTO_GENERATED_SUMMARY_TEXT=$($AUTO_SUMMARY_CMD 2>/dev/null)

    if [ -n "$AUTO_GENERATED_SUMMARY_TEXT" ]; then
      ESCAPED_AUTO_SUMMARY=$(escape_for_sed "$AUTO_GENERATED_SUMMARY_TEXT")
      NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_AUTO_SUMMARY|$ESCAPED_AUTO_SUMMARY|g")
      echo "[DEBUG prepare-commit-msg] Replaced <sum> with auto-summary."
    else
      NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_AUTO_SUMMARY|(Auto-summary not generated)|g")
      echo "[DEBUG prepare-commit-msg] Auto-summary command did not produce output. Replaced <sum> with note."
    fi
  else
    NEW_MSG_CONTENT=$(echo "$NEW_MSG_CONTENT" | sed "s|$PLACEHOLDER_AUTO_SUMMARY|(Auto-summary skipped due to check failures)|g")
    echo "[DEBUG prepare-commit-msg] Auto-summary skipped because one or more checks failed. Replaced <sum> with note."
  fi
elif [ "$APPEND_IF_NO_PLACEHOLDER_SUM" = true ] && [ "$ALL_CHECKS_OK" = true ] && [ -n "$AUTO_SUMMARY_CMD" ]; then
    echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_AUTO_SUMMARY' not found in NEW_MSG_CONTENT. Appending auto-summary."
    AUTO_GENERATED_SUMMARY_TEXT=$($AUTO_SUMMARY_CMD 2>/dev/null)
    if [ -n "$AUTO_GENERATED_SUMMARY_TEXT" ]; then
        NEW_MSG_CONTENT=$(printf '%s\n\nAuto Summary:\n%s' "$NEW_MSG_CONTENT" "$AUTO_GENERATED_SUMMARY_TEXT")
    fi
else
    echo "[DEBUG prepare-commit-msg] Placeholder '$PLACEHOLDER_AUTO_SUMMARY' not found in NEW_MSG_CONTENT and not appending, or AUTO_SUMMARY_CMD not set, or checks failed."
fi
echo "[DEBUG prepare-commit-msg] NEW_MSG_CONTENT after <sum> processing: [$NEW_MSG_CONTENT]"

# ... (rest of the script: Write changes back, Final Decision) ...
if [ "$NEW_MSG_CONTENT" != "$ORIGINAL_MSG_CONTENT" ]; then
  echo "[DEBUG prepare-commit-msg] Modifying commit message in $COMMIT_MSG_FILE."
  if [ -n "$NEW_MSG_CONTENT" ]; then
      printf '%s\n' "$NEW_MSG_CONTENT" > "$COMMIT_MSG_FILE"
  else
      echo "" > "$COMMIT_MSG_FILE"
  fi
  echo "[DEBUG prepare-commit-msg] Write complete. Content of $COMMIT_MSG_FILE is now:"
  cat "$COMMIT_MSG_FILE"
  echo "[DEBUG prepare-commit-msg] --- (end of COMMIT_EDITMSG content) ---"
else
  echo "[DEBUG prepare-commit-msg] No changes made to commit message content."
fi

if [ "$ANY_CRITICAL_FAILED" = true ]; then
  echo "" >&2
  echo "[prepare-commit-msg] ---------------------------------------------------------------------" >&2
  echo "[prepare-commit-msg] ❌ CRITICAL CHECKS FAILED. Commit ABORTED." >&2
  echo "[prepare-commit-msg] ---------------------------------------------------------------------" >&2
  echo "Please fix the issues and try committing again." >&2
  echo "Detailed check outputs are temporarily stored in $TEMP_CHECK_OUTPUT_DIR" >&2
  echo "(These will be cleaned on the next successful commit or next prepare-commit attempt)." >&2
  exit 1
else
  echo "[prepare-commit-msg] ✅ Checks completed (or no critical failures). Proceeding with commit."
  exit 0
fi
