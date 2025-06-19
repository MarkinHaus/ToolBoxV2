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
  # Auto-tagging logic will be called later, even if report is skipped
else
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
    # Set _versions_raw_output to empty if file not found, so auto-tagging can check it
    _versions_raw_output=""
  fi

  # === Erstelle/Aktualisiere "latest" Report Symlink ===
  # Use relative path for symlink for better portability if project moves
  cd "$REPORT_DIR" || exit 1 # Enter report dir to make relative symlink
  ln -snf "$(basename "$CURRENT_REPORT_FILE")" "$LATEST_REPORT_NAME"
  cd "$PROJECT_ROOT" || exit 1 # Go back to project root

  echo "[post-commit] Report created: $CURRENT_REPORT_FILE"
  echo "[post-commit] Latest report link: $LATEST_REPORT_LINK"
fi # End of main report generation block

# === Clean up temporary files (moved here to ensure _versions_output_file is available for tagging) ===
# cleanup_temp_check_outputs # Original position
# We need _versions_output_file for tagging, so cleanup will happen after tagging attempts.


# === START: Auto-Tagging Erweiterung ===
echo "[post-commit] Checking for auto-tagging conditions..."

AUTO_TAGGING_ENABLED=true # Flag to control if tagging proceeds

# 1. Nur auf 'main' Branch taggen
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "master" ]; then
  echo "[auto-tagging] Not on 'master' branch (current: '$CURRENT_BRANCH'). Skipping auto-tagging."
  AUTO_TAGGING_ENABLED=false
fi

# 2. Commit-Nachricht auslesen und Tag-Typ extrahieren
COMMIT_MSG=""
TAG_TYPE_MARKER=""
if [ "$AUTO_TAGGING_ENABLED" = true ]; then
  COMMIT_MSG=$(git log -1 --pretty=%B)
  # Extrahiere [t:X] und dann X. Unterstützt d, a, r.
  TAG_TYPE_MARKER=$(echo "$COMMIT_MSG" | grep -oE '\[t:[dar]\]' | sed -e 's/\[t:\(.\)\]/\1/')

  if [ -z "$TAG_TYPE_MARKER" ]; then
    echo "[auto-tagging] No valid tag marker [t:d], [t:a], or [t:r] found in commit message. Skipping auto-tagging."
    AUTO_TAGGING_ENABLED=false
  else
    echo "[auto-tagging] Found tag marker: [t:$TAG_TYPE_MARKER]"
  fi
fi

# 3. App Version extrahieren
APP_VERSION=""
if [ "$AUTO_TAGGING_ENABLED" = true ]; then
  # Stelle sicher, dass _versions_output_file auch gesetzt ist, falls der Report-Teil übersprungen wurde
  if [ -z "$_versions_output_file" ]; then
      _versions_output_file="$TEMP_CHECK_OUTPUT_DIR/Versions.output"
  fi

  if [ -f "$_versions_output_file" ]; then
    # Nimm die erste Zeile aus der Versionsausgabe, die wie eine Version aussieht (z.B. x.y.z)
    # Dies ist eine vereinfachte Regex; ggf. anpassen.
    # Es wird die erste gefundene Version genommen.
    APP_VERSION=$(cat "$_versions_output_file" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?([a-zA-Z0-9.-]*)?' | head -n 1)
    if [ -z "$APP_VERSION" ]; then
      echo "[auto-tagging] Could not extract app version from '$_versions_output_file'. Skipping auto-tagging."
      AUTO_TAGGING_ENABLED=false
    else
      echo "[auto-tagging] Extracted app version: $APP_VERSION"
    fi
  else
    echo "[auto-tagging] Version file '$_versions_output_file' not found. Skipping auto-tagging."
    AUTO_TAGGING_ENABLED=false
  fi
fi

# 4. Tag erstellen, wenn alle Bedingungen erfüllt sind
if [ "$AUTO_TAGGING_ENABLED" = true ]; then
  SHORT_COMMIT_HASH=$(git rev-parse --short HEAD)
  BASE_TAG_NAME=""
  TAG_PREFIX=""

  case "$TAG_TYPE_MARKER" in
    d)
      TAG_PREFIX="dev"
      BASE_TAG_NAME="${TAG_PREFIX}-v${APP_VERSION}-${SHORT_COMMIT_HASH}"
      ;;
    a)
      TAG_PREFIX="alpha"
      BASE_TAG_NAME="${TAG_PREFIX}-v${APP_VERSION}-${SHORT_COMMIT_HASH}"
      ;;
    r)
      # Kein Präfix für Release, nur vVERSION
      BASE_TAG_NAME="v${APP_VERSION}"
      ;;
    *)
      # Sollte durch vorherige Prüfung nicht erreicht werden, aber sicher ist sicher
      echo "[auto-tagging] Internal error: Invalid tag type marker '$TAG_TYPE_MARKER'. Skipping."
      AUTO_TAGGING_ENABLED=false # Stoppt die Tag-Erstellung
      ;;
  esac

  if [ "$AUTO_TAGGING_ENABLED" = true ] && [ -n "$BASE_TAG_NAME" ]; then
    FINAL_TAG_NAME="$BASE_TAG_NAME"
    COUNT=1
    # Prüfen, ob der Tag (oder eine Variante mit Suffix) bereits existiert
    # git show-ref --tags "$FINAL_TAG_NAME" --quiet gibt 0 zurück, wenn Tag existiert
    while git show-ref --tags "$FINAL_TAG_NAME" --quiet; do
      FINAL_TAG_NAME="${BASE_TAG_NAME}-${COUNT}"
      COUNT=$((COUNT + 1))
    done

    echo "[auto-tagging] Attempting to create tag: $FINAL_TAG_NAME"
    if git tag "$FINAL_TAG_NAME"; then
      echo "[auto-tagging] Successfully created tag: $FINAL_TAG_NAME"
      # Optional: git push origin "$FINAL_TAG_NAME"
    else
      echo "[auto-tagging] Failed to create tag: $FINAL_TAG_NAME"
    fi
  fi
fi
# === END: Auto-Tagging Erweiterung ===


# === Clean up temporary files (jetzt hier, damit Versionsinfo für Tagging verfügbar war) ===
cleanup_temp_check_outputs


# === Abschlussmeldung ===
if [ -f "$CURRENT_REPORT_FILE" ]; then # Nur wenn ein Report erstellt wurde
    if [ "$ALL_CHECKS_PASSED_IN_PREPARE" = false ]; then
    # Note: This reflects status from prepare-commit-msg. Commit itself succeeded.
    echo "[post-commit] ⚠️ Note: One or more checks had issues during pre-commit phase. Review report $LATEST_REPORT_LINK"
    else
    echo "[post-commit] ✅ All checks noted in pre-commit phase passed. Report generated."
    fi
else
    echo "[post-commit] No detailed report was generated."
fi


exit 0
