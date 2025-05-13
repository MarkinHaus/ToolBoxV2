#!/bin/sh

echo "[post-commit] Starting quality checks..."

# === Konfiguration ===
REPORT_DIR="local-reports"
TIMESTAMP=$(date +"%Y-%m-%d-%H%M")
REPORT_FILE="$REPORT_DIR/$TIMESTAMP.txt"
mkdir -p "$REPORT_DIR"

# === Tools ===
CHECKS="ruff check .|Ruff bandit -r .|Bandit safety check --full-report|Safety"
VERSIONS_CMD="uv run tb -l -v"
AUTO_SUMMARY_CMD="uv run tb -c isaa auto_commit_msg"

# === Zustandstracking ===
SUMMARY=""
ALL_OK=true

# === Check-Funktion ===
run_and_record() {
  CMD=$1
  NAME=$2
  echo "[check] Running $NAME..."
  OUTPUT=$(eval "$CMD" 2>&1)
  EXIT_CODE=$?
  echo "[$NAME] Exit Code: $EXIT_CODE" >> "$REPORT_FILE"
  echo "[$NAME] Output:" >> "$REPORT_FILE"
  echo "$OUTPUT" >> "$REPORT_FILE"
  echo "" >> "$REPORT_FILE"

  if [ $EXIT_CODE -ne 0 ]; then
    ALL_OK=false
    SUMMARY="$SUMMARY $NAME❌"
  else
    SUMMARY="$SUMMARY $NAME✅"
  fi
}

# === Ausführen aller Checks ===
OLD_IFS=$IFS
IFS=$'\n'
for LINE in $(echo "$CHECKS"); do
  CMD=$(echo "$LINE" | cut -d'|' -f1)
  NAME=$(echo "$LINE" | cut -d'|' -f2)
  run_and_record "$CMD" "$NAME"
done
IFS=$OLD_IFS

# === Versions-Check separat (für echten Output) ===
echo "[check] Running Versions..."
VERSIONS_RAW=$($VERSIONS_CMD 2>&1)
echo "[Versions] Exit Code: $?" >> "$REPORT_FILE"
echo "[Versions] Output:" >> "$REPORT_FILE"
echo "$VERSIONS_RAW" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# === Nur relevanten Teil extrahieren ===
VERSIONS_FILTERED=$(echo "$VERSIONS_RAW" | awk '/--+ Version --+/,/^[^ ]/{ if ($0 ~ /--+ Version --+/) next; if ($0 ~ /^working on/) exit; print }')
SUMMARY="$SUMMARY
$VERSIONS_FILTERED"

# === Commit-Message lesen ===
COMMIT_MSG=$(git log -1 --pretty=%B)

# === Platzhalter <#> ersetzen ===
ESC_SUMMARY=$(printf '%s\n' "$SUMMARY" | sed -e 's/[\/&]/\\&/g')
NEW_MSG=$(printf '%s\n' "$COMMIT_MSG" | sed "s|<#>|$ESC_SUMMARY|g")

# === <sum> ersetzen, falls alle Checks OK ===
if echo "$NEW_MSG" | grep -q "<sum>" && [ "$ALL_OK" = true ]; then
  echo "[sum] All checks passed. Running auto_commit_msg..."
  SUM_TEXT=$($AUTO_SUMMARY_CMD 2>/dev/null)
  ESC_SUM=$(printf '%s\n' "$SUM_TEXT" | sed -e 's/[\/&]/\\&/g')
  NEW_MSG=$(printf '%s\n' "$NEW_MSG" | sed "s|<sum>|$ESC_SUM|g")
fi

# === Commit aktualisieren ===
git commit --amend -m "$NEW_MSG" --no-edit >/dev/null 2>&1

# === Report-Hinweis ===
if [ "$ALL_OK" = false ]; then
  echo "[post-commit] ❌ One or more checks failed. Report saved to $REPORT_FILE"
else
  echo "[post-commit] ✅ All checks passed. Commit updated. Report saved to $REPORT_FILE"
fi
