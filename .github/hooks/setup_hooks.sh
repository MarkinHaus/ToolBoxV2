# .github/hooks/setup_hooks.sh
#!/bin/bash

echo "Installing git hooks..."
HOOKS_DIR=".github/hooks"
GIT_HOOKS_DIR=".git/hooks"

cp "$HOOKS_DIR/pre-commit.sh" "$GIT_HOOKS_DIR/pre-commit"
chmod +x "$GIT_HOOKS_DIR/pre-commit"

cp "$HOOKS_DIR/prepare-commit-msg.sh" "$GIT_HOOKS_DIR/prepare-commit-msg"
chmod +x "$GIT_HOOKS_DIR/prepare-commit-msg"

cp "$HOOKS_DIR/common_quality_utils.sh" "$GIT_HOOKS_DIR/common_quality_utils.sh"

