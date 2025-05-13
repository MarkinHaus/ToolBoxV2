# .github/hooks/setup_hooks.sh
#!/bin/bash

echo "Installing git hooks..."
HOOKS_DIR=".github/hooks"
GIT_HOOKS_DIR=".git/hooks"

cp "$HOOKS_DIR/pre-commit.sh" "$GIT_HOOKS_DIR/pre-commit"
chmod +x "$GIT_HOOKS_DIR/pre-commit"
