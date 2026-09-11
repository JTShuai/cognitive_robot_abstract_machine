#!/usr/bin/env bash
# Clone and build Fast Downward into the configured planner directory.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${RESYM_FAST_DOWNWARD_DIRECTORY:-$PROJECT_ROOT/vendor/downward}"
if [[ -x "$TARGET/fast-downward.py" && -d "$TARGET/builds/release" ]]; then
  echo "Fast Downward already installed at $TARGET"
  exit 0
fi
git clone --depth 1 https://github.com/aibasel/downward.git "$TARGET"
cd "$TARGET"
uv run python build.py release
echo "Fast Downward ready: $TARGET/fast-downward.py"
