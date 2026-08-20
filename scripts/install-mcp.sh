#!/usr/bin/env bash
# Write a .mcp.json pointing at this checkout, for Claude Code / any MCP client.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"
DB="${INFERWATCH_DB:-${XDG_DATA_HOME:-$HOME/.local/share}/inferwatch/inferwatch.db}"
cat > "$REPO/.mcp.json" <<JSON
{
  "mcpServers": {
    "inferwatch": {
      "command": "$PYTHON",
      "args": ["-m", "inferwatch.mcp_server"],
      "cwd": "$REPO",
      "env": { "INFERWATCH_DB": "$DB" }
    }
  }
}
JSON
echo "wrote $REPO/.mcp.json"
