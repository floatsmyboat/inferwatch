#!/usr/bin/env bash
# Create the API key the MCP server's HTTP transports require.
#
# The key lives in /etc, never in this repository and never in the metrics
# database: the repo is public and the database is a backup target.
#
#   sudo ./scripts/install-mcp-key.sh            # create if absent
#   sudo ./scripts/install-mcp-key.sh --rotate   # replace an existing one
#   sudo ./scripts/install-mcp-key.sh --group inferwatch
#
# Prints the key once, so it can be pasted into a client. Read it back later
# with `sudo cat`, or rotate it.
set -euo pipefail

DIR="${INFERWATCH_MCP_KEY_DIR:-/etc/inferwatch}"
FILE="$DIR/mcp-api-key"
GROUP=""
ROTATE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --rotate) ROTATE=1 ;;
        --group)  GROUP="${2:?--group needs a value}"; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$(id -u)" -ne 0 ]; then
    echo "this writes to $DIR, so it needs root: re-run with sudo" >&2
    exit 1
fi

if [ -e "$FILE" ] && [ "$ROTATE" -eq 0 ]; then
    echo "$FILE already exists; pass --rotate to replace it." >&2
    echo "current value: sudo cat $FILE" >&2
    exit 1
fi

command -v openssl >/dev/null || { echo "openssl is required" >&2; exit 1; }

install -d -m 750 "$DIR"
KEY="$(openssl rand -hex 32)"
# Written via a temp file in the same directory so the real one is never
# briefly readable with the wrong mode.
TMP="$(mktemp "$DIR/.mcp-api-key.XXXXXX")"
chmod 640 "$TMP"
printf '%s\n' "$KEY" > "$TMP"
if [ -n "$GROUP" ]; then
    chown "root:$GROUP" "$TMP"
    chmod 640 "$TMP"
fi
mv -f "$TMP" "$FILE"

echo "wrote $FILE ($(stat -c '%a %U:%G' "$FILE"))"
echo
echo "key: $KEY"
echo
echo "serve the MCP server over HTTP with:"
echo "  inferwatch-mcp --transport streamable-http --port 7071"
echo
echo "and point a client at http://127.0.0.1:7071/mcp with header:"
echo "  Authorization: Bearer $KEY"
if [ -n "$GROUP" ]; then
    echo
    echo "the unit's user must be in the '$GROUP' group to read it."
fi
