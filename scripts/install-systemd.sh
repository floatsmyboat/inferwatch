#!/usr/bin/env bash
# Render a systemd unit for THIS machine and install it.
#
#   ./scripts/install-systemd.sh              # the collector, system-wide
#   ./scripts/install-systemd.sh --user       # the collector, systemctl --user
#   ./scripts/install-systemd.sh --mcp        # the MCP server over HTTP
#   ./scripts/install-systemd.sh --mcp --user
#
# Override any of: HOST, PORT, MCPHOST, MCPPORT, DATADIR, PYTHON
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="system"
UNIT="inferwatch"
for arg in "$@"; do
  case "$arg" in
    --user) MODE="user" ;;
    --mcp)  UNIT="inferwatch-mcp" ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

TEMPLATE="$REPO/systemd/$UNIT.service.in"
[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE" >&2; exit 1; }

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7070}"
MCPHOST="${MCPHOST:-127.0.0.1}"
MCPPORT="${MCPPORT:-7071}"
DATADIR="${DATADIR:-${XDG_DATA_HOME:-$HOME/.local/share}/inferwatch}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

mkdir -p "$DATADIR"

render() {
  sed -e "s|@@USER@@|$(id -un)|g" \
      -e "s|@@GROUP@@|$(id -gn)|g" \
      -e "s|@@DIR@@|$REPO|g" \
      -e "s|@@PYTHON@@|$PYTHON|g" \
      -e "s|@@DATADIR@@|$DATADIR|g" \
      -e "s|@@HOST@@|$HOST|g" \
      -e "s|@@PORT@@|$PORT|g" \
      -e "s|@@MCPHOST@@|$MCPHOST|g" \
      -e "s|@@MCPPORT@@|$MCPPORT|g" \
      "$TEMPLATE"
}

if [ "$MODE" = "user" ]; then
  # A --user unit runs as the invoking user by definition; User=/Group= and
  # SupplementaryGroups= are rejected there, so drop them.
  DEST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT.service"
  mkdir -p "$(dirname "$DEST")"
  render | grep -vE '^(User|Group|SupplementaryGroups)=' > "$DEST"
  systemctl --user daemon-reload
  echo "installed $DEST"
  echo
  echo "next:"
  echo "  systemctl --user enable --now $UNIT"
  echo "  loginctl enable-linger $(id -un)   # so it runs without an active login"
else
  DEST=/etc/systemd/system/$UNIT.service
  render | sudo tee "$DEST" >/dev/null
  sudo systemctl daemon-reload
  echo "installed $DEST"
  echo
  echo "next:"
  echo "  sudo systemctl enable --now $UNIT"
fi
if [ "$UNIT" = "inferwatch-mcp" ]; then
  echo "  MCP will listen on http://$MCPHOST:$MCPPORT/mcp"
  echo "  it needs a key first, readable by $(id -gn):"
  echo "    sudo $REPO/scripts/install-mcp-key.sh --rotate --group $(id -gn)"
  if [ "$MCPHOST" = "0.0.0.0" ]; then
    echo "  NOTE: bound to all interfaces. The API key is required, but it"
    echo "        crosses the wire in clear text, so restrict it at your"
    echo "        firewall and put TLS in front if it leaves this machine."
  fi
else
  echo "  dashboard will listen on http://$HOST:$PORT"
  [ "$HOST" = "0.0.0.0" ] && echo "  NOTE: bound to all interfaces and unauthenticated - restrict it at your firewall"
fi
exit 0
