#!/usr/bin/env bash
# Render systemd/inferwatch.service.in for THIS machine and install it.
#
#   ./scripts/install-systemd.sh              # system-wide (needs sudo)
#   ./scripts/install-systemd.sh --user       # systemctl --user, no sudo
#
# Override any of: HOST, PORT, DATADIR, PYTHON
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$REPO/systemd/inferwatch.service.in"
[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE" >&2; exit 1; }

MODE="system"
[ "${1:-}" = "--user" ] && MODE="user"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7070}"
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
      "$TEMPLATE"
}

if [ "$MODE" = "user" ]; then
  # A --user unit runs as the invoking user by definition; User=/Group= and
  # SupplementaryGroups= are rejected there, so drop them.
  DEST="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/inferwatch.service"
  mkdir -p "$(dirname "$DEST")"
  render | grep -vE '^(User|Group|SupplementaryGroups)=' > "$DEST"
  systemctl --user daemon-reload
  echo "installed $DEST"
  echo
  echo "next:"
  echo "  systemctl --user enable --now inferwatch"
  echo "  loginctl enable-linger $(id -un)   # so it runs without an active login"
else
  DEST=/etc/systemd/system/inferwatch.service
  render | sudo tee "$DEST" >/dev/null
  sudo systemctl daemon-reload
  echo "installed $DEST"
  echo
  echo "next:"
  echo "  sudo systemctl enable --now inferwatch"
fi
echo "  dashboard will listen on http://$HOST:$PORT"
[ "$HOST" = "0.0.0.0" ] && echo "  NOTE: bound to all interfaces and unauthenticated - restrict it at your firewall"
