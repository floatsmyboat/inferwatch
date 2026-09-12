"""API-key authentication for the MCP server's HTTP transports.

WHY THIS EXISTS
---------------
Over stdio the MCP server needs no authentication: the client spawns it as a
subprocess and owns both ends of the pipe.  Over HTTP it is a network service
answering questions about the whole metrics database -- client addresses, which
models each one used, prompt sizes -- and `run_sql` is a general read-only query
tool over all of it.  So an HTTP transport without a key is a data leak waiting
for someone to find the port, which is why one is REQUIRED rather than offered.

WHERE THE KEY LIVES
-------------------
Never in the repository and never in the metrics database.  Resolution order:

    1. --api-key-file PATH          (explicit, wins)
    2. INFERWATCH_MCP_API_KEY       (the value; for a systemd EnvironmentFile)
    3. INFERWATCH_MCP_API_KEY_FILE  (a path)
    4. /etc/inferwatch/mcp-api-key  (the default)

A world-readable key file is refused rather than warned about: a secret every
local user can read is not a secret, and failing to start is the only response
that cannot be ignored.
"""

from __future__ import annotations

import hmac
import logging
import os
import stat

log = logging.getLogger("inferwatch.mcp_auth")

DEFAULT_KEY_FILE = "/etc/inferwatch/mcp-api-key"
ENV_KEY = "INFERWATCH_MCP_API_KEY"
ENV_KEY_FILE = "INFERWATCH_MCP_API_KEY_FILE"
# Shorter than this is not worth the ceremony of having a key at all.
MIN_KEY_LEN = 16


class KeyError_(Exception):
    """A key was configured but cannot be used."""


def read_key_file(path: str) -> str:
    """Read a key from a file, refusing an unsafe one.

    The file is allowed to be group-readable (a service group is the normal way
    to share it with the unit) but never world-readable.
    """
    try:
        st = os.stat(path)
    except OSError as e:
        raise KeyError_(f"cannot read {path}: {e}") from None
    if st.st_mode & stat.S_IROTH:
        raise KeyError_(
            f"{path} is world-readable (mode {stat.S_IMODE(st.st_mode):o}); "
            f"run: chmod 640 {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        raise KeyError_(f"cannot read {path}: {e}") from None
    # Tolerate a trailing newline and an accidental KEY=value line, since an
    # EnvironmentFile and a bare key file look alike to a tired operator.
    text = text.strip()
    if "=" in text.split("\n", 1)[0] and not text.startswith("="):
        head = text.split("\n", 1)[0]
        name, _, value = head.partition("=")
        if name.strip().isidentifier():
            text = value.strip().strip('"').strip("'")
    if not text:
        raise KeyError_(f"{path} is empty")
    return text


def load_api_key(explicit_file: str | None = None, environ=None) -> tuple[str | None, str]:
    """Find the API key.  Returns (key, where-it-came-from).

    (None, reason) when nothing is configured, so the caller can decide whether
    that is fatal -- it is, for an HTTP transport.
    """
    env = environ if environ is not None else os.environ
    if explicit_file:
        return read_key_file(explicit_file), f"--api-key-file {explicit_file}"
    if env.get(ENV_KEY):
        return env[ENV_KEY].strip(), f"${ENV_KEY}"
    if env.get(ENV_KEY_FILE):
        path = env[ENV_KEY_FILE]
        return read_key_file(path), f"${ENV_KEY_FILE} ({path})"
    if os.path.exists(DEFAULT_KEY_FILE):
        return read_key_file(DEFAULT_KEY_FILE), DEFAULT_KEY_FILE
    return None, f"no key: set ${ENV_KEY} or create {DEFAULT_KEY_FILE}"


def check_key_strength(key: str) -> None:
    if len(key) < MIN_KEY_LEN:
        raise KeyError_(
            f"the API key is only {len(key)} characters; use at least "
            f"{MIN_KEY_LEN} (openssl rand -hex 32)")


def presented_key(headers: list) -> str | None:
    """The key a request presented, from either accepted header.

    `Authorization: Bearer <key>` is the MCP convention; `X-API-Key` is
    accepted too because some clients can only set arbitrary headers.
    """
    for raw_name, raw_value in headers or []:
        name = raw_name.decode("latin-1").lower() if isinstance(raw_name, bytes) else str(raw_name).lower()
        value = raw_value.decode("latin-1") if isinstance(raw_value, bytes) else str(raw_value)
        if name == "authorization":
            scheme, _, token = value.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()
        elif name == "x-api-key" and value.strip():
            return value.strip()
    return None


class BearerAuth:
    """ASGI middleware requiring a matching API key on every HTTP request.

    A plain ASGI wrapper rather than a framework middleware, so it works
    unchanged whichever app the MCP SDK hands back and can be tested without
    starting a server.
    """

    def __init__(self, app, key: str):
        self.app = app
        self.key = key
        self._denied = 0

    async def __call__(self, scope, receive, send):
        # lifespan and anything non-HTTP must pass straight through, or the
        # app never starts.
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        presented = presented_key(scope.get("headers") or [])
        # compare_digest on every path, so a missing header and a wrong key
        # take the same time and neither reveals the key's length.
        if presented is not None and hmac.compare_digest(presented, self.key):
            return await self.app(scope, receive, send)
        self._denied += 1
        if self._denied in (1, 10) or self._denied % 100 == 0:
            log.warning("MCP request denied (%d so far) from %s: %s",
                        self._denied,
                        (scope.get("client") or ("?",))[0],
                        "no credentials" if presented is None else "bad key")
        return await self._unauthorized(send)

    @staticmethod
    async def _unauthorized(send):
        body = b'{"error":"unauthorized"}'
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"www-authenticate", b'Bearer realm="inferwatch"'),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
