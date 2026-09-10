"""Runtime configuration: a typed spec, stored in SQLite, editable from the UI.

Precedence, lowest to highest:

    spec default  <  database  <  environment  <  command line

The database layer is what the settings screen writes.  Anything supplied by
the environment or the command line *wins over* it, so a key pinned by a
systemd unit or a flag cannot be silently changed from the browser -- the
screen shows it read-only, with the reason.  `effective()` resolves the whole
chain; `describe()` returns the spec plus provenance for rendering.

Monitored engines live in their own `sources` table rather than in a config
key, because they are a variable-length list the UI needs to create and delete
rows in.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

# --------------------------------------------------------------------------
# setting spec
# --------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhdw]$")
# A backfill window may also be given the way journalctl spells it, because
# that is what the --since flag has always accepted: "-7 days", "-1h",
# "2 hours ago", "@1787000000".  Those are passed through untouched; the
# compact form is canonical and each reader converts it to its own dialect.
_JOURNALCTL_SINCE_RE = re.compile(
    r"^(?:@\d+|-\s*\d+\s*[a-z]*|\d+\s*[a-z]+\s+ago|today|yesterday)$",
    re.IGNORECASE)


class Setting:
    def __init__(self, key: str, label: str, type_: str, default: Any, group: str,
                 help: str = "", choices: list | None = None,
                 minimum: float | None = None, maximum: float | None = None,
                 restart: bool = False, env: str | None = None):
        self.key = key
        self.label = label
        self.type = type_          # int | float | bool | str | enum | duration
        self.default = default
        self.group = group
        self.help = help
        self.choices = choices
        self.minimum = minimum
        self.maximum = maximum
        self.restart = restart     # takes effect only after a process restart
        self.env = env or ("INFERWATCH_" + key.upper().replace(".", "_"))

    def coerce(self, value: Any) -> Any:
        """Validate and convert an incoming value, raising ValueError if bad."""
        if self.type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("1", "true", "yes", "on"):
                    return True
                if low in ("0", "false", "no", "off"):
                    return False
            raise ValueError(f"{self.key}: expected a boolean, got {value!r}")
        if self.type in ("int", "float"):
            try:
                out = int(value) if self.type == "int" else float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.key}: expected a number, got {value!r}") from None
            if self.minimum is not None and out < self.minimum:
                raise ValueError(f"{self.key}: must be >= {self.minimum}")
            if self.maximum is not None and out > self.maximum:
                raise ValueError(f"{self.key}: must be <= {self.maximum}")
            return out
        if self.type == "enum":
            if value not in (self.choices or []):
                raise ValueError(f"{self.key}: must be one of {self.choices}")
            return value
        if self.type == "duration":
            text = str(value).strip().lower()
            if not _DURATION_RE.match(text):
                raise ValueError(
                    f"{self.key}: expected a duration like 30s, 15m, 6h, 7d, got {value!r}")
            return text
        if self.type == "backfill":
            text = str(value).strip()
            if _DURATION_RE.match(text.lower()) or _JOURNALCTL_SINCE_RE.match(text):
                return text
            raise ValueError(
                f"{self.key}: expected a window like 7d, 6h, or a journalctl "
                f"form such as '-2 days' or '@1787000000', got {value!r}")
        return str(value)

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "type": self.type,
                "default": self.default, "group": self.group, "help": self.help,
                "choices": self.choices, "min": self.minimum, "max": self.maximum,
                "restart": self.restart, "env": self.env}


# Groups are rendered in this order by the settings screen.
GROUPS = ["Collection", "Retention", "Dashboard", "Server"]

SPEC: list[Setting] = [
    Setting("collection.poll_interval_s", "GPU / model poll interval", "float", 5.0,
            "Collection", minimum=1.0, maximum=300.0,
            help="How often nvidia-smi and the engine's own status endpoint are sampled."),
    Setting("collection.scrape_interval_s", "vLLM scrape interval", "float", 10.0,
            "Collection", minimum=1.0, maximum=300.0,
            help="How often each vLLM instance's /metrics endpoint is read. vLLM "
                 "counters are cumulative, so this sets the resolution of every "
                 "rate and histogram derived from them."),
    Setting("collection.backfill", "Log backfill window", "backfill", "2d",
            "Collection", restart=True,
            help="How far back to read on a first run, before any resume state "
                 "exists. Accepts 7d / 6h, or a journalctl form like '-2 days'."),
    Setting("collection.rollup_interval_s", "Rollup rebuild interval", "float", 60.0,
            "Collection", minimum=10.0, maximum=3600.0,
            help="How often the 1-minute and 1-hour aggregates are recomputed."),

    Setting("retention.raw_days", "Keep raw request rows", "float", 7.0,
            "Retention", minimum=0.5, maximum=3650.0,
            help="Per-request detail older than this is deleted. Rollups are kept "
                 "indefinitely regardless, so long-range charts survive."),
    Setting("retention.sample_days", "Keep GPU / engine samples", "float", 30.0,
            "Retention", minimum=0.5, maximum=3650.0,
            help="GPU samples, engine samples and the event log are trimmed to this."),

    Setting("dashboard.default_window", "Default time range", "enum", "1h",
            "Dashboard", choices=["15m", "1h", "6h", "24h", "7d", "30d"],
            help="Range selected when the dashboard is opened."),
    Setting("dashboard.include_health", "Count health-check traffic", "bool", False,
            "Dashboard",
            help="Include HEAD / and status polling in request rates. Off by "
                 "default because on a polled instance it can be 90%+ of hits."),
    Setting("dashboard.refresh_s", "Auto-refresh interval", "float", 10.0,
            "Dashboard", minimum=2.0, maximum=600.0,
            help="How often the open dashboard refetches. The live request feed "
                 "is pushed separately and is not affected by this."),

    Setting("server.host", "Bind address", "str", "127.0.0.1", "Server", restart=True,
            help="0.0.0.0 exposes the dashboard on the network. There is no "
                 "authentication, so restrict it at your firewall."),
    Setting("server.port", "Port", "int", 7070, "Server", restart=True,
            minimum=1, maximum=65535,
            help="Port the dashboard and API listen on."),
]

BY_KEY = {s.key: s for s in SPEC}


# --------------------------------------------------------------------------
# source spec (monitored engines)
# --------------------------------------------------------------------------

OLLAMA_READERS = ["journald", "file", "docker"]
# SwarmUI can be monitored over HTTP alone, so its log is optional in a way
# ollama's is not -- ollama has no metrics endpoint at all, while SwarmUI and
# ComfyUI expose queue depth, VRAM and the generations themselves over their
# APIs.  Reading the journal adds the prep/gen timing split and the backend
# stderr behind a failure.
SWARM_READERS = ["journald", "file", "docker", "none"]

SOURCE_KINDS = {
    "ollama": {
        "label": "Ollama",
        "fields": [
            {"key": "reader", "label": "Log source", "type": "enum",
             "choices": OLLAMA_READERS, "default": "journald",
             "help": "Where to read ollama's log from. Per-request metrics come "
                     "from llama.cpp's debug lines, so one of these is required."},
            {"key": "unit", "label": "systemd unit", "type": "str", "default": "ollama",
             "when": {"reader": "journald"},
             "help": "Also used to attribute GPUs: every process in this unit's "
                     "cgroup is matched against nvidia-smi, so only the cards "
                     "ollama actually holds appear on its pane."},
            {"key": "path", "label": "Log file path", "type": "str", "default": "",
             "when": {"reader": "file"},
             "help": "Followed like tail -F, so rotation and truncation are handled."},
            {"key": "container", "label": "Container name or id", "type": "str",
             "default": "ollama", "when": {"reader": "docker"}},
            {"key": "url", "label": "API base URL", "type": "str",
             "default": "http://127.0.0.1:11434",
             "help": "Used to poll /api/ps for resident models."},
            {"key": "models_dir", "label": "Models directory", "type": "str", "default": "",
             "help": "Optional. Resolves blob digests to model names on load events. "
                     "Defaults to $OLLAMA_MODELS or ~/.ollama/models."},
        ],
    },
    "swarmui": {
        "label": "SwarmUI / ComfyUI",
        "fields": [
            {"key": "url", "label": "SwarmUI base URL", "type": "str",
             "default": "http://127.0.0.1:7801",
             "help": "Polled for queue depth and backend health. Its API needs "
                     "no key for a local install."},
            {"key": "reader", "label": "Log source", "type": "enum",
             "choices": SWARM_READERS, "default": "journald",
             "help": "Optional, unlike ollama's. Adds SwarmUI's prep-vs-gen "
                     "timing split, WebAPI failures that never reached a "
                     "backend, and the Python stderr behind a failed "
                     "generation. 'none' polls the APIs only."},
            {"key": "unit", "label": "systemd unit", "type": "str",
             "default": "swarmui", "when": {"reader": "journald"},
             "help": "Also the fallback for GPU attribution, though each "
                     "self-started ComfyUI is usually resolved exactly by its "
                     "own port."},
            {"key": "path", "label": "Log file path", "type": "str", "default": "",
             "when": {"reader": "file"},
             "help": "SwarmUI's own rotated logs under Data/Logs work here."},
            {"key": "container", "label": "Container name or id", "type": "str",
             "default": "", "when": {"reader": "docker"}},
            {"key": "backends", "label": "ComfyUI backend URLs", "type": "str",
             "default": "",
             "help": "Optional, comma separated. Leave empty and the backend "
                     "ports are discovered from the log, which is the only "
                     "place SwarmUI publishes them. Required when the reader "
                     "is 'none'."},
            {"key": "history_limit", "label": "Generations per poll", "type": "str",
             "default": "64",
             "help": "How many /history entries to read each tick. ComfyUI "
                     "keeps this in memory only, so a larger number costs "
                     "little and survives a burst between polls."},
        ],
    },
    "vllm": {
        "label": "vLLM",
        "fields": [
            {"key": "url", "label": "Base URL", "type": "str",
             "default": "http://127.0.0.1:8000",
             "help": "The OpenAI-compatible server root. /metrics is read from here."},
            {"key": "unit", "label": "systemd unit (optional)", "type": "str", "default": "",
             "help": "The unit running the ENGINE, which is what its GPUs are "
                     "attributed by -- every process in that unit's cgroup is "
                     "matched against nvidia-smi. Set it to the engine's unit, "
                     "not a proxy in front of it: with a proxy on the URL there "
                     "are no GPUs behind that port and attribution reports "
                     "'could not attribute'. The journal is also read from it "
                     "for HTTP status codes, client addresses and engine errors, "
                     "which /metrics does not expose."},
            {"key": "api_key", "label": "API key (optional)", "type": "str",
             "default": "", "secret": True,
             "help": "Sent as a bearer token if the server requires one. "
                     "Prefer an indirection like ${VLLM_API_KEY} over pasting "
                     "the value: what is stored here goes into the database in "
                     "plain text, and a reference keeps the secret in the "
                     "environment or an EnvironmentFile instead."},
        ],
    },
}


# What a redacted secret reads as, and the value a client may send back to mean
# "leave it as it is".  A round-trip through the settings screen must not
# overwrite a real key with its own mask, which is the classic way redaction
# turns into data loss.
REDACTED = "***redacted***"

# "${NAME}" or "$NAME" -- an indirection, not a secret, so it is shown rather
# than masked: seeing WHICH variable is referenced is the useful part.
_VAR_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def secret_fields(kind: str) -> set:
    """Keys in a source's config whose value must never be served back."""
    return {f["key"] for f in SOURCE_KINDS.get(kind, {}).get("fields", [])
            if f.get("secret")}


def is_secret_ref(value) -> bool:
    """True when the whole value is an environment reference, not a literal."""
    text = str(value or "").strip()
    if not text:
        return False
    return _VAR_REF.sub("", text) == ""


def resolve_secret(value, environ=None) -> str:
    """Expand ${VAR} / $VAR references in a stored secret.

    Lets the secret live in the environment (or a systemd EnvironmentFile)
    while the database holds only a pointer to it.  An unset variable resolves
    to empty rather than to the literal "${VAR}", so the request goes out with
    no Authorization header and fails as a clean 401 instead of sending a
    nonsense bearer token.
    """
    text = str(value or "")
    if not text:
        return ""
    env = environ if environ is not None else __import__("os").environ
    missing = []

    def sub(m):
        name = m.group(1) or m.group(2)
        got = env.get(name)
        if got is None:
            missing.append(name)
            return ""
        return got

    out = _VAR_REF.sub(sub, text)
    if missing:
        import logging
        logging.getLogger("inferwatch.config").warning(
            "secret references unset environment variable(s) %s; requesting "
            "without credentials", ", ".join(sorted(set(missing))))
    return out


def redact_config(kind: str, cfg: dict) -> dict:
    """A copy of `cfg` safe to serve: literal secrets replaced by REDACTED.

    An empty value stays empty, so a caller can still tell "nothing is set"
    from "set, and you may not read it".
    """
    out = dict(cfg or {})
    for key in secret_fields(kind):
        value = out.get(key)
        if value and not is_secret_ref(value):
            out[key] = REDACTED
    return out


def redact_sources(rows: list) -> list:
    """Redact every source in a list_sources() result."""
    return [{**s, "config": redact_config(s.get("kind", ""), s.get("config") or {})}
            for s in rows]


def merge_secrets(kind: str, incoming: dict, stored: dict | None) -> dict:
    """Apply a client's config edit without letting a mask overwrite a secret.

    REDACTED means "unchanged", so the stored value is kept.  Any other value
    -- including the empty string -- is taken at face value, so clearing a key
    still works.
    """
    out = dict(incoming or {})
    for key in secret_fields(kind):
        if out.get(key) == REDACTED:
            if stored and stored.get(key) is not None:
                out[key] = stored[key]
            else:
                out.pop(key, None)
    return out


def source_defaults(kind: str) -> dict:
    fields = SOURCE_KINDS[kind]["fields"]
    return {f["key"]: f.get("default", "") for f in fields}


def validate_source(kind: str, name: str, cfg: dict) -> dict:
    """Check a source definition, returning the cleaned config."""
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind {kind!r}; "
                         f"expected one of {sorted(SOURCE_KINDS)}")
    if not (name or "").strip():
        raise ValueError("a source needs a name")
    out = source_defaults(kind)
    for f in SOURCE_KINDS[kind]["fields"]:
        if f["key"] in cfg and cfg[f["key"]] is not None:
            val = cfg[f["key"]]
            if f["type"] == "enum" and val not in f["choices"]:
                raise ValueError(f"{f['key']}: must be one of {f['choices']}")
            out[f["key"]] = val if isinstance(val, (int, float, bool)) else str(val).strip()

    if kind == "ollama":
        reader = out.get("reader") or "journald"
        required = {"journald": "unit", "file": "path", "docker": "container"}[reader]
        if not out.get(required):
            raise ValueError(f"reader '{reader}' requires '{required}' to be set")
    if kind == "vllm" and not out.get("url"):
        raise ValueError("a vLLM source needs a base URL")
    if kind == "swarmui":
        if not out.get("url"):
            raise ValueError("a SwarmUI source needs a base URL")
        reader = out.get("reader") or "journald"
        required = {"journald": "unit", "file": "path", "docker": "container"}.get(reader)
        if required and not out.get(required):
            raise ValueError(f"reader '{reader}' requires '{required}' to be set")
        # With no log there is nothing to discover the backend ports from, so
        # they have to be given -- otherwise the source would silently collect
        # queue depth and no generations at all.
        if reader == "none" and not out.get("backends"):
            raise ValueError(
                "reader 'none' requires 'backends' (comma-separated ComfyUI "
                "URLs); the backend ports are otherwise only discoverable from "
                "the log")
        limit = str(out.get("history_limit") or "64")
        if not limit.isdigit() or not (1 <= int(limit) <= 1000):
            raise ValueError("history_limit must be a whole number from 1 to 1000")
    return out


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

class Config:
    """Resolves the precedence chain and persists UI edits."""

    def __init__(self, store, environ: dict | None = None, overrides: dict | None = None):
        self.store = store
        self.environ = environ if environ is not None else __import__("os").environ
        # Keys pinned on the command line; these outrank everything.
        self.overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        self._listeners: list[Callable[[], None]] = []

    # -- resolution ------------------------------------------------------

    def _from_env(self, s: Setting):
        raw = self.environ.get(s.env)
        if raw is None:
            legacy = s.env.replace("INFERWATCH_", "OLLAMON_", 1)
            raw = self.environ.get(legacy)
        if raw is None:
            return None
        try:
            return s.coerce(raw)
        except ValueError:
            return None

    def get(self, key: str):
        s = BY_KEY.get(key)
        if s is None:
            raise KeyError(key)
        if key in self.overrides:
            return s.coerce(self.overrides[key])
        env_val = self._from_env(s)
        if env_val is not None:
            return env_val
        raw = self.store.get_config(key)
        if raw is not None:
            try:
                return s.coerce(json.loads(raw))
            except (ValueError, TypeError):
                pass
        return s.default

    def origin(self, key: str) -> str:
        """Which layer supplied the effective value."""
        s = BY_KEY[key]
        if key in self.overrides:
            return "flag"
        if self._from_env(s) is not None:
            return "env"
        if self.store.get_config(key) is not None:
            return "database"
        return "default"

    def locked(self, key: str) -> bool:
        """True when a higher layer pins the value, so the UI must not edit it."""
        return self.origin(key) in ("flag", "env")

    def effective(self) -> dict:
        return {s.key: self.get(s.key) for s in SPEC}

    def describe(self) -> dict:
        """Everything the settings screen needs to render itself."""
        return {
            "groups": GROUPS,
            "settings": [
                {**s.to_dict(), "value": self.get(s.key), "origin": self.origin(s.key),
                 "locked": self.locked(s.key)}
                for s in SPEC
            ],
            "source_kinds": SOURCE_KINDS,
            # Redacted: this is what the settings screen renders, and it is
            # served by an API with no authentication.
            "sources": redact_sources(self.store.list_sources()),
        }

    # -- writes ----------------------------------------------------------

    def set_many(self, values: dict) -> dict:
        """Validate and persist a batch of edits.  All-or-nothing.

        Returns which of the applied keys need a restart to take effect.
        """
        cleaned = {}
        errors = {}
        for key, value in values.items():
            s = BY_KEY.get(key)
            if s is None:
                errors[key] = "unknown setting"
                continue
            if self.locked(key):
                errors[key] = f"pinned by {self.origin(key)}; cannot be changed here"
                continue
            try:
                cleaned[key] = s.coerce(value)
            except ValueError as e:
                errors[key] = str(e)
        if errors:
            raise ConfigError(errors)
        for key, value in cleaned.items():
            self.store.set_config(key, json.dumps(value))
        self.store.commit()
        self.notify()
        return {"applied": cleaned,
                "restart_required": sorted(k for k in cleaned if BY_KEY[k].restart)}

    def reset(self, key: str) -> None:
        if key not in BY_KEY:
            raise KeyError(key)
        self.store.delete_config(key)
        self.store.commit()
        self.notify()

    # -- change notification --------------------------------------------

    def on_change(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:  # noqa: BLE001 - a bad listener must not block a save
                import logging
                logging.getLogger("inferwatch.config").exception(
                    "config change listener failed")


class ConfigError(Exception):
    """Per-key validation errors from a settings save."""

    def __init__(self, errors: dict):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


def now() -> float:
    return time.time()
