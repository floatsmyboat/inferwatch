"""inferwatch service entry point.

    python -m inferwatch.main serve      # collectors + dashboard + API
    python -m inferwatch.main ingest     # one-shot ollama log backfill, no follow
    python -m inferwatch.main stats      # what is in the database
    python -m inferwatch.main sources    # list / add / remove monitored engines

One process runs everything: a log reader per Ollama source, a /metrics scraper
per vLLM source, the host GPU poller, the rollup/prune maintainer, and uvicorn.
They share one SQLite connection in WAL mode; the MCP server opens the same file
read-only.

Command-line flags OUTRANK the settings screen: a key passed here is shown
read-only in the UI with its origin, so a value pinned by a systemd unit cannot
be silently changed from a browser.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time

log = logging.getLogger("inferwatch")

DEFAULT_UNIT = "ollama"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"

# The project was called "ollamon" before it grew vLLM support.  Old paths and
# environment variables keep working, with a one-line notice, so an upgrade
# never silently starts collecting into an empty database.
LEGACY_NAME = "ollamon"

# argparse evaluates defaults once per subparser, so notices are emitted once
# each rather than once per parser.
_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logging.getLogger("inferwatch").warning(msg, *args)


def _xdg_data_home() -> str:
    return os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share")


def default_data_dir() -> str:
    """XDG data dir, so nothing is hardcoded to one machine's home."""
    return os.path.join(_xdg_data_home(), "inferwatch")


def default_db() -> str:
    """Preferred database path, falling back to the pre-rename location."""
    new = os.path.join(default_data_dir(), "inferwatch.db")
    if os.path.exists(new):
        return new
    legacy = os.path.join(_xdg_data_home(), LEGACY_NAME, f"{LEGACY_NAME}.db")
    if os.path.exists(legacy):
        _warn_once("legacy-db",
                   "using the pre-rename database at %s; to adopt the new location "
                   "run: mkdir -p %s && mv %s* %s/", legacy, default_data_dir(),
                   legacy, default_data_dir())
        return legacy
    return new


def env(name: str, default=None):
    """Read INFERWATCH_<name>, honouring the legacy OLLAMON_<name> as well."""
    val = os.environ.get(f"INFERWATCH_{name}")
    if val is not None:
        return val
    val = os.environ.get(f"OLLAMON_{name}")
    if val is not None:
        _warn_once(f"legacy-env-{name}",
                   "OLLAMON_%s is deprecated; rename it to INFERWATCH_%s", name, name)
        return val
    return default


def detect_debug_logging(unit: str = DEFAULT_UNIT) -> bool | None:
    """Is OLLAMA_DEBUG set for the ollama service?

    Without it llama.cpp prints no timing lines, so TTFT and token counts are
    unavailable and the dashboard should say so rather than show zeros.  Returns
    None when it cannot be determined -- which is the honest answer for a
    non-systemd install, where there is no unit to interrogate.
    """
    if os.environ.get("OLLAMA_DEBUG"):
        return True
    try:
        out = subprocess.run(["systemctl", "show", unit, "--property=Environment"],
                             capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    if "Environment=" not in out:
        return None
    return "OLLAMA_DEBUG=" in out


# --------------------------------------------------------------------------
# CLI -> config overrides
# --------------------------------------------------------------------------

# Flags that pin a config key.  Their argparse default is None so that "not
# passed" is distinguishable from "passed the same value as the default" --
# otherwise every key would look pinned and the settings screen would be
# entirely read-only.
FLAG_TO_KEY = {
    "host": "server.host",
    "port": "server.port",
    "retention_days": "retention.raw_days",
    "poll_interval": "collection.poll_interval_s",
    "scrape_interval": "collection.scrape_interval_s",
    "since": "collection.backfill",
}


def overrides_from_args(args) -> dict:
    out = {}
    for flag, key in FLAG_TO_KEY.items():
        val = getattr(args, flag, None)
        if val is not None:
            out[key] = val
    return out


def seed_sources(store, args) -> None:
    """On a first run, create one Ollama source so the tool works immediately."""
    if store.list_sources():
        return
    cfg = {
        "reader": "journald",
        "unit": args.unit or DEFAULT_UNIT,
        "url": args.ollama_url or DEFAULT_OLLAMA,
        "models_dir": args.models_dir or "",
    }
    store.add_source("ollama", "ollama", cfg, enabled=True)
    log.info("seeded an Ollama source reading the %r systemd unit; "
             "edit or add sources in the dashboard's Settings tab", cfg["unit"])


def build(args):
    from .api import AppState, LiveHub, create_app
    from .config import Config
    from .store import Store
    from .supervisor import Supervisor

    store = Store(args.db)
    config = Config(store, overrides=overrides_from_args(args))
    seed_sources(store, args)
    hub = LiveHub()
    sup = Supervisor(store, config, hub)
    config.on_change(sup.request_reload)
    state = AppState(store, hub, config=config, supervisor=sup,
                     debug_enabled_getter=lambda: detect_debug_logging(
                         _ollama_unit(store) or DEFAULT_UNIT))
    app = create_app(state)
    return store, config, sup, app, state


def _ollama_unit(store) -> str | None:
    for s in store.list_sources():
        if s["kind"] == "ollama" and (s.get("config") or {}).get("reader") == "journald":
            return (s.get("config") or {}).get("unit")
    return None


async def serve(args) -> None:
    import uvicorn

    store, config, sup, app, state = build(args)
    host = config.get("server.host")
    port = int(config.get("server.port"))
    log.info("db=%s", store.path)
    for s in store.list_sources():
        log.info("source %-16s %-7s %s", s["name"], s["kind"],
                 "enabled" if s["enabled"] else "disabled")

    class GracefulServer(uvicorn.Server):
        """Flush before uvicorn re-raises the signal.

        uvicorn's capture_signals() restores the default handler and then calls
        signal.raise_signal(), so the process dies inside serve() and no
        `finally` further up ever runs.  handle_exit is invoked first, which
        makes it the only place a flush is guaranteed to happen.  Exiting via
        the re-raised signal is uvicorn's intended behaviour, so systemd
        recording ExecMainStatus=15 alongside Result=success is expected.
        """

        def handle_exit(self, sig, frame):
            if store.commit_nowait():
                log.info("flushed collector state on signal %s", sig)
            else:
                log.info("signal %s arrived during a write; flush deferred "
                         "(inserts are idempotent, so the overlap is discarded "
                         "on restart)", sig)
            super().handle_exit(sig, frame)

    config_uv = uvicorn.Config(app, host=host, port=port, log_level="warning",
                               access_log=False, timeout_graceful_shutdown=5)
    server = GracefulServer(config_uv)
    state.should_exit = lambda: server.should_exit

    tasks = [
        asyncio.create_task(sup.run(), name="supervisor"),
        asyncio.create_task(server.serve(), name="http"),
    ]
    log.info("dashboard on http://%s:%d", host, port)
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            if t.exception():
                raise t.exception()
    finally:
        try:
            store.commit()
        except Exception:
            log.exception("failed to flush on shutdown")


def ingest(args) -> None:
    """One-shot: read an Ollama log into the database and build rollups."""
    from .collect import Correlator
    from .models import ModelIndex
    from .readers import message_text
    from .store import Store

    store = Store(args.db)
    index = ModelIndex(args.models_dir) if args.models_dir else ModelIndex()
    n_req = n_ev = 0

    def on_request(r):
        nonlocal n_req
        store.insert_request(r); n_req += 1

    def on_event(e):
        nonlocal n_ev
        store.insert_event(e); n_ev += 1

    corr = Correlator(on_request=on_request, on_event=on_event, model_index=index)
    lines = 0
    t0 = time.time()

    if args.log_file:
        from .readers import derive_timestamp
        last = None
        with open(args.log_file, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                last = derive_timestamp(line, last)
                corr.feed(last, line)
                lines += 1
    else:
        argv = ["journalctl", "-u", args.unit, "-o", "json", "--no-pager",
                "--since", args.since or "-2 days"]
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True, bufsize=1 << 16)
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            ts_us = entry.get("__REALTIME_TIMESTAMP")
            if ts_us is None:
                continue
            msg = message_text(entry.get("MESSAGE"))
            if not msg:
                continue
            corr.feed(int(ts_us) / 1e6, msg)
            lines += 1
        proc.wait()

    corr.tick(time.time() + 1e6)   # force-flush pending joins
    store.commit()
    rows = store.query("SELECT MIN(ts) a, MAX(ts) b FROM requests")[0]
    if rows["a"]:
        store.rebuild_rollups(rows["a"], rows["b"] + 3600)
    print(f"read {lines} log lines in {time.time()-t0:.1f}s -> "
          f"{n_req} requests, {n_ev} events")
    print("correlator:", dict(corr.stats))


def stats(args) -> None:
    from .store import Store
    if not os.path.exists(args.db):
        print(f"no database at {args.db}; run `ingest` or `serve` first")
        return
    store = Store(args.db, read_only=True)
    for table in ("requests", "events", "gpu_samples", "ps_samples", "rollup_1m",
                  "rollup_1h", "vllm_samples", "vllm_hist", "sources", "config"):
        try:
            n = store.query(f"SELECT COUNT(*) n FROM {table}")[0]["n"]
        except Exception as e:
            n = f"error: {e}"
        print(f"  {table:14} {n}")
    row = store.query("SELECT MIN(ts) a, MAX(ts) b FROM requests")[0]
    if row["a"]:
        print(f"  span           {time.ctime(row['a'])}  ->  {time.ctime(row['b'])}")
    print(f"  db bytes       {os.path.getsize(args.db) if os.path.exists(args.db) else 0}")


def sources_cmd(args) -> None:
    """List, add, enable/disable or remove monitored engines from the CLI."""
    from .config import SOURCE_KINDS, validate_source
    from .store import Store
    store = Store(args.db)

    if args.action in (None, "list"):
        rows = store.list_sources()
        if not rows:
            print("no sources configured")
            return
        for s in rows:
            flag = "on " if s["enabled"] else "off"
            detail = " ".join(f"{k}={v}" for k, v in (s["config"] or {}).items() if v)
            print(f"  [{s['id']:>3}] {flag} {s['kind']:<7} {s['name']:<16} {detail}")
        return

    if args.action == "add":
        if args.kind not in SOURCE_KINDS:
            raise SystemExit(f"--kind must be one of {sorted(SOURCE_KINDS)}")
        cfg = {}
        for item in args.set or []:
            if "=" not in item:
                raise SystemExit(f"--set expects key=value, got {item!r}")
            k, _, v = item.partition("=")
            cfg[k.strip()] = v.strip()
        try:
            cfg = validate_source(args.kind, args.name, cfg)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        sid = store.add_source(args.kind, args.name, cfg)
        print(f"added source {args.name!r} (id {sid})")
        return

    target = next((s for s in store.list_sources()
                   if s["name"] == args.name or str(s["id"]) == str(args.name)), None)
    if target is None:
        raise SystemExit(f"no source named {args.name!r}")
    if args.action == "remove":
        store.delete_source(target["id"])
        print(f"removed {target['name']!r}")
    elif args.action in ("enable", "disable"):
        store.update_source(target["id"], enabled=(args.action == "enable"))
        print(f"{args.action}d {target['name']!r}")


def _common(p) -> None:
    """Flags accepted both before and after the subcommand."""
    p.add_argument("--db", default=env("DB") or default_db())
    p.add_argument("--unit", default=env("UNIT", DEFAULT_UNIT),
                   help="systemd unit for the seeded Ollama source / for ingest")
    p.add_argument("--ollama-url", default=env("OLLAMA_URL", DEFAULT_OLLAMA))
    p.add_argument("--models-dir", default=os.environ.get("OLLAMA_MODELS"),
                   help="ollama models dir (resolves blob digests to model names)")
    p.add_argument("--log-file", default=None,
                   help="ingest: read this log file instead of the journal")
    # These pin config keys; default None means "not specified".
    p.add_argument("--since", default=None, help="log backfill window, e.g. '-2 days'")
    p.add_argument("--retention-days", type=float, default=None,
                   help="raw request retention; rollups are kept forever")
    p.add_argument("--poll-interval", type=float, default=None)
    p.add_argument("--scrape-interval", type=float, default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="inferwatch",
                                description="Local LLM serving metrics")
    _common(p)
    sub = p.add_subparsers(dest="cmd")
    for name, help_text in (("serve", "run collectors + dashboard"),
                            ("ingest", "one-shot log backfill"),
                            ("stats", "show database contents")):
        _common(sub.add_parser(name, help=help_text))
    sp = sub.add_parser("sources", help="manage monitored engines")
    _common(sp)
    sp.add_argument("action", nargs="?", default="list",
                    choices=["list", "add", "remove", "enable", "disable"])
    sp.add_argument("name", nargs="?", help="source name (or id)")
    sp.add_argument("--kind", default="vllm", help="ollama | vllm")
    sp.add_argument("--set", action="append",
                    help="config entry as key=value; repeatable")
    args = p.parse_args(argv)
    if args.cmd is None:
        args.cmd = "serve"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.cmd == "ingest":
        ingest(args)
    elif args.cmd == "stats":
        stats(args)
    elif args.cmd == "sources":
        sources_cmd(args)
    else:
        try:
            asyncio.run(serve(args))
        except KeyboardInterrupt:
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
