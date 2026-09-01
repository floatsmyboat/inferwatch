"""HTTP API + dashboard host.

The dashboard fetches ONE composite endpoint (/api/dashboard) per refresh
rather than one call per panel.  That is deliberate: every tile, chart and
table then describes the same time slice, so the numbers on screen always
agree with each other even while traffic is arriving.

/api/stream is a server-sent-event feed for the live row: per-request
completions, mid-generation token rates, and GPU samples.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import metrics, vllm_metrics
from .config import SOURCE_KINDS, ConfigError, validate_source

log = logging.getLogger("inferwatch.api")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
MAX_SSE_QUEUE = 500


class LiveHub:
    """Fan-out for SSE subscribers.  Slow clients get dropped events, not backpressure."""

    def __init__(self):
        self.subscribers: set[asyncio.Queue] = set()
        self.recent: list[dict] = []

    def publish(self, event: dict) -> None:
        if event.get("type") in ("request", "gen"):
            self.recent.append(event)
            del self.recent[:-200]
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=MAX_SSE_QUEUE)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)


class AppState:
    """Shared handles the routes read from."""

    def __init__(self, store, hub: LiveHub, config=None, supervisor=None,
                 debug_enabled_getter=None, should_exit=None):
        self.store = store
        self.hub = hub
        self.config = config
        self.supervisor = supervisor
        self._debug_enabled_getter = debug_enabled_getter or (lambda: None)
        self._debug_cached = None
        self._debug_checked = 0.0
        # Long-lived SSE responses would otherwise hold uvicorn's graceful
        # shutdown open until systemd loses patience and SIGKILLs us, which
        # drops the un-committed journal cursor.  The stream loop polls this so
        # it can close itself the moment shutdown begins.
        self.should_exit = should_exit or (lambda: False)
        self.started = time.time()

    @property
    def debug_enabled(self):
        """Cached: shelling out to systemctl on every request would be silly."""
        now = time.time()
        if now - self._debug_checked > 60:
            self._debug_checked = now
            self._debug_cached = self._debug_enabled_getter()
        return self._debug_cached

    def ollama_corr(self):
        rt = self.supervisor.ollama_runtime() if self.supervisor else None
        return rt.corr if rt else None


def _window(window: str) -> tuple[float, float]:
    try:
        return metrics.bounds(window)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="inferwatch", docs_url="/api/docs", redoc_url=None)
    app.state.inferwatch = state

    # ---------------------------------------------------------------- static

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    # ---------------------------------------------------------------- health

    @app.get("/api/health")
    async def health():
        st = state.store
        row = st.query("SELECT COUNT(*) n, MAX(ts) last FROM requests")[0]
        ev = st.query("SELECT MAX(ts) last FROM events")[0]
        sup = state.supervisor.status() if state.supervisor else {}
        corr = state.ollama_corr()
        db_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            path = st.path + suffix
            if os.path.exists(path):
                db_bytes += os.path.getsize(path)
        return {
            "ok": True,
            "uptime_s": time.time() - state.started,
            "rows": row["n"],
            "last_request_ts": row["last"],
            "last_event_ts": ev["last"],
            "inflight": getattr(corr, "inflight", None),
            "correlator_stats": dict(getattr(corr, "stats", {}) or {}),
            # If OLLAMA_DEBUG is off, llama.cpp stops printing timing lines and
            # every token metric goes silently null.  Surface it loudly.
            "debug_logging": state.debug_enabled,
            "db_path": st.path,
            "db_bytes": db_bytes,
            "supervisor": sup,
            "lines_read": sum(s.get("lines_read") or 0 for s in sup.get("sources", [])),
        }

    @app.get("/api/status")
    async def status():
        return state.supervisor.status() if state.supervisor else {}

    # ------------------------------------------------------------- settings

    @app.get("/api/config")
    async def get_config():
        if not state.config:
            raise HTTPException(status_code=503, detail="configuration unavailable")
        return state.config.describe()

    @app.put("/api/config")
    async def put_config(values: dict = Body(...)):
        """Save a batch of settings.  All-or-nothing, so a typo in one field
        cannot leave a half-applied configuration behind."""
        if not state.config:
            raise HTTPException(status_code=503, detail="configuration unavailable")
        try:
            return state.config.set_many(values or {})
        except ConfigError as e:
            raise HTTPException(status_code=400, detail={"errors": e.errors}) from None

    @app.post("/api/config/reset")
    async def reset_config(body: dict = Body(...)):
        key = (body or {}).get("key")
        if not state.config:
            raise HTTPException(status_code=503, detail="configuration unavailable")
        try:
            state.config.reset(key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown setting {key!r}") from None
        return {"reset": key, "value": state.config.get(key)}

    # -------------------------------------------------------------- sources

    @app.get("/api/sources")
    async def list_sources():
        return {"sources": state.store.list_sources(), "kinds": SOURCE_KINDS,
                "running": (state.supervisor.status() if state.supervisor else {})}

    @app.post("/api/sources")
    async def add_source(body: dict = Body(...)):
        kind = (body or {}).get("kind")
        name = (body or {}).get("name")
        cfg = (body or {}).get("config") or {}
        try:
            clean = validate_source(kind, name, cfg)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        if any(s["name"] == name for s in state.store.list_sources()):
            raise HTTPException(status_code=409, detail=f"a source named {name!r} exists")
        sid = state.store.add_source(kind, name, clean,
                                    enabled=bool((body or {}).get("enabled", True)))
        _reload()
        return {"id": sid}

    @app.put("/api/sources/{source_id}")
    async def edit_source(source_id: int, body: dict = Body(...)):
        existing = next((s for s in state.store.list_sources() if s["id"] == source_id),
                        None)
        if existing is None:
            raise HTTPException(status_code=404, detail="no such source")
        cfg = body.get("config")
        if cfg is not None:
            try:
                cfg = validate_source(existing["kind"], body.get("name")
                                      or existing["name"], cfg)
            except (ValueError, KeyError) as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
        state.store.update_source(source_id, name=body.get("name"), config=cfg,
                                  enabled=body.get("enabled"))
        _reload()
        return {"ok": True}

    @app.delete("/api/sources/{source_id}")
    async def remove_source(source_id: int):
        state.store.delete_source(source_id)
        _reload()
        return {"ok": True}

    @app.post("/api/sources/probe")
    async def probe_source(body: dict = Body(...)):
        """Check a source definition before it is saved, so a typo shows up
        here rather than as silence in the charts."""
        kind = (body or {}).get("kind")
        cfg = (body or {}).get("config") or {}
        loop = asyncio.get_running_loop()
        try:
            validate_source(kind, (body or {}).get("name") or "probe", cfg)
        except (ValueError, KeyError) as e:
            return {"ok": False, "detail": str(e)}
        return await loop.run_in_executor(None, _probe, kind, cfg)

    def _reload():
        if state.supervisor:
            state.supervisor.request_reload()

    # ----------------------------------------------------------------- vllm

    @app.get("/api/vllm/instances")
    async def vllm_instances():
        return {"instances": vllm_metrics.instances(state.store)}

    @app.get("/api/vllm/dashboard")
    async def vllm_dashboard(source: str | None = None, window: str = Query("1h"),
                             step: int | None = None):
        start, end = _window(window)
        st = state.store
        loop = asyncio.get_running_loop()
        inst = vllm_metrics.instances(st)
        if source is None:
            source = inst[0]["source"] if inst else None
        if source is None:
            return {"window": window, "start": start, "end": end, "instances": [],
                    "source": None, "summary": None, "timeseries": None,
                    "hint": "No vLLM source configured. Add one in Settings."}

        def build():
            return {
                "window": window, "start": start, "end": end, "now": time.time(),
                "source": source, "instances": inst,
                "summary": vllm_metrics.summary(st, source, start, end),
                "timeseries": vllm_metrics.timeseries(st, source, start, end, step),
                "gpu": metrics.gpu_series(st, start, end, step),
            }

        return await loop.run_in_executor(None, build)

    @app.get("/api/vllm/summary")
    async def vllm_summary(source: str, window: str = "1h"):
        start, end = _window(window)
        return vllm_metrics.summary(state.store, source, start, end)

    @app.get("/api/vllm/timeseries")
    async def vllm_timeseries(source: str, window: str = "1h", step: int | None = None):
        start, end = _window(window)
        return vllm_metrics.timeseries(state.store, source, start, end, step)

    # ------------------------------------------------------------- composite

    @app.get("/api/prefs")
    async def prefs():
        """Dashboard defaults, so the client honours the settings screen."""
        cfg = state.config
        if not cfg:
            return {"default_window": "1h", "refresh_s": 10.0, "include_health": False}
        return {"default_window": cfg.get("dashboard.default_window"),
                "refresh_s": cfg.get("dashboard.refresh_s"),
                "include_health": cfg.get("dashboard.include_health")}

    @app.get("/api/dashboard")
    async def dashboard(window: str = Query("1h"), model: str | None = None,
                        step: int | None = None, include_health: bool | None = None):
        start, end = _window(window)
        st = state.store
        loop = asyncio.get_running_loop()
        if include_health is None:
            include_health = bool(state.config.get("dashboard.include_health")) \
                if state.config else False

        raw_from = metrics.raw_coverage(st)

        def build():
            return {
                "window": window,
                "start": start,
                "end": end,
                "now": time.time(),
                "summary": metrics.summary(st, start, end, model),
                "timeseries": metrics.timeseries(st, start, end, step, model),
                "models": metrics.by_model(st, start, end),
                "endpoints": metrics.by_endpoint(st, start, end),
                "clients": metrics.by_client(st, start, end),
                # by_client can only answer back to the oldest raw row; the
                # dashboard says so rather than implying a full history.
                "clients_complete": raw_from is not None and raw_from <= start,
                "clients_from": raw_from,
                "statuses": metrics.status_breakdown(st, start, end),
                "errors": metrics.recent_errors(st, start, end, 25),
                "slowest": metrics.slowest(st, start, end, "ttft_ms", 10),
                "slowest_queue": metrics.slowest(st, start, end, "queue_ms", 10),
                "requests": metrics.recent_requests(st, start, end, 100, include_health),
                "include_health": include_health,
                "events": metrics.events(st, start, end, None, 40),
                "gpu": metrics.gpu_series(st, start, end, step),
                "cache": metrics.cache_summary(st, start, end),
                "cache_series": metrics.cache_series(st, start, end, step),
                "live": metrics.loaded_models(st),
                "model_names": metrics.model_names(st),
            }

        # SQLite reads are blocking; keep the event loop free for SSE.
        return await loop.run_in_executor(None, build)

    # -------------------------------------------------------------- granular

    @app.get("/api/summary")
    async def summary(window: str = "1h", model: str | None = None):
        start, end = _window(window)
        return metrics.summary(state.store, start, end, model)

    @app.get("/api/timeseries")
    async def timeseries(window: str = "1h", step: int | None = None,
                         model: str | None = None):
        start, end = _window(window)
        return metrics.timeseries(state.store, start, end, step, model)

    @app.get("/api/models")
    async def models(window: str = "1h"):
        start, end = _window(window)
        return metrics.by_model(state.store, start, end)

    @app.get("/api/errors")
    async def errors(window: str = "24h", limit: int = 100):
        start, end = _window(window)
        return metrics.recent_errors(state.store, start, end, limit)

    @app.get("/api/slowest")
    async def slowest(window: str = "1h", by: str = "ttft_ms", limit: int = 20):
        start, end = _window(window)
        try:
            return metrics.slowest(state.store, start, end, by, limit)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.get("/api/requests")
    async def requests_(window: str = "1h", limit: int = 200,
                        include_health: bool = False):
        start, end = _window(window)
        return metrics.recent_requests(state.store, start, end, limit, include_health)

    @app.get("/api/events")
    async def events(window: str = "24h", kind: str | None = None, limit: int = 100):
        start, end = _window(window)
        return metrics.events(state.store, start, end, kind, limit)

    @app.get("/api/gpu")
    async def gpu(window: str = "1h", step: int | None = None):
        start, end = _window(window)
        return metrics.gpu_series(state.store, start, end, step)

    @app.get("/api/clients")
    async def clients(window: str = "1h", limit: int = 25):
        """Per-client detail: models requested, context sizes, tokens, TTFT."""
        start, end = _window(window)
        rows = metrics.by_client(state.store, start, end, limit)
        covers_from = metrics.raw_coverage(state.store)
        return {"window": window, "start": start, "end": end, "clients": rows,
                # Client identity lives only on raw rows, so the answer is
                # complete only back to the oldest surviving one.  Stated
                # outright so a short list is not read as "few clients called".
                "covers_from": covers_from,
                "complete": covers_from is not None and covers_from <= start}

    @app.get("/api/cache")
    async def cache(window: str = "1h", step: int | None = None):
        """Ollama's prompt-cache occupancy: the one KV gauge it publishes."""
        start, end = _window(window)
        return {"summary": metrics.cache_summary(state.store, start, end),
                "series": metrics.cache_series(state.store, start, end, step),
                # Reported for context, not as the explanation: it reflects
                # whether OLLAMA_DEBUG is set on the unit, and ollama 0.32.x
                # logs these lines even when it is not.  Zero samples usually
                # means no cache updates ran, which `summary.note` says.
                "debug_logging": state.debug_enabled}

    @app.get("/api/ps")
    async def ps():
        return metrics.loaded_models(state.store)

    # ------------------------------------------------------------------- SSE

    @app.get("/api/stream")
    async def stream(request: Request):
        q = await state.hub.subscribe()

        async def gen():
            try:
                # Prime the connection so the client renders immediately.
                yield _sse({"type": "hello", "ts": time.time(),
                            "recent": state.hub.recent[-40:],
                            "live": metrics.loaded_models(state.store)})
                idle = 0.0
                while True:
                    if state.should_exit() or await request.is_disconnected():
                        break
                    try:
                        # Short poll so shutdown is noticed promptly; the
                        # keepalive still only goes out every ~15s.
                        ev = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        idle += 1.0
                        if idle >= 15.0:
                            idle = 0.0
                            yield ": keepalive\n\n"  # stops proxies idling us out
                        continue
                    idle = 0.0
                    yield _sse(ev)
            finally:
                state.hub.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        })

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


def _probe(kind: str, cfg: dict) -> dict:
    """Synchronously check that a source is actually reachable."""
    if kind == "vllm":
        from .vllm import ScrapeError, Snapshot, fetch, parse_prometheus
        url = (cfg.get("url") or "").rstrip("/")
        try:
            text = fetch(f"{url}/metrics", cfg.get("api_key") or "", timeout=6.0)
        except ScrapeError as e:
            return {"ok": False, "detail": f"could not read {url}/metrics: {e}"}
        snap = Snapshot(time.time(), parse_prometheus(text))
        if not snap.hist and not snap.counters:
            return {"ok": False, "detail": f"{url}/metrics responded but exposed no "
                                           "vllm: metrics -- is this a vLLM server?"}
        return {"ok": True, "detail": f"reachable; model {snap.model or 'unknown'}, "
                                      f"{len(snap.hist)} histograms, "
                                      f"{len(snap.counters)} counters"}
    if kind == "ollama":
        reader = cfg.get("reader") or "journald"
        notes = []
        if reader == "journald":
            try:
                out = subprocess.run(
                    ["journalctl", "-u", cfg.get("unit") or "ollama", "-n", "1",
                     "--no-pager", "-o", "cat"],
                    capture_output=True, text=True, timeout=8)
            except (subprocess.SubprocessError, OSError) as e:
                return {"ok": False, "detail": f"journalctl failed: {e}"}
            if out.returncode != 0:
                return {"ok": False, "detail": (out.stderr or "journalctl failed").strip()[:200]}
            if not out.stdout.strip():
                notes.append("unit found but its journal is empty")
        elif reader == "file":
            path = os.path.expanduser(cfg.get("path") or "")
            if not os.path.isfile(path):
                return {"ok": False, "detail": f"no such file: {path}"}
            if not os.access(path, os.R_OK):
                return {"ok": False, "detail": f"not readable: {path}"}
        elif reader == "docker":
            exe = shutil.which("docker") or shutil.which("podman")
            if not exe:
                return {"ok": False, "detail": "docker/podman not found on PATH"}
            try:
                out = subprocess.run([exe, "logs", "--tail", "1",
                                      cfg.get("container") or ""],
                                     capture_output=True, text=True, timeout=10)
            except (subprocess.SubprocessError, OSError) as e:
                return {"ok": False, "detail": f"{exe} logs failed: {e}"}
            if out.returncode != 0:
                return {"ok": False, "detail": (out.stderr or "container not found").strip()[:200]}

        url = (cfg.get("url") or "").rstrip("/")
        if url:
            try:
                with urllib.request.urlopen(url + "/api/version", timeout=5) as r:
                    ver = json.loads(r.read()).get("version")
                notes.append(f"ollama {ver} reachable at {url}")
            except Exception:
                notes.append(f"log source OK, but {url} did not answer /api/version")
        return {"ok": True, "detail": "; ".join(notes) or f"{reader} source looks readable"}
    return {"ok": False, "detail": f"unknown source kind {kind!r}"}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=_json_default)}\n\n"


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)
