"""Collecting exllama (tabbyAPI): correlating its log, polling what it answers.

tabbyAPI has no `/metrics`, so everything measured here comes from the log
(see parse_exllama for the shapes).  Two things are polled rather than read:
`/health` for reachability and `/v1/model` for the model it is serving,
because a log says nothing while an engine is merely idle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from . import gpuproc, parse_exllama

log = logging.getLogger("inferwatch.exllama")


class ExllamaCorrelator:
    """Joins a request's submission to its completion by the id tabbyAPI prints.

    The id restarts at 1 with the server (each startup writes a new log file),
    so it is unique only within a run.  A "Serving OAI API on ..." line opens
    a new epoch: whatever is still pending belonged to a process that is gone
    and is retired as abandoned.
    """

    def __init__(self, source: str, on_request=None, on_event=None,
                 on_live=None):
        self.source = source
        self.on_request = on_request or (lambda row: None)
        self.on_event = on_event or (lambda row: None)
        self.on_live = on_live or (lambda payload: None)
        self.pending: dict[int, dict] = {}
        self.epoch = 0
        self.model: str | None = None
        self.max_seq_len: int | None = None
        self.cache_size: int | None = None
        self.last_load_ms: float | None = None
        self.stats = {"requests": 0, "errors": 0, "joined": 0,
                      "unpaired": 0, "abandoned": 0}

    def feed(self, ts: float, msg: str) -> None:
        rec = parse_exllama.parse(msg)
        if rec is None:
            return
        kind = rec.pop("kind")
        getattr(self, "_" + kind, self._ignore)(ts, rec)

    def _ignore(self, ts, rec):
        pass

    def _listening(self, ts, rec):
        self._retire_all(ts)
        self.epoch += 1
        self.on_event({"ts": ts, "kind": "start", "model": self.model,
                       "msg": f"listening on {rec.get('url')}"})

    def _loaded(self, ts, rec):
        self.last_load_ms = rec.get("load_ms")
        self.on_event({"ts": ts, "kind": "load", "model": self.model,
                       "duration_ms": rec.get("load_ms"),
                       "msg": "model loaded"})

    def _loading(self, ts, rec):
        self.model = rec.get("model") or self.model

    def _context(self, ts, rec):
        self.max_seq_len = rec.get("max_seq_len")
        self.cache_size = rec.get("cache_size")

    def _submit(self, ts, rec):
        self.pending[rec["req_id"]] = {"started_ts": ts, **rec}

    def _tool_call(self, ts, rec):
        sub = self.pending.get(rec["req_id"])
        if sub is not None:
            sub["tool_calls"] = rec.get("tool_calls")

    def _done(self, ts, rec):
        sub = self.pending.pop(rec["req_id"], None)
        if sub is None:
            self.stats["unpaired"] += 1
        else:
            self.stats["joined"] += 1
        row = self._row(ts, rec, sub)
        self.stats["requests"] += 1
        self.on_request(row)
        self.on_live({"type": "exllama_request", **row})

    def _disconnect(self, ts, rec):
        sub = self.pending.pop(rec["req_id"], None)
        row = self._row(ts, {"finish": "disconnected"}, sub)
        self.stats["requests"] += 1
        self.on_request(row)
        self.on_event({"ts": ts, "kind": "disconnect", "model": self.model,
                       "msg": "client disconnected, generation cancelled"})

    def _error(self, ts, rec):
        self.stats["errors"] += 1
        self.on_event({"ts": ts, "kind": "error", "model": self.model,
                       "msg": rec.get("error") or "request failed"})

    def _row(self, ts: float, rec: dict, sub: dict | None) -> dict:
        sub = sub or {}
        row = {
            "ts": ts,
            "source": self.source,
            "epoch": self.epoch,
            "req_id": rec.get("req_id"),
            "model": self.model,
            "endpoint": sub.get("endpoint") or rec.get("endpoint")
                        or "chat/completions",
            "stream": sub.get("stream"),
            "max_tokens": sub.get("max_tokens"),
            "tool_calls": sub.get("tool_calls"),
            "paired": 1 if sub else 0,
        }
        for k in ("finish", "error", "prompt_tokens", "cached_tokens",
                  "prompt_tokens_total", "output_tokens", "ttft_ms",
                  "latency_ms", "prefill_ms", "decode_ms", "prefill_tps",
                  "decode_tps", "draft_accept", "draft_mean_len"):
            row[k] = rec.get(k)
        if row.get("latency_ms"):
            row["started_ts"] = ts - row["latency_ms"] / 1000.0
        else:
            row["started_ts"] = sub.get("started_ts")
        return row

    def _retire_all(self, ts: float) -> None:
        if self.pending:
            self.stats["abandoned"] += len(self.pending)
            log.info("[%s] %d request(s) abandoned by a server restart",
                     self.source, len(self.pending))
        self.pending.clear()

    @property
    def inflight(self) -> int:
        return len(self.pending)

    def tick(self, now: float) -> None:
        return None


def sample_connections(port: int) -> dict[str, int] | None:
    """Peers holding an established connection to `port`, counted per address."""
    exe = shutil.which("ss")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "-Htn", "state", "established", f"( sport = :{port} )"],
            capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    counts: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        peer = parts[3]
        host = (peer[1:peer.index("]")] if peer.startswith("[")
                else peer.rsplit(":", 1)[0])
        if host:
            counts[host] = counts.get(host, 0) + 1
    return counts


def fetch_json(url: str, api_key: str = "", timeout: float = 5.0):
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


class ExllamaPoller:
    """Reachability, the served model, and GPU attribution."""

    GPU_RECHECK_S = 60.0

    def __init__(self, store, source: str, cfg: dict, corr: ExllamaCorrelator,
                 interval_getter=None, on_live=None):
        self.store = store
        self.source = source
        self.url = (cfg.get("url") or "http://127.0.0.1:8003").rstrip("/")
        self.api_key = cfg.get("api_key") or ""
        self.corr = corr
        self.interval_getter = interval_getter or (lambda: 10.0)
        self.on_live = on_live or (lambda payload: None)
        self.reachable = False
        self.error: str | None = None
        self.gpu_indices: list[int] | None = None
        self.gpu_source = "unavailable"
        self.gpu_ts: float | None = None
        self._gpu_checked = 0.0
        self.connections: dict | None = None
        self.last: dict = {}

    def probe(self) -> dict:
        out: dict = {"reachable": False, "error": None, "model": None}
        try:
            fetch_json(f"{self.url}/health", self.api_key)
            out["reachable"] = True
        except (urllib.error.URLError, OSError, ValueError) as e:
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        try:
            got = fetch_json(f"{self.url}/v1/model", self.api_key)
            if isinstance(got, dict):
                out["model"] = got.get("id") or got.get("model")
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return out

    def resolve_gpus(self, now: float) -> list[int] | None:
        if now - self._gpu_checked < self.GPU_RECHECK_S:
            return self.gpu_indices
        self._gpu_checked = now
        self.gpu_ts = now
        port = (gpuproc.port_of(self.url) if gpuproc.is_local(self.url)
                else None)
        try:
            self.gpu_indices, self.gpu_source = gpuproc.resolve(
                unit=None, port=port)
        except Exception:
            log.debug("gpu attribution failed for %r", self.source,
                      exc_info=True)
            self.gpu_indices, self.gpu_source = None, "unavailable"
        return self.gpu_indices

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                ts = time.time()
                got = await loop.run_in_executor(None, self.probe)
                await loop.run_in_executor(None, self.resolve_gpus, ts)
                self.reachable = got["reachable"]
                self.error = got["error"]
                model = got.get("model") or self.corr.model
                if model:
                    self.corr.model = model
                self.store.upsert_exllama_instance(
                    self.source, last_seen=ts, model=model,
                    reachable=1 if self.reachable else 0, error=self.error,
                    max_seq_len=self.corr.max_seq_len,
                    cache_size=self.corr.cache_size,
                    load_ms=self.corr.last_load_ms,
                    gpu_indices=(json.dumps(self.gpu_indices)
                                 if self.gpu_indices is not None else None),
                    gpu_source=self.gpu_source, gpu_ts=self.gpu_ts)
                port = gpuproc.port_of(self.url)
                if port is not None and gpuproc.is_local(self.url):
                    conns = await loop.run_in_executor(
                        None, sample_connections, port)
                    if conns:
                        self.store.insert_exllama_client_samples(
                            ts, self.source, conns)
                    self.connections = conns
                self.store.commit()
                self.last = {"ts": ts, "reachable": self.reachable,
                             "connections": self.connections,
                             "model": model, "error": self.error,
                             "gpu_indices": self.gpu_indices,
                             "gpu_source": self.gpu_source}
                self.on_live({"type": "exllama_status", **self.last})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("exllama poller tick failed")
            await asyncio.sleep(max(1.0, float(self.interval_getter())))