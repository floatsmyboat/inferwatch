"""Collecting NInfer: correlating its log, and polling what it will answer.

NInfer has no `/metrics`, so everything measured here comes from the log (see
parse_ninfer for the shapes).  Two things are polled rather than read: `/health`
for reachability and `/v1/models` for the model it is serving, because a log
says nothing while an engine is merely idle.
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

from . import gpuproc, parse_ninfer

log = logging.getLogger("inferwatch.ninfer")


class NinferCorrelator:
    """Joins a request's submission to its completion by the id NInfer prints.

    Unlike the vLLM proxy alongside it, every line here carries `[req N]`, so
    this is a lookup and not an ordering guess -- which matters, because that
    proxy's requests overlap essentially always and the same guess there would
    be wrong most of the time.

    The id restarts at 1 with the server, so it is unique only within a run.
    `listening on ...` opens a new epoch: whatever is still pending belonged to
    a process that is gone and is retired as abandoned, rather than left to be
    completed by a same-numbered request from the new one.
    """

    def __init__(self, source: str, on_request=None, on_sample=None,
                 on_event=None, on_live=None):
        self.source = source
        self.on_request = on_request or (lambda row: None)
        self.on_sample = on_sample or (lambda row: None)
        self.on_event = on_event or (lambda row: None)
        self.on_live = on_live or (lambda payload: None)
        self.pending: dict[int, dict] = {}
        self.epoch = 0
        self.model: str | None = None
        self.kv_tokens: int | None = None
        self.last_load_ms: float | None = None
        self.last_sample: dict | None = None
        self.stats = {"requests": 0, "errors": 0, "samples": 0,
                      "joined": 0, "unpaired": 0, "abandoned": 0}

    # -- one line ---------------------------------------------------------

    def feed(self, ts: float, msg: str) -> None:
        rec = parse_ninfer.parse(msg)
        if rec is None:
            return
        kind = rec.pop("kind")
        getattr(self, "_" + kind, self._ignore)(ts, rec)

    def _ignore(self, ts, rec):
        pass

    def _listening(self, ts, rec):
        """A new server process: the id space restarts here."""
        self._retire_all(ts)
        self.epoch += 1
        self.model = rec.get("model") or self.model
        self.on_event({"ts": ts, "kind": "start", "model": self.model,
                       "msg": f"listening on {rec.get('url')}"})

    def _loaded(self, ts, rec):
        self.last_load_ms = rec.get("load_ms")
        self.on_event({"ts": ts, "kind": "load", "model": self.model,
                       "duration_ms": rec.get("load_ms"),
                       "msg": "model loaded"})

    def _kv(self, ts, rec):
        self.kv_tokens = rec.get("kv_tokens")

    def _submit(self, ts, rec):
        self.pending[rec["req_id"]] = {"started_ts": ts, **rec}

    def _done(self, ts, rec):
        sub = self.pending.pop(rec["req_id"], None)
        if sub is None:
            self.stats["unpaired"] += 1
        else:
            self.stats["joined"] += 1
        row = self._row(ts, rec, sub)
        self.stats["requests"] += 1
        self.on_request(row)
        self.on_live({"type": "ninfer_request", **row})

    def _error(self, ts, rec):
        sub = self.pending.pop(rec["req_id"], None)
        row = self._row(ts, {"finish": "error", **rec}, sub)
        self.stats["requests"] += 1
        self.stats["errors"] += 1
        self.on_request(row)
        self.on_event({"ts": ts, "kind": "error", "model": self.model,
                       "msg": rec.get("error") or "request failed"})

    def _throughput(self, ts, rec):
        row = {"ts": ts, "source": self.source, **rec}
        self.stats["samples"] += 1
        self.last_sample = row
        self.on_sample(row)
        self.on_live({"type": "ninfer_sample", **row})

    # -- helpers ----------------------------------------------------------

    def _row(self, ts: float, rec: dict, sub: dict | None) -> dict:
        """A stored request row: the completion, plus what submission knew.

        Submission carries the shape of the ask (streaming, tool count, how
        many messages, the cap requested); completion carries what it cost.
        A completion whose submission was never seen -- the reader started mid
        request -- still stores everything the completion states rather than
        being dropped for want of the other half.
        """
        sub = sub or {}
        row = {
            "ts": ts,
            "source": self.source,
            "epoch": self.epoch,
            "req_id": rec.get("req_id"),
            "model": self.model,
            "endpoint": sub.get("endpoint") or "openai_chat_completions",
            "stream": sub.get("stream"),
            "messages": sub.get("messages"),
            "max_tokens": sub.get("max_tokens"),
            "tools": sub.get("tools"),
            "thinking": sub.get("thinking"),
            "paired": 1 if sub else 0,
        }
        for k in ("finish", "tool_calls", "prompt_tokens", "cached_tokens",
                  "prompt_tokens_total", "output_tokens", "reuse", "ttft_ms",
                  "prefill_tps", "decode_tps", "latency_ms", "decode_ms",
                  "speculator", "draft_mean_len", "draft_accept", "error"):
            row[k] = rec.get(k)
        # The completion states its own wall time, so when the request began
        # is known whether or not its submission was ever read.  Deriving this
        # only for paired requests left an unpaired one spanning zero seconds,
        # which no connection sample can cover -- so it could never be
        # attributed to a client, for want of a line that says nothing about
        # when it started.
        if row.get("latency_ms"):
            row["started_ts"] = ts - row["latency_ms"] / 1000.0
        else:
            row["started_ts"] = sub.get("started_ts")
        return row

    def _retire_all(self, ts: float) -> None:
        """Drop requests left open by a process that has gone away.

        They are counted, not stored: nothing is known about how they ended,
        and inventing a finish reason for them would put a guess beside
        measurements.
        """
        if self.pending:
            self.stats["abandoned"] += len(self.pending)
            log.info("[%s] %d request(s) abandoned by a server restart",
                     self.source, len(self.pending))
        self.pending.clear()

    @property
    def inflight(self) -> int:
        return len(self.pending)

    def tick(self, now: float) -> None:
        """Periodic flush hook, kept for symmetry with the ollama correlator."""
        return None


def sample_connections(port: int) -> dict[str, int] | None:
    """Peers holding an established connection to `port`, counted per address.

    The only source of client identity NInfer has: it logs none, and unlike
    vLLM there is no proxy in front to ask.  None means the socket table could
    not be read at all, which is different from {} meaning nobody is connected.

    Ports are dropped and addresses counted, so two keep-alive sockets from one
    machine read as one client holding two connections rather than as two.
    """
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
        # IPv6 peers are bracketed, so splitting on ":" would cut the address.
        host = peer[1:peer.index("]")] if peer.startswith("[") else peer.rsplit(":", 1)[0]
        if host:
            counts[host] = counts.get(host, 0) + 1
    return counts


def fetch_json(url: str, api_key: str = "", timeout: float = 5.0):
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


class NinferPoller:
    """Reachability, the served model, and GPU attribution.

    The log goes quiet when the engine is merely idle, which is
    indistinguishable from the engine being gone if nothing asks.  `/health`
    asks.
    """

    GPU_RECHECK_S = 60.0

    def __init__(self, store, source: str, cfg: dict, corr: NinferCorrelator,
                 interval_getter=None, on_live=None):
        self.store = store
        self.source = source
        self.url = (cfg.get("url") or "http://127.0.0.1:8011").rstrip("/")
        self.api_key = cfg.get("api_key") or ""
        self.unit = (cfg.get("unit") or "").strip()
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
            got = fetch_json(f"{self.url}/v1/models", self.api_key)
            data = (got or {}).get("data") or []
            if data:
                out["model"] = data[0].get("id")
        except (urllib.error.URLError, OSError, ValueError):
            pass            # health answered; the model name is a bonus
        return out

    def resolve_gpus(self, now: float) -> list[int] | None:
        if now - self._gpu_checked < self.GPU_RECHECK_S:
            return self.gpu_indices
        self._gpu_checked = now
        self.gpu_ts = now
        port = (gpuproc.port_of(self.url) if gpuproc.is_local(self.url) else None)
        try:
            self.gpu_indices, self.gpu_source = gpuproc.resolve(
                unit=self.unit or None, port=port)
        except Exception:
            log.debug("gpu attribution failed for %r", self.source, exc_info=True)
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
                self.store.upsert_ninfer_instance(
                    self.source, last_seen=ts, model=model,
                    reachable=1 if self.reachable else 0, error=self.error,
                    kv_tokens=self.corr.kv_tokens,
                    load_ms=self.corr.last_load_ms,
                    gpu_indices=(json.dumps(self.gpu_indices)
                                 if self.gpu_indices is not None else None),
                    gpu_source=self.gpu_source, gpu_ts=self.gpu_ts)
                port = gpuproc.port_of(self.url)
                if port is not None and gpuproc.is_local(self.url):
                    conns = await loop.run_in_executor(
                        None, sample_connections, port)
                    if conns:
                        self.store.insert_ninfer_client_samples(
                            ts, self.source, conns)
                    self.connections = conns
                self.store.commit()
                self.last = {"ts": ts, "reachable": self.reachable,
                             "connections": self.connections,
                             "model": model, "error": self.error,
                             "gpu_indices": self.gpu_indices,
                             "gpu_source": self.gpu_source}
                self.on_live({"type": "ninfer_status", **self.last})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ninfer poller tick failed")
            await asyncio.sleep(max(1.0, float(self.interval_getter())))
