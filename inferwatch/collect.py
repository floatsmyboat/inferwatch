"""Journal follower, request correlator, and GPU/model pollers.

THE CORRELATION PROBLEM
-----------------------
Ollama's logs split one logical request across three voices that never share an
id.  For a single /v1/chat/completions call this host emits, in order:

    slot ... | task 6763 | new prompt, ... task.n_tokens = 1408
    slot ... | task 6763 | prompt eval time = 1254.52 ms / 55 tokens (...)
    slot ... | task 6763 | eval time = 14591.31 ms / 416 tokens (...)
    slot ... | task 6763 | total time = 15845.83 ms / 471 tokens
    slot ... | task 6763 | stop processing: n_tokens = 131071, truncated = 1
    [GIN] ... | 200 | 16.862061865s | 192.0.2.10 | POST "/v1/chat/completions"
    time=... msg="context for request finished" runner.name=.../llama3.2:3b

The task id is llama.cpp's; the access line has the status/client/endpoint but
no task id; the model NAME only appears afterwards on the scheduler line.  We
join them by arrival order: a task whose `total time` has printed is queued,
and the next inference access line claims the oldest queued task.

That join is exact while one request is in flight.  With OLLAMA_NUM_PARALLEL=2
(this host) two tasks can finish before either access line prints, and nothing
in the log disambiguates them -- so those rows are stored with
attribution='ambiguous' rather than pretending.  Per-request numbers stay
correct in aggregate either way; only the model/client pairing is uncertain.
Set attribution='none' when an inference request had no timings at all (it
failed before reaching the runner, e.g. the 500s in this host's history).
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import shutil
import subprocess
import time
import urllib.request

from .models import ModelIndex
from .parse import parse_line

log = logging.getLogger("inferwatch.collect")

# A finished task waits this long for an access line to claim it before being
# written unattributed.  Generous: the access line follows within milliseconds.
JOIN_TIMEOUT_S = 30.0
# A joined row waits this long for the scheduler line that names the model.
MODEL_TIMEOUT_S = 2.0
# How long a task that printed "total time" may wait for its "release" line
# (which carries the truncation flag) before being retired without it.
TOTAL_GRACE_S = 5.0


class Correlator:
    """Turns a stream of (timestamp, message) into request rows and events.

    Pure and synchronous -- no I/O -- so a captured journal can be replayed
    through it in tests and produce byte-identical rows.
    """

    def __init__(self, on_request=None, on_event=None, on_live=None,
                 model_index: ModelIndex | None = None):
        self.on_request = on_request or (lambda r: None)
        self.on_event = on_event or (lambda e: None)
        self.on_live = on_live or (lambda x: None)
        # Resolves the blob digests on load lines to real model names.
        self.model_index = model_index

        self.tasks: dict[int, dict] = {}          # task_id -> accumulating metrics
        self.finished: collections.deque = collections.deque()  # awaiting access line
        self.awaiting_model: collections.deque = collections.deque()  # awaiting model name
        self.blob_to_model: dict[str, str] = {}
        self.loaded_models: set[str] = set()
        self.loaded_count: int = 0
        self.recent_model: str | None = None
        self.pending_load: dict | None = None
        self._last_load: tuple[str | None, float] = (None, 0.0)
        self.last_ts: float = 0.0
        self.stats = collections.Counter()

    # -- public entry point --------------------------------------------------

    def feed(self, ts: float, msg: str) -> None:
        self.last_ts = ts
        ev = parse_line(msg)
        if ev is None:
            return
        kind = ev["kind"]
        if kind == "gin":
            self._on_gin(ts, ev)
        elif kind == "go":
            self._on_go(ts, ev)
        else:
            self._on_slot(ts, ev)
        self._expire(ts)

    def tick(self, now: float | None = None) -> None:
        """Flush rows whose join/model windows have elapsed (call periodically)."""
        self._expire(now if now is not None else time.time(), force_model=True)

    @property
    def inflight(self) -> int:
        return len(self.tasks)

    # -- slot lines ----------------------------------------------------------

    def _on_slot(self, ts: float, ev: dict) -> None:
        tid = ev["task_id"]
        if tid < 0:  # task -1: slot bookkeeping, not a request
            return
        kind = ev["kind"]

        if kind == "gen_progress":
            # Live decode rate, mid-stream.  Not stored per row; drives the
            # realtime view only.
            self.on_live({"type": "gen", "ts": ts, "task_id": tid,
                          "n_gen": ev["n_gen"], "tps": ev["tg"], "tps_3s": ev["tg_3s"],
                          "model": self.tasks.get(tid, {}).get("model") or self.recent_model})
            return

        t = self.tasks.get(tid)
        if t is None:
            t = self.tasks[tid] = {"task_id": tid, "slot_id": ev.get("slot_id"),
                                   "first_ts": ts, "model": None}
        t["slot_id"] = ev.get("slot_id", t.get("slot_id"))

        if kind == "launch":
            t["launch_ts"] = ts
        elif kind == "new_prompt":
            t["prompt_tokens_total"] = ev["prompt_tokens_total"]
            t["n_ctx_slot"] = ev.get("n_ctx_slot")
        elif kind == "prefill_progress":
            t["prefill_tps"] = ev.get("prefill_tps")
        elif kind == "prompt_eval":
            t["ttft_ms"] = ev["ttft_ms"]
            t["prompt_tokens"] = ev["prompt_tokens"]
            t["prefill_tps"] = ev["prefill_tps"]
        elif kind == "eval":
            t["decode_ms"] = ev["decode_ms"]
            t["output_tokens"] = ev["output_tokens"]
            t["decode_tps"] = ev["decode_tps"]
        elif kind == "draft":
            t["draft_accept"] = ev["draft_accept"]
            t["draft_mean_len"] = ev["draft_mean_len"]
        elif kind == "total":
            t["total_ms"] = ev["total_ms"]
            t["done_ts"] = ts
        elif kind == "release":
            t["context_tokens"] = ev["context_tokens"]
            t["truncated"] = ev["truncated"]
            if ev["truncated"]:
                self.on_event({"ts": ts, "kind": "truncation", "level": "WARN",
                               "model": t.get("model") or self.recent_model,
                               "source": "llama-server",
                               "msg": f"context truncated at {ev['context_tokens']} tokens",
                               "detail": {"task_id": tid}})
            # release is the last slot line for the task: queue it for joining.
            self._retire(tid, ts)
            return

    def _retire(self, tid: int, ts: float) -> None:
        t = self.tasks.pop(tid, None)
        if t is None:
            return
        if "total_ms" not in t and "ttft_ms" not in t:
            return  # nothing measurable; drop it
        t["retire_ts"] = ts
        self.finished.append(t)
        self.stats["tasks_retired"] += 1

    # -- access lines --------------------------------------------------------

    def _on_gin(self, ts: float, ev: dict) -> None:
        cls = ev["class"]
        row = {
            "ts": ts,
            "endpoint": ev["endpoint"],
            "method": ev["method"],
            "class": cls,
            "status": ev["status"],
            "client_ip": ev["client_ip"],
            "latency_ms": ev["latency_ms"],
        }
        if ev["latency_ms"] is not None:
            row["started_ts"] = ts - ev["latency_ms"] / 1000.0

        if cls in ("inference", "embed"):
            claimed = self.finished.popleft() if self.finished else None
            if claimed is not None:
                ambiguous = len(self.finished) > 0
                row.update({k: v for k, v in claimed.items() if k in (
                    "ttft_ms", "decode_ms", "total_ms", "prompt_tokens",
                    "prompt_tokens_total", "output_tokens", "prefill_tps", "decode_tps",
                    "draft_accept", "draft_mean_len", "truncated", "context_tokens",
                    "slot_id", "task_id")})
                if row.get("prompt_tokens_total") is not None and row.get("prompt_tokens") is not None:
                    row["cached_tokens"] = max(0, row["prompt_tokens_total"] - row["prompt_tokens"])
                if row.get("latency_ms") is not None and row.get("total_ms") is not None:
                    # Wall clock minus runner time = scheduler + queue + HTTP.
                    row["queue_ms"] = max(0.0, row["latency_ms"] - row["total_ms"])
                row["attribution"] = "ambiguous" if ambiguous else "exact"
                row["model"] = claimed.get("model")
                self.stats["joined_" + row["attribution"]] += 1
            else:
                row["attribution"] = "none"
                self.stats["joined_none"] += 1
            # Hold briefly: the model name arrives on the NEXT line.
            self.awaiting_model.append((ts, row))
        else:
            self._emit(row)

    # -- go lines ------------------------------------------------------------

    def _on_go(self, ts: float, ev: dict) -> None:
        model = ev.get("model")
        blob = ev.get("blob")
        if model:
            self.recent_model = model
            if blob:
                self.blob_to_model[blob] = model
        elif blob:
            model = self.blob_to_model.get(blob)
            if not model and self.model_index is not None:
                model = self.model_index.resolve(blob)
                if model:
                    self.blob_to_model[blob] = model

        event = ev.get("event")

        if event == "request_finished" and model:
            # Names the model for the access line we just held.  A row with
            # attribution 'none' never reached a runner (e.g. an unknown-model
            # 404), so this line does not belong to it -- skip past it rather
            # than labelling it with whatever ran most recently.
            for idx, (_ts, row) in enumerate(self.awaiting_model):
                if row.get("attribution") == "none":
                    continue
                if not row.get("model"):
                    row["model"] = model
                del self.awaiting_model[idx]
                self._emit(row)
                break
            return

        if event == "load_start":
            self.pending_load = {"ts": ts, "blob": blob, "num_ctx": ev.get("num_ctx"),
                                 "parallel": ev.get("parallel"), "spec_type": ev.get("spec_type"),
                                 "port": ev.get("port")}
        elif event == "model_loaded":
            detail = dict(self.pending_load or {})
            blob_ref = detail.get("blob")
            name = model or self.blob_to_model.get(blob_ref or "")
            if not name and self.model_index is not None:
                name = self.model_index.resolve(blob_ref)
            if name:
                self.loaded_models.add(name)
            # Ollama prints "llama-server started in Ns" once per llama-server
            # handle (main model and projector), so the same load surfaces
            # twice a few ms apart.  Keep the first.
            # Keep pending_load intact: the second line arrives with no
            # load_start of its own, and clearing it would strip the blob the
            # dedup key needs.  load_start overwrites it on the next real load.
            prev_blob, prev_ts = self._last_load
            if ts - prev_ts < 2.0 and (blob_ref is None or prev_blob == blob_ref):
                return
            self._last_load = (blob_ref or prev_blob, ts)
            self.on_event({"ts": ts, "kind": "model_loaded", "level": "INFO", "model": name,
                           "source": ev.get("source"), "msg": ev["msg"],
                           "duration_ms": ev.get("load_ms"), "detail": detail or None})
        elif event == "unload":
            if model:
                self.loaded_models.discard(model)
            self.on_event({"ts": ts, "kind": "unload", "level": "INFO", "model": model,
                           "source": ev.get("source"), "msg": ev["msg"], "detail": None})
        elif event == "runner_count":
            self.loaded_count = ev.get("count") or 0
        elif event == "idle_timer":
            self.on_event({"ts": ts, "kind": "idle_timer", "level": "INFO", "model": model,
                           "source": ev.get("source"), "msg": ev["msg"],
                           "detail": {"keep_alive": ev.get("keep_alive")}})
        elif event == "problem":
            self.on_event({"ts": ts, "kind": "problem", "level": ev["level"], "model": model,
                           "source": ev.get("source"), "msg": ev["msg"],
                           "detail": ev.get("kv") or None})

    # -- flush ---------------------------------------------------------------

    def _expire(self, now: float, force_model: bool = False) -> None:
        # Rows still waiting for a model name.
        while self.awaiting_model:
            ts, row = self.awaiting_model[0]
            if not force_model and now - ts < MODEL_TIMEOUT_S:
                break
            self.awaiting_model.popleft()
            if not row.get("model") and row.get("attribution") != "none":
                # Unambiguous only when exactly one model is loaded.
                if len(self.loaded_models) == 1:
                    row["model"] = next(iter(self.loaded_models))
                else:
                    row["model"] = self.recent_model
                    if row.get("attribution") == "exact":
                        row["attribution"] = "ambiguous"
            self._emit(row)

        # Tasks nobody claimed (client disconnected before the access line).
        while self.finished and now - self.finished[0]["retire_ts"] > JOIN_TIMEOUT_S:
            t = self.finished.popleft()
            row = {"ts": t["retire_ts"], "endpoint": "(unclaimed)", "method": None,
                   "class": "inference", "status": None, "client_ip": None,
                   "attribution": "orphan", "model": t.get("model") or self.recent_model}
            for k in ("ttft_ms", "decode_ms", "total_ms", "prompt_tokens", "prompt_tokens_total",
                      "output_tokens", "prefill_tps", "decode_tps", "draft_accept",
                      "draft_mean_len", "truncated", "context_tokens", "slot_id", "task_id"):
                if k in t:
                    row[k] = t[k]
            self.stats["orphans"] += 1
            self._emit(row)

        # A task that printed its timings but never released (client aborted,
        # runner died) is retired after a short grace so its metrics are not
        # lost -- 'release' normally follows 'total time' within milliseconds.
        for tid, t in [(k, v) for k, v in self.tasks.items()
                       if v.get("done_ts") and now - v["done_ts"] > TOTAL_GRACE_S]:
            self._retire(tid, t["done_ts"])

        # Stale in-progress tasks (runner died mid-request).
        for tid in [k for k, v in self.tasks.items() if now - v["first_ts"] > 3600]:
            self.tasks.pop(tid, None)

    def _emit(self, row: dict) -> None:
        self.stats["requests"] += 1
        self.on_request(row)
        if row["class"] in ("inference", "embed"):
            self.on_live({"type": "request", **{k: row.get(k) for k in (
                "ts", "model", "endpoint", "status", "ttft_ms", "decode_tps",
                "output_tokens", "prompt_tokens", "latency_ms", "queue_ms",
                "attribution", "client_ip")}})


# --------------------------------------------------------------------------
# journal following
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# pollers
# --------------------------------------------------------------------------

_NVIDIA_QUERY = ("index,name,utilization.gpu,memory.used,memory.total,"
                 "temperature.gpu,power.draw")


def sample_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        gpus.append({
            "index": _num(parts[0], int), "name": parts[1],
            "util_pct": _num(parts[2], float), "mem_used": _num(parts[3], int),
            "mem_total": _num(parts[4], int), "temp_c": _num(parts[5], float),
            "power_w": _num(parts[6], float),
        })
    return gpus


def _num(s, cast):
    try:
        return cast(float(s))
    except (TypeError, ValueError):
        return None


def sample_ps(base_url: str) -> list[dict]:
    """GET /api/ps -- currently loaded models and their VRAM footprint."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/ps", timeout=3) as r:
            data = json.loads(r.read())
    except Exception:
        return []
    out = []
    for m in data.get("models", []):
        out.append({"name": m.get("name") or m.get("model"),
                    "size": m.get("size"), "size_vram": m.get("size_vram"),
                    "expires_at": m.get("expires_at"),
                    "context_length": m.get("context_length")})
    return out


class GpuPoller:
    """Samples nvidia-smi.  GPUs belong to the host, not to a source, so this
    runs once regardless of how many engines are being monitored."""

    def __init__(self, store, interval_getter, on_live=None):
        self.store = store
        self.interval_getter = interval_getter
        self.on_live = on_live or (lambda x: None)
        self.last: dict = {}

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                ts = time.time()
                gpus = await loop.run_in_executor(None, sample_gpus)
                if gpus:
                    self.store.insert_gpu_samples(ts, gpus)
                    self.store.commit()
                self.last = {"ts": ts, "gpus": gpus}
                self.on_live({"type": "gpu", **self.last})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("gpu poller tick failed")
            await asyncio.sleep(max(1.0, float(self.interval_getter())))


class PsPoller:
    """Polls one ollama instance's /api/ps for resident models."""

    def __init__(self, store, corr: Correlator, base_url: str, interval_getter,
                 on_live=None):
        self.store = store
        self.corr = corr
        self.base_url = base_url
        self.interval_getter = interval_getter
        self.on_live = on_live or (lambda x: None)
        self.last: dict = {}

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                ts = time.time()
                ps = await loop.run_in_executor(None, sample_ps, self.base_url)
                inflight = self.corr.inflight if self.corr else 0
                self.store.insert_ps_sample(ts, len(ps), ps, inflight)
                self.store.commit()
                self.last = {"ts": ts, "models": ps, "inflight": inflight}
                self.on_live({"type": "sample", **self.last})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ps poller tick failed")
            await asyncio.sleep(max(1.0, float(self.interval_getter())))


class Maintainer:
    """Periodic rollup rebuild + retention prune.

    Reads its intervals and retention through getters so a change in the
    settings screen takes effect on the next tick, with no restart.
    """

    def __init__(self, store, raw_retention_getter, sample_retention_getter,
                 interval_getter):
        self.store = store
        self.raw_retention_getter = raw_retention_getter
        self.sample_retention_getter = sample_retention_getter
        self.interval_getter = interval_getter

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        # On startup, roll up everything already in the table (a first run has a
        # full backfill to fold in).
        await loop.run_in_executor(None, self._initial)
        last_prune = 0.0
        while True:
            await asyncio.sleep(max(5.0, float(self.interval_getter())))
            now = time.time()
            try:
                # Re-roll the trailing window so late-joined rows are picked up.
                await loop.run_in_executor(
                    None, self.store.rebuild_rollups, now - 900, now)
                if now - last_prune > 3600:
                    dropped = self.store.prune(float(self.raw_retention_getter()),
                                               float(self.sample_retention_getter()))
                    if any(dropped.values()):
                        log.info("pruned %s", dropped)
                    last_prune = now
            except Exception:
                log.exception("maintenance tick failed")

    def _initial(self) -> None:
        row = self.store.query("SELECT MIN(ts) a, MAX(ts) b FROM requests")
        if row and row[0]["a"]:
            self.store.rebuild_rollups(row[0]["a"], row[0]["b"] + 60)
