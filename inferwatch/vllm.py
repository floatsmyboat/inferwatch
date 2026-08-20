"""vLLM collection, via its native Prometheus endpoint.

WHY THIS IS A SEPARATE SECTION
------------------------------
Ollama is monitored by reconstructing per-request rows from llama.cpp's debug
log.  vLLM cannot be monitored that way: it publishes /metrics, which is
pre-aggregated and carries no request identity at all.  There is no honest way
to turn counters back into per-request rows -- you cannot recover which TTFT
belonged to which request -- so no attempt is made.  vLLM data lives in its own
tables (`vllm_samples`, `vllm_hist`, `vllm_instances`) and its own dashboard
tab, showing what vLLM actually reports.

vLLM's histogram buckets are also not a refinement of the ones used for ollama
(vLLM has bounds at 1ms/20ms/40ms/250ms/2.5s/40s/640s; ollama's are at
25ms/200ms/1.5s/15s/60s), so the two cannot share a bucket layout without
interpolating between bounds.  Each scrape therefore stores vLLM's own bounds
alongside its own counts.

WHAT IS WRITTEN, AND HOW OFTEN
------------------------------
The endpoint is scraped every `collection.scrape_interval_s` (default 10s),
which drives the live view.  Rows are flushed to disk aggregated per MINUTE:
gauges as mean/min/max, counters as the summed delta, histograms as summed
bucket deltas.  Storing every scrape would add millions of rows a month for no
extra insight at the resolutions the dashboard draws.

Counters are cumulative and restart at zero when the engine restarts, so
`process_start_time_seconds` is watched and the interval spanning a restart is
dropped rather than emitted as a huge negative or bogus delta.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import urllib.error
import urllib.request

from .gpuproc import gpus_for_port, is_local, port_of

log = logging.getLogger("inferwatch.vllm")

# Gauges: current state.  Recorded as-is.
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_waiting_by_reason",
    "vllm:kv_cache_usage_perc",
    "vllm:engine_sleep_state",
    "vllm:lora_requests_info",
)

# Counters: monotonic.  Recorded as a per-interval delta plus a per-second rate.
COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)

# Histograms: recorded as per-interval bucket deltas against vLLM's own bounds.
# Inter-token latency has been spelled three different ways across vLLM
# releases; all are collected and the query layer uses whichever has data, so
# this works against old and new servers without configuration.
HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:time_per_output_token_seconds",          # <= 0.6-ish
    "vllm:inter_token_latency_seconds",            # current name
    "vllm:request_time_per_output_token_seconds",  # per-request variant
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "vllm:iteration_tokens_total",
)

# Latency histograms are in seconds; the dashboard and MCP layer speak
# milliseconds everywhere else, so these are converted on read.
SECONDS_HISTOGRAMS = frozenset({
    "vllm:time_to_first_token_seconds",
    "vllm:time_per_output_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
})

RESET_SENTINEL = "process_start_time_seconds"


# --------------------------------------------------------------------------
# Prometheus text parsing
# --------------------------------------------------------------------------

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+(?P<value>[^\s]+)(?:\s+[^\s]+)?\s*$"
)
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


def _parse_value(text: str) -> float | None:
    t = text.strip()
    if t in ("+Inf", "Inf"):
        return math.inf
    if t == "-Inf":
        return -math.inf
    if t == "NaN":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_prometheus(text: str) -> list[tuple[str, dict, float]]:
    """Parse exposition-format text into (name, labels, value) triples.

    Comments and malformed lines are skipped; a broken line must never take the
    whole scrape down.
    """
    out: list[tuple[str, dict, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        value = _parse_value(m.group("value"))
        if value is None:
            continue
        labels = {}
        raw = m.group("labels")
        if raw:
            for k, v in _LABEL.findall(raw):
                labels[k] = v.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
        out.append((m.group("name"), labels, value))
    return out


def json_bounds(bounds: list[float]) -> str:
    """Serialise bucket bounds as strict JSON.

    The last Prometheus bucket is `le="+Inf"`, and json.dumps would write it as
    the bare token `Infinity`, which is valid for Python's json module but not
    for JSON.parse -- it would throw in the dashboard.  The overflow bound is
    written as null instead, and read back as "no upper bound".
    """
    return json.dumps([None if math.isinf(b) else b for b in bounds])


def label_key(labels: dict, drop=("le",)) -> str:
    """Canonical, stable rendering of a label set for use as a row key."""
    items = sorted((k, v) for k, v in labels.items() if k not in drop)
    return ",".join(f"{k}={v}" for k, v in items)


class Snapshot:
    """One scrape, organised by metric family."""

    def __init__(self, ts: float, samples: list[tuple[str, dict, float]]):
        self.ts = ts
        self.gauges: dict[tuple[str, str], float] = {}
        self.counters: dict[tuple[str, str], float] = {}
        # (metric) -> {bound -> cumulative count}; vLLM emits one series per
        # model, and a single instance serves one model, so labels beyond `le`
        # are folded into the key.
        self.hist: dict[str, dict[float, float]] = {}
        self.hist_count: dict[str, float] = {}
        self.hist_sum: dict[str, float] = {}
        self.model: str | None = None
        self.engine_start: float | None = None
        self.info: dict = {}

        for name, labels, value in samples:
            if name == RESET_SENTINEL:
                self.engine_start = value
                continue
            if labels.get("model_name") and not self.model:
                self.model = labels["model_name"]

            base = name
            for suffix in ("_bucket", "_sum", "_count"):
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    break

            if base in HISTOGRAMS:
                if name.endswith("_bucket"):
                    le = _parse_value(labels.get("le", ""))
                    if le is not None:
                        self.hist.setdefault(base, {})[le] = value
                elif name.endswith("_count"):
                    self.hist_count[base] = value
                elif name.endswith("_sum"):
                    self.hist_sum[base] = value
                continue
            if name in GAUGES:
                self.gauges[(name, label_key(labels))] = value
            elif name in COUNTERS:
                self.counters[(name, label_key(labels))] = value
            elif name == "vllm:cache_config_info":
                self.info = labels

    def bounds(self, metric: str) -> list[float]:
        return sorted(self.hist.get(metric, {}))


# --------------------------------------------------------------------------
# scraping
# --------------------------------------------------------------------------

class ScrapeError(Exception):
    pass


def fetch(url: str, api_key: str = "", timeout: float = 8.0) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise ScrapeError(str(e)) from e


class MinuteAccumulator:
    """Buckets scrapes into one-minute rows for the database."""

    def __init__(self):
        self.bucket: int | None = None
        self.gauge: dict[tuple[str, str], list] = {}      # -> [sum, n, min, max]
        self.counter: dict[tuple[str, str], float] = {}   # summed delta
        self.hist: dict[str, dict] = {}                   # -> {bounds, counts, n, sum}
        self.model: str | None = None

    def start(self, bucket: int) -> None:
        self.bucket = bucket
        self.gauge.clear()
        self.counter.clear()
        self.hist.clear()

    def add_gauge(self, key, value) -> None:
        acc = self.gauge.get(key)
        if acc is None:
            self.gauge[key] = [value, 1, value, value]
        else:
            acc[0] += value; acc[1] += 1
            acc[2] = min(acc[2], value); acc[3] = max(acc[3], value)

    def add_counter(self, key, delta) -> None:
        self.counter[key] = self.counter.get(key, 0.0) + delta

    def add_hist(self, metric, bounds, deltas, n, total) -> None:
        h = self.hist.get(metric)
        if h is None:
            self.hist[metric] = {"bounds": bounds, "counts": list(deltas),
                                 "n": n, "sum": total}
            return
        if h["bounds"] != bounds:
            # vLLM changed its bucket layout mid-minute (a restart with new
            # config).  Start the row over rather than adding mismatched arrays.
            self.hist[metric] = {"bounds": bounds, "counts": list(deltas),
                                 "n": n, "sum": total}
            return
        for i, d in enumerate(deltas):
            h["counts"][i] += d
        h["n"] += n
        h["sum"] += total

    def rows(self, source: str) -> tuple[list, list]:
        ts = float(self.bucket or 0)
        samples = []
        for (metric, labels), acc in self.gauge.items():
            total, n, lo, hi = acc
            samples.append((ts, source, metric, labels, self.model, total / n, None))
            if hi != lo:  # only worth storing when it actually moved
                samples.append((ts, source, metric + ":max", labels, self.model, hi, None))
        for (metric, labels), delta in self.counter.items():
            samples.append((ts, source, metric, labels, self.model, delta, delta / 60.0))
        hists = []
        for metric, h in self.hist.items():
            hists.append((ts, source, metric, self.model, json_bounds(h["bounds"]),
                          json.dumps(h["counts"]), h["n"], h["sum"]))
        return samples, hists


class VllmCollector:
    """Scrapes one vLLM instance on a loop and writes minute-aggregated rows."""

    def __init__(self, store, source_name: str, cfg: dict, interval_getter,
                 on_live=None):
        self.store = store
        self.name = source_name
        self.url = (cfg.get("url") or "http://127.0.0.1:8000").rstrip("/")
        self.api_key = cfg.get("api_key") or ""
        self.interval_getter = interval_getter
        self.on_live = on_live or (lambda x: None)
        self.prev: Snapshot | None = None
        self.acc = MinuteAccumulator()
        self.last: dict = {}
        self.errors = 0
        self.scrapes = 0
        # Which GPUs this instance's processes hold.  Resolved by walking the
        # process tree, which is not free, so it is refreshed on an interval and
        # whenever the engine restarts (workers get new pids).
        self.gpu_indices: list[int] | None = None
        self._gpu_checked = 0.0
        self._gpu_for_engine_start: float | None = None

    GPU_RECHECK_S = 60.0

    def resolve_gpus(self, engine_start: float | None, now: float) -> list[int] | None:
        """GPU indices for this instance, refreshed when stale or after a restart."""
        fresh = (now - self._gpu_checked) < self.GPU_RECHECK_S
        same_engine = engine_start == self._gpu_for_engine_start
        if fresh and same_engine and self.gpu_indices is not None:
            return self.gpu_indices
        self._gpu_checked = now
        self._gpu_for_engine_start = engine_start
        if not is_local(self.url):
            self.gpu_indices = None      # a remote engine's GPUs are not ours
            return None
        port = port_of(self.url)
        if port is None:
            self.gpu_indices = None
            return None
        try:
            self.gpu_indices = gpus_for_port(port)
        except Exception:
            log.debug("gpu attribution failed for %r", self.name, exc_info=True)
            self.gpu_indices = None
        return self.gpu_indices

    # -- one scrape ------------------------------------------------------

    def scrape_once(self) -> Snapshot:
        text = fetch(f"{self.url}/metrics", self.api_key)
        snap = Snapshot(time.time(), parse_prometheus(text))
        self.scrapes += 1
        return snap

    def _reset_detected(self, snap: Snapshot) -> bool:
        if self.prev is None:
            return False
        if (snap.engine_start is not None and self.prev.engine_start is not None
                and snap.engine_start != self.prev.engine_start):
            return True
        # Fallback for builds that do not export a start time: a counter that
        # went backwards can only mean the series restarted.
        for key, value in snap.counters.items():
            old = self.prev.counters.get(key)
            if old is not None and value < old:
                return True
        return False

    def ingest(self, snap: Snapshot) -> dict:
        """Fold one scrape into the current minute; returns the live view."""
        bucket = int(snap.ts // 60) * 60
        if self.acc.bucket is None:
            self.acc.start(bucket)
        elif bucket != self.acc.bucket:
            self.flush()
            self.acc.start(bucket)
        self.acc.model = snap.model or self.acc.model

        reset = self._reset_detected(snap)
        if reset:
            log.warning("vllm source %r restarted; dropping the interval that "
                        "spans the restart", self.name)

        for key, value in snap.gauges.items():
            self.acc.add_gauge(key, value)

        if self.prev is not None and not reset:
            dt = max(1e-6, snap.ts - self.prev.ts)
            for key, value in snap.counters.items():
                old = self.prev.counters.get(key)
                if old is not None and value >= old:
                    self.acc.add_counter(key, value - old)
            for metric, buckets in snap.hist.items():
                old_buckets = self.prev.hist.get(metric)
                if not old_buckets:
                    continue
                bounds = sorted(buckets)
                if sorted(old_buckets) != bounds:
                    continue
                # Cumulative ("le") buckets -> per-bucket counts, then delta.
                deltas = []
                prev_cum = 0.0
                prev_cum_old = 0.0
                ok = True
                for b in bounds:
                    cur = buckets[b] - prev_cum
                    old = old_buckets[b] - prev_cum_old
                    prev_cum = buckets[b]
                    prev_cum_old = old_buckets[b]
                    d = cur - old
                    if d < 0:
                        ok = False
                        break
                    deltas.append(d)
                if not ok:
                    continue
                n = (snap.hist_count.get(metric, 0.0)
                     - self.prev.hist_count.get(metric, 0.0))
                total = (snap.hist_sum.get(metric, 0.0)
                         - self.prev.hist_sum.get(metric, 0.0))
                self.acc.add_hist(metric, bounds, deltas, max(0.0, n), max(0.0, total))
            live_rate = {}
            for key, value in snap.counters.items():
                old = self.prev.counters.get(key)
                if old is not None and value >= old:
                    live_rate[f"{key[0]}|{key[1]}"] = (value - old) / dt
        else:
            live_rate = {}

        self.prev = snap
        self.last = {
            "source": self.name, "ts": snap.ts, "model": snap.model,
            "running": snap.gauges.get(("vllm:num_requests_running", label_key(
                {"engine": "0", "model_name": snap.model or ""}))),
            "gauges": {f"{k[0]}|{k[1]}": v for k, v in snap.gauges.items()},
            "rates": live_rate,
        }
        return self.last

    def flush(self) -> None:
        if self.acc.bucket is None:
            return
        samples, hists = self.acc.rows(self.name)
        if samples:
            self.store.insert_vllm_samples(samples)
        if hists:
            self.store.insert_vllm_hist(hists)
        if samples or hists:
            self.store.commit()

    # -- loop ------------------------------------------------------------

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            interval = max(1.0, float(self.interval_getter()))
            try:
                snap = await loop.run_in_executor(None, self.scrape_once)
                live = self.ingest(snap)
                gpus = await loop.run_in_executor(
                    None, self.resolve_gpus, snap.engine_start, snap.ts)
                live["gpu_indices"] = gpus
                self.store.upsert_vllm_instance(
                    self.name, last_seen=snap.ts, engine_start=snap.engine_start,
                    model=snap.model, reachable=1, error=None,
                    info_json=json.dumps(snap.info) if snap.info else None,
                    gpu_indices=json.dumps(gpus) if gpus is not None else None)
                self.store.commit()
                self.errors = 0
                self.on_live({"type": "vllm", **live})
            except ScrapeError as e:
                self.errors += 1
                if self.errors in (1, 5) or self.errors % 30 == 0:
                    log.warning("vllm source %r unreachable at %s: %s",
                                self.name, self.url, e)
                self.store.upsert_vllm_instance(
                    self.name, last_seen=time.time(), reachable=0, error=str(e)[:300])
                self.store.commit()
            except asyncio.CancelledError:
                self.flush()
                raise
            except Exception:
                log.exception("vllm scrape failed for %r", self.name)
            await asyncio.sleep(interval)
