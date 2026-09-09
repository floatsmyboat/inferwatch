"""SwarmUI / ComfyUI collection: generations, queue depth, VRAM, errors.

A third engine family, and deliberately not squeezed into either of the others.
Image generation has no tokens, no time-to-first-token and no context window;
its unit of work is a generation with a duration, a set of models the workflow
loaded, and a node that may have thrown.

TWO SOURCES, ONE COUNT
----------------------
ComfyUI's `/history` is the authority for generations.  It is the only place
with a stable identity for one (`prompt_id`), millisecond event timestamps, the
failing node with its class and exception, and the model names -- which are read
out of the workflow graph rather than reported anywhere directly.

SwarmUI's journal describes the same work from the orchestrator's side and adds
things ComfyUI cannot see: the prep-versus-gen timing split, failures that never
reached a backend, and the backend ports.  Joining the two would mean guessing
which log line belongs to which prompt_id, so it is NOT joined: the log feeds an
event timeline and never a second generation count.

WHAT IS POLLED
--------------
    SwarmUI   /API/GetCurrentStatus      queue depth, backend health
              /API/ListBackends          per-backend status, GPU_ID, idle time
    ComfyUI   /system_stats              version, per-device VRAM
              /queue                     running / pending
              /history                   the generations themselves

`/history` is an in-memory ring in ComfyUI, so it is polled rather than
subscribed to and each entry is stored under its prompt_id; re-reading the same
window is a no-op.  Anything still running is stored too, and superseded when it
finishes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request

from . import gpuproc

log = logging.getLogger("inferwatch.images")

# Values that look like a model file, whatever node claimed to load them.  A
# fixed list of loader classes would miss every custom node, and this ecosystem
# is mostly custom nodes.
MODEL_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf",
                  ".sft", ".onnx")
# Input keys whose value names a model even without a recognised suffix.
MODEL_KEYS = ("ckpt_name", "unet_name", "vae_name", "lora_name", "clip_name",
              "control_net_name", "model_name", "style_model_name")


class FetchError(Exception):
    """Any failure to read one of these HTTP endpoints."""


def fetch_json(url: str, timeout: float = 8.0, payload: dict | None = None):
    """GET, or POST when `payload` is given.  Raises FetchError, never bare."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
        raise FetchError(str(e)) from None


# --------------------------------------------------------------------------
# pure: reading one /history entry
# --------------------------------------------------------------------------

def extract_models(graph: dict) -> list[dict]:
    """Every model the workflow loads, as {role, name, node_class}.

    Detected by the SHAPE of a node's inputs rather than a list of loader
    classes: a string input whose key is a known model key, or whose value ends
    in a weights suffix.  Custom nodes are the norm in this ecosystem and a
    class list would silently miss all of them.
    """
    out = []
    for node_id, node in sorted((graph or {}).items()):
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type")
        for key, value in (node.get("inputs") or {}).items():
            if not isinstance(value, str) or not value.strip():
                continue
            if key in MODEL_KEYS or value.lower().endswith(MODEL_SUFFIXES):
                out.append({"role": key, "name": value, "node_class": cls})
    return out


def primary_model(models: list[dict]) -> str | None:
    """The one model worth grouping a generation by.

    A workflow loads several (checkpoint, VAE, a stack of LoRAs); the checkpoint
    or diffusion model is the one a person means by "which model was that".
    """
    for key in ("ckpt_name", "unet_name"):
        for m in models:
            if m["role"] == key:
                return m["name"]
    return models[0]["name"] if models else None


def parse_history_entry(prompt_id: str, entry: dict) -> dict | None:
    """Turn one /history entry into a generation row.

    Durations come from the message timestamps rather than from any duration
    field, because ComfyUI publishes none: `execution_start` to whichever of
    `execution_success` / `execution_error` arrived.  An entry with neither is
    still in flight and is stored as such rather than given a made-up end.
    """
    if not isinstance(entry, dict):
        return None
    status = entry.get("status") or {}
    messages = {}
    for m in status.get("messages") or []:
        # [name, payload] pairs; a repeated name keeps the first occurrence,
        # which is the one that opened the phase.
        if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] not in messages:
            messages[m[0]] = m[1] or {}

    start = (messages.get("execution_start") or {}).get("timestamp")
    err = messages.get("execution_error") or {}
    done = messages.get("execution_success") or err
    end = done.get("timestamp")

    # ComfyUI reports epoch MILLISECONDS here.
    started_ts = start / 1000.0 if start else None
    ts = end / 1000.0 if end else started_ts
    if ts is None:
        return None

    graph = {}
    prompt = entry.get("prompt")
    if isinstance(prompt, list) and len(prompt) > 2 and isinstance(prompt[2], dict):
        graph = prompt[2]
    models = extract_models(graph)
    cached = (messages.get("execution_cached") or {}).get("nodes") or []

    row = {
        "prompt_id": prompt_id,
        "ts": ts,
        "started_ts": started_ts,
        "status": status.get("status_str") or ("running" if not end else None),
        "total_ms": (end - start) if (start and end) else None,
        "model": primary_model(models),
        "models_json": json.dumps(models) if models else None,
        "node_count": len(graph) or None,
        "cached_nodes": len(cached) if cached else 0,
        "error_node": err.get("node_id"),
        "error_type": err.get("node_type"),
        # Tracebacks can be enormous; the message is the part a person reads.
        "error_message": (err.get("exception_message") or "")[:2000] or None,
    }
    if not end:
        row["status"] = "running"
    return row


# --------------------------------------------------------------------------
# clients
# --------------------------------------------------------------------------

class ComfyClient:
    """One ComfyUI backend's HTTP API."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def system_stats(self) -> dict:
        return fetch_json(f"{self.url}/system_stats") or {}

    def queue(self) -> dict:
        return fetch_json(f"{self.url}/queue") or {}

    def history(self, limit: int = 64) -> dict:
        return fetch_json(f"{self.url}/history?max_items={int(limit)}") or {}


class SwarmClient:
    """SwarmUI's API, which needs a session id on every call.

    The session is fetched lazily and refreshed whenever a call is rejected,
    since SwarmUI expires them on restart and there is no way to ask whether
    one is still good short of using it.
    """

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.session: str | None = None

    def new_session(self) -> str:
        data = fetch_json(f"{self.url}/API/GetNewSession", payload={}) or {}
        sid = data.get("session_id")
        if not sid:
            raise FetchError("GetNewSession returned no session_id")
        self.session = sid
        return sid

    def call(self, endpoint: str, **params) -> dict:
        if not self.session:
            self.new_session()
        body = {"session_id": self.session, **params}
        data = fetch_json(f"{self.url}/API/{endpoint}", payload=body) or {}
        if isinstance(data, dict) and data.get("error_id") == "invalid_session_id":
            self.new_session()
            body["session_id"] = self.session
            data = fetch_json(f"{self.url}/API/{endpoint}", payload=body) or {}
        return data if isinstance(data, dict) else {}

    def status(self) -> dict:
        return self.call("GetCurrentStatus")

    def backends(self) -> dict:
        return self.call("ListBackends")


# --------------------------------------------------------------------------
# log side
# --------------------------------------------------------------------------

class SwarmLogCollector:
    """Turns SwarmUI journal lines into an event timeline.

    Pure and synchronous, like the Ollama correlator, so a captured journal can
    be replayed through it in tests.

    It also learns the backend PORTS, which is the only place they are
    published -- "Self-Start ComfyUI-0 on port 7821 started." -- so the poller
    can find the ComfyUI instances without being told each one by hand.

    The request and finish lines are deliberately NOT paired into one row. A
    request for N images produces N finish lines, so a 1:1 join would silently
    misattribute every batch; and the authoritative per-generation record
    already exists in /history. The prep/gen split is therefore kept as its own
    aggregate rather than being grafted onto a generation it might not belong
    to.
    """

    def __init__(self, source: str, on_event=None):
        self.source = source
        self.on_event = on_event or (lambda e: None)
        self.backend_ports: dict[int, int] = {}
        self.backend_status: dict[int, str] = {}
        self.version: str | None = None
        self.stats: dict[str, int] = {}

    def feed(self, ts: float, msg: str) -> None:
        from .parse_swarm import parse_line
        ev = parse_line(msg)
        if ev is None:
            return
        kind = ev["kind"]
        self.stats[kind] = self.stats.get(kind, 0) + 1

        if kind in ("backend_up", "backend_loading"):
            self.backend_ports[ev["backend_index"]] = ev["port"]
            self.backend_status[ev["backend_index"]] = (
                "running" if kind == "backend_up" else "loading")
        elif kind == "backend_down":
            self.backend_status[ev["backend_index"]] = "down"
        elif kind == "server_running":
            self.version = ev.get("version")

        row = {
            "ts": ts, "source": self.source, "kind": kind,
            "level": ev.get("level"),
            "backend_index": ev.get("backend_index"),
            "model": ev.get("model"), "user": ev.get("user"),
            "route": ev.get("route"),
            "prep_ms": ev.get("prep_ms"), "gen_ms": ev.get("gen_ms"),
            "msg": self._message(ev),
        }
        detail = {k: v for k, v in ev.items()
                  if k not in ("kind", "level", "clock", "tag", "backend_index",
                               "model", "user", "route", "prep_ms", "gen_ms")}
        row["detail"] = detail or None
        self.on_event(row)

    @staticmethod
    def _message(ev: dict) -> str:
        """A one-line human summary, since `kind` alone does not say what broke."""
        kind = ev["kind"]
        if kind == "webapi_error":
            return f"{ev.get('route')}: {ev.get('reason')}"
        if kind == "backend_stderr":
            return ev.get("text") or ""
        if kind == "generation_requested":
            return (f"{ev.get('user')} requested {ev.get('images')} image(s) "
                    f"with {ev.get('model')}")
        if kind == "generation_finished":
            return (f"generated in {ev.get('prep_ms', 0) / 1000:.2f}s prep + "
                    f"{ev.get('gen_ms', 0) / 1000:.2f}s gen")
        if kind in ("backend_up", "backend_loading"):
            return f"ComfyUI-{ev.get('backend_index')} on port {ev.get('port')} {kind[8:]}"
        if kind == "backend_stopping":
            return f"stopping port {ev.get('port')} (pid {ev.get('pid')})"
        if kind == "server_running":
            return f"SwarmUI {ev.get('version')} running ({ev.get('mode')})"
        return ev.get("text") or kind


# --------------------------------------------------------------------------
# poll side
# --------------------------------------------------------------------------

class ImagesPoller:
    """Polls SwarmUI and each ComfyUI backend on an interval."""

    GPU_RECHECK_S = 30.0

    def __init__(self, store, source_name: str, cfg: dict, interval_getter,
                 on_live=None, log_collector: SwarmLogCollector | None = None):
        self.store = store
        self.name = source_name
        self.url = (cfg.get("url") or "http://127.0.0.1:7801").rstrip("/")
        self.swarm = SwarmClient(self.url)
        self.interval_getter = interval_getter
        self.on_live = on_live or (lambda x: None)
        self.log = log_collector
        self.unit = (cfg.get("unit") or "").strip()
        # Explicit backend URLs win over what the log discovered, so an install
        # whose journal is not being read is still fully monitorable.
        self.configured = [u.strip().rstrip("/") for u in
                           (cfg.get("backends") or "").split(",") if u.strip()]
        self.history_limit = int(cfg.get("history_limit") or 64)
        # Backend ports are announced ONCE, when SwarmUI starts its backends.
        # A collector restart resumes the journal past those lines, so without
        # remembering them a restart would discover nothing and quietly collect
        # no generations at all until SwarmUI itself was restarted.
        self._ports_key = f"swarm_backends:{source_name}"
        self._ports: dict[int, int] = self._load_ports()
        self._probed = False
        self.last: dict = {}
        self.errors = 0
        self.polls = 0
        self._gpus: dict[str, tuple[list[int] | None, str]] = {}
        self._gpu_checked = 0.0

    # -- backend discovery ---------------------------------------------

    def _load_ports(self) -> dict[int, int]:
        try:
            raw = self.store.get_meta(self._ports_key) if self.store else None
            return {int(k): int(v) for k, v in json.loads(raw or "{}").items()}
        except (ValueError, TypeError, AttributeError):
            return {}

    def _save_ports(self) -> None:
        try:
            self.store.set_meta(self._ports_key,
                                json.dumps({str(k): v for k, v in self._ports.items()}))
        except Exception:
            log.debug("could not persist backend ports", exc_info=True)

    # SwarmUI numbers its self-started backends from here, one per backend.
    PROBE_BASE = 7821
    PROBE_COUNT = 10

    def probe_ports(self) -> dict[int, int]:
        """Last-resort discovery: look for ComfyUI on the conventional ports.

        Needed because the ports are announced only when SwarmUI starts its
        backends.  A source added long after that has no log line to learn
        from, no remembered value, and would otherwise collect queue depth and
        no generations at all.

        Bounded to ten localhost ports, tried once, and each hit must actually
        answer as ComfyUI -- so this can add a backend but never mislabel one.
        """
        found: dict[int, int] = {}
        host = self.url.split("//", 1)[-1].split(":")[0] or "127.0.0.1"
        for i in range(self.PROBE_COUNT):
            port = self.PROBE_BASE + i
            try:
                stats = fetch_json(f"http://{host}:{port}/system_stats", timeout=1.5)
            except FetchError:
                continue
            if isinstance(stats, dict) and (stats.get("system") or {}).get("comfyui_version"):
                found[i] = port
        if found:
            log.info("discovered ComfyUI backends by probe: %s", found)
        return found

    def backend_urls(self) -> dict[str, str]:
        """{backend label: base url}, from config, the log, what was learned on
        a previous run, or a probe."""
        if self.configured:
            return {f"backend-{i}": u for i, u in enumerate(self.configured)}
        learned = dict(self._ports)
        if self.log and self.log.backend_ports:
            learned.update(self.log.backend_ports)
        if not learned and not self._probed:
            self._probed = True
            learned.update(self.probe_ports())
        if learned != self._ports:
            self._ports = learned
            self._save_ports()
        host = self.url.split("//", 1)[-1].split(":")[0] or "127.0.0.1"
        return {f"ComfyUI-{idx}": f"http://{host}:{port}"
                for idx, port in sorted(learned.items())}

    def resolve_gpus(self, label: str, url: str, now: float) -> tuple[list[int] | None, str]:
        """Which cards ONE backend holds.

        The port is tried before the unit here, which is the opposite of the
        vLLM case, because the two answer different questions.  A SwarmUI
        install runs every self-started ComfyUI inside the one `swarmui.service`
        cgroup, so the unit reports the cards the whole install holds and hands
        each backend an identical, too-wide answer -- both of this host's
        backends came back as [0, 1] when the unit led.  Each backend is its own
        process listening on its own port and holding its own card, so the port
        walk is exact, and the unit is kept only as the fallback for a backend
        whose port cannot be walked.

        Cached briefly because it reads /proc once per backend per tick.
        """
        if now - self._gpu_checked > self.GPU_RECHECK_S:
            self._gpus = {}
            self._gpu_checked = now
        if label in self._gpus:
            return self._gpus[label]
        port = gpuproc.port_of(url) if gpuproc.is_local(url) else None
        try:
            got = gpuproc.resolve(port=port)
            if got[0] is None and self.unit:
                got = gpuproc.resolve(unit=self.unit)
        except Exception:
            log.debug("gpu attribution failed for %s", label, exc_info=True)
            got = (None, "unavailable")
        self._gpus[label] = got
        return got

    # -- one tick -------------------------------------------------------

    def poll_once(self, now: float) -> dict:
        live: dict = {"ts": now, "source": self.name, "backends": []}

        # SwarmUI's own view: queue depth and orchestrator health.
        try:
            st = self.swarm.status()
            s = st.get("status") or {}
            bs = st.get("backend_status") or {}
            row = {"ts": now, "source": self.name, "backend": "",
                   "status": bs.get("status"),
                   "live_gens": s.get("live_gens"),
                   "waiting_gens": s.get("waiting_gens"),
                   "loading_models": s.get("loading_models")}
            self.store.insert_image_sample(row)
            live["swarm"] = row
        except FetchError as e:
            live["swarm_error"] = str(e)

        # Backend health, which names the GPU each one was pinned to.
        pinned: dict[int, str] = {}
        try:
            for idx, b in (self.swarm.backends() or {}).items():
                try:
                    pinned[int(idx)] = (b.get("settings") or {}).get("GPU_ID")
                except (TypeError, ValueError):
                    continue
        except FetchError:
            pass

        for label, url in self.backend_urls().items():
            entry = {"backend": label, "url": url}
            try:
                stats = ComfyClient(url).system_stats()
                q = ComfyClient(url).queue()
                dev = (stats.get("devices") or [{}])[0]
                gpus, how = self.resolve_gpus(label, url, now)
                row = {
                    "ts": now, "source": self.name, "backend": label,
                    "status": "running",
                    "queue_running": len(q.get("queue_running") or []),
                    "queue_pending": len(q.get("queue_pending") or []),
                    "vram_total": dev.get("vram_total"),
                    "vram_free": dev.get("vram_free"),
                    "torch_vram_total": dev.get("torch_vram_total"),
                    "torch_vram_free": dev.get("torch_vram_free"),
                    "gpu_indices": json.dumps(gpus) if gpus is not None else None,
                }
                self.store.insert_image_sample(row)
                entry.update({k: row[k] for k in
                              ("status", "queue_running", "queue_pending",
                               "vram_total", "vram_free")})
                entry["gpu_indices"] = gpus
                entry["gpu_source"] = how
                entry["version"] = (stats.get("system") or {}).get("comfyui_version")
                entry["device"] = dev.get("name")
                entry["pinned_gpu"] = pinned.get(_index_of(label))
                self._ingest_history(url, label)
            except FetchError as e:
                entry.update({"status": "unreachable", "error": str(e)})
                self.store.insert_image_sample(
                    {"ts": now, "source": self.name, "backend": label,
                     "status": "unreachable"})
            live["backends"].append(entry)

        self.store.commit()
        self.polls += 1
        self.last = live
        return live

    def _ingest_history(self, url: str, label: str) -> None:
        """Store every generation the backend still remembers.

        Keyed by prompt_id, so re-reading the same window costs nothing and a
        generation seen mid-flight is replaced by its finished form.
        """
        hist = ComfyClient(url).history(self.history_limit)
        if not isinstance(hist, dict):
            return
        for prompt_id, entry in hist.items():
            row = parse_history_entry(prompt_id, entry)
            if row is None:
                continue
            row["source"] = self.name
            self.store.insert_image_generation(row)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            try:
                live = await loop.run_in_executor(None, self.poll_once, time.time())
                self.on_live({"type": "images", **live})
                self.errors = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                self.errors += 1
                if self.errors in (1, 5) or self.errors % 30 == 0:
                    log.exception("images poll failed for %r", self.name)
            await asyncio.sleep(max(1.0, float(self.interval_getter())))


def _index_of(label: str) -> int | None:
    """'ComfyUI-1' -> 1, so a backend can be matched to ListBackends."""
    tail = label.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None
