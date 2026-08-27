"""Parsers for ollama journal lines.

Every regex here was written against real journal output from this host
(ollama 0.32.14 with OLLAMA_DEBUG=1).  Three families of line matter:

  1. llama.cpp slot lines  -- "slot print_timing: id 0 | task 42 | ..."
     token counts and timings.  This is where TTFT and tok/s come from.
  2. gin access lines      -- "[GIN] ... | 200 | 1.2s | 192.0.2.10 | POST "/api/chat""
     status, wall latency, client, endpoint.
  3. ollama go lines       -- 'time=... level=INFO source=sched.go:1 msg="..." k=v'
     model identity, load/unload, warnings.
  4. llama.cpp srv lines   -- "srv  update:  - cache state: 30 prompts, 8010.969 MiB"
     ollama's PROMPT CACHE: how full the saved-prompt pool is, and what its
     maintenance pass costs.  This is the closest thing ollama has to vLLM's
     kv_cache_usage_perc, and unlike everything else here it is a GAUGE
     sampled whenever ollama happens to run a cache update, not per request.
  5. llama_kv_cache lines  -- "llama_kv_cache: size = 4608.00 MiB ( 32768 cells, ...)"
     the LIVE KV cache, sized once at load.  Its GPU/CPU split matters a lot:
     a cache that spilled to host RAM decodes far slower than one that fit.

Those last two come from the llama-server runner, and appear only at its higher
log verbosity.  Ollama 0.32.x passes `--log-verbosity 4` on every load, so they
are present by default there; OLLAMA_DEBUG=1 is the documented way to be sure of
it, and additionally turns on ollama's own DEBUG-level Go lines.  Either way an
instance that serves no traffic logs none of them, so their absence is not by
itself evidence of a misconfiguration.

Parsers are pure: they take a message string and return a dict, or None.
No I/O, so they can be replayed over a captured journal in tests.
"""

from __future__ import annotations

import re
import shlex

# --------------------------------------------------------------------------
# duration parsing (go's time.Duration string form)
# --------------------------------------------------------------------------

_DUR_UNITS = {
    "ns": 1e-6,
    "us": 1e-3,
    "µs": 1e-3,  # go emits U+00B5 MICRO SIGN
    "μs": 1e-3,  # ...and occasionally GREEK SMALL LETTER MU
    "ms": 1.0,
    "s": 1000.0,
    "m": 60_000.0,
    "h": 3_600_000.0,
}
_DUR_PART = re.compile(r"([0-9]*\.?[0-9]+)(ns|us|µs|μs|ms|h|m|s)")


def parse_duration_ms(text: str) -> float | None:
    """'16.862061865s' -> 16862.06, '33.313µs' -> 0.033, '1m30s' -> 90000."""
    if not text:
        return None
    parts = _DUR_PART.findall(text.strip())
    if not parts:
        return None
    # reject trailing garbage: the parts must account for the whole string
    if sum(len(v) + len(u) for v, u in parts) != len(text.strip()):
        return None
    return sum(float(v) * _DUR_UNITS[u] for v, u in parts)


# --------------------------------------------------------------------------
# endpoint classification
# --------------------------------------------------------------------------

_INFERENCE_PATHS = (
    "/api/chat",
    "/api/generate",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
)
_EMBED_PATHS = ("/api/embed", "/api/embeddings", "/v1/embeddings")
# Polled constantly by dashboards and load balancers.  Counting these in
# req/sec would drown the real traffic (on this host they were 96% of hits),
# so they get their own class and are excluded from inference rates.
_HEALTH_PATHS = ("/", "/api/ps", "/api/status", "/api/version")


def classify_endpoint(method: str, path: str) -> str:
    base = path.split("?", 1)[0]
    if any(base.startswith(p) for p in _INFERENCE_PATHS):
        return "inference"
    if any(base.startswith(p) for p in _EMBED_PATHS):
        return "embed"
    if base in _HEALTH_PATHS or (method == "HEAD" and base == "/"):
        return "health"
    return "admin"


# --------------------------------------------------------------------------
# gin access log
# --------------------------------------------------------------------------

_GIN = re.compile(
    r"^\[GIN\]\s+\S+\s+-\s+\S+\s+\|\s*(?P<status>\d+)\s*\|"
    r"\s*(?P<latency>\S+)\s*\|\s*(?P<ip>\S+)\s*\|\s*(?P<method>[A-Z]+)\s+"
    r'"(?P<path>[^"]*)"'
)


def parse_gin(msg: str) -> dict | None:
    m = _GIN.match(msg)
    if not m:
        return None
    method, path = m.group("method"), m.group("path")
    return {
        "kind": "gin",
        "status": int(m.group("status")),
        "latency_ms": parse_duration_ms(m.group("latency")),
        "client_ip": m.group("ip"),
        "method": method,
        "endpoint": path,
        "class": classify_endpoint(method, path),
    }


# --------------------------------------------------------------------------
# llama.cpp slot lines
# --------------------------------------------------------------------------

_SLOT_HEAD = re.compile(r"^slot\s+\S+\s*:\s*id\s+(?P<slot>\d+)\s*\|\s*task\s+(?P<task>-?\d+)\s*\|\s*(?P<rest>.*)$")

_R_PROMPT_EVAL = re.compile(
    r"^prompt eval time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tok>\d+)\s*tokens"
    r"\s*\(\s*(?P<mspt>[\d.]+)\s*ms per token,\s*(?P<tps>[\d.]+)\s*tokens per second\)"
)
_R_EVAL = re.compile(
    r"^eval time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tok>\d+)\s*tokens"
    r"\s*\(\s*(?P<mspt>[\d.]+)\s*ms per token,\s*(?P<tps>[\d.]+)\s*tokens per second\)"
)
_R_TOTAL = re.compile(r"^total time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tok>\d+)\s*tokens")
_R_NGEN = re.compile(r"^n_gen\s*=\s*(?P<n>\d+),\s*tg\s*=\s*(?P<tg>[\d.]+)\s*t/s,\s*tg_3s\s*=\s*(?P<tg3>[\d.]+)\s*t/s")
_R_NEWPROMPT = re.compile(
    r"^new prompt, n_ctx_slot\s*=\s*(?P<nctx>\d+), n_keep\s*=\s*(?P<nkeep>-?\d+),\s*task\.n_tokens\s*=\s*(?P<ntok>\d+)"
)
_R_RELEASE = re.compile(r"^stop processing: n_tokens\s*=\s*(?P<ntok>\d+), truncated\s*=\s*(?P<trunc>\d+)")
_R_DRAFT = re.compile(
    r"^draft acceptance\s*=\s*(?P<rate>[\d.]+)\s*\(\s*(?P<acc>\d+)\s*accepted\s*/\s*(?P<gen>\d+)\s*generated\),"
    r"\s*mean len\s*=\s*(?P<mean>[\d.]+)"
)
_R_LAUNCH = re.compile(r"^processing task")
# Context checkpoints are llama.cpp's within-slot KV snapshots.  There is a
# bounded number of them ("2 of 32"), and the eviction lines below are the
# pressure signal -- ollama's analogue of vLLM's preemptions.
_R_CKPT_CREATE = re.compile(
    r"^created context checkpoint\s+(?P<n>\d+)\s+of\s+(?P<total>\d+)\s*\("
    r".*?n_tokens\s*=\s*(?P<ntok>\d+),.*?size\s*=\s*(?P<mib>[\d.]+)\s*MiB")
_R_CKPT_RESTORE = re.compile(
    r"^restored context checkpoint\s*\(.*?n_tokens\s*=\s*(?P<ntok>\d+),"
    r".*?size\s*=\s*(?P<mib>[\d.]+)\s*MiB")
# Two spellings, two reasons: one is capacity pressure (a new checkpoint
# crowded out a neighbour), the other is correctness (the cached positions no
# longer apply).  Counting them together would blur those apart.
_R_CKPT_CROWDED = re.compile(
    r"^erasing context checkpoint too close to an earlier one\s*\("
    r".*?size\s*=\s*(?P<mib>[\d.]+)\s*MiB")
_R_CKPT_INVALID = re.compile(
    r"^erased invalidated context checkpoint\s*\("
    r".*?size\s*=\s*(?P<mib>[\d.]+)\s*MiB")
_R_PROMPT_PROGRESS = re.compile(
    r"^prompt processing, n_tokens\s*=\s*(?P<tok>\d+), progress\s*=\s*(?P<prog>[\d.]+),"
    r"\s*t\s*=\s*(?P<sec>[\d.]+)\s*s\s*/\s*(?P<tps>[\d.]+)\s*tokens per second"
)


def parse_slot(msg: str) -> dict | None:
    """Parse a llama.cpp per-slot line into a typed event."""
    head = _SLOT_HEAD.match(msg)
    if not head:
        return None
    base = {
        "slot_id": int(head.group("slot")),
        "task_id": int(head.group("task")),
    }
    rest = head.group("rest").strip()

    if m := _R_PROMPT_EVAL.match(rest):
        # prompt eval time IS time-to-first-token, measured server side:
        # the model cannot emit token 1 until the prompt is prefilled.
        return {**base, "kind": "prompt_eval", "ttft_ms": float(m.group("ms")),
                "prompt_tokens": int(m.group("tok")), "prefill_tps": float(m.group("tps"))}
    if m := _R_EVAL.match(rest):
        return {**base, "kind": "eval", "decode_ms": float(m.group("ms")),
                "output_tokens": int(m.group("tok")), "decode_tps": float(m.group("tps"))}
    if m := _R_TOTAL.match(rest):
        return {**base, "kind": "total", "total_ms": float(m.group("ms")),
                "total_tokens": int(m.group("tok"))}
    if m := _R_NGEN.match(rest):
        return {**base, "kind": "gen_progress", "n_gen": int(m.group("n")),
                "tg": float(m.group("tg")), "tg_3s": float(m.group("tg3"))}
    if m := _R_NEWPROMPT.match(rest):
        # task.n_tokens is the FULL prompt; prompt_eval tokens is only the
        # uncached remainder.  The difference is the prompt-cache hit.
        return {**base, "kind": "new_prompt", "n_ctx_slot": int(m.group("nctx")),
                "prompt_tokens_total": int(m.group("ntok"))}
    if m := _R_RELEASE.match(rest):
        return {**base, "kind": "release", "context_tokens": int(m.group("ntok")),
                "truncated": int(m.group("trunc"))}
    if m := _R_DRAFT.match(rest):
        return {**base, "kind": "draft", "draft_accept": float(m.group("rate")),
                "draft_accepted": int(m.group("acc")), "draft_generated": int(m.group("gen")),
                "draft_mean_len": float(m.group("mean"))}
    if m := _R_PROMPT_PROGRESS.match(rest):
        return {**base, "kind": "prefill_progress", "tokens": int(m.group("tok")),
                "progress": float(m.group("prog")), "prefill_tps": float(m.group("tps"))}
    if m := _R_CKPT_CREATE.match(rest):
        return {**base, "kind": "ckpt_create", "ckpt_index": int(m.group("n")),
                "ckpt_total": int(m.group("total")), "ckpt_tokens": int(m.group("ntok")),
                "ckpt_mib": float(m.group("mib"))}
    if m := _R_CKPT_RESTORE.match(rest):
        return {**base, "kind": "ckpt_restore", "ckpt_tokens": int(m.group("ntok")),
                "ckpt_mib": float(m.group("mib"))}
    if m := _R_CKPT_CROWDED.match(rest):
        return {**base, "kind": "ckpt_evict", "reason": "crowded",
                "ckpt_mib": float(m.group("mib"))}
    if m := _R_CKPT_INVALID.match(rest):
        return {**base, "kind": "ckpt_evict", "reason": "invalidated",
                "ckpt_mib": float(m.group("mib"))}
    if _R_LAUNCH.match(rest):
        return {**base, "kind": "launch"}
    return None


# --------------------------------------------------------------------------
# llama.cpp srv lines: ollama's prompt cache
# --------------------------------------------------------------------------

_SRV_HEAD = re.compile(r"^srv\s+(?P<fn>\S+)\s*:\s*(?P<rest>.*)$")

# The one line that states occupancy outright.  `est` is llama.cpp's own guess
# at how many tokens the pool could still hold, which is not the same as the
# hard `tokens` limit next to it, so both are kept.
_R_CACHE_STATE = re.compile(
    r"^-\s*cache state:\s*(?P<prompts>\d+)\s+prompts?,\s*(?P<used>[\d.]+)\s*MiB"
    r"\s*\(limits:\s*(?P<limit>[\d.]+)\s*MiB,\s*(?P<tok_limit>\d+)\s*tokens,"
    r"\s*(?P<est>\d+)\s*est\)")
# The maintenance pass is synchronous and sits inside the request that triggered
# it, so on a full cache this lands directly in TTFT.
_R_CACHE_UPDATE_MS = re.compile(r"^prompt cache update took\s*(?P<ms>[\d.]+)\s*ms")
_R_PROMPT_SAVE = re.compile(
    r"^-\s*saving prompt with length\s*(?P<len>\d+),\s*total state size\s*=\s*"
    r"(?P<mib>[\d.]+)\s*MiB")


def parse_srv(msg: str) -> dict | None:
    """Parse a llama.cpp `srv` line.  Only the prompt-cache ones are claimed."""
    head = _SRV_HEAD.match(msg)
    if not head:
        return None
    rest = head.group("rest").strip()

    if m := _R_CACHE_STATE.match(rest):
        limit = float(m.group("limit"))
        used = float(m.group("used"))
        return {"kind": "cache_state", "prompts": int(m.group("prompts")),
                "used_mib": used, "limit_mib": limit,
                # Reported alongside the raw pair rather than instead of it: a
                # limit of zero means "unbounded", not "100% full".
                "usage": (used / limit) if limit > 0 else None,
                "token_limit": int(m.group("tok_limit")),
                "est_tokens": int(m.group("est"))}
    if m := _R_CACHE_UPDATE_MS.match(rest):
        return {"kind": "cache_update", "update_ms": float(m.group("ms"))}
    if m := _R_PROMPT_SAVE.match(rest):
        return {"kind": "cache_save", "prompt_tokens": int(m.group("len")),
                "state_mib": float(m.group("mib"))}
    return None


# --------------------------------------------------------------------------
# llama_kv_cache lines: the live KV cache, sized once at load
# --------------------------------------------------------------------------

_KV_HEAD = re.compile(r"^llama_kv_cache:\s*(?P<rest>.*)$")
_R_KV_SIZE = re.compile(
    r"^size\s*=\s*(?P<mib>[\d.]+)\s*MiB\s*\(\s*(?P<cells>\d+)\s*cells,"
    r"\s*(?P<layers>\d+)\s*layers,\s*(?P<seqs>\d+)\s*/\s*(?P<seqs_max>\d+)\s*seqs\)")
_R_KV_TYPES = re.compile(
    r"K\s*\((?P<ktype>[^)]*)\):\s*(?P<kmib>[\d.]+)\s*MiB.*?"
    r"V\s*\((?P<vtype>[^)]*)\):\s*(?P<vmib>[\d.]+)\s*MiB")
# "CUDA0 KV buffer size = 1024.00 MiB" / "CPU KV buffer size = 4096.00 MiB".
# The device name is what makes this worth collecting: KV on the CPU is the
# single loudest explanation for a model that decodes slowly.
_R_KV_BUFFER = re.compile(
    r"^(?P<device>\S+)\s+KV buffer size\s*=\s*(?P<mib>[\d.]+)\s*MiB")


def parse_kv_cache(msg: str) -> dict | None:
    """Parse a `llama_kv_cache:` line from a model load."""
    head = _KV_HEAD.match(msg)
    if not head:
        return None
    rest = head.group("rest").strip()

    if m := _R_KV_SIZE.match(rest):
        out = {"kind": "kv_size", "kv_mib": float(m.group("mib")),
               "cells": int(m.group("cells")), "layers": int(m.group("layers")),
               "seqs": int(m.group("seqs")), "seqs_max": int(m.group("seqs_max"))}
        if t := _R_KV_TYPES.search(rest):
            out.update({"k_type": t.group("ktype"), "k_mib": float(t.group("kmib")),
                        "v_type": t.group("vtype"), "v_mib": float(t.group("vmib"))})
        return out
    if m := _R_KV_BUFFER.match(rest):
        return {"kind": "kv_buffer", "device": m.group("device"),
                "kv_mib": float(m.group("mib"))}
    return None


# --------------------------------------------------------------------------
# ollama go structured lines
# --------------------------------------------------------------------------

_GO_LINE = re.compile(r'^time=(?P<ts>\S+)\s+level=(?P<level>\w+)\s+source=(?P<source>\S+)\s+msg=(?P<tail>.*)$')


def _split_kv(tail: str) -> tuple[str, dict]:
    """Split 'msg="a b" k=v k2="x y"' into (msg, {k: v}).  shlex handles quotes."""
    try:
        tokens = shlex.split(tail)
    except ValueError:
        tokens = tail.split()
    if not tokens:
        return "", {}
    msg = tokens[0]
    kv: dict[str, str] = {}
    for tok in tokens[1:]:
        if "=" in tok:
            k, _, v = tok.partition("=")
            kv[k] = v
    return msg, kv


_LOAD_DONE = re.compile(r"^llama-server started in (?P<sec>[\d.]+) seconds$")


def parse_go(msg: str) -> dict | None:
    m = _GO_LINE.match(msg)
    if not m:
        return None
    text, kv = _split_kv(m.group("tail"))
    ev = {
        "kind": "go",
        "level": m.group("level"),
        "source": m.group("source"),
        "msg": text,
        "kv": kv,
    }

    if d := _LOAD_DONE.match(text):
        ev["event"] = "model_loaded"
        ev["load_ms"] = float(d.group("sec")) * 1000.0
    elif text == "starting llama-server":
        ev["event"] = "load_start"
        ev.update(_parse_llama_cmd(kv.get("cmd", "")))
    elif text == "loading model via llama-server":
        ev["event"] = "load_model"
        ev["blob"] = kv.get("model")
    elif text == "loaded runners":
        ev["event"] = "runner_count"
        ev["count"] = _int(kv.get("count"))
    elif text in ("context for request finished", "after processing request finished event"):
        # Emitted immediately after the gin line for an inference request and
        # carries the model NAME -- the slot lines only know a task id.
        ev["event"] = "request_finished"
    elif text == "llama-server completion request":
        ev["event"] = "completion_request"
        ev["prompt_len"] = _int(kv.get("prompt_len"))
    elif text in ("sending an unloaded event", "timer expired, expiring to unload"):
        ev["event"] = "unload"
    elif text == "runner with non-zero duration has gone idle, adding timer":
        ev["event"] = "idle_timer"
        ev["keep_alive"] = kv.get("duration")
    elif m.group("level") in ("WARN", "ERROR"):
        ev["event"] = "problem"

    # Model identity travels on most sched.go lines.
    if "runner.name" in kv:
        ev["model"] = _clean_model(kv["runner.name"])
    if "runner.model" in kv:
        ev["blob"] = kv["runner.model"]
    for src, dst in (("runner.vram", "vram"), ("runner.size", "size")):
        if src in kv:
            ev[dst] = kv[src]
    for src, dst in (("runner.num_ctx", "num_ctx"), ("runner.parallel", "parallel"),
                     ("runner.pid", "runner_pid")):
        if src in kv:
            ev[dst] = _int(kv[src])
    return ev


_LLAMA_CMD_FLAGS = {"--model": "blob", "--port": "port", "-c": "num_ctx",
                    "-np": "parallel", "--spec-type": "spec_type"}


def _parse_llama_cmd(cmd: str) -> dict:
    """Pull useful config out of the llama-server argv ollama logs on load."""
    out: dict = {}
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return out
    for flag, key in _LLAMA_CMD_FLAGS.items():
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                out[key] = argv[i + 1]
    for key in ("port", "num_ctx", "parallel"):
        if key in out:
            out[key] = _int(out[key])
    return out


def _clean_model(name: str) -> str:
    """'registry.ollama.ai/library/llama3.2:3b' -> 'llama3.2:3b'."""
    for prefix in ("registry.ollama.ai/library/", "registry.ollama.ai/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_line(msg: str) -> dict | None:
    """Dispatch one journal MESSAGE to whichever parser owns it."""
    if not msg:
        return None
    if msg.startswith("[GIN]"):
        return parse_gin(msg)
    if msg.startswith("slot "):
        return parse_slot(msg)
    if msg.startswith("time="):
        return parse_go(msg)
    if msg.startswith("srv "):
        return parse_srv(msg)
    if msg.startswith("llama_kv_cache:"):
        return parse_kv_cache(msg)
    return None
