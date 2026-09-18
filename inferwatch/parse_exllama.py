"""Parsing tabbyAPI (exllamav3) server logs.

tabbyAPI logs with loguru:

    2026-09-18 08:33:01.755 | INFO     | #2 chat/completions (stream): 107,134 prompt tokens · ...

Per-request, on start:
    #2 chat/completions (stream): 107,134 prompt tokens · temperature: 0.8 (preset),
        top_k: 40 (preset), top_p: 0.95 (preset), min_p: 0.05 (preset), max_tokens: 32000 (req)

Per-request, on completion:
    #2 chat/completions (stream): 1,225 tokens generated at 49.8 T/s · prompt 107,134
        tokens, none cached, 107,134 new in 146.4 s (732 T/s) · first token 146.4 s,
        total 171.0 s · draft 825/1200 accepted (69%)

The request id restarts at 1 with the process (each startup writes a new log
file), so ids are unique only within one run.  A "Serving OAI API on ..." line
opens a new epoch.

WHAT IS NOT IN THE LOG
----------------------
No client address and no HTTP status.  Errors are logged without a request id
("Sent to request: ..."), so they cannot be joined to a specific request.
Disconnects do carry the id ("#21 chat/completions: client disconnected").
"""

from __future__ import annotations

import re

# #2 chat/completions (stream): 107,134 prompt tokens · temperature: 0.8 (preset), ...
#         max_tokens: 32000 (req)
_SUBMIT = re.compile(
    r"#(?P<id>\d+)\s+(?P<endpoint>[\w/]+)\s+\((?P<mode>stream|non-stream)\):\s+"
    r"(?P<prompt_tokens>[\d,]+)\s+prompt tokens"
    r".*?max_tokens:\s*(?P<max_tokens>[\d,]+)"
)

# #2 chat/completions (stream): 1,225 tokens generated at 49.8 T/s · prompt 107,134
#     tokens, none cached, 107,134 new in 146.4 s (732 T/s) · first token 146.4 s,
#     total 171.0 s · draft 825/1200 accepted (69%)
#
# The prefill speed "(732 T/s)" is absent when the prefill was too short to
# measure, so it is optional.  The draft section is absent when speculative
# decoding is not in use.
_DONE = re.compile(
    r"#(?P<id>\d+)\s+(?P<endpoint>[\w/]+)\s+\((?P<mode>stream|non-stream)\):\s+"
    r"(?P<output_tokens>[\d,]+)\s+tokens generated at (?P<decode_tps>[\d.]+) T/s"
    r"\s+·\s+prompt (?P<prompt_total>[\d,]+) tokens,"
    r"\s+(?P<cache>none cached|\d+% cached),"
    r"\s+(?P<new_tokens>[\d,]+) new in (?P<prefill_time>[\d.]+) s"
    r"(?:\s+\((?P<prefill_tps>[\d,]+) T/s\))?"
    r"\s+·\s+first token (?P<ttft>[\d.]+) s,"
    r"\s+total (?P<total>[\d.]+) s"
    r"(?:\s+·\s+draft (?P<draft_acc>\d+)/(?P<draft_tot>\d+) accepted"
    r"\s+\((?P<draft_pct>\d+)%\))?"
)

# #2 chat/completions (stream): parsed 1 tool call (qwen3_5)
# #25 chat/completions (stream): parsed 4 tool calls (qwen3_5)
_TOOL_CALL = re.compile(
    r"#(?P<id>\d+)\s+(?P<endpoint>[\w/]+)\s+\((?P<mode>stream|non-stream)\):\s+"
    r"parsed (?P<tool_calls>\d+) tool calls?"
)

# #21 chat/completions: client disconnected, generation cancelled
_DISCONNECT = re.compile(
    r"#(?P<id>\d+)\s+(?P<endpoint>[\w/]+):\s+client disconnected"
)

# Sent to request: Prompt length 115605 exceeds the available context size of 114688 tokens
_ERROR = re.compile(r"Sent to request:\s+(?P<msg>.+?)\s*$")

# Serving OAI API on http://0.0.0.0:8003 (docs at http://0.0.0.0:8003/redoc)
_LISTENING = re.compile(
    r"Serving OAI API on\s+(?P<url>\S+?)(?:\s+\(docs at)"
)

# Model loaded in 23.6 s
_LOADED = re.compile(r"Model loaded in\s+(?P<seconds>[\d.]+)\s*s")

# Context: max_seq_len 114,688 tokens (configured), cache_size 458,752 tokens
_CONTEXT = re.compile(
    r"Context:\s+max_seq_len\s+(?P<max_seq_len>[\d,]+)\s+tokens"
    r".*?cache_size\s+(?P<cache_size>[\d,]+)\s+tokens"
)

# Loading model /path/to/model (tensor parallel)
_LOADING = re.compile(
    r"Loading model\s+(?P<path>\S+)(?:\s+\((?P<mode>[\w ]+)\))?"
)


def _num(raw, cast=float):
    """A field that may be absent; commas in numbers are stripped first."""
    if raw is None:
        return None
    try:
        return cast(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse(msg: str) -> dict | None:
    """One log line as a typed record, or None if it is not one we read.

    Returns a dict with `kind` in: submit, done, tool_call, disconnect,
    error, listening, loaded, context, loading.  The caller correlates
    submit/done/disconnect by their request id; everything else stands alone.
    """
    msg = msg or ""

    m = _DONE.search(msg)
    if m:
        prompt_total = _num(m.group("prompt_total"), int) or 0
        new_tokens = _num(m.group("new_tokens"), int) or 0
        output_tokens = _num(m.group("output_tokens"), int) or 0
        ttft_s = float(m.group("ttft"))
        total_s = float(m.group("total"))
        prefill_s = float(m.group("prefill_time"))
        # "none cached" means zero; "N% cached" is rounded, so the exact
        # cached count is the difference rather than the percentage.
        cached = prompt_total - new_tokens
        return {
            "kind": "done",
            "req_id": int(m.group("id")),
            "endpoint": m.group("endpoint"),
            "stream": int(m.group("mode") == "stream"),
            "output_tokens": output_tokens,
            "decode_tps": float(m.group("decode_tps")),
            "prompt_tokens": new_tokens,
            "cached_tokens": cached,
            "prompt_tokens_total": prompt_total,
            "ttft_ms": ttft_s * 1000.0,
            "latency_ms": total_s * 1000.0,
            "prefill_ms": prefill_s * 1000.0,
            "prefill_tps": _num(m.group("prefill_tps")),
            "decode_ms": max(0.0, (total_s - ttft_s) * 1000.0),
            "draft_accept": (int(m.group("draft_pct")) / 100.0
                             if m.group("draft_pct") is not None else None),
            "draft_mean_len": (
                (int(m.group("draft_acc")) / int(m.group("draft_tot")))
                if m.group("draft_tot") else None
            ),
        }

    m = _TOOL_CALL.search(msg)
    if m:
        return {
            "kind": "tool_call",
            "req_id": int(m.group("id")),
            "tool_calls": int(m.group("tool_calls")),
        }

    m = _DISCONNECT.search(msg)
    if m:
        return {"kind": "disconnect", "req_id": int(m.group("id"))}

    m = _ERROR.search(msg)
    if m:
        return {"kind": "error", "error": m.group("msg")}

    m = _SUBMIT.search(msg)
    if m:
        return {
            "kind": "submit",
            "req_id": int(m.group("id")),
            "endpoint": m.group("endpoint"),
            "stream": int(m.group("mode") == "stream"),
            "prompt_tokens": _num(m.group("prompt_tokens"), int),
            "max_tokens": _num(m.group("max_tokens"), int),
        }

    m = _LISTENING.search(msg)
    if m:
        return {"kind": "listening", "url": m.group("url")}

    m = _LOADED.search(msg)
    if m:
        return {"kind": "loaded", "load_ms": float(m.group("seconds")) * 1000.0}

    m = _CONTEXT.search(msg)
    if m:
        return {
            "kind": "context",
            "max_seq_len": _num(m.group("max_seq_len"), int),
            "cache_size": _num(m.group("cache_size"), int),
        }

    m = _LOADING.search(msg)
    if m:
        path = m.group("path")
        name = path.rstrip("/").rsplit("/", 1)[-1] if "/" in path else path
        return {"kind": "loading", "model_path": path, "model": name}

    return None