"""Parsing NInfer's server log.

A THIRD SHAPE, AND THE RICHEST OF THEM
--------------------------------------
NInfer serves an OpenAI-compatible API but publishes no `/metrics` (verified:
404, while `/health` and `/v1/models` answer).  So like ollama it is read from
its log -- and unlike either of the others, that log carries BOTH halves:

  per-request, on completion:
    [req 2] done finish=tool_calls tool_calls=1 prompt=41854 gen=2177 cache=0
            reuse=full_reset ttft=19893ms prefill=2113.1tok/s decode=70.8tok/s
            wall=50.67s speculative=mtp 2.69tok/round (56.3%)

  evenly sampled, every 5s:
    throughput interval=5.000s prefill=0.0tok/s decode=65.2tok/s running=1
            prefilling=0 decode_ready=1 waiting=0 avg_decode_batch=1.00

Ollama gives the first and not the second; vLLM gives the second and not the
first.  NInfer gives both, with TTFT, token counts, prompt-cache reuse and
speculative acceptance per request, and queue depth and batch occupancy as
gauges.

THE REQUEST ID IS REAL, WHICH CHANGES WHAT MAY BE JOINED
--------------------------------------------------------
Every line carries `[req N]`, so submission, completion and error join exactly.
That is worth stating because the vLLM proxy alongside it does NOT have one --
there, status and latency cannot be tied to a client at all and the pane says
so.  Here they can, and the join is a lookup rather than a guess.

The counter restarts from 1 whenever the server does, so ids are unique only
within one run.  A `listening on ...` line opens a new epoch and the correlator
retires whatever was still open, rather than letting a fresh `[req 1]` collide
with an abandoned one from the previous process.

WHAT IS NOT IN THE LOG
----------------------
No client address and no HTTP status.  `(client)` on the submission line marks
where `max_tokens` came from, not who called -- it is not an address, and must
not be read as one.  Finish reasons stand in for status: `stop_token`,
`output_limit`, `tool_calls`, and an `error` line for a request that failed.
"""

from __future__ import annotations

import re

# [req 3] openai_chat_completions stream msgs=58 max_tokens=32000 (client)
#         tools=11 tool_choice=auto tool_history=yes thinking=on ... → submitted
_SUBMIT = re.compile(
    r"\[req (?P<id>\d+)\]\s+(?P<endpoint>\w+)\s+(?P<mode>non-stream|stream)\s+"
    r"msgs=(?P<msgs>\d+)\s+max_tokens=(?P<max_tokens>\d+)"
    r"(?:\s+\(\w+\))?"
    r"(?:\s+tools=(?P<tools>\d+))?"
    r"(?:[^\n]*?\bthinking=(?P<thinking>on|off))?"
)

# [req 2] done finish=... prompt=... gen=... cache=... reuse=... ttft=...ms
#         prefill=...tok/s decode=...tok/s wall=...s speculative=mtp X (Y%)
_DONE = re.compile(
    r"\[req (?P<id>\d+)\]\s+done\s+finish=(?P<finish>\S+)"
    r"(?:\s+tool_calls=(?P<tool_calls>\d+))?"
    r"\s+prompt=(?P<prompt>\d+)\s+gen=(?P<gen>\d+)\s+cache=(?P<cache>\d+)"
    r"\s+reuse=(?P<reuse>\S+)"
    r"\s+ttft=(?P<ttft>\d+)ms"
    # The unit is only printed when there IS a rate: a request too short to
    # have one logs a bare `decode=n/a`, with no `tok/s` after it.  Requiring
    # the suffix silently dropped 7 of 16 completions in a day's log -- every
    # single-token one, which is exactly the set worth seeing.
    r"\s+prefill=(?:(?P<prefill>[\d.]+)tok/s|n/a)"
    r"\s+decode=(?:(?P<decode>[\d.]+)tok/s|n/a)"
    r"\s+wall=(?P<wall>[\d.]+)s"
    r"(?:\s+speculative=(?P<spec>\S+)"
    r"(?:\s+(?P<accept_len>[\d.]+)tok/round\s+\((?P<accept_pct>[\d.]+)%\))?)?"
)

_ERROR = re.compile(r"\[req (?P<id>\d+)\]\s+error\s+(?P<msg>.+?)\s*$")

_THROUGHPUT = re.compile(
    r"throughput\s+interval=(?P<interval>[\d.]+)s"
    r"\s+prefill=(?P<prefill>[\d.]+)tok/s"
    r"\s+decode=(?P<decode>[\d.]+)tok/s"
    r"\s+running=(?P<running>\d+)"
    r"\s+prefilling=(?P<prefilling>\d+)"
    r"\s+decode_ready=(?P<decode_ready>\d+)"
    r"\s+waiting=(?P<waiting>\d+)"
    r"(?:\s+avg_decode_batch=(?P<batch>[\d.]+|n/a))?"
)

# KV capacity explicit resolved=217600 tokens pages=3400/8192 ...
_KV = re.compile(
    r"KV capacity\s+\S+\s+resolved=(?P<tokens>\d+)\s+tokens"
    r"\s+pages=(?P<pages_used>\d+)/(?P<pages_total>\d+)")

_LISTENING = re.compile(
    r"listening on\s+(?P<url>\S+)\s+\(model id:\s*(?P<model>[^,]+),")

_LOADED = re.compile(r"model loaded in\s+(?P<seconds>[\d.]+)\s*s")


def _num(raw, cast=float):
    """A field the server may render as `n/a`, which is unknown, not zero."""
    if raw is None or raw in ("n/a", "none", ""):
        return None
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return None


def parse(msg: str) -> dict | None:
    """One log line as a typed record, or None if it is not one we read.

    Returns a dict with `kind` in: submit, done, error, throughput, kv,
    listening, loaded.  The caller correlates `submit`/`done`/`error` by their
    request id; everything else stands alone.
    """
    msg = msg or ""

    m = _DONE.search(msg)
    if m:
        gen = int(m.group("gen"))
        prompt = int(m.group("prompt"))
        cache = int(m.group("cache"))
        wall_ms = float(m.group("wall")) * 1000.0
        ttft_ms = float(m.group("ttft"))
        return {
            "kind": "done",
            "req_id": int(m.group("id")),
            "finish": m.group("finish"),
            "tool_calls": _num(m.group("tool_calls"), int),
            # Tokens actually evaluated; `cache` is what prefix reuse saved, so
            # the full prompt the client sent is the two together.
            "prompt_tokens": prompt,
            "cached_tokens": cache,
            "prompt_tokens_total": prompt + cache,
            "output_tokens": gen,
            "reuse": m.group("reuse"),
            "ttft_ms": ttft_ms,
            "prefill_tps": _num(m.group("prefill")),
            # n/a whenever a request produced too few tokens to have a rate.
            "decode_tps": _num(m.group("decode")),
            "latency_ms": wall_ms,
            # Time spent generating rather than waiting for the first token.
            "decode_ms": max(0.0, wall_ms - ttft_ms),
            "speculator": m.group("spec"),
            "draft_mean_len": _num(m.group("accept_len")),
            "draft_accept": (_num(m.group("accept_pct")) / 100.0
                             if m.group("accept_pct") is not None else None),
        }

    m = _ERROR.search(msg)
    if m:
        return {"kind": "error", "req_id": int(m.group("id")),
                "error": m.group("msg")}

    m = _SUBMIT.search(msg)
    if m:
        return {
            "kind": "submit",
            "req_id": int(m.group("id")),
            "endpoint": m.group("endpoint"),
            "stream": int(m.group("mode") == "stream"),
            "messages": int(m.group("msgs")),
            "max_tokens": int(m.group("max_tokens")),
            "tools": _num(m.group("tools"), int),
            "thinking": (None if m.group("thinking") is None
                         else int(m.group("thinking") == "on")),
        }

    m = _THROUGHPUT.search(msg)
    if m:
        return {
            "kind": "throughput",
            "interval_s": float(m.group("interval")),
            "prefill_tps": float(m.group("prefill")),
            "decode_tps": float(m.group("decode")),
            "running": int(m.group("running")),
            "prefilling": int(m.group("prefilling")),
            "decode_ready": int(m.group("decode_ready")),
            "waiting": int(m.group("waiting")),
            # n/a while nothing is decoding -- unknown, not a batch of zero.
            "avg_decode_batch": _num(m.group("batch")),
        }

    m = _KV.search(msg)
    if m:
        return {"kind": "kv", "kv_tokens": int(m.group("tokens")),
                "pages_used": int(m.group("pages_used")),
                "pages_total": int(m.group("pages_total"))}

    m = _LISTENING.search(msg)
    if m:
        return {"kind": "listening", "url": m.group("url"),
                "model": m.group("model").strip()}

    m = _LOADED.search(msg)
    if m:
        return {"kind": "loaded", "load_ms": float(m.group("seconds")) * 1000.0}

    return None
