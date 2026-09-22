"""Parsing the client-facing proxy that sits in front of vLLM.

WHY THIS EXISTS AT ALL
----------------------
vLLM publishes Prometheus metrics and nothing else: they are pre-aggregated and
carry no request identity, so no client address exists anywhere in them.  Its
own journal has uvicorn access lines, but behind a proxy every one of them reads
`127.0.0.1` -- the proxy is the only client vLLM ever sees.  On the development
host that is 100% of 3,879 lines in six hours.

The proxy, however, knows exactly who called it, and says so on every request:

    REQ backend=local v1/chat/completions from=10.0.0.96 model=qwen3.8 msgs=29
        prompt=19,172 limit=131072 max_tokens=16384 max_completion_tokens=None
        stream=True

`prompt` is exact rather than estimated -- the proxy tokenises upstream before
forwarding -- which makes it the one token figure on the vLLM side that is not a
bucket bound or a rate.

WHAT CANNOT BE PARSED, AND WHY IT IS NOT ATTEMPTED
--------------------------------------------------
The proxy logs `RESP` and `DONE` lines with the status and the duration, but
they are SEPARATE lines carrying no request id, and only `REQ` carries `from=`.
Joining them would mean assuming the order requests complete in, and that
assumption does not survive this workload: of 731 chat requests in 24 hours,
729 began while another was still in flight, peaking at 72 concurrent.  So a
status or a latency pinned to a client would be a guess dressed as a fact, and
this module does not make it -- the pane reports what `REQ` states and says
plainly that the rest is not attributable.

Giving the proxy a request id would close that gap; until it has one, these are
the fields that exist.
"""

from __future__ import annotations

import re

# The proxy prefixes its own timestamp, and journald prefixes the unit, so the
# marker is matched wherever it falls rather than anchored to the line start.
_REQ = re.compile(
    r"\bREQ\s+"
    r"(?:backend=(?P<backend>\S+)\s+)?"
    r"(?P<path>\S+)\s+"
    r"from=(?P<client>\S+)\s+"
    r"model=(?P<model>\S+)\s+"
    r"msgs=(?P<msgs>\d+)\s+"
    r"prompt=(?P<prompt>[\d,]+|unknown)\s+"
    r"limit=(?P<limit>\d+)"
    r"(?:\s+max_tokens=(?P<max_tokens>\S+))?"
    r"(?:\s+max_completion_tokens=(?P<max_completion_tokens>\S+))?"
    r"(?:\s+stream=(?P<stream>\S+))?"
)


def _int(raw: str | None) -> int | None:
    """A field the proxy renders with thousands separators, or as None/unknown."""
    if not raw or raw in ("None", "unknown", "?"):
        return None
    try:
        return int(raw.replace(",", ""))
    except ValueError:
        return None


def _opt(raw: str | None) -> str | None:
    return None if not raw or raw in ("None", "?") else raw


def parse_req(msg: str) -> dict | None:
    """One client request from a proxy REQ line, or None if it is not one.

    Unknown values stay None rather than becoming zero: `prompt=unknown` means
    the proxy could not tokenise the body, which is not the same as a request
    that carried no prompt, and averaging zeros in would quietly drag the
    per-client token figures down.
    """
    m = _REQ.search(msg or "")
    if not m:
        return None
    stream = m.group("stream")
    return {
        "endpoint": m.group("path"),
        "backend": _opt(m.group("backend")),
        "client": m.group("client"),
        "model": _opt(m.group("model")),
        "messages": _int(m.group("msgs")),
        "prompt_tokens": _int(m.group("prompt")),
        "context_limit": _int(m.group("limit")),
        "max_tokens": _int(m.group("max_tokens")),
        "max_completion_tokens": _int(m.group("max_completion_tokens")),
        # Absent on older proxy builds, which is unknown rather than False.
        "stream": None if stream is None else int(stream == "True"),
    }
