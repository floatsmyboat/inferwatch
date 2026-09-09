"""Parsers for SwarmUI journal lines.

Every regex here was written against real journal output from this host
(SwarmUI 0.9.8.3 with two self-started ComfyUI backends).  The format is

    HH:MM:SS.mmm [Level] [OptionalTag] message

and four families matter:

  1. generation lifecycle -- "User local requested 1 image with model '...'"
     and "Generated an image in 4.03 sec (prep) and 94.66 sec (gen)".  The
     prep/gen split is SwarmUI's own and has no equivalent in ComfyUI's API:
     prep is queueing plus model load, gen is the sampling itself, and a slow
     generation is a very different problem depending on which half grew.
  2. backend lifecycle    -- "Self-Start ComfyUI-0 on port 7821 started."
     This is where the backend PORTS come from, so a source can find the
     ComfyUI instances to poll without being told each one by hand.
  3. WebAPI errors        -- "[Error] [WebAPI] Error handling API request ..."
     Failures that never reach a backend, so ComfyUI's /history cannot see them.
  4. backend stderr       -- "[Warning] [ComfyUI-0/STDERR] RuntimeError: ..."
     The Python traceback behind a failed generation, tagged with the backend
     that produced it.

A NOTE ON THE CLOCK
-------------------
These lines carry a time but NO DATE, so the line alone cannot be placed in
history.  journald stamps every entry itself, which is why the journald reader
is the supported path; a plain log file gets its date from the
"== SwarmUI logs 2026-09-09 15:19 ==" banner SwarmUI writes on rotation, which
`parse_banner_date` reads.

Parsers are pure: they take a message string and return a dict, or None.
"""

from __future__ import annotations

import re

# ComfyUI colours its output, so the tag and the message arrive wrapped in SGR
# escapes ("\x1b[31m[ERROR]\x1b[0m").  Stored raw they render as mojibake in
# the dashboard and defeat any grouping of similar errors.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text or "")


_HEAD = re.compile(
    r"^(?P<clock>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+\[(?P<level>\w+)\]"
    r"(?:\s+\[(?P<tag>[A-Za-z0-9_/.-]+)\])?\s?(?P<rest>.*)$")

# "== SwarmUI logs 2026-09-09 15:19 ==" -- the only line carrying a date.
_BANNER = re.compile(r"^==\s*SwarmUI logs\s+(?P<date>\d{4}-\d{1,2}-\d{1,2})")

# -- generation lifecycle --------------------------------------------------
_REQUESTED = re.compile(
    r"^User (?P<user>\S+) requested (?P<n>\d+) images? with model '(?P<model>[^']*)'")
_GENERATED = re.compile(
    r"^Generated an image in (?P<prep>[\d.]+) sec \(prep\) and (?P<gen>[\d.]+) sec \(gen\)")

# -- backend lifecycle ----------------------------------------------------
_BACKEND_PORT = re.compile(
    r"^Self-Start (?P<backend>ComfyUI-(?P<index>\d+)) on port (?P<port>\d+)"
    r"\s+(?P<state>started|is loading)")
_BACKEND_INIT = re.compile(r"^Initializing backend #(?P<index>\d+) - (?P<kind>.+?)\.\.\.")
_BACKEND_STOP = re.compile(
    r"^Shutting down self-start ComfyUI \(port=(?P<port>\d+)\) process #(?P<pid>\d+)")
_BACKEND_DOWN = re.compile(r"^ComfyUI backend (?P<index>\d+) shutting down")

# -- server / errors ------------------------------------------------------
_WEBAPI_ERROR = re.compile(
    r"^Error handling API request '(?P<route>[^']*)'"
    r"(?: for user '(?P<user>[^']*)')?: (?P<reason>.*)$")
_RUNNING = re.compile(r"^SwarmUI v(?P<version>\S+) - (?P<mode>.+?) is now running\.")
_STDERR_TAG = re.compile(r"^ComfyUI-(?P<index>\d+)/STDERR$")
# ComfyUI's own start-up output reaches the journal without SwarmUI's
# timestamp/level prefix, so a traceback raised during boot takes this shape.
_BARE_STDERR = re.compile(r"^\[ComfyUI-(?P<index>\d+)/STDERR\]\s?(?P<text>.*)$")


def parse_banner_date(msg: str) -> str | None:
    """The date from a rotation banner, for readers whose lines lack one."""
    m = _BANNER.match(msg or "")
    return m.group("date") if m else None


def parse_line(msg: str) -> dict | None:
    """Parse one SwarmUI log line into a typed event, or None if unclaimed."""
    if not msg:
        return None
    head = _HEAD.match(msg)
    if not head:
        # Unprefixed backend output, which is how start-up stderr arrives.
        if m := _BARE_STDERR.match(msg):
            text = strip_ansi(m.group("text")).strip()
            return ({"level": "Warning", "tag": f"ComfyUI-{m.group('index')}/STDERR",
                     "clock": None, "kind": "backend_stderr",
                     "backend_index": int(m.group("index")), "text": text}
                    if text else None)
        return None
    level, tag = head.group("level"), head.group("tag")
    rest = head.group("rest").strip()
    base = {"level": level, "tag": tag, "clock": head.group("clock")}

    # Backend stderr: the tag identifies it, and the body is free text that
    # often continues a traceback over several lines.  Blank ones are dropped
    # rather than stored as empty events.
    if tag and (m := _STDERR_TAG.match(tag)):
        text = strip_ansi(rest).strip()
        if not text:
            return None
        return {**base, "kind": "backend_stderr",
                "backend_index": int(m.group("index")), "text": text}

    if tag == "WebAPI" and (m := _WEBAPI_ERROR.match(rest)):
        return {**base, "kind": "webapi_error", "route": m.group("route"),
                "user": m.group("user"), "reason": m.group("reason")}

    if m := _REQUESTED.match(rest):
        return {**base, "kind": "generation_requested", "user": m.group("user"),
                "images": int(m.group("n")), "model": m.group("model")}

    if m := _GENERATED.match(rest):
        # Reported in seconds; converted here so every duration in the store is
        # milliseconds, as it is for both other engines.
        return {**base, "kind": "generation_finished",
                "prep_ms": float(m.group("prep")) * 1000.0,
                "gen_ms": float(m.group("gen")) * 1000.0}

    if m := _BACKEND_PORT.match(rest):
        return {**base, "kind": "backend_up" if m.group("state") == "started"
                                else "backend_loading",
                "backend": m.group("backend"),
                "backend_index": int(m.group("index")), "port": int(m.group("port"))}

    if m := _BACKEND_INIT.match(rest):
        return {**base, "kind": "backend_init",
                "backend_index": int(m.group("index")), "backend_kind": m.group("kind")}

    if m := _BACKEND_STOP.match(rest):
        return {**base, "kind": "backend_stopping", "port": int(m.group("port")),
                "pid": int(m.group("pid"))}

    if m := _BACKEND_DOWN.match(rest):
        return {**base, "kind": "backend_down",
                "backend_index": int(m.group("index"))}

    if m := _RUNNING.match(rest):
        return {**base, "kind": "server_running", "version": m.group("version"),
                "mode": m.group("mode")}

    # Anything at Error or Warning that no rule above claimed is still worth
    # keeping: an unrecognised failure is the one you most want to see.
    if level in ("Error", "Warning") and rest:
        return {**base, "kind": "problem", "text": strip_ansi(rest).strip()}
    return None
