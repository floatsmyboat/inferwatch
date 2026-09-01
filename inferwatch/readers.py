"""Log readers: where an engine's log lines come from.

Ollama's per-request metrics only exist in its log, so the reader is the part
that decides whether this tool works on a given install at all.  Three are
supported:

    journald  ollama running under systemd  (full microsecond timestamps)
    file      `ollama serve` in a terminal, or any install writing to a file
    docker    ollama in a container

All three yield `(timestamp, message)` and feed the same `Correlator`, so the
parsing and joining logic is identical regardless of source.

A NOTE ON TIMESTAMP FIDELITY
----------------------------
journald stamps every entry with microsecond precision.  A plain log file does
not: ollama's own Go lines carry `time=...`, but the llama.cpp `slot` lines --
the ones holding the token counts and TTFT -- carry no timestamp at all.  The
file reader therefore carries the most recent timestamp it saw forward, from a
Go line's `time=` or a `[GIN]` line's clock (second resolution), and falls back
to wall clock. Ordering, which is what the correlator's joins depend on, is
always preserved; absolute precision is lower than journald's. Docker's `-t`
flag gives real per-line timestamps, so it sits between the two.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone

log = logging.getLogger("inferwatch.readers")

_GO_TS = re.compile(r"^time=(\S+)")
_GIN_TS = re.compile(r"^\[GIN\]\s+(\d{4}/\d{2}/\d{2})\s+-\s+(\d{2}:\d{2}:\d{2})")
_DOCKER_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)\s+(.*)$")


def parse_iso_dt(text: str):
    """Parse an ISO-8601 timestamp to a datetime, or None.

    Docker emits nanoseconds and datetime accepts at most microseconds, so the
    fraction is truncated rather than the whole value rejected.
    """
    try:
        t = text.rstrip("Z")
        was_utc = text.endswith("Z")
        if "." in t:
            head, frac = t.split(".", 1)
            tz = ""
            for i, ch in enumerate(frac):
                if ch in "+-":
                    tz = frac[i:]
                    frac = frac[:i]
                    break
            frac = (frac + "000000")[:6]
            t = f"{head}.{frac}{tz}"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None and was_utc:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def parse_iso(text: str) -> float | None:
    dt = parse_iso_dt(text)
    return dt.timestamp() if dt is not None else None


# A line whose own clock is this far behind the previous line is treated as a
# resolution artefact and clamped forward; anything older than this is trusted
# as a genuine jump (a rotated-in older file, or a clock change).
_BACKWARDS_TOLERANCE_S = 2.0


def log_timezone(message: str):
    """The UTC offset a log line declares, if it declares one.

    Only ollama's Go lines carry an offset (`-05:00`, or `Z`).  Learning it
    matters because gin lines do not: a naive clock belongs to the timezone of
    the process that WROTE it, which need not be the zone of the process
    reading it -- an engine in a UTC container read from a host in another zone
    is the ordinary case, not an exotic one.
    """
    m = _GO_TS.match(message)
    if not m:
        return None
    dt = parse_iso_dt(m.group(1))
    return dt.tzinfo if dt is not None else None


def derive_timestamp(message: str, previous: float | None, tz=None) -> float:
    """Best available timestamp for a log line lacking its own.

    Ollama's Go lines carry a microsecond clock WITH an offset; gin lines carry
    a second-resolution clock with NO offset; llama.cpp's slot lines carry none
    at all and inherit the last timestamp seen.

    A naive gin clock is interpreted in `tz` when one has been learned from a Go
    line, falling back to this process's local zone only when nothing better is
    known.  Always interpreting it locally is wrong by the whole offset
    difference whenever writer and reader disagree -- and silently correct on a
    host that shares the log's zone, which is how the bug hid.

    Mixing resolutions also makes time appear to go backwards: a gin line logged
    at 11:17:18.9 parses as 11:17:18.0, earlier than the Go line before it.
    Stored that way a request row could predate the task that produced it, so a
    small regression is clamped to the previous timestamp.
    """
    ts = None
    m = _GO_TS.match(message)
    if m:
        ts = parse_iso(m.group(1))
    if ts is None:
        m = _GIN_TS.match(message)
        if m:
            dt = parse_iso_dt(f"{m.group(1).replace('/', '-')}T{m.group(2)}")
            if dt is not None:
                if dt.tzinfo is None and tz is not None:
                    dt = dt.replace(tzinfo=tz)
                ts = dt.timestamp()
    if ts is None:
        return previous if previous is not None else time.time()
    if previous is not None and 0 < (previous - ts) <= _BACKWARDS_TOLERANCE_S:
        return previous
    return ts


class TimestampTracker:
    """Stateful timestamp derivation for one log stream.

    Remembers the last timestamp seen (for lines with no clock) and the
    timezone the log declares (for lines with a clock but no offset).
    """

    def __init__(self):
        self.last: float | None = None
        self.tz = None

    def feed(self, message: str) -> float:
        tz = log_timezone(message)
        if tz is not None:
            self.tz = tz
        self.last = derive_timestamp(message, self.last, self.tz)
        return self.last


_COMPACT_WINDOW = re.compile(r"^(\d+(?:\.\d+)?)([smhdw])$", re.IGNORECASE)


def to_journalctl_since(value: str) -> str:
    """journalctl accepts '-7d' but rejects '7d', so sign the compact form.

    Anything already in a journalctl dialect ('-2 days', '@1787000000',
    '1h ago') is passed through untouched.
    """
    text = (value or "").strip()
    if _COMPACT_WINDOW.match(text):
        return "-" + text
    return text


_WINDOW_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def to_docker_since(value: str, now: float | None = None) -> str:
    """Absolute RFC3339 timestamp for `docker logs --since`.

    docker takes a Go duration or a timestamp, and Go durations have no day or
    week unit -- `--since 7d` is an error there.  Resolving the window to an
    absolute time avoids the whole dialect problem, and anything unparseable
    yields "" so the caller omits --since and reads the full log rather than
    failing in a loop.
    """
    text = (value or "").strip()
    base = now if now is not None else time.time()
    if text.startswith("@"):
        try:
            return datetime.fromtimestamp(float(text[1:])).astimezone().isoformat()
        except ValueError:
            return ""
    m = _COMPACT_WINDOW.match(text.lstrip("-").strip())
    if m:
        seconds = float(m.group(1)) * _WINDOW_SECONDS[m.group(2).lower()]
        return datetime.fromtimestamp(base - seconds).astimezone().isoformat()
    # journalctl-only spellings such as "2 hours ago" have no docker equivalent.
    return ""


class Reader:
    """Base: yields (timestamp, message) and persists its own resume state."""

    kind = "base"

    def __init__(self, store, source: str, cfg: dict):
        self.store = store
        self.source = source
        self.cfg = cfg
        self.lines = 0
        self.errors = 0

    @property
    def state_key(self) -> str:
        return f"reader_state:{self.source}"

    def load_state(self) -> dict:
        raw = self.store.get_meta(self.state_key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError:
            return {}

    def save_state(self, state: dict) -> None:
        self.store.set_meta(self.state_key, json.dumps(state))

    def describe(self) -> str:
        return self.kind

    async def run(self, on_line, on_flush) -> None:      # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------
# journald
# --------------------------------------------------------------------------

class JournaldReader(Reader):
    """`journalctl -u <unit> -f -o json`, resuming from a stored cursor."""

    kind = "journald"

    def __init__(self, store, source: str, cfg: dict, backfill: str = "-2 days"):
        super().__init__(store, source, cfg)
        self.unit = cfg.get("unit") or "ollama"
        self.backfill = backfill
        self.cursor_stale = False
        self.empty_attempts = 0

    def describe(self) -> str:
        return f"journald unit={self.unit}"

    def _resume_since(self) -> str:
        try:
            row = self.store.query("SELECT MAX(ts) t FROM requests")
        except Exception:
            row = None
        if row and row[0]["t"]:
            return "@" + str(int(row[0]["t"] - 60))
        return to_journalctl_since(self.backfill)

    def _usable_cursor(self) -> str | None:
        if self.cursor_stale:
            return None
        cursor = self.load_state().get("cursor")
        if not cursor:
            return None
        stamp = cursor_timestamp_us(cursor)
        if stamp is not None and stamp > (time.time() + 60) * 1e6:
            log.warning("[%s] stored cursor is stamped in the future; resuming "
                        "by timestamp instead", self.source)
            self.cursor_stale = True
            return None
        return cursor

    def _argv(self) -> list[str]:
        argv = ["journalctl", "-u", self.unit, "-o", "json", "--no-pager", "-f"]
        cursor = self._usable_cursor()
        if cursor:
            argv += [f"--after-cursor={cursor}"]
        else:
            argv += ["--since", self._resume_since()]
        return argv

    async def run(self, on_line, on_flush) -> None:
        while True:
            argv = self._argv()
            log.info("[%s] following %s", self.source, " ".join(argv[:6]))
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL)
            before = self.lines
            try:
                await self._pump(proc.stdout, on_line, on_flush)
            except asyncio.CancelledError:
                proc.terminate()
                raise
            except Exception:
                log.exception("[%s] journald reader failed", self.source)
            if self.lines == before:
                self.empty_attempts += 1
                if self.empty_attempts >= 2 and not self.cursor_stale:
                    log.warning("[%s] follow produced nothing twice; abandoning "
                                "the stored cursor", self.source)
                    self.cursor_stale = True
            await asyncio.sleep(5)

    async def _pump(self, stream, on_line, on_flush) -> None:
        last_commit = time.time()
        cursor = None
        while True:
            raw = await stream.readline()
            if not raw:
                return
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            ts_us = entry.get("__REALTIME_TIMESTAMP")
            if ts_us is None:
                continue
            self.empty_attempts = 0
            cursor = entry.get("__CURSOR", cursor)
            self.lines += 1
            on_line(int(ts_us) / 1e6, message_text(entry.get("MESSAGE")))
            now = time.time()
            if now - last_commit > 1.0:
                if cursor:
                    self.save_state({"cursor": cursor})
                on_flush(now)
                last_commit = now


def message_text(m) -> str:
    """journalctl gives MESSAGE as a string, or a byte array if not UTF-8."""
    if isinstance(m, str):
        return m
    if isinstance(m, list):
        try:
            return bytes(m).decode("utf-8", "replace")
        except (TypeError, ValueError):
            return ""
    return ""


def cursor_timestamp_us(cursor: str | None) -> int | None:
    """A journald cursor embeds its own realtime stamp as ';t=<hex>'."""
    if not cursor:
        return None
    m = re.search(r"(?:^|;)t=([0-9a-fA-F]+)", cursor)
    return int(m.group(1), 16) if m else None


# --------------------------------------------------------------------------
# plain file
# --------------------------------------------------------------------------

class FileReader(Reader):
    """Follows a log file like `tail -F`, surviving rotation and truncation.

    Resume state is (inode, offset).  If the inode changed the file was rotated
    and reading restarts at the beginning of the new one; if the file shrank it
    was truncated in place, so the offset is reset.

    Only COMPLETE lines are consumed.  A read can land between a writer's
    write() and its newline, and treating that fragment as a line would deliver
    half a record, then its remainder as another -- so a `[GIN]` line would be
    split and neither half would parse.  The offset therefore never advances
    past a fragment; it is left for the next pass.  The one exception is a file
    being rotated away, whose trailing fragment will never be completed and is
    flushed rather than stranded.
    """

    kind = "file"

    def __init__(self, store, source: str, cfg: dict, poll: float = 0.5):
        super().__init__(store, source, cfg)
        self.path = os.path.expanduser(cfg.get("path") or "")
        self.poll = poll
        self.clock = TimestampTracker()

    def describe(self) -> str:
        return f"file {self.path}"

    async def run(self, on_line, on_flush) -> None:
        if not self.path:
            log.error("[%s] file reader needs a path", self.source)
            return
        state = self.load_state()
        offset = int(state.get("offset") or 0)
        inode = state.get("inode")
        fh = None
        last_commit = time.time()
        try:
            while True:
                try:
                    st = os.stat(self.path)
                except OSError:
                    if fh:
                        fh.close(); fh = None
                    await asyncio.sleep(2.0)
                    continue

                if fh is None or inode != st.st_ino:
                    if fh:
                        # This file gets no more writes, so any held-back
                        # fragment is final rather than half-written.
                        self._drain(fh, on_line)
                        fh.close()
                    fh = open(self.path, "r", encoding="utf-8", errors="replace")
                    if inode == st.st_ino and offset <= st.st_size:
                        fh.seek(offset)          # same file, carry on
                    else:
                        # New or rotated file: read it from the start so the
                        # history in it is not silently skipped.
                        inode = st.st_ino
                        offset = 0
                        fh.seek(0)
                    log.info("[%s] following file %s (inode %s, offset %s)",
                             self.source, self.path, inode, offset)
                elif st.st_size < offset:
                    log.info("[%s] %s was truncated; restarting from the top",
                             self.source, self.path)
                    offset = 0
                    fh.seek(0)

                read_any = self._consume(fh, on_line)
                # tell() is the start of any fragment left behind, so the
                # persisted offset re-reads it rather than skipping it.
                offset = fh.tell()
                inode = st.st_ino

                now = time.time()
                if read_any or now - last_commit > 1.0:
                    self.save_state({"inode": inode, "offset": offset})
                    on_flush(now)
                    last_commit = now
                if not read_any:
                    await asyncio.sleep(self.poll)
        except asyncio.CancelledError:
            if fh:
                self.save_state({"inode": inode, "offset": offset})
            raise
        finally:
            if fh:
                fh.close()

    def _emit(self, line: str, on_line) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        ts = self.clock.feed(line)
        self.lines += 1
        on_line(ts, line)

    def _consume(self, fh, on_line) -> bool:
        """Emit every complete line available, leaving a partial one in place.

        Returns whether anything was emitted.  Reading line by line rather than
        with readlines() is what makes holding the fragment possible: the
        position before each read is recorded so an unterminated line can be
        rewound to.  Text-mode seek only accepts a value tell() produced, which
        is why the offset cannot simply be arithmetic on the fragment's length.
        """
        emitted = False
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                return emitted
            if not line.endswith("\n"):
                fh.seek(pos)           # incomplete: wait for the newline
                return emitted
            self._emit(line, on_line)
            emitted = True

    def _drain(self, fh, on_line) -> None:
        """Flush whatever is left in a file we are about to stop reading.

        Called only when the inode changed, i.e. the file was rotated away.  Its
        last line may lack a newline because the writer moved on mid-record;
        emitting it keeps the record when it was merely unterminated, and a
        genuinely truncated one simply fails to parse downstream -- which beats
        losing it silently.
        """
        try:
            rest = fh.read()
        except OSError:
            return
        for line in rest.split("\n"):
            self._emit(line, on_line)


# --------------------------------------------------------------------------
# docker
# --------------------------------------------------------------------------

class DockerReader(Reader):
    """`docker logs -f -t <container>`, resuming with --since.

    `-t` makes docker prefix each line with an RFC3339 timestamp, so precision
    is better than a plain file and no timestamp has to be inferred.
    """

    kind = "docker"

    def __init__(self, store, source: str, cfg: dict, backfill: str = "-2 days"):
        super().__init__(store, source, cfg)
        self.container = cfg.get("container") or ""
        self.binary = cfg.get("docker_binary") or ""
        self.backfill = backfill
        self.clock = TimestampTracker()

    def describe(self) -> str:
        return f"docker container={self.container}"

    def _exe(self) -> str | None:
        return self.binary or shutil.which("docker") or shutil.which("podman")

    def _since(self) -> str:
        state = self.load_state()
        if state.get("since"):
            return state["since"]
        row = self.store.query("SELECT MAX(ts) t FROM requests")
        if row and row[0]["t"]:
            return datetime.fromtimestamp(row[0]["t"] - 60).astimezone().isoformat()
        return to_docker_since(self.backfill)

    async def run(self, on_line, on_flush) -> None:
        exe = self._exe()
        if not exe:
            log.error("[%s] docker/podman not found on PATH", self.source)
            return
        if not self.container:
            log.error("[%s] docker reader needs a container name", self.source)
            return
        while True:
            since = self._since()
            argv = [exe, "logs", "-f", "-t"]
            if since:
                argv += ["--since", since]
            argv.append(self.container)
            log.info("[%s] following %s", self.source, " ".join(argv))
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT)
            try:
                last_commit = time.time()
                while True:
                    raw = await proc.stdout.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", "replace").rstrip("\n")
                    m = _DOCKER_TS.match(line)
                    if m:
                        body = m.group(2)
                        # `docker logs -t` stamps every line absolutely, which
                        # beats anything inferable from the body.
                        ts = parse_iso(m.group(1)) or self.clock.feed(body)
                        self.clock.last = ts
                        tz = log_timezone(body)
                        if tz is not None:
                            self.clock.tz = tz
                    else:
                        body = line
                        ts = self.clock.feed(body)
                    self.lines += 1
                    on_line(ts, body)
                    now = time.time()
                    if now - last_commit > 1.0:
                        self.save_state({
                            "since": datetime.fromtimestamp(ts).astimezone().isoformat()})
                        on_flush(now)
                        last_commit = now
            except asyncio.CancelledError:
                proc.terminate()
                raise
            except Exception:
                log.exception("[%s] docker reader failed", self.source)
            # Container restarts are normal; retry rather than giving up.
            await asyncio.sleep(5)


READERS = {"journald": JournaldReader, "file": FileReader, "docker": DockerReader}


def build_reader(store, source: str, cfg: dict, backfill: str = "-2 days") -> Reader:
    kind = (cfg.get("reader") or "journald").lower()
    cls = READERS.get(kind)
    if cls is None:
        raise ValueError(f"unknown reader {kind!r}; expected one of {sorted(READERS)}")
    if cls is FileReader:
        return cls(store, source, cfg)
    return cls(store, source, cfg, backfill=backfill)
