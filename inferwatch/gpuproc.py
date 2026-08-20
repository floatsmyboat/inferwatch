"""Attributing GPUs to the process that is using them.

`nvidia-smi` reports GPUs for the whole host, but a box often runs more than one
engine: on the development host vLLM holds two cards while Ollama has the third.
Plotting all three cards on an instance's pane would imply it uses all of them.

So each engine's GPUs are resolved by following processes:

    the port it serves  ->  listening pid  ->  its descendants
                        ->  intersect with nvidia-smi's compute processes
                        ->  the GPU indices those processes hold

Everything here degrades to None rather than guessing: without `ss`, without
nvidia-smi process visibility (common inside containers), or for a remote
instance, attribution is simply unavailable and the caller shows every GPU
without emphasis.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess

log = logging.getLogger("inferwatch.gpuproc")

_SS_PID = re.compile(r"pid=(\d+)")


def gpu_uuid_to_index() -> dict[str, int]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        out = subprocess.run([exe, "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    mapping = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                mapping[parts[1]] = int(parts[0])
            except ValueError:
                continue
    return mapping


def compute_apps() -> list[dict]:
    """Processes currently holding GPU memory, with the GPU index they hold."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    idx = gpu_uuid_to_index()
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory,gpu_uuid,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            mem = int(float(parts[1]))
        except ValueError:
            continue
        apps.append({"pid": pid, "mem_mib": mem, "gpu_index": idx.get(parts[2]),
                     "name": parts[3]})
    return apps


def listener_pid(port: int) -> int | None:
    """PID listening on a TCP port, via `ss`."""
    exe = shutil.which("ss")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-ltnp"], capture_output=True, text=True,
                             timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    for line in out.splitlines():
        # Match ":<port>" as the local address' port, not a peer or a substring
        # of a longer number.
        if not re.search(rf"[:\[]{port}\s", line):
            continue
        m = _SS_PID.search(line)
        if m:
            return int(m.group(1))
    return None


def _ppid(pid: int) -> int | None:
    """Parent of a pid, read from /proc/<pid>/stat.

    The comm field can contain spaces and parentheses, so the fields after it
    are taken from the last ')' rather than by splitting the whole line.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    close = data.rfind(")")
    if close < 0:
        return None
    rest = data[close + 2:].split()
    if len(rest) < 2:
        return None
    try:
        return int(rest[1])
    except ValueError:
        return None


def descendants(root: int, limit: int = 4096) -> set[int]:
    """All pids under `root`, inclusive.

    Built by walking every process's parent upward rather than scanning for
    children repeatedly, so the whole tree costs one pass over /proc.
    """
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return {root}
    parents = {pid: _ppid(pid) for pid in pids[:limit]}
    out = {root}
    for pid in parents:
        seen = set()
        cur = pid
        while cur and cur not in seen:
            seen.add(cur)
            if cur == root:
                out.add(pid)
                break
            cur = parents.get(cur)
    return out


def gpus_for_port(port: int) -> list[int] | None:
    """GPU indices used by the process tree serving a local TCP port.

    Returns None when attribution cannot be established -- no `ss`, no visible
    compute processes, or the port is served from somewhere this host cannot
    see -- so callers can tell "no GPUs" apart from "unknown".
    """
    apps = compute_apps()
    if not apps:
        return None
    pid = listener_pid(port)
    if pid is None:
        return None
    tree = descendants(pid)
    hits = sorted({a["gpu_index"] for a in apps
                   if a["pid"] in tree and a["gpu_index"] is not None})
    return hits or None


def port_of(url: str) -> int | None:
    m = re.search(r":(\d{2,5})(?:/|$)", url or "")
    return int(m.group(1)) if m else None


def is_local(url: str) -> bool:
    """Attribution only works for an engine on this host."""
    host = re.sub(r"^\w+://", "", url or "").split(":")[0].split("/")[0]
    return host in ("", "localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]")
