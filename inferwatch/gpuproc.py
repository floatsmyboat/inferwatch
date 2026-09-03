"""Attributing GPUs to the engine that is using them.

`nvidia-smi` reports GPUs for the whole host, but a box often runs more than one
engine: on the development host vLLM holds two cards, ollama is confined to two
others, and an unrelated image generator holds those same two. Plotting every
card on one engine's pane would imply it uses them all.

THREE WAYS TO ASK, IN DESCENDING ORDER OF TRUST
-----------------------------------------------
1. `gpus_for_unit(unit)`  -- every pid in a systemd unit's cgroup.
2. `gpus_for_pids(pids)`  -- an exact pid set the caller already knows, e.g.
   ollama logs `runner.pid` for the llama-server it spawned.
3. `gpus_for_port(port)`  -- the listening pid, then its descendants.

(3) was the original and it is the weakest, because it assumes the processes
holding the GPUs are children of whatever answers the port.  That is false for
anything non-trivial: vLLM v1 runs its engine core and workers as separate
processes, and a reverse proxy in front of the API server severs the link
completely.  On the development host the listener for the vLLM URL lives in
`vllm-proxy.service` while the workers live in `vllm-qwen38.service`, both
reparented to init -- so the descendant walk finds nothing at all.  A cgroup is
the thing that actually survives reparenting, which is why (1) leads.

NONE AND EMPTY ARE DIFFERENT ANSWERS
------------------------------------
`None` means attribution was not possible: no `ss`, no nvidia-smi, no cgroup
visibility, a remote engine.  `[]` means it WAS possible and the engine holds no
GPU -- a CPU-only instance.  Callers render those differently (every card
undifferentiated versus none highlighted), so the two are never merged.
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


def compute_apps() -> list[dict] | None:
    """Processes currently holding GPU memory, with the GPU index they hold.

    None when nvidia-smi could not be asked at all, which is different from an
    empty list meaning "asked, and nothing is on a GPU right now".
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    idx = gpu_uuid_to_index()
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory,gpu_uuid,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, OSError):
        return None
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


def descendants(root: int) -> set[int]:
    """All pids under `root`, inclusive.

    Built by walking every process's parent upward rather than scanning for
    children repeatedly, so the whole tree costs one pass over /proc.

    Every pid is examined.  An earlier version capped the scan at 4096 entries,
    but os.listdir("/proc") has no meaningful order, so on a busy host that
    dropped arbitrary processes from the parent map and silently under-reported
    the tree -- which reads downstream as an engine holding fewer GPUs than it
    does.  Nothing here caches, so a full pass is the honest option and costs
    one stat per process.
    """
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return {root}
    parents = {pid: _ppid(pid) for pid in pids}
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


# "0::/system.slice/ollama.service", or a .scope for a transient unit.
_CGROUP_UNIT = re.compile(r"/([A-Za-z0-9_.@\\-]+\.(?:service|scope))")


def unit_of(pid: int) -> str | None:
    """The systemd unit a pid belongs to, from its cgroup."""
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    m = _CGROUP_UNIT.search(text)
    return m.group(1) if m else None


def normalise_unit(unit: str) -> str:
    """'ollama' -> 'ollama.service'; anything already suffixed is left alone."""
    unit = (unit or "").strip()
    if not unit or unit.endswith((".service", ".scope", ".slice")):
        return unit
    return unit + ".service"


def pids_in_unit(unit: str) -> set[int] | None:
    """Every visible pid whose cgroup is `unit`.  None if /proc is unreadable."""
    want = normalise_unit(unit)
    if not want:
        return None
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return None
    return {pid for pid in pids if unit_of(pid) == want}


def _gpus_of(pids) -> list[int] | None:
    """GPU indices held by `pids`.  None when nvidia-smi could not be asked."""
    apps = compute_apps()
    if apps is None:
        return None
    return sorted({a["gpu_index"] for a in apps
                   if a["pid"] in pids and a["gpu_index"] is not None})


def gpus_for_unit(unit: str) -> list[int] | None:
    """GPU indices held by a systemd unit's processes.

    The most reliable of the three: a cgroup is unaffected by reparenting, so it
    still describes an engine whose workers are separate processes.
    """
    pids = pids_in_unit(unit)
    # An empty set is not "this unit holds no GPU": a running unit always has at
    # least one process in its cgroup, so nothing there means the unit is not
    # running or the name is wrong -- which is unknown, not zero.
    if not pids:
        return None
    return _gpus_of(pids)


def gpus_for_pids(pids) -> list[int] | None:
    """GPU indices held by an exact pid set the caller already knows."""
    pids = {int(p) for p in (pids or []) if p}
    if not pids:
        return None
    return _gpus_of(pids)


def gpus_for_port(port: int) -> list[int] | None:
    """GPU indices used by the process tree serving a local TCP port.

    The weakest of the three -- see the module docstring -- and kept as the last
    resort for a single-process engine with no unit configured.
    """
    pid = listener_pid(port)
    if pid is None:
        return None
    return _gpus_of(descendants(pid))


def resolve(unit: str | None = None, pids=None,
            port: int | None = None) -> tuple[list[int] | None, str]:
    """Best available attribution, with the method that produced it.

    Tried in descending order of trust.  For the two precise methods the FIRST
    answer wins even when it is `[]`: a unit or a known pid set that
    demonstrably holds no GPU is a real result, and falling through to a weaker
    method could match somebody else's processes instead.

    The port walk is treated differently.  An empty result there means "nothing
    in the process tree behind this port is on a GPU", and the overwhelmingly
    likelier explanation is that it is the wrong tree -- a proxy in front of the
    engine, or workers that were reparented -- rather than a genuinely CPU-only
    engine.  So `[]` from the weakest method degrades to None, "could not
    attribute", instead of asserting that an engine holds no card.

    The method name travels with the answer so the UI can say how it knows, and
    so a surprising attribution is diagnosable without re-running this.
    """
    if unit:
        got = gpus_for_unit(unit)
        if got is not None:
            return got, f"cgroup:{normalise_unit(unit)}"
    if pids:
        got = gpus_for_pids(pids)
        if got is not None:
            return got, "pids"
    if port is not None:
        got = gpus_for_port(port)
        if got:
            return got, f"port:{port}"
    return None, "unavailable"


def port_of(url: str) -> int | None:
    m = re.search(r":(\d{2,5})(?:/|$)", url or "")
    return int(m.group(1)) if m else None


def is_local(url: str) -> bool:
    """Attribution only works for an engine on this host.

    IPv6 literals are bracketed, and splitting the authority on ":" would leave
    "[" as the host -- so "http://[::1]:8000" never matched the loopback list it
    was already meant to be in, and a v6-bound instance silently lost its GPU
    attribution.  The brackets are stripped before the port is.
    """
    host = re.sub(r"^\w+://", "", url or "").split("/")[0]
    if host.startswith("["):
        host = host[1:].split("]")[0]
    else:
        host = host.split(":")[0]
    return host in ("", "localhost", "127.0.0.1", "0.0.0.0", "::1", "::")
