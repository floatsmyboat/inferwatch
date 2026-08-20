"""Blob-digest -> model-name resolution.

The scheduler logs a model name (`runner.name=.../llama3.2:3b`) on most
lines, but the *load* lines only carry the blob path llama-server was handed:

    --model /var/lib/ollama/models/blobs/sha256-f5f1dd89...

Without a map, every cold-load event is anonymous until some later line happens
to mention the name.  Ollama's own manifest tree is the authority, so read it:

    $OLLAMA_MODELS/manifests/registry.ollama.ai/library/<name>/<tag>

is JSON whose layers include one of mediaType `application/vnd.ollama.image.model`
-- that layer's digest is the blob llama-server loads.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

log = logging.getLogger("inferwatch.models")

MODEL_LAYER = "application/vnd.ollama.image.model"
_BLOB_RE = re.compile(r"sha256[-:]([0-9a-f]{64})")


def default_models_dir() -> str:
    return os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")


def blob_digest(text: str | None) -> str | None:
    """Extract the bare hex digest from a blob path, URL, or 'sha256:...' ref."""
    if not text:
        return None
    m = _BLOB_RE.search(text)
    return m.group(1) if m else None


class ModelIndex:
    """Maps blob digest -> 'name:tag', refreshed from the manifest tree.

    Rescans at most every `ttl` seconds, and only when the tree's mtimes have
    changed, so a `ollama pull` shows up without paying for a walk per lookup.
    """

    def __init__(self, models_dir: str | None = None, ttl: float = 60.0):
        self.models_dir = models_dir or default_models_dir()
        self.ttl = ttl
        self.by_digest: dict[str, str] = {}
        self.names: set[str] = set()
        self._last_scan = 0.0
        self.refresh(force=True)

    @property
    def manifest_root(self) -> str:
        return os.path.join(self.models_dir, "manifests")

    def refresh(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_scan < self.ttl:
            return
        self._last_scan = now
        root = self.manifest_root
        if not os.path.isdir(root):
            return
        by_digest: dict[str, str] = {}
        names: set[str] = set()
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, root)
                parts = rel.split(os.sep)
                if len(parts) < 2:
                    continue
                # <registry>/<namespace>/<name>/<tag>  ->  name:tag
                name = f"{parts[-2]}:{parts[-1]}"
                names.add(name)
                try:
                    with open(path, "rb") as fh:
                        manifest = json.load(fh)
                except (OSError, ValueError):
                    continue
                for layer in manifest.get("layers", []):
                    if layer.get("mediaType") == MODEL_LAYER:
                        d = blob_digest(layer.get("digest"))
                        if d:
                            by_digest[d] = name
        if by_digest or names:
            self.by_digest = by_digest
            self.names = names

    def resolve(self, blob_or_path: str | None) -> str | None:
        d = blob_digest(blob_or_path)
        if not d:
            return None
        if d not in self.by_digest:
            self.refresh()
        return self.by_digest.get(d)
