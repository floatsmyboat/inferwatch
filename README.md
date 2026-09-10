# inferwatch

[![ci](https://github.com/floatsmyboat/inferwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/floatsmyboat/inferwatch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Real-time and historical metrics for locally served LLMs — **Ollama** and
**vLLM** — with a browser dashboard, a settings screen, and an MCP server so an
agent can query the same data.

One Python process, one SQLite file. No Docker, no Node, no Prometheus, no
external services. It never sits in the request path, so it cannot slow down or
break inference.

```
┌── Ollama ──────────────┐        ┌── vLLM ────────────────┐
│ journald / file /      │        │ GET /metrics           │
│ docker logs            │        │ (native Prometheus)    │
└──────────┬─────────────┘        └──────────┬─────────────┘
           │ per-request rows                │ pre-aggregated
           ▼                                 ▼
        ┌──────────────── SQLite (WAL) ────────────────┐
        │  requests · rollups · vllm_samples/hist      │
        └───────┬──────────────────────────┬───────────┘
                ▼                          ▼
         dashboard :7070            MCP server (stdio)
```

---

## The engines are not symmetric, and the tool does not pretend otherwise

This is the central design fact, so it is worth stating plainly.

| | Ollama | vLLM |
|---|---|---|
| Source | its log | `/metrics` |
| Per-request rows | **yes** | **no** — none exist to collect |
| TTFT / latency | exact, per request | histograms only |
| Tokens | per request | cumulative counters |
| Errors | HTTP status per request | `request_success_total{finished_reason}` |
| Client address | yes | no |
| Percentiles | exact within retention | bucket upper bounds; **means are exact** |
| KV / prompt cache | occupancy + eviction counts, sampled from the log | occupancy gauge, scraped |
| Unique extras | prompt cache reuse, draft accept, cold-load time, KV VRAM/RAM split | preemptions, batch occupancy, waiting-by-reason |

Both tabs show GPU utilisation, VRAM, temperature and power draw, since those
are measured by `nvidia-smi` rather than by either engine. Temperature and power
get separate charts rather than sharing an axis, and each is aggregated the way
its unit demands: utilisation averages across cards, VRAM and watts sum,
temperature reports the hottest card. Temperature is the one series not plotted
from zero — a 33–68 °C range starting at 0 wastes most of the plot.

### And a third family: image generation

SwarmUI and its ComfyUI backends are further from either of those than they are
from each other. There are **no tokens, no time-to-first-token and no context
window**; the unit of work is a generation with a duration, the models a
workflow loaded, and a node that may have thrown. So it gets its own tab, its
own tables and its own MCP tools rather than blank columns in someone else's.

Two sources feed it, and only one of them counts:

| | ComfyUI `/history` | SwarmUI journal |
|---|---|---|
| Generations | **the count** — stable `prompt_id`, ms timestamps | a timeline, never a second count |
| Models | read out of the workflow graph | the name on the request line |
| Errors | failing node, its class, the exception | WebAPI failures + backend stderr |
| Timing | total, measured | prep vs gen, split |

Joining them would mean guessing which log line belongs to which `prompt_id`,
and a request for N images produces N finish lines — so a 1:1 pairing would
misattribute every batch. They are kept apart instead: generations come from
`/history`, and SwarmUI's prep-versus-gen split is reported as its own
aggregate. Durations here are **exact**; every one is measured, so unlike the
token engines there is no bucketed-percentile caveat.

Backend ports are published exactly once, when SwarmUI starts its backends
(`Self-Start ComfyUI-0 on port 7821 started.`). They are learned from the log,
**remembered across restarts**, and as a last resort probed on the conventional
range — a collector restarted mid-life resumes the journal past those lines and
would otherwise collect queue depth and no generations at all.

So they get **separate dashboard tabs, separate tables and separate MCP tools**.
No attempt is made to reconstruct per-request rows for vLLM by differencing
counters: you cannot recover which TTFT belonged to which request, and faking it
would put invented rows beside real ones.

### Ollama: where the numbers come from

Ollama exposes no `/metrics` endpoint (verified — the route is not in the
binary). With `OLLAMA_DEBUG=1`, the embedded llama.cpp prints a timing block per
request, which is combined with the access line and the scheduler line:

```
slot print_timing: id 0 | task 6763 | prompt eval time = 1254.52 ms /  55 tokens
slot print_timing: id 0 | task 6763 |        eval time = 14591.31 ms / 416 tokens
[GIN] ... | 200 | 16.862061865s | 192.0.2.10 | POST "/v1/chat/completions"
time=... msg="context for request finished" runner.name=.../llama3.2:3b
```

That yields TTFT, prefill/decode split, token counts, decode rate, status,
client, endpoint and model — for every request, from every client, without
touching the request path. Two numbers fall out of the combination that neither
source has alone:

- **queue wait** = wall latency − runner time: time spent waiting rather than
  generating. A proxy cannot separate these.
- **prompt cache reuse** = full prompt length − tokens actually evaluated.

**`OLLAMA_DEBUG=1` is required.** Without it llama.cpp prints no timing lines:
request rates, statuses and GPU metrics still work, but TTFT and token counts
stay empty. The dashboard says so in a banner instead of showing zeros.

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_DEBUG=1"
```

### Ollama: KV and prompt cache

Ollama publishes no metrics endpoint, so there is nothing to scrape — but its
runner *logs* the cache state outright, and that line is collected into
`ollama_cache_samples`:

```
srv  update:  - cache state: 30 prompts, 8010.969 MiB (limits: 8192.000 MiB, 32768 tokens, 68068 est)
srv  get_availabl: prompt cache update took 364.22 ms
```

That is 97.8% of an 8 GiB pool, and the maintenance pass that produced it cost
364 ms **inside the request that triggered it** — cache pressure shows up as
TTFT, which is why the cost is stored next to the occupancy rather than
separately.

There are two different caches here and the tool keeps them apart:

| | What it is | Where it comes from |
|---|---|---|
| **Prompt cache** | the bounded pool of saved prompt states that lets a returning conversation skip prefill | `cache state` lines → `ollama_cache_samples.usage` |
| **Live KV cache** | the slot's own context memory, preallocated at load | `context_tokens / n_ctx_slot` per request → `ctx_usage` |

Prompt-cache occupancy is the closer analogue of vLLM's `kv_cache_usage_perc`.
Live KV occupancy is the distance to a **context shift or truncation** — a
request at 99.9% is one token from losing history.

Alongside the occupancy gauge, the same lines yield eviction pressure, split by
reason because the two mean different things: `evict_crowded` says the pool is
too small, `evict_invalidated` says the cached positions no longer applied. Each
eviction is a prefill somebody pays for again later.

Two caveats that shape every figure:

- **Sampling is ollama's, not ours.** A `cache state` line appears only when
  ollama runs a cache update, so an empty window means *no updates happened*,
  not *the cache was empty*. `samples` is always reported alongside, and the
  eviction counters are stored as **deltas** between samples and summed rather
  than turned into rates — a rate over unevenly spaced deltas would be fiction.
- **The lines need the runner at high log verbosity.** They come from
  llama-server, not from ollama's Go code. Ollama 0.32.x passes
  `--log-verbosity 4` on every load, so they are present by default there;
  `OLLAMA_DEBUG=1` is the documented way to be sure of it. Note that
  `/api/health`'s `debug_logging` only reports whether `OLLAMA_DEBUG` is set on
  the unit, so it can read `false` on a host that is logging these lines
  perfectly well.

Model loads also record how the KV cache was *placed*:

```
llama_kv_cache:      CUDA0 KV buffer size =  512.00 MiB
llama_kv_cache:        CPU KV buffer size = 4096.00 MiB
```

A cache that did not fit in VRAM caps decode throughput for the whole life of
the load, so that raises a `kv_offload` **WARN event** ("89% of the KV cache is
on host RAM, not VRAM") rather than being left buried in the load event's
detail.

### vLLM: where the numbers come from

vLLM's native Prometheus endpoint is scraped every
`collection.scrape_interval_s` (default 10s). Cumulative counters are
differenced; histogram buckets are differenced per bucket; both are written
**aggregated per minute**, because storing every scrape would add millions of
rows a month at resolutions no chart uses.

Three details worth knowing:

- **vLLM's own bucket bounds are stored with the counts.** Its bounds step
  1ms/20ms/250ms/2.5s/40s/640s; Ollama's step 25ms/200ms/1.5s/15s/60s. Neither
  is a refinement of the other, so re-bucketing one into the other would require
  interpolating between bounds — inventing numbers. Percentiles are computed
  against each source's own bounds and reported as **bucket upper bounds**.
- **`_sum` and `_count` are exact**, so the **mean is exact**. Since vLLM's
  buckets are coarse in the seconds range, the dashboard and MCP tools lead with
  the mean and label percentiles as "at most".
- **Restarts are detected** via `process_start_time_seconds` (and by a counter
  going backwards). The interval spanning a restart is dropped rather than
  emitted as a bogus delta.

Inter-token latency has been spelled `time_per_output_token_seconds`,
`inter_token_latency_seconds` and `request_time_per_output_token_seconds` across
vLLM releases. All are collected and whichever has data is used, so this works
against old and new servers with no configuration.

---

## Install

Requires Python 3.10 or newer — not because of this code (which is 3.9-clean)
but because fastapi, uvicorn, starlette and mcp all require it.

```bash
pip install git+https://github.com/floatsmyboat/inferwatch     # or:
git clone https://github.com/floatsmyboat/inferwatch && cd inferwatch
python3 -m venv --upgrade-deps .venv && .venv/bin/pip install -e ".[dev]"
```

Installing gives you two commands:

| Command | What it is |
|---|---|
| `inferwatch` | the collector, dashboard and API (`serve`, `ingest`, `stats`, `sources`) |
| `inferwatch-mcp` | the MCP server, over stdio |

Not on PyPI yet; install from git for now.

Backfill from a journal you already have, then look at the database:

```bash
inferwatch ingest --since 2d      # or: python -m inferwatch.main ingest
inferwatch stats
```

Run it:

```bash
inferwatch serve                  # http://127.0.0.1:7070
```

As a service — the unit is rendered from `systemd/inferwatch.service.in` for the
current user, checkout path and interpreter, so nothing is hardcoded:

```bash
./scripts/install-systemd.sh          # system service (uses sudo)
sudo systemctl enable --now inferwatch

./scripts/install-systemd.sh --user   # or per-user, no sudo
systemctl --user enable --now inferwatch
```

Override with `HOST=0.0.0.0 PORT=7070 DATADIR=... ./scripts/install-systemd.sh`.

### Exposing it on a network

There is **no authentication**, and the API is **not read-only**. It stores no
prompt or response text — only counts, timings, model names and client addresses
— but anyone who can reach the port can also write:

| Endpoint | What an unauthenticated caller can do |
|---|---|
| `PUT /api/config`, `POST /api/config/reset` | change any setting not pinned by env or flag |
| `POST /api/sources`, `PUT`/`DELETE /api/sources/{id}` | add, edit or remove a monitored engine |
| `POST /api/sources/probe` | make the server fetch an arbitrary URL, with an arbitrary bearer token |

That last one is a server-side request forgery primitive: the probe exists so a
typo surfaces before a source is saved, and it will dial whatever it is given.

So the firewall is not belt-and-braces, it is the only control:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 7070 proto tcp comment "inferwatch"
```

Better still, leave `server.host` at `127.0.0.1` and put something that
authenticates in front of it.

#### Secrets

A vLLM source's `api_key` is stored in the `sources` table **in plain text** —
SQLite has no encryption here, and the collector needs the value to send it as a
bearer token. So prefer an indirection:

```bash
inferwatch sources add --kind vllm vllm-prod \
    --set url=http://127.0.0.1:8000 --set 'api_key=${VLLM_API_KEY}'
```

`${VAR}` and `$VAR` are expanded when the request is made, so the database holds
only the pointer and the secret stays in the environment (a systemd
`EnvironmentFile=` is the natural home). An unset variable resolves to empty and
logs a warning, so the request fails as a clean 401 rather than sending a
literal `${VAR}` as the token.

Whatever is stored, it is **masked on every read**: `/api/sources`,
`/api/config`, the Settings tab, `inferwatch sources list`, and the MCP
`list_sources` and `run_sql` tools all report `***redacted***` instead of the
value. A reference like `${VLLM_API_KEY}` is shown as itself, since knowing
which variable is referenced is useful and the reference is not the secret.
Sending `***redacted***` back in a `PUT` means "leave it unchanged", so editing
a source through the API cannot overwrite a key with its own mask; sending an
empty string still clears it.

Note that a literal passed as `--set api_key=…` is visible in this process's
command line to any local user for as long as the command runs, which is a
second reason to prefer the reference form.

---

## Configuring what is monitored

Everything is editable from the dashboard's **Settings** tab, or from the CLI:

```bash
python -m inferwatch.main sources                       # list
python -m inferwatch.main sources add --kind vllm --name qwen \
    --set url=http://127.0.0.1:8000
python -m inferwatch.main sources add --kind ollama --name box \
    --set reader=file --set path=~/.ollama/logs/server.log
python -m inferwatch.main sources disable qwen
```

**Ollama log readers.** Not everyone runs Ollama under systemd:

| reader | for | timestamp fidelity |
|---|---|---|
| `journald` | `ollama.service` | microsecond, from journald |
| `file` | `ollama serve` in a terminal, or any install logging to a file | derived; see below |
| `docker` | Ollama in a container | per-line, from `docker logs -t` |

The `file` reader follows like `tail -F`, surviving rotation (inode change) and
truncation, and persists an offset so a restart does not replay. Ollama's Go
lines carry `time=`, but llama.cpp's `slot` lines — the ones holding the token
counts — carry no timestamp, so the most recent one seen is carried forward.
Ordering, which the joins depend on, always holds; absolute precision is lower
than journald's, and a second-resolution `[GIN]` line is clamped forward so time
never appears to run backwards.

### Settings precedence

```
spec default  <  database (Settings tab)  <  environment  <  command line
```

A key supplied by the environment or a flag is shown **read-only** in the
Settings tab with its origin, because the process was told to use it and a
browser must not override that silently. Saving is all-or-nothing, so a typo in
one field cannot leave a half-applied configuration. Changes to sources,
intervals and retention apply **without a restart**; `server.host` and
`server.port` are marked as needing one, and the API says so after saving.

### Configuration reference

Every setting below is editable in the Settings tab, settable as an environment
variable, and some are pinnable with a flag. The tables are generated from the
code (`scripts/gen-config-docs.py`), so they cannot drift from what the program
actually accepts.

<!-- BEGIN generated: configuration reference -->

_Generated by `scripts/gen-config-docs.py` — do not edit by hand._

#### Collection

| Setting | Default | Accepts | Environment variable | Notes |
|---|---|---|---|---|
| `collection.poll_interval_s` | `5.0` | 1–300 | `INFERWATCH_COLLECTION_POLL_INTERVAL_S` | How often nvidia-smi and the engine's own status endpoint are sampled. |
| `collection.scrape_interval_s` | `10.0` | 1–300 | `INFERWATCH_COLLECTION_SCRAPE_INTERVAL_S` | How often each vLLM instance's /metrics endpoint is read. vLLM counters are cumulative, so this sets the resolution of every rate and histogram derived from them. |
| `collection.backfill` | `2d` | `7d`, or `-2 days` / `@epoch` | `INFERWATCH_COLLECTION_BACKFILL` | **restart required** — How far back to read on a first run, before any resume state exists. Accepts 7d / 6h, or a journalctl form like '-2 days'. |
| `collection.rollup_interval_s` | `60.0` | 10–3600 | `INFERWATCH_COLLECTION_ROLLUP_INTERVAL_S` | How often the 1-minute and 1-hour aggregates are recomputed. |

#### Retention

| Setting | Default | Accepts | Environment variable | Notes |
|---|---|---|---|---|
| `retention.raw_days` | `7.0` | 0.5–3650 | `INFERWATCH_RETENTION_RAW_DAYS` | Per-request detail older than this is deleted. Rollups are kept indefinitely regardless, so long-range charts survive. |
| `retention.sample_days` | `30.0` | 0.5–3650 | `INFERWATCH_RETENTION_SAMPLE_DAYS` | GPU samples, engine samples and the event log are trimmed to this. |

#### Dashboard

| Setting | Default | Accepts | Environment variable | Notes |
|---|---|---|---|---|
| `dashboard.default_window` | `1h` | `15m`, `1h`, `6h`, `24h`, `7d`, `30d` | `INFERWATCH_DASHBOARD_DEFAULT_WINDOW` | Range selected when the dashboard is opened. |
| `dashboard.include_health` | `false` | — | `INFERWATCH_DASHBOARD_INCLUDE_HEALTH` | Include HEAD / and status polling in request rates. Off by default because on a polled instance it can be 90%+ of hits. |
| `dashboard.refresh_s` | `10.0` | 2–600 | `INFERWATCH_DASHBOARD_REFRESH_S` | How often the open dashboard refetches. The live request feed is pushed separately and is not affected by this. |

#### Server

| Setting | Default | Accepts | Environment variable | Notes |
|---|---|---|---|---|
| `server.host` | `127.0.0.1` | — | `INFERWATCH_SERVER_HOST` | **restart required** — 0.0.0.0 exposes the dashboard on the network. There is no authentication, so restrict it at your firewall. |
| `server.port` | `7070` | 1–65535 | `INFERWATCH_SERVER_PORT` | **restart required** — Port the dashboard and API listen on. |

#### Source fields

Set these with `--set key=value` on `sources add`, or in the Settings tab.

**Ollama** (`--kind ollama`)

| Field | Default | Required when | Notes |
|---|---|---|---|
| `reader` | `journald` | — | Where to read ollama's log from. Per-request metrics come from llama.cpp's debug lines, so one of these is required. One of `journald`, `file`, `docker`. |
| `unit` | `ollama` | `reader=journald` | Also used to attribute GPUs: every process in this unit's cgroup is matched against nvidia-smi, so only the cards ollama actually holds appear on its pane. |
| `path` | — | `reader=file` | Followed like tail -F, so rotation and truncation are handled. |
| `container` | `ollama` | `reader=docker` |  |
| `url` | `http://127.0.0.1:11434` | — | Used to poll /api/ps for resident models. |
| `models_dir` | — | — | Optional. Resolves blob digests to model names on load events. Defaults to $OLLAMA_MODELS or ~/.ollama/models. |

**SwarmUI / ComfyUI** (`--kind swarmui`)

| Field | Default | Required when | Notes |
|---|---|---|---|
| `url` | `http://127.0.0.1:7801` | — | Polled for queue depth and backend health. Its API needs no key for a local install. |
| `reader` | `journald` | — | Optional, unlike ollama's. Adds SwarmUI's prep-vs-gen timing split, WebAPI failures that never reached a backend, and the Python stderr behind a failed generation. 'none' polls the APIs only. One of `journald`, `file`, `docker`, `none`. |
| `unit` | `swarmui` | `reader=journald` | Also the fallback for GPU attribution, though each self-started ComfyUI is usually resolved exactly by its own port. |
| `path` | — | `reader=file` | SwarmUI's own rotated logs under Data/Logs work here. |
| `container` | — | `reader=docker` |  |
| `backends` | — | — | Optional, comma separated. Leave empty and the backend ports are discovered from the log, which is the only place SwarmUI publishes them. Required when the reader is 'none'. |
| `history_limit` | `64` | — | How many /history entries to read each tick. ComfyUI keeps this in memory only, so a larger number costs little and survives a burst between polls. |

**vLLM** (`--kind vllm`)

| Field | Default | Required when | Notes |
|---|---|---|---|
| `url` | `http://127.0.0.1:8000` | — | The OpenAI-compatible server root. /metrics is read from here. |
| `unit` | — | — | The unit running the ENGINE, which is what its GPUs are attributed by -- every process in that unit's cgroup is matched against nvidia-smi. Set it to the engine's unit, not a proxy in front of it: with a proxy on the URL there are no GPUs behind that port and attribution reports 'could not attribute'. The journal is also read from it for HTTP status codes, client addresses and engine errors, which /metrics does not expose. |
| `api_key` | — | — | Sent as a bearer token if the server requires one. Prefer an indirection like ${VLLM_API_KEY} over pasting the value: what is stored here goes into the database in plain text, and a reference keeps the secret in the environment or an EnvironmentFile instead. |

#### Command-line flags

| Flag | Purpose | Pins the setting |
|---|---|---|
| `--db` | — | — |
| `--unit` | systemd unit for the seeded Ollama source / for ingest | — |
| `--ollama-url` | — | — |
| `--models-dir` | ollama models dir (resolves blob digests to model names) | — |
| `--log-file` | ingest: read this log file instead of the journal | — |
| `--since` | log backfill window, e.g. '-2 days' | `collection.backfill` |
| `--retention-days` | raw request retention; rollups are kept forever | `retention.raw_days` |
| `--poll-interval` | — | `collection.poll_interval_s` |
| `--scrape-interval` | — | `collection.scrape_interval_s` |
| `--host` | — | `server.host` |
| `--port` | — | `server.port` |
| `-v`, `--verbose` | — | — |

Flags that pin a setting outrank both the environment and the Settings tab; the tab shows those keys read-only with their origin.

<!-- END generated: configuration reference -->

### Scope: one Ollama source; many vLLM and SwarmUI sources

vLLM and SwarmUI rows are keyed by source throughout, so any number of those
instances can be monitored side by side. The Ollama tables (`requests`,
`events`, `ps_samples`) are **not** source-partitioned, so exactly one Ollama
source runs at a time; enabling a second logs a warning and ignores it rather
than silently blending two instances into one set of numbers. Partitioning those
tables is a schema change worth doing deliberately.

One SwarmUI source covers all of that install's ComfyUI backends, which are
discovered rather than configured; the per-backend rows are keyed by backend
name within the source.

---

## Dashboard

`http://127.0.0.1:7070` — four tabs: **Ollama**, **vLLM**, **Images**, **Settings**.

### Which GPUs belong to which engine

A host often runs more than one engine, so plotting every card on an engine's
pane would imply it uses all of them. **Both** panes scope their GPU charts and
tiles to the cards their engine actually holds: those carry the series colour
and the aggregates (VRAM, watts, hottest card) count only them, while the host's
other cards stay visible in grey, labelled "other engine".

GPUs are resolved by three methods, in descending order of trust:

| Method | How | When it applies |
|---|---|---|
| **cgroup** | every pid in a unit's cgroup, intersected with `nvidia-smi`'s compute processes | a `unit` is configured on the source — the reliable one |
| **pids** | ollama's own `runner.pid` log lines | Ollama, for the `llama-server` it spawned |
| **port** | the listening pid → its descendants | last resort, no unit configured |

Prefer configuring `unit`. The port walk assumes the processes holding the GPUs
are children of whatever answers the port, and that is false for anything
non-trivial: vLLM v1 runs its engine core and workers as separate processes, and
a reverse proxy in front of the API server severs the link entirely. A cgroup is
what survives reparenting. On the development host the URL's listener is in
`vllm-proxy.service` while the workers are in `vllm-qwen38.service`, so only the
cgroup method finds them — set `unit` to the unit running the **engine**, never
the proxy.

Three outcomes, rendered differently, because collapsing them is how a wrong
answer gets presented as a right one:

- **specific cards** — "holds GPU 2, 3 of 4 · via cgroup:vllm-qwen38.service",
  with the method named so a surprising answer is diagnosable
- **none** — attributed, and the engine is on no card (a CPU-only instance)
- **unknown** — no nvidia-smi, no cgroup visibility, a remote engine, or an
  empty result from the weakest method (far likelier to be the wrong process
  tree than a genuinely idle engine). Every card is shown without emphasis and
  the pane says it could not attribute.

Attribution is re-resolved on an interval rather than cached once, and the
stored answer records **when and how** it was learned. An earlier version cached
the first success and let a failure be merged away, so a topology change never
propagated and the pane kept presenting weeks-old indices as current fact.

One filter row scopes everything below it. Every chart has a **Table** toggle
showing the same series as numbers, so no value is reachable only by hovering.
A live SSE feed drives the request ticker and the current-rate figure.

URL parameters: `?tab=vllm`, `?window=6h`, `?model=llama3.2:3b`, `?source=name`,
`?nostream=1` (disables the live feed — useful for kiosk displays and screenshot
tools, which otherwise wait forever on an open stream).

### API

| Endpoint | Returns |
|---|---|
| `/api/dashboard?window=1h&model=` | everything the Ollama tab needs, one time slice |
| `/api/vllm/dashboard?window=1h&source=` | same for one vLLM instance |
| `/api/summary`, `/api/timeseries`, `/api/models`, `/api/slowest?by=queue_ms` | Ollama breakdowns |
| `/api/vllm/summary`, `/api/vllm/timeseries`, `/api/vllm/instances` | vLLM breakdowns |
| `/api/requests`, `/api/errors`, `/api/events`, `/api/gpu`, `/api/ps` | raw rows and timelines |
| `/api/cache?window=1h` | Ollama prompt-cache occupancy, evictions, update cost |
| `/api/clients?window=1h&limit=25` | per-client detail: models requested, context sizes, tokens, TTFT |
| `/api/endpoints?window=1h` | traffic by endpoint and class |
| `/api/images/dashboard?window=24h&source=` | everything the Images tab needs, one time slice |
| `/api/images/summary`, `/api/images/timeseries`, `/api/images/models`, `/api/images/failures`, `/api/images/generations`, `/api/images/sources` | image breakdowns |
| `/api/config` (GET/PUT), `/api/config/reset` | settings |
| `/api/sources` (GET/POST/PUT/DELETE), `/api/sources/probe` | monitored engines |
| `/api/prefs`, `/api/status`, `/api/health` | dashboard defaults, collector state |
| `/api/stream` | SSE live feed |

`/api/sources/probe` checks a definition **before** it is saved, so a typo
surfaces there rather than as silence in the charts.

---

## MCP server

```bash
./scripts/install-mcp.sh      # writes .mcp.json for this checkout (gitignored)
```

or `claude mcp add inferwatch -- /path/to/.venv/bin/python -m inferwatch.mcp_server`.

Opens the same SQLite file read-only (`mode=ro` plus `PRAGMA query_only`) and
answers through the same query layer as the dashboard, so a number it reports
always matches the number on screen.

| Tool | Purpose |
|---|---|
| `get_summary`, `get_timeseries` | Ollama headline metrics and series |
| `compare_models`, `list_models` | per-model breakdown; what is resident |
| `recent_requests`, `slowest_requests`, `recent_errors` | Ollama per-request detail |
| `get_events` | cold loads, evictions, truncations, warnings |
| `vllm_summary`, `vllm_timeseries`, `vllm_instances` | vLLM metrics, reachability, GPU attribution |
| `gpu_status` | per-device util/VRAM/temp/power |
| `cache_status` | Ollama prompt-cache occupancy and eviction pressure, plus live KV usage |
| `client_stats` | who is calling, for which models, at what context size |
| `image_summary` | SwarmUI/ComfyUI throughput, durations, models, backends |
| `image_failures` | what failed, by node class, and what never reached a backend |
| `image_generations` | recent generations with every model the workflow loaded |
| `list_sources`, `get_settings` | what is monitored, and how it is configured |
| `health` | is collection working, is debug logging on |
| `run_sql`, `describe_schema` | read-only SELECT escape hatch, with units |

---

## Retention

- Raw per-request rows (Ollama): **7 days** (`retention.raw_days`).
- GPU samples, events, prompt-cache samples, image samples and the image event
  timeline, vLLM rows: **30 days** (`retention.sample_days`).
- Image generations follow the **raw** window, being the image equivalent of a
  request row. There are no rollups for them, so past retention the answer is
  "not stored" and `complete`/`covers_from` say so.
- `rollup_1m` and `rollup_1h`: **kept indefinitely**.

Rollups store fixed-bucket **histograms** of TTFT and latency, not pre-computed
percentiles. Histograms add, so a percentile over any range is computed by
summing buckets and walking to the target rank. Percentiles of percentiles would
be meaningless; this is not.

Queries inside the raw window return exact percentiles; beyond it they come from
histograms and are reported as the **upper bound of the containing bucket**.
Every response carries `exact: true|false`.

## Restarts and reboots

Resume state is per source — a journald cursor, a file inode+offset, or a docker
timestamp — and is flushed on SIGTERM. Two things make that safe rather than
merely likely:

**Writes are idempotent.** Every request and event row carries a `dedupe_key`
under a UNIQUE index, and inserts are `INSERT OR IGNORE`. Re-reading lines that
were already stored is a no-op, so `ingest` can be run repeatedly and a resume
can safely overlap.

**A cursor that cannot be used is not trusted.** If the journal a cursor points
into was rotated away, journalctl silently repositions using the timestamp
embedded in the cursor and resumes correctly. But a cursor stamped in the
*future* (clock skew, a restored database) makes journalctl wait for entries that
will not arrive, stalling collection silently; such a cursor is rejected on
startup. A follow attempt yielding nothing twice running does the same.

`systemctl stop` completes in well under a second. systemd records
`ExecMainStatus=15` alongside `Result=success`: uvicorn deliberately re-raises
the signal after shutting down, so exiting *by* SIGTERM is expected, not a crash.

---

## Honest limitations

**Attribution under parallelism (Ollama).** The task id in llama.cpp's timing
lines and the access line's status never appear together, so they are joined by
arrival order. With one request in flight that is exact. When two finish before
either access line prints, nothing in the log disambiguates them — those rows are
stored as `attribution='ambiguous'` rather than guessed. Values: `exact`,
`ambiguous`, `none` (failed before reaching the runner — no model is guessed
either), `orphan` (timings with no access line). A 2-day backfill on the
development host scored 174 exact, 9 ambiguous, 4 orphan, with
`OLLAMA_NUM_PARALLEL=1` in effect; expect a higher ambiguous share the more
requests run concurrently.

**vLLM's token and request counters are not per-request aligned.**
`generation_tokens_total` advances as tokens stream; `request_success_total`
advances only when a request completes. Over a short window they therefore
describe overlapping but different sets of requests, and dividing one by the
other does *not* give tokens per request. The API marks this with
`counters_aligned: false`, and the dashboard says so on the vLLM tab.

**No per-request anything for vLLM.** Covered above. If you need per-request
detail from vLLM, its request-level logging is the only source, and it logs
prompt text — which this tool deliberately never stores.

**Some breakdowns reach back only as far as raw retention.** Endpoint, status
code, client address and per-request identity exist only on raw request rows —
the rollups aggregate by `(bucket, model, class)` and carry none of them — so
`by_endpoint`, `status_breakdown`, `by_client`, `recent_errors`, `slowest` and
`recent_requests` have nothing to degrade to. Past retention the honest answer
is *not stored*, which is a different statement from *nothing happened*, so
those responses carry `covers_from` (the oldest surviving raw row) and
`complete`, and the dashboard's empty states name the cutoff instead of reading
as an idle window.

Note this bound is `retention.raw_days` (7 by default), **not** the 6-hour
`RAW_WINDOW_S` that decides when *other* queries switch to the rollups for
speed. A 7-day client breakdown is complete even though `summary()['exact']` is
false for that span; conflating the two reports a full answer as a partial one.

**A client's model mix is only as good as ollama's logging.** Ollama names the
model on a per-request scheduler line, and when that line is absent the request
is counted with its model left null — reported as `unattributed` per client
rather than dropped or guessed. On this host the rate has ranged from 100% named
to 0% named on different days, so a client showing mostly `unattributed` is a
statement about the log, not about the client.

**Image generations are counted from ComfyUI, never from SwarmUI's log.**
The log describes the same work from the orchestrator's side, but a request for
N images produces N "Generated an image" lines and nothing ties either to a
`prompt_id`. Counting both would double-count; joining them would guess. So the
log is a timeline and the count comes from `/history` alone — which does mean a
generation driven through ComfyUI directly, bypassing SwarmUI, still appears
(correctly), while one that failed inside SwarmUI before reaching a backend
appears only as a log error.

**ComfyUI's history is in memory, not on disk.** It is polled, so a backend
restarting between polls loses whatever it had not yet reported. At the default
interval that is a few seconds' exposure; `history_limit` controls how much of
the ring is re-read each tick.

**The Ollama prompt-cache gauge is sampled on ollama's schedule.** A `cache
state` line is logged only when ollama runs a cache update, so the series is
unevenly spaced and a window with no samples means "no cache updates happened",
not "the cache was empty". Its counters are therefore stored as deltas and
reported as totals, never as rates, and `samples` accompanies every figure.

**Live KV occupancy exists only inside the raw window.** It is computed from
`context_tokens / n_ctx_slot` on per-request rows. The rollups aggregate per
model and class rather than per slot, so beyond retention `ctx_usage` is null
rather than back-computed from an assumed context size. Rows written before the
`n_ctx_slot` column existed are skipped for the same reason.

**Logs are the feed, not the archive.** A journal may hold only a day or two
depending on `journald.conf`; the SQLite file is the historian. If logs rotate
faster than inferwatch runs, that gap is unrecoverable.

**Two byte-identical events in the same microsecond collapse to one.** The
dedupe key for events is built from their values, so an identical warning logged
twice within a microsecond keeps one row. A deliberate trade for guaranteed
idempotency — dropping a repeated warning beats duplicating history.

**Health-check traffic is separated, not counted.** `HEAD /` and `GET /api/ps`
were 96% of requests on the development host. They are stored with
`class='health'` and excluded from inference rates unless
`dashboard.include_health` is on; `requests_all` always includes them.

**Depends on log formats and metric names.** Ollama's timing lines are debug
output, not a contract, and vLLM renames metrics between releases.
`tests/test_parse.py` holds verbatim fixture lines and `tests/test_vllm.py` a
real `/metrics` excerpt; if an upgrade breaks parsing, those tests fail and show
what changed.

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

CI runs this on Python 3.10 through 3.14, plus a packaging job that builds the
wheel, asserts the dashboard HTML is inside it, and installs it into a clean
environment from an empty directory so the source tree cannot mask a packaging
mistake. No network, GPU or engine is required. Parser fixtures are verbatim real log lines and a
real `/metrics` excerpt. Coverage includes the correlator's join and
its ambiguous/orphan/failed cases, histogram percentiles and rollup idempotency,
the schema migration, signal-safe commits, cursor validation, file rotation and
truncation, timestamp monotonicity, counter-reset detection, and config
precedence and locking. The configuration reference in this file is generated
from the spec and a test fails if it drifts. The dashboard's JavaScript is
syntax-checked with a pure-Python parser and its formatters executed in a real
JS engine (both optional — no Node needed).

```bash
.venv/bin/python scripts/gen-config-docs.py --check   # docs match the code?
```

## Layout

```
inferwatch/parse.py        ollama log line parsers (pure, fixture-tested)
inferwatch/readers.py      journald / file / docker log readers
inferwatch/collect.py      correlator, GPU + model pollers, maintainer
inferwatch/vllm.py         Prometheus scraper, delta and reset handling
inferwatch/parse_swarm.py  SwarmUI log line parsers (pure, fixture-tested)
inferwatch/images.py       SwarmUI/ComfyUI poller, history ingest, log collector
inferwatch/image_metrics.py image-generation query layer
inferwatch/gpuproc.py      maps GPUs to the process tree holding them
inferwatch/vllm_metrics.py vLLM query layer
inferwatch/metrics.py      ollama query layer (shared by API and MCP)
inferwatch/store.py        SQLite schema, rollups, histograms, retention
inferwatch/config.py       typed settings spec, precedence, source validation
inferwatch/supervisor.py   builds and rebuilds collectors from the sources table
inferwatch/api.py          FastAPI endpoints + SSE
inferwatch/web/index.html  dashboard (single file, no CDN, no build step)
inferwatch/mcp_server.py   MCP server (read-only)
inferwatch/main.py         serve / ingest / stats / sources
```

---

## Contributing

Issues and pull requests are welcome. Two things make a change easy to accept:

- `python -m unittest discover -s tests -t .` passes.
- If you touched `inferwatch/config.py`, run `python scripts/gen-config-docs.py`
  so the README's configuration reference matches the code — a test enforces it.

Parser changes should come with a fixture line copied verbatim from real engine
output, the way the existing tests do. Log formats and metric names are not
contracts, and a real fixture is what makes a future break obvious.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
