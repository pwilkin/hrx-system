# loom-launch (native)

`loom-launch` finds published Loom profiles on a model hub, installs them with
`loom-fetch`, serves them with `loom-serve` behind an instrumented
OpenAI-compatible endpoint, and chats with and benchmarks them. It is the native
counterpart of the Python launcher in `loom/py/loom/launch/`: the same verbs,
the same deployment documents, and the same state directory, so either launcher
can read what the other wrote.

The endpoint belongs to a **launcher service**, not to a model: the launcher
keeps its listener, dashboard and instrumentation up with or without a model,
and models are loaded and unloaded under it (see
[Launcher and model lifecycle](#launcher-and-model-lifecycle)).

## Build

```sh
cd loom/rust/loom-launch
cargo build --release          # target/release/loom-launch
cargo test                     # unit tests + GPU-free end-to-end supervision tests
```

Cargo is the supported build. Dependencies: `clap`, `serde`, `serde_json`,
`tiny_http`, `ureq` (rustls), `url` (the URL parser ureq itself uses, for
`/api/fetch`) and `libc` (signals, `setsid` for `--detach`, interface
addresses). There is no async runtime.

## Commands

Global options go before or after the verb: `--state DIR`, `--hub URL`, `--json`,
`--profiles DIR` (repeatable; see [Local profiles](#local-profiles)).
Exit codes: `0` success, `1` not found / not running / refused, `2` usage.

| Verb | What it does |
| --- | --- |
| `search [QUERY]` | Lists hub repositories that publish a root `model.json` (tag `loom-serve`) and the [local profiles](#local-profiles) on this machine, with quant, served context (and the model's native context when it differs), card memory, published decode and a fit score for this machine. `--author`, `--tag` (repeatable), `--limit` (20, hub results), `--no-filter`, `--no-details`, `--offline` (cached profiles only), `--wait SECONDS` (per hub request). An owner prefix (`loom-fun/`) becomes an author filter. A hub that does not answer is a warning: local and cached profiles are still listed. `--json` rows carry `source` (`hub`/`local`), `path`, `installed`, `scenario`, `scenario_devices`. |
| `show TARGET` | One profile (owner/name, a local directory or its card, an install name, or a local profile's name): identity, quant, architecture, context (`40960 tokens (model native 262144)`: the preset's served context, then the model's ceiling), memory, measured-on device, the scenario it carries, preset capacities, chat policy, features, gates, published measurements with baseline ratio and roofline %. Ends with `install with: loom-launch install OWNER/NAME` for a hub profile and `serve with: loom-launch serve /abs/dir` for a local one. `--prompt TEXT [--system TEXT] [--thinking on\|off]` renders the prompt the package's own chat policy produces. `--raw`, `--revision`, `--offline`. |
| `install OWNER/NAME` | Runs `loom-fetch huggingface:<repo>[:<preset>][@<revision>] --output=<dir> --materialize=<mode> --no-input --json`, verifies `model.json`, then gives the install the profile's own `llm_config.json`/`info.json` (see [Chat policy and measurements](#chat-policy-and-measurements)) and `deployment.json`/`devices.json` when it publishes them (see [Installed scenarios](#installed-scenarios)) and records the install. `--output`, `--preset`, `--revision`, `--materialize auto\|copy\|share`, `--loom-fetch PATH`, `--offline` (`loom-fetch --offline`; no hub reads). |
| `install --from-path DIR [OWNER/NAME]` | Registers a package directory on this machine so it is listed and servable without the hub, under `OWNER/NAME` (default `local/<directory name>`). By default it is registered in place and nothing is copied; `--copy` copies it into `<state>/models/` (GGUF files are linked to their source, never copied). |
| `launcher start` | Starts the launcher service: the client-facing endpoint, the dashboard, `/loom/*` and the jobs, with no model. `--host` (127.0.0.1), `--proxy-port` (0 = ephemeral; the client-facing port), `--port` (8080; the native port of the models it loads, always on 127.0.0.1), `--api-key` (or `LOOM_LAUNCH_API_KEY`), `--allow-web`, `--loom-serve PATH`, `--loom-fetch PATH`; `--detach` runs it in its own session (output in `logs/launcher.log`), `--wait` (30 s) for it to answer. Refused while a launcher of the same state directory runs. |
| `launcher status` | The launcher (endpoint, pid, uptime, what it answered itself, the last model that exited on its own) and a one-line model summary. Exit 1 when none runs. |
| `launcher stop` | SIGTERM to the launcher, which unloads its model and exits; SIGKILL after `--wait` (45 s), then its children and recorded model server are stopped. Also stops a launch recorded by a launcher of before the split, and clears `launcher.json` and `current.json`. |
| `start [TARGET]` | Loads a model under the running launcher, or starts a launcher in the background first (with `--host`, `--proxy-port`, `--port`, `--api-key`, `--loom-serve`). Takes `serve`'s model options (`--context`, `--max-tokens`, `--parallel`, `--draft-depth`, `--session-cache`, `--prefix-cache`, `--checkpoint-capacity`, `--prefill-capacity`, `--device`, `--workers`, `--memory-bytes`, `--pool-backing`, `--allow-web` (this model), `--wait`, `--no-install`, `--port` (this model's native port)); installs and refusals happen before anything starts. Prints the server's output while it prepares; ^C cancels the load. Refused while a model is loaded. |
| `unload` | Unloads the model (SIGTERM, SIGKILL after `--wait` 30 s) or cancels one still preparing; the launcher keeps serving. Exit 1 when no model is loaded. |
| `serve [TARGET]` | One step: a launcher and its model together (^C stops both; `unload` leaves the launcher serving). Writes the deployment document, starts `loom-serve`, waits for readiness, serves it behind the launcher's endpoint; a [prepared scenario](#prepared-scenarios) runs its own deployment instead. With `--detach` and a launcher already running, the model is loaded under it (as `start`); a foreground `serve` is refused then. TARGET is owner/name, a local directory or its card, an install name, or a local profile's name. `--context`, `--max-tokens`, `--parallel`, `--draft-depth`, `--session-cache`, `--prefix-cache`, `--checkpoint-capacity`, `--prefill-capacity`, `--device`, `--host` (127.0.0.1; the address the proxy and dashboard listen on, see [Serving on the local network](#serving-on-the-local-network)), `--port` (8080, native listener, always on 127.0.0.1), `--proxy-port` (0 = ephemeral), `--workers`, `--memory-bytes` (default: the card's), `--pool-backing`, `--no-proxy`, `--api-key` (or `LOOM_LAUNCH_API_KEY`), `--allow-web` (the dashboard chat's `web_fetch` tool), `--wait` (1800 s), `--detach`, `--no-install`, `--loom-serve PATH`. |
| `chat [TARGET]` | Streams a conversation (REPL without `--prompt`). Uses `--base-url`, else the running launch, else starts one for the duration of the command (`--keep-server` leaves it running, detached). `--think on\|off\|auto`, `--system`, `--max-tokens`, `--temperature`, `--session` (`X-Loom-Session`), `--template-kwargs JSON` (`chat_template_kwargs`), `--save PATH`. REPL: `/help /reset /think /system /prompt /usage /save /exit`. Each turn ends with a usage line `[N prompt, N completion, R tok/s, ttft T ms]  (P% of <served context>)`. |
| `bench [TARGET]` | Runs a suite against the same endpoint resolution as `chat`: `--suite quick\|standard\|full` (standard), `--repeats N` (median of N samples), `--think` (off), `--session`; `--list`, `--show ID`, `--no-store`. Prints the measured table against the card's published rows and roofline. |
| `status` | The launcher (from `launcher.json`, probed) and the model (from `current.json`) with its counters; "none running" / "none loaded" for either. `--json`: `{launcher, model, current, state}`. Exit 1 only when neither runs (or the model's probe fails). |
| `logs` | Tail of the current (else latest) run's `serve.log`; `--lines`, `--run ID`, `--follow`, `--events`. |
| `stop` | Unloads the model (SIGTERM, SIGKILL after `--wait` 30 s) and says that the launcher is still running, naming `launcher stop`. `--all` stops the launcher too (as `launcher stop`). With no launcher service, stops a launch of before the split as it always did. |
| `hardware`, `doctor` | Machine description; environment check (`loom-serve`, `loom-fetch`, hub, state directory, device). |
| `dashboard` | A launcher in the foreground on the dashboard's port (`--port` 8090, `--host` (127.0.0.1), `--allow-web`, `--loom-serve PATH`, `--loom-fetch PATH`); models loaded from the page get a free native port each. ^C also unloads a model the page loaded. See below. |

`--context` defaults to the preset's served `context_capacity` and may widen it
up to the model's native context (`llm_config.json` `context_length`, else the
measured `context.native`), never beyond. `show`, `search`, the dashboard and
the chat usage line all report the served context.

Tool discovery for `loom-serve` / `loom-fetch`: the explicit flag, then
`LOOM_SERVE` / `LOOM_FETCH`, then a sibling of this executable, then `PATH`.

`loom-fetch` must understand `huggingface:` selectors (the checkout's
`loom/py/loom/model/fetch_reference.py`). The PyPI `hrx-loom==0.1.0` wheel
predates them; with that wheel, put the checkout's `loom/py` first on
`PYTHONPATH` when installing.

## Launcher and model lifecycle

The **launcher** is one long-lived process per state directory. It owns the
client-facing listener (`--host`/`--proxy-port`) and everything on it: the
OpenAI routes it forwards, `/loom/*`, the dashboard and its jobs, the request
log and counters, and the lifecycle of the model servers it runs as children.
It records itself in `launcher.json` and never needs a model. A **model** is
one `loom-serve` child listening on `127.0.0.1:<port>`: `start` (or the page)
loads it, the launcher points its upstream at it once it is ready (before
`current.json` announces it), and `unload` (or a crash) clears the upstream and
`current.json` again. Clients keep one address throughout: no restart, no port
change.

| Verb | What it does to the launcher | ... to the model |
| --- | --- | --- |
| `launcher start [--detach]` | starts it (refused if one runs) | none loaded |
| `start TARGET` | starts one in the background if none runs | loads it (refused while one is loaded) |
| `unload` | keeps it serving | SIGTERM, SIGKILL after 30 s; cancels one still preparing |
| `stop` | keeps it serving, and says so | as `unload` |
| `stop --all`, `launcher stop` | stops it | unloaded first |
| `serve TARGET` | starts one in the foreground (^C stops it) | loads it; with `--detach`, both run in the background |
| `status`, `launcher status` | reported | reported (`status`) |

Exact output against the `fake-serve` stand-in (paths shortened to `<dir>`):

```
$ loom-launch launcher start --detach --proxy-port 18300 --port 18301 --loom-serve <dir>/fake-loom-serve
launcher  http://127.0.0.1:18300 (pid 126326)
  dashboard http://127.0.0.1:18300/
  models    served on 127.0.0.1:18301 when loaded
  state     <dir>/state
  log       <dir>/state/logs/launcher.log
  no model is loaded; load one with: loom-launch start PROFILE
  stop it with: loom-launch launcher stop

$ loom-launch launcher status
launcher  running at http://127.0.0.1:18300/ (pid 126326, up 0s)
  started   2026-11-12T23:17:31Z
  listening 127.0.0.1:18300
  answered  page 0, api 0, loom 1, health 2, /v1 without a model 0
  log       <dir>/state/logs/launcher.log
  model     none loaded (`loom-launch start PROFILE` loads one)

$ loom-launch start <dir>/card --device task
loading   test/model under the launcher at http://127.0.0.1:18300 (native port 18301)
  context   8192 tokens, max 512 output, 2 rows
  | launching test/model on task (context 8192, max 512 output tokens, port 18301)
  | <dir>/fake-loom-serve --model=<dir>/state/runs/20261112T231731-126326/serve.json --listen=127.0.0.1:18301 --device=task
  | fake-serve: serving test-model (no device is used)
  | {"event":"ready","address":"127.0.0.1:18301","models":1}
  | serving at http://127.0.0.1:18300
serving test/model at http://127.0.0.1:18300
  deployment test-model
  server     http://127.0.0.1:18301 (native listener)
  dashboard  http://127.0.0.1:18300/
  run        <dir>/state/runs/20261112T231731-126326
  prepared in 0.9s
  unload with: loom-launch unload (the launcher keeps running; `loom-launch launcher stop` ends it)

$ loom-launch status
launcher  running at http://127.0.0.1:18300/ (pid 126326, up 1s)
  started   2026-11-12T23:17:31Z
  listening 127.0.0.1:18300
  answered  page 0, api 7, loom 3, health 5, /v1 without a model 0
  log       <dir>/state/logs/launcher.log
model     test/model @ http://127.0.0.1:18300
  deployment test-model
  server    http://127.0.0.1:18301
  pid       126352
  started   2026-11-12T23:17:31Z
  traffic   0 request(s), 0 completion tokens, in flight 0
  latency   ttft p50 - p95 -, decode mean -

$ loom-launch unload
unloaded test/model (pid 126352, exit status 143)
the launcher keeps running at http://127.0.0.1:18300/ (pid 126326); `loom-launch launcher stop` ends it

$ loom-launch stop                     # after another start
unloaded test/model (pid 126392, exit status 143)
the launcher is still running at http://127.0.0.1:18300/ (pid 126326); `loom-launch launcher stop` ends it (or `loom-launch stop --all`)

$ loom-launch stop                     # exit 1
no model is loaded
the launcher is still running at http://127.0.0.1:18300/ (pid 126326); `loom-launch launcher stop` ends it (or `loom-launch stop --all`)

$ loom-launch launcher stop
stopped the launcher (pid 126326) at http://127.0.0.1:18300; no model was loaded

$ loom-launch serve <dir>/card --device task --port 18301 --proxy-port 18300 --loom-serve <dir>/fake-loom-serve --detach
serving test/model at http://127.0.0.1:18300
  deployment test-model
  server     http://127.0.0.1:18301 (native listener)
  dashboard  http://127.0.0.1:18300/
  launcher   pid 126440; it keeps serving when the model is unloaded
  log        <dir>/state/logs/detached.log
  unload with: loom-launch unload; stop everything with: loom-launch stop --all

$ loom-launch stop --all
stopped the launcher (pid 126440) at http://127.0.0.1:18300; unloaded test/model
```

(`exit status 143` is the stand-in dying of SIGTERM; `loom-serve` drains and
exits 0.)

**With no model loaded**, the launcher answers `/v1/*` itself:

```
HTTP/1.1 503
{"error": {"message": "no model is loaded", "type": "loom_launch_error", "code": "UNAVAILABLE"}}
```

and `/healthz` reports itself healthy: `{"status": "ok", "launcher": {"pid",
"uptime_seconds"}, "model": false, "state": "none", "repo": null}` (`state` is
`starting` while a model prepares, `serving` once it is loaded, with `model:
true`). `/loom/*`, `/api/*`, the dashboard and its assets keep working; the
503s are counted (`/v1 without a model`) and the counters run on across models.

**Control.** The CLI talks to a running launcher through its own dashboard
routes (`/api/serve`, `/api/unload`, `/api/jobs/<id>`) with the control token
from `launcher.json` as its bearer token, so it works under `--api-key`
without the key, and its loads and unloads are Activity rows like the page's.
That token also lets `start` load a package directory the page could not
name; `chat` and `bench` use it when they talk to the launcher's model without
a key of their own. With a launcher running and no model, `chat` and `bench`
load the model under it for the command and unload it afterwards
(`--keep-server` leaves it loaded).

**Signals and exits.** ^C (SIGINT) or SIGTERM ends a launcher: it unloads its
model (or cancels one preparing) and exits, clearing both records. A launcher
started with `--detach` (or by `start`, or `serve --detach`) runs in its own
session and ignores SIGHUP, so it outlives the shell; `launcher stop` (or `stop
--all`) ends it. An unload of a model still preparing cancels it: the server is
stopped, its load job fails as cancelled, and nothing is reported as a crash. A
model that exits on its own leaves the launcher serving with no model: an
Activity row (`exit`, failed, with its last output), `last_exit` in
`/api/status`, `/loom/status`, `launcher status` and `launcher.json`. The
launcher records the server it spawns in `launcher.json` (`model`) before it is
ready, so if the launcher itself is killed outright, the next `launcher stop`,
`stop` or `launcher start` stops that server; a forced `launcher stop` also
stops the launcher's children.

**Records of before the split.** A `current.json` written by an older
`loom-launch serve` (its `launcher_pid` is that serve process, and there is no
`launcher.json`) is still stopped by `stop` (and `unload`) exactly as before,
and `launcher stop` cleans it up too.

## Serving

A load (`start`, `serve`, the page) runs

```
loom-serve --model=<run>/serve.json --listen=<host>:<port> --device=<uri> \
  [--workers=N] [--pool_backing=X] [--memory_bytes=<card memory_bytes or --memory-bytes>]
```

with the device defaulting to `amdgpu://` when `rocminfo` reports a GPU agent
(else `task`). The deployment document is the Python launcher's:

```json
{"name": "<repo slug>", "model": {"path": "/abs/model.json", "preset": "chat"},
 "definition": {"kind": "text", "rows": 4, "max_sessions": 4, "context_capacity": 32768,
                "max_tokens": 4096, "draft_depth": 3, "...": "only what was chosen or declared"}}
```

`--context` may not exceed the package's native context and the output limit
must be smaller than the context; `--parallel` sets `rows` and `max_sessions`.

### Prepared scenarios

A package directory holding both `deployment.json` and `devices.json` is a
*scenario*: it was prepared for one machine shape, such as a GPU and an XDNA NPU
sharing an exchange pool (`device_placement` and `exchange_pool` in its card).
`serve` runs it exactly as prepared, still behind the proxy and dashboard:

```
loom-serve --devices=<pkg>/devices.json --model=<pkg>/deployment.json \
  --listen=127.0.0.1:<port> [--workers=N]
```

There is no composed `serve.json` and no `--device`. The scenario owns its
capacities and devices: `--context`, `--max-tokens`, `--parallel`,
`--draft-depth`, `--checkpoint-capacity`, `--session-cache`, `--prefix-cache`,
`--prefill-capacity` and `--device` are refused (also under `--detach`, before
anything starts) with a message naming `deployment.json` and `devices.json` and
the capacities they declare. `--port`, `--proxy-port`, `--host`, `--wait`,
`--workers`, `--detach`, `--api-key` and `--no-proxy` work as usual.
`loom-serve` refuses `--memory_bytes` and `--pool_backing` next to `--devices`,
so `--memory-bytes` / `--pool-backing` write a copy of `devices.json` into the
run directory with the deployment's pool (`deployment.json` `pool`) changed, and
the command reads that copy; the package's own file is never modified.

The capacities a scenario serves are its card preset's `execution` overlaid with
the deployment's `definition`. `serve` prints which deployment runs on which
devices (`scenario  running the package's own deployment qwen3.8-27b across
amdgpu + xdna`), and `current.json`, the run's `meta.json` and `/api/status`
record `kind: "scenario"`, `device` (`amdgpu + xdna`), `deployment`,
`context_length`, `rows`, `max_tokens`, `command` (the exact argv) and a
`scenario` object (`directory`, `deployment_path`, `devices_path`,
`served_devices_path`, `device_uris`, `pool`, `pools`, `overrides`). Ordinary
launches record `kind: "profile"`.

### Installed scenarios

An installed scenario carries its own deployment documents. `loom-fetch`
installs only what the card declares, and a card does not declare
`deployment.json` or `devices.json`, so after a successful fetch `install`
completes the package itself, in this order:

1. Both files already in the installed directory: used as they are (no network).
2. Both in the local Hugging Face cache's snapshot of the pinned revision
   (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`):
   copied from there (no network).
3. With `--offline`: nothing more; the install is an ordinary profile.
4. Otherwise both are read from the hub at the revision the fetch pinned (the
   commit `loom-fetch` reports, or `--revision`) and written next to the card;
   a pair that does not describe a servable scenario is removed again.

A profile that publishes neither (most do) installs exactly as before: nothing
is printed and the install record gains no keys. A profile that publishes only
one of them, or a hub that cannot be read, gets a `warning:` on stderr, but the
install that already succeeded stands:

```
  scenario  deployment qwen3.8-27b on amdgpu + xdna (deployment.json + devices.json from the hub at 3d5342e…)
warning: ilintar/x@v1 publishes deployment.json but not devices.json; installed as an ordinary profile (a scenario needs both)
warning: could not check ilintar/x@3d5342e… for deployment.json and devices.json: cannot reach https://…/deployment.json: …. The install itself succeeded, but if the profile is a GPU+NPU (or other multi-device) scenario, `serve` will run a plain single-device deployment until both files are in <dir>; rerun the install when the hub is reachable, or copy them from the repository at that revision
```

The install record says `"scenario": true` and `"scenario_revision"` (the
revision the documents came from); `install --json` reports
`result.scenario_documents` (`status` present/cache/hub/absent/offline/not-hub/
incomplete/invalid/failed, `revision`, `source`, `devices`, `warning`).
`install --from-path` records `"scenario": true` for a directory holding both.
`serve OWNER/NAME` then runs the scenario command above, `show OWNER/NAME`
explains the installed package (its scenario line included) rather than the
hub's, and `search`/`/api/models` judge an installed profile's `scenario` by its
installed directory.

### Chat policy and measurements

A card does not declare `llm_config.json` (the generator's configuration: the
chat policy with its thinking mode and delimiters, the tool-call format and the
drafter recommendation) or `info.json` (the publisher's measurements), so
`loom-fetch` does not install them either. Without `llm_config.json` the
launcher has no chat policy for the package: the dashboard cannot tell whether
the model thinks, so its thinking switch stays off (with the reason shown).
After a successful fetch `install` therefore gives the package both documents,
each on its own, in this order:

1. Already next to the card: used as it is.
2. In the local Hugging Face cache's snapshot of the pinned revision: copied.
3. In the launcher's own cache of the profile's hub documents
   (`<state>/cache/profiles/`), when that copy was read at the same revision:
   written out from there.
4. With `--offline`: nothing more.
5. Otherwise read from the hub at the pinned revision and written next to the
   card (a body that is not a JSON object is refused).

Neither is required: a profile that publishes none installs as before, and a
hub that cannot be read leaves the install standing with a `warning:` naming
what the package lacks (`could not read llm_config.json of <repo>@<rev>: …. The
install itself succeeded, but without llm_config.json it has no chat policy …;
rerun the install when the hub is reachable`). `install` prints a `policy` line (`llm_config.json +
info.json from the hub at 3d5342e…`), and `install --json` reports
`result.profile_documents` (`name`, `status`
present/cache/launcher-cache/hub/absent/offline/not-hub/invalid/failed,
`installed`, `source`, `warning`).

`install --from-path` never asks the hub: it looks next to the card, then in the
local hub cache and the launcher's cache at the revision the package's
`model.json.lock` records for its card (`origins.document.requested.url`,
`…/<owner>/<name>/resolve/<revision>/`), and writes what it finds next to the
card. When the package still has no `llm_config.json` it says so and names the
remedy (`loom-launch install <owner>/<name>`, or copying the file).

### Local profiles

Every directory holding a `model.json` under a profile root is a local profile:
`--profiles DIR` (repeatable, a global option) and `$LOOM_LAUNCH_PROFILES` (a
path list), searched four levels deep (hidden directories and a package's own
subdirectories are skipped), plus every package registered with
`install --from-path`. Local profiles are read from their own files only: they
never trigger a hub request, need no `--offline`, and a hub that does not answer
does not hide them. `search` and the dashboard's `/api/models` list them next to
hub results with `source: "local"`, `path`, `installed` (under `<state>/models/`
or registered), `scenario` and `scenario_devices`, annotated like hub rows
(quant, served and native context, memory, published decode, fit). `serve`,
`show`, `chat` and `bench` accept a local profile's directory, install name,
card model id or directory name (an ambiguous name is refused, listing the
paths).

### Readiness and the proxy

Readiness needs the server to be listening (its `{"event":"ready",...}` line, or
`/healthz` if that line never appears) and `/v1/models` to list the deployment
with registration `ready`. A `failed` registration (`data[].loom.registration`
or `/v1/status`) ends the wait at once with the server's error message, and the
server is stopped.

The launcher's listener (tiny_http, one thread per request) forwards everything
except `/loom/*`, `/healthz` and the dashboard routes (`/`, `/api/*`) to the
loaded model's native listener, and answers `/v1/*` with 503 while none is
loaded. SSE
responses are relayed line by line with chunked framing and a flush per line.
Each request is appended to `requests.jsonl` with the keys `id, time, method,
path, route, model, status, stream, duration_seconds, ttft_seconds,
prompt_tokens, completion_tokens, cached_tokens, decode_tokens_per_second,
chunks, request_bytes, response_bytes, client, error`. Launcher routes:
`/loom/metrics` (Prometheus: `loom_launch_requests_total{route,status,stream}`,
`loom_launch_{prompt,completion,cached}_tokens_total`,
`loom_launch_ttft_seconds{quantile}`,
`loom_launch_decode_tokens_per_second{quantile}`, `loom_launch_in_flight`,
`loom_launch_upstream_up`, `loom_launch_errors_total`,
`loom_launch_{request,response}_bytes_total`, `loom_launch_uptime_seconds`),
`/loom/status` (`proxy`, `launcher`, and the loaded model's `launch`,
`settings`, `scenario`, or `launch: null`), `/loom/requests?limit=N`. With
`--api-key`, every route except `/healthz` and the static dashboard page
requires `Authorization: Bearer` (the key, or the launcher's control token). A
dead upstream answers 502 with an OpenAI-shaped `loom_launch_proxy_error`. A
launch `chat`/`bench` start for themselves when no launcher runs keeps a proxy
of its own, with the same routes.

### Serving on the local network

`serve --host 0.0.0.0` (or `launcher start --host 0.0.0.0`) binds the
launcher's endpoint and its dashboard there; `loom-serve` itself always listens on `127.0.0.1`, so inference
is only reachable through the instrumented, optionally keyed proxy.
`--no-proxy` with a non-loopback `--host` is refused. After the `endpoint`
line (which stays a loopback address for local clients), `serve` prints one
`network   http://<ip>:<port>/` line per IPv4 interface address that is up
(everything but loopback and link-local), and `current.json`, the run record and
`/api/status` record `bind_host` and `urls`. Without `--api-key` it warns that
anyone who can reach the address can use the endpoint. With `--api-key` the
dashboard and network URLs are printed with `?access_token=<key>` so they open
signed in (the page moves the key to its session storage and out of the
address bar); the key is printed only when stdout is a terminal, so
`detached.log`, `serve.log`, `events.jsonl` and the run record never carry it.
`dashboard --host 0.0.0.0` prints the same URLs (it takes no key); models loaded
from the page stay on loopback behind it.

The request guard accepts a LAN deployment: `Host: 192.168.0.200:8099` is an IP
address, and a POST whose `Origin` is `http://192.168.0.200:8099` matches it
(default ports normalized). A foreign `Origin` (`http://evil.example`, or
another machine on the LAN) is refused, as is a `Host` that is a DNS name other
than `localhost`.

## State directory

`--state`, else `$LOOM_LAUNCH_STATE`, else `~/.local/state/loom-launch`
(always made absolute):

```
installs.json                 {"installs": {"<repo>": {repo, path, preset, revision, installed_at, size_bytes,
                              source?, scenario?, scenario_revision?}}}  (source "local" for
                              install --from-path; scenario true when the install carries its own
                              deployment.json + devices.json)
launcher.json                 the launcher service (mode 0600): pid, endpoint, dashboard, bind_host,
                              port, urls, server_port (models' native port; 0 = a free one per load),
                              started_at, auth (whether an API key is required; never the key),
                              allow_web, detached, log, command (key redacted), state_directory,
                              version, control_token (the CLI's bearer token; kept out of
                              `status --json`), model ({pid, listen, run_id, repo} of the server it
                              runs, from spawn to exit), last_exit ({repo, run_id, exit_code, detail,
                              at} of the last model that exited on its own)
current.json                  the loaded model (mode 0600): run_id, run_directory, endpoint (the
                              launcher's), bind_host, urls, server_endpoint, pid (server),
                              launcher_pid, repo, deployment, device, started_at, context_length,
                              max_tokens, max_sessions, auth, allow_web, launcher_command, kind,
                              command, rows, profile_path, scenario
models/<owner>-<name>/        default install destinations
runs/<YYYYmmddTHHMMSS>-<pid>/ serve.json (composed launches), devices.json (scenario pool overrides),
                              serve.log, events.jsonl, requests.jsonl, meta.json
benchmarks/index.jsonl        one line per stored run
benchmarks/<owner>-<name>/<id>.json
cache/profiles/<owner>-<name>.json   hub documents for --offline
logs/fetch.log, logs/detached.log (`serve --detach`), logs/launcher.log (`launcher start --detach`, `start`)
```

Benchmark records keep the Python launcher's shape (`id, kind, suite, profile,
revision, model, started_at, elapsed_seconds, endpoint, harness, settings,
device, metrics, comparison, ok`) and add `harness.parallel` and `notes`.
Comparison rows carry `metric, label, measured, published, ratio, ceiling,
roofline_fraction` plus `value` (= measured) and `baseline`. Workloads come from
the seeded `loom-launch/text-v1` word stream, byte-identical to the Python
launcher's.

## Differences from the Python launcher

- The launcher service exists only here. The Python launcher's `serve` is still
  the model's supervisor and its proxy in one process: stopping or losing the
  model ends its endpoint and dashboard, and it has no `launcher`, `start` or
  `unload` verbs. Both read and write `current.json` the same way; the Python
  launcher ignores `launcher.json`.

- Local profile discovery (`--profiles`, `$LOOM_LAUNCH_PROFILES`),
  `install --from-path`, the dashboard and the `source`/`path`/`scenario` search
  fields exist only here. Both launchers install a profile's own
  `deployment.json`/`devices.json` the same way (same order, messages and
  install-record keys) and serve an installed scenario with
  `--devices=… --model=…`; the Python launcher also refuses the same options for
  a scenario and reports it in `show`. Both
  launchers report the served context the same way (`context_length`, `native_context_length`,
  `40960 tokens (model native 262144)`), cap `--context` at the native context,
  and end `show <directory>` with `serve with: loom-launch serve <directory>`.

- An installed profile keeps the installed `model.json` (what the server loads)
  and takes measurements, chat policy and README from the hub (or cache) under
  the install record's repository name. The Python launcher re-reads the card's
  `model.id` as a hub repository, which loses the preset for loom-fun packages.
- GPU detection reads `rocminfo` per agent, so the GPU's marketing name is
  reported rather than the CPU agent's; an `amdgpu` DRM node counts when
  `rocminfo` is missing.
- File reads use `{hub}/{repo}/resolve/{revision}/{path}` (Python: `/raw/`).
- `--offline` accepts cached documents of any age and `search --offline` lists
  the cache without the network.
- `install --revision` is passed to `loom-fetch` as `@<revision>`, and the
  install record keeps the revision the fetch result reports.
- Stopping uses SIGTERM then SIGKILL (Python sends SIGINT first); `current.json`
  is cleared only by the launch that owns it, and `stop` clears a record whose
  launcher is gone.
- A client that disconnects mid-stream ends the upstream request (recorded as
  `client disconnected`) instead of draining it.
- `/healthz` stays anonymous under `--api-key`.
- `chat`/`bench` stop a launch they started themselves (or unload the model they
  loaded under a running launcher); `--keep-server` keeps it so it outlives the
  command.

## Testing without a GPU

`loom-launch fake-serve` (hidden) is a stand-in with `loom-serve`'s argument
shape: it reads the deployment name from `--model`, listens on `--listen`,
prints the ready event and answers `/healthz`, `/v1/models`, `/v1/status` and
streamed or complete `/v1/chat/completions`. Environment hooks:
`LOOM_FAKE_SERVE_TOKENS`, `LOOM_FAKE_SERVE_TOKEN_MS`,
`LOOM_FAKE_SERVE_PREPARE_MS`, `LOOM_FAKE_SERVE_FAIL` (a failed registration
message), `LOOM_FAKE_SERVE_PANEL_MS` (i8-panel derivation lines). A launcher
passes its environment to the models it loads, so set these on the launcher. Like `loom-serve` it needs `--device` or `--devices` and refuses
`--devices` next to `--device`, `--memory_bytes` or `--pool_backing`. Point
`--loom-serve` at a two-line shim that runs `exec loom-launch fake-serve "$@"`;
`tests/supervise.rs` and `tests/launcher.rs` (the launcher lifecycle: no
model, 503, load, unload, `stop`, `launcher stop`, a crash, a cancelled load,
^C, SIGHUP, a SIGKILLed launcher, a record of before the split) do exactly
this, and `tests/scenario.rs` adds a shim that
records its argv to check the exact scenario command against the hybrid
package's documents (`tests/fixtures/`).

A request that offers `tools` can ask the stand-in for a tool call, streamed in
pieces the way OpenAI servers do: a user message `/js CODE` is answered with a
`run_javascript` call, `/fetch URL` with a `web_fetch` call, and `/loop` with a
`run_javascript` call every round (to reach the page's round limit); once a tool
message follows, the answer quotes what the tool returned.

## Web dashboard

`src/dashboard.rs` (with `src/dashboard/{events,jobs,progress,fetch,page}.rs`)
serves one embedded page and its JSON/SSE routes. The page is `src/dashboard/page/
{index.html,style.css,app.js}`, joined into one HTML document with inline style
and script at build time (`include_str!`): no CDN, no external fonts, no build
step. It is served with a `Content-Security-Policy` that allows nothing but this
listener, plus `worker-src blob:` for the chat's JavaScript sandbox.

The page is an application shell: a sticky top bar (the model's name and
state, the endpoint with a copy button, device/context/sessions/uptime chips,
upstream health, live-update state, the theme switch), tabs (**Overview**:
the current launch, what it allows, traffic; **Models**: search, install,
register, launch; **Chat**; **Benchmarks**; **Logs**; the address keeps the
tab, arrow keys move between them) and an **Activity** rail. On a launcher's
page the Overview starts with a **Launcher** card (state, endpoint, pid,
uptime, listener, the models' native port, the requests it answered itself:
page, api, loom, health and `/v1` without a model, and the last model that
exited on its own). The action under
way is also named in the tab bar, so it is in sight from every tab; below
1180 px the rail moves above the panel. Colours, type sizes and spacing are
tokens for the dark and light themes; motion stops under
`prefers-reduced-motion` (an indeterminate bar becomes a still, striped one);
the layout holds down to 700 px; every control is a native button or input
reachable by keyboard.

**Activity.** Every action is a job the server keeps (the last 30, active ones
always), shown as one row until dismissed, so a reload finds it where it was:
install from the hub, install from a path (registration), launch, unload (stop
on a launch's own page), benchmark, the chat's web fetches, and a model that
exited on its own. A row shows the job's state
(`queued` → `running` → `done` | `failed`), its elapsed time, the current step,
a progress bar, the last output lines, the steps with their times, and on
request the full output; a failed row carries the error, a hint on what to do,
and a Retry that repeats the request that started it.

| Job | Steps | Progress (where the numbers come from) |
| --- | --- | --- |
| install (hub) | read the card · fetch and verify files · install files · fetch the chat policy and measurements · record | `loom-fetch`'s console lines (`loom-fetch  <Stage>  …`): files from its `n/m files`, bytes from its per-file `[bar] p%  done / size`, measured against the total the card's lock (`model.json.lock`, every non-derived file's `size`) declares, read first from the local hub cache or the hub; without a lock, files. |
| install (path) | read the package · add the chat policy and measurements · record | three steps. |
| launch | start loom-serve · open the listener (`{"event":"ready"}`) · load weights and register (registration phases from `/v1/models`, recorded as `registration` events) · derive i8 panels (`i8-panel deriving`/`derived`/`reused`, with the loader's `validation_seconds`) · serve | indeterminate: the stages have no measurable extent, so the bar shows activity and the steps their times until it serves. |
| unload (stop) | ask the server to drain (SIGTERM; SIGKILL after 30 s) · stopped (exit status from the run record); for a model still preparing: cancel the preparation | indeterminate. |
| exit | the model server exited (exit status or signal, with its last output); always failed | none. |
| benchmark | warm-up, then one step per metric with its measured value | requests: warm-up plus each metric's repeats (`decode 1/3`), from the runner's progress events. |
| web fetch | fetch the page | none (one request). |

Install, launch and benchmark share one slot: a second one answers 409, and
the page shows that refusal next to the control that asked. Buttons are
disabled and marked busy while their request or job is in flight.

Where it runs:

- **Launcher** (`mode: "launcher"`) — every launcher (`launcher start`,
  `serve`, `start`, `dashboard`) answers `/` and `/api/*` next to the OpenAI
  routes. Stored benchmarks, installs and the latest run's logs are browsable
  without a model. With no model loaded the Overview says so and offers a load
  form (an installed profile with the card's values; the Models tab has the
  capacity fields), the header keeps the launcher's endpoint, the Chat panel
  explains that nothing answers yet and links to the form, and the benchmark
  suites are disabled with the reason. Loading, unloading and a crash update the
  page through its event stream, without a reload. A launch recorded in
  `current.json` by another process is shown (its counters are read from its
  `/loom/status`, chat goes through its proxy).
- **Hosted** — a launch that `chat` or `bench` start for themselves (no launcher
  running) has a proxy of its own whose page shows that launch. It can stop
  the launch but never starts another (`POST /api/serve` answers 409).

| Route | Answer |
| --- | --- |
| `GET /` | The page. With scripts disabled it shows the panel skeleton and a server-rendered summary (launch, installs, stored runs with links to their JSON). |
| `GET /api/status` | `mode` (`launcher` or `hosted`), `launcher{state,pid,endpoint,dashboard,bind_host,port,urls,started_at,uptime_seconds,auth,allow_web,server_port,counters{page,api,loom,health,no_model},model,repo,last_exit}` (null when hosted), `launch{kind,state,repo,endpoint,server_endpoint,device,deployment,context_length,max_tokens,max_sessions,rows,launch_kind,command,profile_path,started_at,uptime_seconds,up,model,...}` (`kind` is hosted/owned/external/none; `launch_kind` is `scenario` or `profile`), `scenario` (the scenario record, `kind: "scenario"`, or null), `current` (`current.json`), `proxy` (counters), `profile` (quant, context, memory, chat policy, `thinking{available,mode,default,locked,reason}`, `served{preset,draft_depth,rows,context_length,max_tokens,prefill_capacity}`, `drafter{kind,depth,served_depth,served_tps,recommended_tps,note}`), `jobs` (every kept job), `job` (the latest install, launch or benchmark), `hardware`, `capabilities{serve,stop,unload,stopping,chat,chat_reason,bench,bench_reason,install,web_access,web_access_reason,web_access_command,launch_allow_web}`, `suites`, `stored_runs`, `installs`. |
| `GET /api/events` | Server-sent events (`event: snapshot`, one per second): `launch`, `launcher`, `metrics`, `in_flight`, `last_request`, `jobs` (each with its last 6 output lines), `job`, `samples` (counter samples not yet sent; the first event carries up to ten minutes), `log{run,reset,serve[],events[]}` (lines appended since the previous event). |
| `GET /api/jobs`, `/api/jobs/<id>` | The kept jobs, newest first: `{id, kind, target, state, step, steps[{name,label,state,started_at,elapsed_seconds,detail}], progress{done,total,unit,label,fraction,files?,bytes?}, started_at, finished_at, elapsed_seconds, lines[], line_count, result, error, hint, retry{route,body}}` (`total: null` is indeterminate); one job with up to 400 output lines. |
| `GET /api/models?q=&offline=&limit=` | Hub search plus local profiles, with details (quant, served `context_length` and `native_context_length`, memory, published decode, fit score and notes, installed/running, `source`, `path`, `scenario`, `scenario_devices`); fresh cached documents avoid refetching; falls back to the cache with a `warning` when the hub does not answer (local profiles are listed either way). |
| `GET /api/installs` | Installed profiles with quant, served and native context, memory, card capacities, `scenario`, `source`, `running`. |
| `GET /api/benchmarks`, `/api/benchmarks/<id>` | Stored runs (newest first, with headline rows) and one record. |
| `GET /api/logs?lines=&run=` | Tail of `serve.log` and `events.jsonl` (current launch, else latest run). |
| `GET /api/requests?limit=`, `/api/metrics` | Recent requests (live: the model's, or the launcher's since it started; else the latest run's `requests.jsonl`) and Prometheus text. |
| `POST /api/install {repo, preset?}` | 202; runs `loom-fetch` in the background as an install job. |
| `POST /api/install {path, name?}` | 202; registers a local profile the dashboard lists, in place (as `install --from-path`); any other path is 400. |
| `POST /api/serve {repo \| path, context?, max_tokens?, parallel?, draft_depth?, checkpoint_capacity?, session_cache?, prefix_cache?, prefill_capacity?, device?, port?, workers?, pool_backing?, memory_bytes?, wait?, allow_web?}` | 202 with the load job after validating against the card; `path` must be a local profile the dashboard lists (any package directory with the launcher's control token); a scenario refuses the capacity fields and `device` (400); `allow_web` lets the chat fetch public pages while this model serves; 409 while a model is loaded or in hosted mode. |
| `POST /api/unload {wait?}`, `POST /api/stop` | 202 with the unload job (`stop` on a launch's own page); unloads the model (or cancels one still preparing, even one still being planned); the launcher keeps serving; 409 `no model is loaded` when none, or while an unload is in progress. |
| `POST /api/jobs/dismiss {id}` | Hides a finished job's row (409 for an active one, 404 for an unknown id). |
| `POST /api/bench {suite, repeats?, repo?}` | 202; runs the suite through the launch's endpoint and stores the record where `bench --list` reads it. |
| `POST /api/chat` | The body goes to `/v1/chat/completions` (the served model name and `stream_options.include_usage` are filled in when absent; `tools` and tool messages pass through); the answer is relayed unchanged and counted by the proxy like any request. |
| `POST /api/fetch {url}` | With `--allow-web`: one public `http`/`https` URL as `{url, status, content_type, bytes, truncated, text}` (plus `title`, and `location` for a redirect). Without it: 403 naming the flag. See [Chat tools](#chat-tools). |

One install, benchmark or launch runs at a time; a second answers 409. Routes that start work require a JSON body, refuse a foreign
`Origin`, and every `/api/*` route requires a `Host` that is an IP address or
`localhost` (DNS rebinding). With `--api-key` the page asks for the key (or
takes it once from `?access_token=` in its address); the event stream accepts
it as `?access_token=`. `/api/status` also reports `capabilities.web_access` and
`listener{bind_host,port,urls}`.

### Chat tools

The chat panel can let the model call two tools, each behind a switch next to
*Thinking*: **JavaScript** (on by default) and **Web access** (enabled only when
the launcher runs with `--allow-web`, or a standalone dashboard started the
launch with "Allow web access" ticked, and then on). A switch that is off says
why next to it: web access names the remedy and a copyable command (the
launch's own recorded command line with `--allow-web` appended, an API key
left as `"$LOOM_LAUNCH_API_KEY"`); thinking names the package's policy when it
fixes thinking (`thinking: always` / `never`) or, when the package ships no
`llm_config.json`, that there is no policy and how to get one. With both off a turn is
exactly the plain streamed chat. With either on, each request carries the
OpenAI function schemas below; streamed `tool_calls` fragments are merged by
`index`; when a turn ends with calls, the page runs each one, appends
`{"role": "tool", "tool_call_id": ..., "content": "<string>"}` and sends again,
for at most 6 rounds per message (calls past the limit are answered "not run"
so the history stays well formed, and a note says so). Each call is a card in
the transcript: tool name, the code or URL, running/done/failed (or cancelled,
timed out), duration, and the output, collapsed beyond six lines. Stop cancels
the stream and the tool loop, terminating a running worker.

```json
{"type": "function", "function": {"name": "run_javascript",
  "description": "Run a snippet of JavaScript in a sandboxed worker with no DOM, no network and a 2 s limit; return the value of the last expression (use `return` explicitly or leave an expression; console.log output is captured)",
  "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The JavaScript to run, for example: return [3, 1, 2].sort((a, b) => a - b)"}}, "required": ["code"]}}}
{"type": "function", "function": {"name": "web_fetch",
  "description": "Fetch one public http/https URL and return its text content (HTML reduced to text), for facts you cannot answer otherwise",
  "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The absolute http or https URL of a public page, scheme included"}}, "required": ["url"]}}}
```

**The JavaScript sandbox** runs in the page, never on the server: each run
builds a Web Worker from a fresh `Blob` and terminates it afterwards, so no
state survives between runs. The worker inherits the page's
`Content-Security-Policy` (`worker-src blob:` admits it; there is no
`unsafe-eval`, so code in it cannot `eval`, and it may connect only to this
listener), and before the snippet runs it removes `fetch`, `XMLHttpRequest`,
`WebSocket`, `WebSocketStream`, `EventSource`, `WebTransport`, `importScripts`,
`BroadcastChannel`, `RTCPeerConnection`, `Worker`, `SharedWorker`, `indexedDB`,
`caches`, `navigator.sendBeacon`, `navigator.storage`, `postMessage` and
`close` from the worker and its prototype chain, leaving stubs that throw "X is
not available in the sandbox". Because `eval` is unavailable, the snippet is
compiled into the worker's own script: first as one expression
(`return (snippet)`), then with its last top-level statement returned, then as
written; the first form that parses runs. `console.log/info/debug/warn/error`
are captured. After 2000 ms the worker is terminated and the run reports a
timeout. The result goes back to the model as JSON text
(`{"ok", "type", "result" | "error", "console"}`; cycles, functions, bigints,
non-finite numbers, Maps, Sets and Dates become readable stand-ins), capped at
8 KiB with a truncation marker.

**Web access** is a deliberate fetch proxy for the model: `POST /api/fetch`
(`src/dashboard/fetch.rs`) fetches a URL from the launcher's machine on the
model's behalf. It is off unless `--allow-web` is given, and its address checks
are what keep it from reaching this machine's own services (the dashboard,
`loom-serve`, anything on loopback or the LAN): only `http`/`https` URLs without
credentials; `localhost` names and every address the host resolves to must be
public (loopback, private, shared/CGNAT, link-local, unique-local, multicast,
unspecified, documentation and reserved ranges are refused, for IPv4, IPv6 and
IPv4 embedded in IPv6); the HTTP client's resolver is pinned to exactly the
checked addresses, so a second DNS answer cannot substitute a private one; and
redirects are not followed (the answer carries `location`, and asking for it is
checked again). It sends a browser-like `User-Agent` and
`Accept: text/html,text/plain,application/json`, gives up after 10 s, reads at
most 1 MiB, reduces HTML to text (scripts, styles and markup removed, entities
decoded, whitespace collapsed) and returns at most 8 KiB of it. The route sits
behind the same guard as the other `/api/*` routes (JSON body, matching
`Origin`, IP or `localhost` `Host`) and, on a keyed launch, the bearer token.
