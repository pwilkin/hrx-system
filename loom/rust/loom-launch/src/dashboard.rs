//! The web dashboard (spec §8): one embedded page plus the JSON and SSE routes behind it.
//!
//! The page and its routes are served two ways. A launcher service
//! (`loom-launch launcher start`, `serve`, `dashboard`) answers `/` and `/api/*`
//! next to the OpenAI routes it carries ("launcher"): the page stays up whether
//! or not a model is loaded, loads one behind the launcher's own endpoint and
//! unloads it again, and shows a launch recorded in `current.json` by another
//! process when one is running. A launch started for one command (`chat` and
//! `bench` with nothing running) has a proxy of its own whose page watches only
//! that launch ("hosted").
//!
//! Routes that start work (install, launch, stop, benchmark, the chat's web
//! fetches) run it as a job the page shows as one row until it is dismissed:
//! its state, steps, progress and output travel on `/api/events` with
//! everything else the page shows live, so the page has no polling loop and a
//! reload finds every action where it was.

mod events;
mod fetch;
mod jobs;
mod page;
mod progress;
#[cfg(test)]
mod tests;

pub use jobs::{JobState, Jobs};

use crate::bench::{suite_metrics, BenchEvent, Runner, SUITES};
use crate::cli::{self, Ctx, ProfileArgs, SearchArgs};
use crate::client::Client;
use crate::deployment::ContextRequest;
use crate::errors::{Error, Result};
use crate::hardware::{detect, Hardware};
use crate::hub::{Hub, HubModel, PROFILE_TAG};
use crate::install::{
    default_models_directory, install_and_record_observed, published_lock, InstallOptions,
    InstallStage,
};
use crate::metrics::Metrics;
use crate::profile::{Profile, ProfileDocuments};
use crate::proxy::{
    header, json_response, query_value, raw_response, request_headers, Forwarder, HealthProvider,
    LauncherCounters, Route, StatusProvider, Upstream,
};
use crate::server::{read_current, stop_current, LaunchOptions, LaunchPlan, Supervisor};
use crate::signals;
use crate::store::{BenchmarkStore, InstallRegistry, ProfileCache, RunDirectory, RunStore};
use crate::values::{
    arr, child, get, grouped, human_bytes, int_value, iso_utc_millis, now_seconds, number_value,
    obj, str_value, strict_int, string, truthy,
};
use jobs::{Progress, StepState};
use progress::{declared_bytes, FetchTracker, LaunchTracker};
use serde_json::{json, Map, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

type Reply = tiny_http::Response<std::io::Cursor<Vec<u8>>>;

/// Largest request body a dashboard route reads.
const BODY_LIMIT: u64 = 8 * 1024 * 1024;
/// How long a benchmark or chat request may take.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(900);

/// A dashboard riding on a launch's proxy.
pub struct Hosted {
    pub state_root: PathBuf,
    pub hub_endpoint: Option<String>,
    pub(crate) forwarder: Arc<Forwarder>,
    pub status_provider: Option<StatusProvider>,
    /// The proxy's own address (what clients and benchmarks talk to).
    pub endpoint: String,
    pub auth_token: Option<String>,
    /// Serve `POST /api/fetch` for the chat's `web_fetch` tool (`--allow-web`).
    pub allow_web: bool,
    /// Directories local profiles are listed from.
    pub profile_roots: Vec<PathBuf>,
}

/// A launcher service: the page, the client-facing endpoint and the models
/// loaded under it (`loom-launch launcher start`, `serve`, `dashboard`).
#[derive(Clone, Debug)]
pub struct Service {
    pub state_root: PathBuf,
    pub hub_endpoint: Option<String>,
    /// The address the client-facing listener binds.
    pub host: String,
    /// The client-facing port (0: ephemeral).
    pub port: u16,
    /// The native listener port a loaded model gets (0: a free one per load).
    pub server_port: u16,
    pub loom_serve: Option<String>,
    pub loom_fetch: Option<String>,
    /// Serve `POST /api/fetch` for the chat's `web_fetch` tool (`--allow-web`).
    pub allow_web: bool,
    /// Directories local profiles are listed from (`--profiles`, `$LOOM_LAUNCH_PROFILES`).
    pub profile_roots: Vec<PathBuf>,
    /// The command line that started the launcher (an API key redacted), for
    /// the restart command the page offers.
    pub command: Vec<String>,
    /// Seconds a load may prepare before it is given up.
    pub startup_timeout: f64,
    /// Echo loaded models' output (`  | line`) on stdout.
    pub echo: bool,
}

impl Default for Service {
    fn default() -> Self {
        Service {
            state_root: PathBuf::new(),
            hub_endpoint: None,
            host: "127.0.0.1".into(),
            port: 0,
            server_port: 0,
            loom_serve: None,
            loom_fetch: None,
            allow_web: false,
            profile_roots: Vec::new(),
            command: Vec::new(),
            startup_timeout: 1800.0,
            echo: false,
        }
    }
}

/// The listener a launcher carries its models behind: the forwarder that
/// follows its route, its address, and the requests it answered itself.
pub struct Front {
    pub(crate) forwarder: Arc<Forwarder>,
    pub endpoint: String,
    /// Whether the listener requires an API key.
    pub auth: bool,
    pub counters: Arc<LauncherCounters>,
}

/// A model this launcher loaded and supervises.
struct Owned {
    supervisor: Supervisor,
    /// Loaded with "Allow web access": the chat may fetch through this page.
    allow_web: bool,
    /// Set when the launcher unloads it, so its exit is not reported as a crash.
    unloading: AtomicBool,
}

/// Called on a load's thread once its run and deployment are written.
pub type OnPlanned = Box<dyn FnOnce(&LaunchPlan) + Send>;

/// One model load, from the page, `loom-launch start` or `serve`.
pub struct Load {
    pub(crate) resolved: cli::Resolved,
    pub(crate) prepared: cli::Prepared,
    pub device: Option<String>,
    /// The native listener port (default: the launcher's, else a free one).
    pub port: Option<u16>,
    pub allow_web: bool,
    pub workers: Option<i64>,
    pub pool_backing: Option<String>,
    pub memory_bytes: Option<i64>,
    /// Seconds the load may prepare (default: the launcher's).
    pub wait: Option<f64>,
    /// What started it (`dashboard`, `start`, `serve`), recorded in the run.
    pub source: &'static str,
    /// The request a retry repeats.
    pub retry: Value,
    /// Called on the load's thread once its run and deployment are written.
    pub on_planned: Option<OnPlanned>,
}

/// Whether the chat's `web_fetch` tool works on this page; when it does not,
/// why, and the command that would turn it on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WebAccess {
    pub(crate) allowed: bool,
    pub(crate) reason: Option<String>,
    pub(crate) command: Option<String>,
}

/// Words joined into a line a POSIX shell reads back as the same words.
fn shell_join(words: &[String]) -> String {
    words
        .iter()
        .map(|word| {
            let plain = !word.is_empty()
                && word
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "-_./:=@,+%".contains(c));
            // The redacted key placeholder is meant to expand.
            if plain || word.contains("$LOOM_LAUNCH_API_KEY") {
                word.clone()
            } else {
                format!("'{}'", word.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Counters of a launch supervised by another process, read from its proxy.
#[derive(Clone, Debug, Default)]
struct Remote {
    endpoint: String,
    metrics: Value,
    last_request: Value,
}

/// Which launch the dashboard is showing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The launch whose proxy serves this dashboard.
    Hosted,
    /// A model this launcher loaded.
    Owned,
    /// A launch recorded in `current.json` by another process.
    External,
    None,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Hosted => "hosted",
            Kind::Owned => "owned",
            Kind::External => "external",
            Kind::None => "none",
        }
    }
}

/// The live launch, resolved once per request or event.
struct Live {
    kind: Kind,
    record: Option<Map<String, Value>>,
    /// Launch facts from the hosting supervisor (hosted only).
    provided: Map<String, Value>,
    repo: Option<String>,
    /// The client-facing endpoint (the proxy when there is one).
    endpoint: Option<String>,
    /// The native listener.
    server_endpoint: Option<String>,
    metrics: Option<Arc<Metrics>>,
    forwarder: Option<Arc<Forwarder>>,
    /// Bearer token the client-facing endpoint expects.
    token: Option<String>,
}

impl Live {
    fn key(&self) -> String {
        self.endpoint.clone().unwrap_or_default()
    }

    fn fact(&self, key: &str) -> Value {
        let recorded = self
            .record
            .as_ref()
            .map(|r| get(r, key))
            .unwrap_or(&Value::Null);
        if !recorded.is_null() {
            return recorded.clone();
        }
        get(child(&self.provided, "launch"), key).clone()
    }
}

/// What the dashboard routes can see and do.
pub struct Context {
    pub state_root: PathBuf,
    hub_endpoint: Option<String>,
    hosted: Option<Hosted>,
    /// The launcher service this page belongs to (absent when hosted).
    service: Option<Service>,
    /// The launcher's listener (absent when hosted).
    front: Option<Front>,
    /// Whether `POST /api/fetch` fetches (see `dashboard/fetch.rs`).
    allow_web: bool,
    /// Directories local profiles are listed from.
    profile_roots: Vec<PathBuf>,
    /// The address this dashboard's listener is bound to and its LAN URLs.
    listener: Mutex<Value>,
    /// The client-facing address once the listener is bound.
    endpoint: Mutex<Option<String>>,
    /// When the launcher started.
    started_at: f64,
    /// The last model that exited without being unloaded (`last_exit`).
    last_exit: Mutex<Value>,
    /// Test builds: host names `/api/fetch` answers from a local stub.
    #[cfg(test)]
    fetch_stubs: Mutex<Vec<(String, std::net::SocketAddr)>>,
    jobs: Arc<Jobs>,
    owned: Mutex<Option<Arc<Owned>>>,
    /// Threads waiting on owned launches; shutdown joins them so the run
    /// record and `current.json` are finished before the process exits.
    watchers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// A launch still preparing, so it can be cancelled.
    preparing: Mutex<Option<Supervisor>>,
    /// An unload arrived while a load was still being planned (before its
    /// supervisor existed): the load cancels itself as soon as it can.
    cancel_load: AtomicBool,
    stopping: Mutex<Option<String>>,
    remote: Mutex<Option<Remote>>,
    remote_forwarder: Mutex<Option<(String, Arc<Forwarder>)>>,
    history: Mutex<events::History>,
    probe: Mutex<Option<(String, bool)>>,
    served_model: Mutex<Option<(String, String)>>,
    profile: Mutex<Option<(String, Option<Profile>)>>,
    hardware: OnceLock<Hardware>,
    closed: AtomicBool,
}

impl Context {
    /// A hosted dashboard (`hosted`) or a launcher's (`service` and `front`).
    fn new(hosted: Option<Hosted>, service: Option<Service>, front: Option<Front>) -> Arc<Context> {
        let (state_root, hub_endpoint, allow_web, profile_roots) = match (&hosted, &service) {
            (Some(hosted), _) => (
                hosted.state_root.clone(),
                hosted.hub_endpoint.clone(),
                hosted.allow_web,
                hosted.profile_roots.clone(),
            ),
            (None, Some(options)) => (
                options.state_root.clone(),
                options.hub_endpoint.clone(),
                options.allow_web,
                options.profile_roots.clone(),
            ),
            (None, None) => (PathBuf::new(), None, false, Vec::new()),
        };
        let endpoint = front
            .as_ref()
            .map(|f| f.endpoint.clone())
            .filter(|e| !e.is_empty());
        let context = Arc::new(Context {
            state_root,
            hub_endpoint,
            hosted,
            service,
            front,
            allow_web,
            profile_roots,
            listener: Mutex::new(Value::Null),
            endpoint: Mutex::new(endpoint),
            started_at: now_seconds(),
            last_exit: Mutex::new(Value::Null),
            #[cfg(test)]
            fetch_stubs: Mutex::new(Vec::new()),
            jobs: Arc::new(Jobs::default()),
            owned: Mutex::new(None),
            watchers: Mutex::new(Vec::new()),
            preparing: Mutex::new(None),
            cancel_load: AtomicBool::new(false),
            stopping: Mutex::new(None),
            remote: Mutex::new(None),
            remote_forwarder: Mutex::new(None),
            history: Mutex::new(events::History::default()),
            probe: Mutex::new(None),
            served_model: Mutex::new(None),
            profile: Mutex::new(None),
            hardware: OnceLock::new(),
            closed: AtomicBool::new(false),
        });
        spawn_sampler(Arc::downgrade(&context));
        context
    }

    /// The dashboard of a launch's proxy.
    pub fn hosted(hosted: Hosted) -> Arc<Context> {
        Context::new(Some(hosted), None, None)
    }

    /// The dashboard of a launcher service, served on its listener `front`.
    pub fn launcher(service: Service, front: Front) -> Arc<Context> {
        Context::new(None, Some(service), Some(front))
    }

    /// A launcher's dashboard on a listener of the caller's (tests): its
    /// route starts empty and its counters start at zero.
    pub fn standalone(service: &Service) -> Arc<Context> {
        let route = Arc::new(Route::default());
        let front = Front {
            forwarder: Arc::new(Forwarder::routed(
                route,
                Arc::new(Metrics::default()),
                REQUEST_TIMEOUT,
            )),
            endpoint: String::new(),
            auth: false,
            counters: Arc::new(LauncherCounters::default()),
        };
        Context::new(None, Some(service.clone()), Some(front))
    }

    /// Records the address the listener serving these routes is bound to.
    pub fn set_listener(&self, bind_host: &str, port: u16) {
        *self.listener.lock().unwrap() = json!({
            "bind_host": bind_host,
            "port": port,
            "urls": crate::network::reachable_urls(bind_host, port),
        });
        let mut endpoint = self.endpoint.lock().unwrap();
        if endpoint.is_none() {
            *endpoint = Some(crate::network::http_url(
                &crate::network::local_connect_host(bind_host),
                port,
            ));
        }
    }

    /// The launcher's client-facing address, once bound.
    pub fn endpoint(&self) -> Option<String> {
        self.endpoint.lock().unwrap().clone()
    }

    /// Whether this page belongs to a launcher service.
    pub fn is_launcher(&self) -> bool {
        self.service.is_some()
    }

    /// The model this launcher loaded, while it runs.
    pub fn owned_supervisor(&self) -> Option<Supervisor> {
        self.owned
            .lock()
            .unwrap()
            .as_ref()
            .filter(|o| !o.supervisor.exited())
            .map(|o| o.supervisor.clone())
    }

    /// Whether the live model is on its way out: being unloaded, or exited
    /// and not yet released by the thread that ends its run.
    fn going_away(&self, live: &Live) -> bool {
        if self.stopping.lock().unwrap().is_some() {
            return true;
        }
        let owned = self.owned.lock().unwrap().clone();
        live.kind == Kind::Owned && owned.is_some_and(|o| o.supervisor.exited())
    }

    /// Whether a load is still preparing.
    pub fn preparing(&self) -> bool {
        self.preparing.lock().unwrap().is_some()
    }

    /// The launcher's counters (all requests it carried or refused).
    pub fn metrics(&self) -> Option<Arc<Metrics>> {
        self.front.as_ref().map(|f| f.forwarder.metrics.clone())
    }

    /// Answers `/api/fetch` for `host` from a local stub, as if it were public
    /// (test builds only; production always applies the address check).
    #[cfg(test)]
    pub(crate) fn stub_public_host(&self, host: &str, address: std::net::SocketAddr) {
        self.fetch_stubs
            .lock()
            .unwrap()
            .push((host.to_string(), address));
    }

    /// Ends event streams and the sampler (the listener is going away).
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    pub fn closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    pub fn jobs(&self) -> &Arc<Jobs> {
        &self.jobs
    }

    fn mode(&self) -> &'static str {
        if self.hosted.is_some() {
            "hosted"
        } else {
            "launcher"
        }
    }

    fn cli(&self) -> Ctx {
        Ctx {
            state: self.state_root.clone(),
            hub_endpoint: self.hub_endpoint.clone(),
            json: true,
            profile_roots: self.profile_roots.clone(),
        }
    }

    fn hardware(&self) -> &Hardware {
        self.hardware.get_or_init(detect)
    }

    /// Fixes the machine description (tests).
    #[cfg(test)]
    pub(crate) fn set_hardware(&self, hardware: Hardware) {
        let _ = self.hardware.set(hardware);
    }

    // -- the live launch ----------------------------------------------------

    fn live(&self) -> Live {
        let record = read_current(&self.state_root);
        let record_repo = record
            .as_ref()
            .and_then(|r| str_value(get(r, "repo")).map(str::to_string));
        if let Some(hosted) = &self.hosted {
            let provided = hosted
                .status_provider
                .as_ref()
                .map(|provider| provider())
                .unwrap_or_default();
            let repo = str_value(get(child(&provided, "launch"), "repo"))
                .map(str::to_string)
                .or(record_repo);
            return Live {
                kind: Kind::Hosted,
                record,
                provided,
                repo,
                endpoint: Some(hosted.endpoint.clone()),
                server_endpoint: hosted.forwarder.upstream().map(|u| u.base_url()),
                metrics: Some(hosted.forwarder.metrics.clone()),
                forwarder: Some(hosted.forwarder.clone()),
                token: hosted.auth_token.clone(),
            };
        }
        // A model stays loaded until whoever ends its run releases it, so
        // "no model" is never reported before its record is cleared, its
        // route dropped and an exit on its own reported.
        let owned = self.owned.lock().unwrap().clone();
        if let (Some(owned), Some(front)) = (owned, &self.front) {
            let supervisor = &owned.supervisor;
            return Live {
                kind: Kind::Owned,
                record,
                provided: (supervisor.status_provider())(),
                repo: Some(supervisor.plan().profile.repo.clone()),
                endpoint: Some(supervisor.endpoint()),
                server_endpoint: Some(supervisor.server_endpoint()),
                metrics: Some(front.forwarder.metrics.clone()),
                forwarder: Some(front.forwarder.clone()),
                token: None,
            };
        }
        if let Some(current) = &record {
            let alive = strict_int(get(current, "pid")).is_some_and(|pid| !signals::exited(pid));
            if let (true, Some(endpoint)) = (alive, str_value(get(current, "endpoint"))) {
                let token = truthy(get(current, "auth"))
                    .then(|| std::env::var("LOOM_LAUNCH_API_KEY").ok())
                    .flatten()
                    .filter(|t| !t.is_empty());
                return Live {
                    kind: Kind::External,
                    record: record.clone(),
                    provided: Map::new(),
                    repo: record_repo,
                    endpoint: Some(endpoint.to_string()),
                    server_endpoint: string(get(current, "server_endpoint")),
                    metrics: None,
                    forwarder: self.remote_forwarder(endpoint),
                    token,
                };
            }
        }
        Live {
            kind: Kind::None,
            record,
            provided: Map::new(),
            repo: None,
            endpoint: None,
            server_endpoint: None,
            metrics: None,
            forwarder: None,
            token: None,
        }
    }

    /// Forwards to another process's launch through its own proxy, which
    /// counts the request; the local counters stay untouched.
    fn remote_forwarder(&self, endpoint: &str) -> Option<Arc<Forwarder>> {
        let mut slot = self.remote_forwarder.lock().unwrap();
        if let Some((cached, forwarder)) = slot.as_ref() {
            if cached == endpoint {
                return Some(forwarder.clone());
            }
        }
        let forwarder = Arc::new(Forwarder::new(
            upstream_of(endpoint)?,
            Arc::new(Metrics::default()),
            None,
            REQUEST_TIMEOUT,
        ));
        *slot = Some((endpoint.to_string(), forwarder.clone()));
        Some(forwarder)
    }

    /// The counters the page shows: the live launch's, else (a launcher with
    /// no model) the launcher's own, which outlive every model.
    fn live_metrics(&self, live: &Live) -> Option<Arc<Metrics>> {
        live.metrics
            .clone()
            .or_else(|| (live.kind == Kind::None).then(|| self.metrics()).flatten())
    }

    /// The key a counter sample belongs to: a launcher's history runs on
    /// across models, so it is keyed by the launcher's own endpoint.
    fn sample_key(&self, live: &Live) -> String {
        match (live.kind, self.endpoint()) {
            (Kind::None, Some(endpoint)) if self.front.is_some() => endpoint,
            _ => live.key(),
        }
    }

    fn metrics_snapshot(&self, live: &Live) -> Value {
        match (self.live_metrics(live), live.kind) {
            (Some(metrics), _) => metrics.snapshot(),
            (None, Kind::External) => self
                .remote
                .lock()
                .unwrap()
                .as_ref()
                .filter(|r| Some(&r.endpoint) == live.endpoint.as_ref())
                .map(|r| r.metrics.clone())
                .unwrap_or(Value::Null),
            _ => Value::Null,
        }
    }

    fn last_request(&self, live: &Live) -> Value {
        match (self.live_metrics(live), live.kind) {
            (Some(metrics), _) => metrics.recent(1).into_iter().next().unwrap_or(Value::Null),
            (None, Kind::External) => self
                .remote
                .lock()
                .unwrap()
                .as_ref()
                .filter(|r| Some(&r.endpoint) == live.endpoint.as_ref())
                .map(|r| r.last_request.clone())
                .unwrap_or(Value::Null),
            _ => Value::Null,
        }
    }

    fn launch_summary(&self, live: &Live) -> Value {
        let preparing = self
            .jobs
            .running()
            .filter(|(kind, _)| kind == "serve")
            .map(|(_, target)| target);
        let state = match live.kind {
            Kind::None if preparing.is_some() => "starting",
            Kind::None => "none",
            _ if self.going_away(live) => "stopping",
            _ => "serving",
        };
        let repo = live.repo.clone().or(preparing);
        let started_at = live.fact("started_at");
        let uptime = str_value(&started_at)
            .and_then(parse_iso_utc)
            .map(|t| crate::values::round_to((now_seconds() - t).max(0.0), 0));
        let up = self
            .probe
            .lock()
            .unwrap()
            .as_ref()
            .filter(|(key, _)| *key == live.key() && live.kind != Kind::None)
            .map(|(_, up)| *up);
        let model = self
            .served_model
            .lock()
            .unwrap()
            .as_ref()
            .filter(|(key, _)| *key == live.key())
            .map(|(_, name)| name.clone());
        json!({
            "kind": live.kind.name(),
            "state": state,
            "repo": repo,
            "endpoint": live.endpoint,
            "server_endpoint": live.server_endpoint,
            "device": live.fact("device"),
            "deployment": live.fact("deployment"),
            "run_id": live.fact("run_id"),
            "pid": live.fact("pid"),
            "context_length": live.fact("context_length"),
            "max_tokens": live.fact("max_tokens"),
            "max_sessions": live.fact("max_sessions"),
            "started_at": started_at,
            "uptime_seconds": uptime,
            "up": up,
            "auth": live.token.is_some() || truthy(&live.fact("auth")),
            "bind_host": live.fact("bind_host"),
            "urls": live.fact("urls"),
            "model": model,
            "launch_kind": live.fact("kind"),
            "rows": live.fact("rows"),
            "command": live.fact("command"),
            "profile_path": live.fact("profile_path"),
        })
    }

    /// The name the live launch answers to (cached per launch).
    fn served_model(&self, live: &Live) -> Option<String> {
        let key = live.key();
        if let Some((cached, name)) = self.served_model.lock().unwrap().as_ref() {
            if *cached == key {
                return Some(name.clone());
            }
        }
        let base = match live.kind {
            Kind::External => live.endpoint.clone()?,
            _ => live.server_endpoint.clone().or(live.endpoint.clone())?,
        };
        let token = (live.kind == Kind::External)
            .then(|| live.token.clone())
            .flatten();
        let client = Client::new(&base, token.as_deref(), Duration::from_secs(3)).ok()?;
        let names = client.model_names(Some(Duration::from_secs(3)));
        let deployment = str_value(&live.fact("deployment")).map(str::to_string);
        let name = deployment
            .filter(|d| names.contains(d))
            .or_else(|| names.into_iter().next())?;
        *self.served_model.lock().unwrap() = Some((key, name.clone()));
        Some(name)
    }

    /// Whether the chat's `web_fetch` tool may fetch through this dashboard.
    /// A hosted page follows its launch's `--allow-web`; a standalone page its
    /// own, or the "Allow web access" choice of a launch it started.
    fn web_access(&self, live: &Live) -> WebAccess {
        let owned_allows = live.kind == Kind::Owned
            && self
                .owned
                .lock()
                .unwrap()
                .as_ref()
                .is_some_and(|owned| owned.allow_web);
        if self.allow_web || owned_allows {
            return WebAccess {
                allowed: true,
                reason: None,
                command: None,
            };
        }
        let (reason, command) = if self.hosted.is_some() {
            (
                "web access is off for this launch: restart it with --allow-web",
                self.restart_command(live),
            )
        } else {
            let reason = match live.kind {
                Kind::Owned => {
                    "web access is off for this model: unload it and load it again with \
                     \"Allow web access\" ticked, or restart the launcher with --allow-web"
                }
                Kind::External => {
                    "web access is off for this launcher: restart it with --allow-web (the \
                     chat fetches through this launcher, not through the launch it shows)"
                }
                _ => {
                    "web access is off: tick \"Allow web access\" when you launch, or restart \
                     the launcher with --allow-web"
                }
            };
            (reason, self.dashboard_command())
        };
        WebAccess {
            allowed: false,
            reason: Some(reason.to_string()),
            command: Some(command),
        }
    }

    /// The command that restarts the shown launch with `--allow-web`: the
    /// launcher's own recorded command line, else one rebuilt from its record.
    fn restart_command(&self, live: &Live) -> String {
        let recorded: Vec<String> = arr(&live.fact("launcher_command"))
            .iter()
            .filter_map(|word| word.as_str().map(str::to_string))
            .collect();
        let mut words = if recorded.len() > 1 {
            recorded
        } else {
            let target = string(&live.fact("profile_path"))
                .or_else(|| live.repo.clone())
                .unwrap_or_else(|| "OWNER/NAME".into());
            let mut words = vec!["loom-launch".to_string(), "serve".to_string(), target];
            if let Some(host) =
                string(&live.fact("bind_host")).filter(|h| !crate::network::is_loopback_host(h))
            {
                words.extend(["--host".to_string(), host]);
            }
            if let Some(port) = live
                .endpoint
                .as_deref()
                .and_then(upstream_of)
                .map(|u| u.port)
            {
                words.extend(["--proxy-port".to_string(), port.to_string()]);
            }
            words
        };
        if !words.iter().any(|w| w == "--allow-web") {
            words.push("--allow-web".into());
        }
        shell_join(&words)
    }

    /// The command that restarts this launcher with `--allow-web`.
    fn dashboard_command(&self) -> String {
        let command = self
            .service
            .as_ref()
            .map(|s| s.command.clone())
            .unwrap_or_default();
        let mut words = if command.len() > 1 {
            command
        } else {
            let listener = self.listener.lock().unwrap().clone();
            let mut words = vec![
                "loom-launch".to_string(),
                "launcher".to_string(),
                "start".to_string(),
            ];
            if let Some(port) = listener.get("port").and_then(Value::as_u64) {
                words.extend(["--proxy-port".to_string(), port.to_string()]);
            }
            if let Some(host) = listener
                .get("bind_host")
                .and_then(Value::as_str)
                .filter(|h| !crate::network::is_loopback_host(h))
            {
                words.extend(["--host".to_string(), host.to_string()]);
            }
            words
        };
        if !words.iter().any(|w| w == "--allow-web") {
            words.push("--allow-web".into());
        }
        shell_join(&words)
    }

    /// The settings the live launch was planned with (draft depth, rows...).
    fn served_settings(&self, live: &Live) -> Map<String, Value> {
        match live.kind {
            Kind::Hosted => child(&live.provided, "settings").clone(),
            Kind::Owned => child(&live.provided, "settings").clone(),
            Kind::External => current_run(&self.state_root)
                .map(|run| child(&run.read_metadata(), "settings").clone())
                .unwrap_or_default(),
            Kind::None => Map::new(),
        }
    }

    /// The profile of `repo` from local documents only (cached per repo):
    /// its install, else the package directory the launch serves, else the
    /// cached hub documents.
    fn profile_of(&self, repo: &str, directory: Option<&str>) -> Option<Profile> {
        if let Some((cached, profile)) = self.profile.lock().unwrap().as_ref() {
            if cached == repo {
                return profile.clone();
            }
        }
        let ctx = self.cli();
        let profile = match InstallRegistry::new(&self.state_root).get(repo) {
            Some(installed) => cli::local_profile(&ctx, &installed, true).ok(),
            None => directory
                .and_then(|d| Profile::from_local(Path::new(d), Some(repo)).ok())
                .or_else(|| {
                    ProfileCache::new(&self.state_root)
                        .get_any(repo)
                        .map(|cached| cli::from_cache(repo, &cached, None))
                }),
        };
        *self.profile.lock().unwrap() = Some((repo.to_string(), profile.clone()));
        profile
    }

    // -- sampling -----------------------------------------------------------

    fn sample(&self, tick: u64) {
        let live = self.live();
        if live.kind == Kind::External {
            self.refresh_remote(&live);
        }
        let probed = self
            .probe
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|(key, _)| *key == live.key());
        if (tick.is_multiple_of(2) || !probed) && live.kind != Kind::None {
            let target = live
                .server_endpoint
                .clone()
                .or(live.endpoint.clone())
                .unwrap_or_default();
            let up = Client::new(&target, None, Duration::from_secs(1))
                .map(|c| c.healthy(Duration::from_secs(1)))
                .unwrap_or(false);
            *self.probe.lock().unwrap() = Some((live.key(), up));
        }
        let snapshot = self.metrics_snapshot(&live);
        let sample =
            events::Sample::from_snapshot(now_seconds(), &self.sample_key(&live), &snapshot);
        self.history.lock().unwrap().push(sample);
    }

    fn refresh_remote(&self, live: &Live) {
        let Some(endpoint) = live.endpoint.clone() else {
            return;
        };
        let Ok(client) = Client::new(&endpoint, live.token.as_deref(), Duration::from_secs(2))
        else {
            return;
        };
        let timeout = Some(Duration::from_secs(2));
        let metrics = match client.get_json("/loom/status", timeout) {
            Ok((200, status)) => get(child(&status, "proxy"), "metrics").clone(),
            _ => Value::Null,
        };
        let last_request = match client.get_json("/loom/requests?limit=1", timeout) {
            Ok((200, document)) => arr(get(&document, "data"))
                .first()
                .cloned()
                .unwrap_or(Value::Null),
            _ => Value::Null,
        };
        *self.remote.lock().unwrap() = Some(Remote {
            endpoint,
            metrics,
            last_request,
        });
    }

    /// One `/api/events` document.
    fn event(&self, sequence: u64, cursor: &mut events::Cursor) -> Value {
        let live = self.live();
        let metrics = self.metrics_snapshot(&live);
        let samples: Vec<Value> = {
            let history = self.history.lock().unwrap();
            let fresh = history.since(cursor.last_sample);
            if let Some(last) = fresh.last() {
                cursor.last_sample = Some(last.t);
            }
            fresh.iter().map(|s| s.describe()).collect()
        };
        json!({
            "seq": sequence,
            "time": iso_utc_millis(now_seconds()),
            "mode": self.mode(),
            "launch": self.launch_summary(&live),
            "launcher": self.launcher_summary(&live),
            "in_flight": get(obj(&metrics), "in_flight").as_u64().unwrap_or(0),
            "metrics": metrics,
            "last_request": self.last_request(&live),
            "job": self.jobs.describe(40),
            "jobs": self.jobs.list(6),
            "samples": samples,
            "log": cursor.advance(&self.state_root),
        })
    }

    // -- launches started here ----------------------------------------------

    /// Unloads the model this launcher loaded (or cancels one preparing) and
    /// waits until its run record is finished; used on shutdown. Returns
    /// what was unloaded or cancelled, if anything.
    pub fn stop_owned(&self, grace: Duration) -> Option<String> {
        let mut stopped = None;
        if let Some(preparing) = self.preparing.lock().unwrap().clone() {
            stopped = Some(preparing.plan().profile.repo.clone());
            preparing.cancel(grace);
        }
        let owned = self.owned.lock().unwrap().take();
        if let Some(owned) = owned {
            owned.unloading.store(true, Ordering::SeqCst);
            if !owned.supervisor.exited() {
                stopped = Some(owned.supervisor.plan().profile.repo.clone());
            }
            owned.supervisor.stop(grace);
        }
        // A load still on its way to ready finishes (as cancelled) before
        // the process may exit: nothing it started is left behind.
        let deadline = std::time::Instant::now() + grace + Duration::from_secs(15);
        while self.jobs.active("serve").is_some() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        let watchers: Vec<_> = self.watchers.lock().unwrap().drain(..).collect();
        for watcher in watchers {
            let _ = watcher.join();
        }
        stopped
    }

    // -- the launcher ----------------------------------------------------------

    /// The launcher itself: its listener, uptime, the requests it answered
    /// itself, the model state and the last model that exited on its own.
    /// Null for a hosted page.
    fn launcher_summary(&self, live: &Live) -> Value {
        let Some(front) = &self.front else {
            return Value::Null;
        };
        let listener = self.listener.lock().unwrap().clone();
        let endpoint = self.endpoint();
        let model = match live.kind {
            Kind::Owned | Kind::External if self.going_away(live) => "stopping",
            Kind::Owned | Kind::External => "serving",
            _ if self.preparing() || self.jobs.active("serve").is_some() => "starting",
            _ => "none",
        };
        json!({
            "state": "running",
            "pid": std::process::id(),
            "endpoint": endpoint,
            "dashboard": endpoint.as_ref().map(|e| format!("{e}/")),
            "bind_host": listener.get("bind_host"),
            "port": listener.get("port"),
            "urls": listener.get("urls"),
            "started_at": crate::values::iso_utc(self.started_at),
            "uptime_seconds": crate::values::round_to(now_seconds() - self.started_at, 0),
            "auth": front.auth,
            "allow_web": self.allow_web,
            "server_port": self.service.as_ref().map(|s| s.server_port),
            "counters": front.counters.snapshot(),
            "model": model,
            "repo": live.repo,
            "last_exit": self.last_exit.lock().unwrap().clone(),
        })
    }

    /// `/loom/status` of a launcher: the launcher, and the loaded model's
    /// launch facts when one is loaded.
    pub fn status_provider(self: &Arc<Self>) -> StatusProvider {
        let weak = Arc::downgrade(self);
        Arc::new(move || {
            let Some(context) = weak.upgrade() else {
                return Map::new();
            };
            let live = context.live();
            let mut status = Map::new();
            status.insert("launcher".into(), context.launcher_summary(&live));
            if live.kind == Kind::Owned {
                for (key, value) in &live.provided {
                    status.insert(key.clone(), value.clone());
                }
            } else {
                status.insert("launch".into(), Value::Null);
            }
            status
        })
    }

    /// `/healthz` of a launcher: healthy whenever it answers, and whether a
    /// model is loaded.
    pub fn health_provider(self: &Arc<Self>) -> HealthProvider {
        let weak = Arc::downgrade(self);
        Arc::new(move || {
            let Some(context) = weak.upgrade() else {
                return json!({"status": "ok", "model": false});
            };
            let live = context.live();
            let summary = context.launcher_summary(&live);
            json!({
                "status": "ok",
                "launcher": {"pid": std::process::id(), "uptime_seconds": summary["uptime_seconds"]},
                "model": matches!(live.kind, Kind::Owned | Kind::External),
                "state": summary["model"],
                "repo": live.repo,
            })
        })
    }

    /// Reports a loaded model that exited without being unloaded: a failed
    /// Activity row, `last_exit` in the status and `launcher.json`.
    fn model_exited(&self, supervisor: &Supervisor, code: i32) {
        let repo = supervisor.plan().profile.repo.clone();
        let run = supervisor.run();
        let detail = exit_detail(code);
        let message = format!(
            "{repo} exited on its own ({detail}); the launcher keeps serving with no model loaded"
        );
        if let Ok(mut job) = self.jobs.start("exit", &repo) {
            job.set_fact("run", json!(run.run_id));
            job.begin();
            job.step("exit", "The model server exited");
            for line in supervisor.tail(8) {
                job.line(&line);
            }
            job.end_step(
                "exit",
                "The model server exited",
                StepState::Failed,
                Some(detail.clone()),
            );
            job.set_hint(
                "Its last output is above and in serve.log (Logs tab). Load it again from the \
                 Models tab or with `loom-launch start`.",
            );
            job.finish(Err(message.clone()));
        }
        let record = json!({
            "repo": repo,
            "run_id": run.run_id,
            "exit_code": code,
            "detail": detail,
            "at": crate::values::iso_utc(now_seconds()),
        });
        *self.last_exit.lock().unwrap() = record.clone();
        crate::launcher::note_exit(&self.state_root, &record);
        if self.service.as_ref().is_some_and(|s| s.echo) {
            println!("  model     {message}");
        }
    }
}

/// `exit status 3`, or `killed by signal 9, exit status 137`.
pub(crate) fn exit_detail(code: i32) -> String {
    if code > 128 && code < 160 {
        format!("killed by signal {}, exit status {code}", code - 128)
    } else {
        format!("exit status {code}")
    }
}

/// Samples counters once a second, probes the upstream every other second,
/// and reads another process's launch through its proxy.
fn spawn_sampler(context: Weak<Context>) {
    let _ = std::thread::Builder::new()
        .name("loom-launch-dashboard".into())
        .spawn(move || {
            let mut tick: u64 = 0;
            loop {
                {
                    let Some(context) = context.upgrade() else {
                        return;
                    };
                    if context.closed() {
                        return;
                    }
                    context.sample(tick);
                }
                tick += 1;
                std::thread::sleep(Duration::from_secs(1));
            }
        });
}

/// The run the logs come from: the current launch's, else the latest.
pub(crate) fn current_run(state_root: &Path) -> Option<RunDirectory> {
    let store = RunStore::new(state_root);
    read_current(state_root)
        .and_then(|record| str_value(get(&record, "run_id")).map(|id| store.open(id)))
        .filter(|run| run.root.is_dir())
        .or_else(|| store.latest())
}

/// `http://host:port` → the proxy's upstream address.
fn upstream_of(url: &str) -> Option<Upstream> {
    let rest = url.strip_prefix("http://").unwrap_or(url);
    let authority = rest.split('/').next()?;
    let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
        let (host, tail) = bracketed.split_once(']')?;
        (host.to_string(), tail.strip_prefix(':')?.parse().ok()?)
    } else {
        let (host, port) = authority.rsplit_once(':')?;
        (host.to_string(), port.parse().ok()?)
    };
    Some(Upstream { host, port })
}

/// Seconds since the epoch for `YYYY-mm-ddTHH:MM:SS[.fff]Z`.
fn parse_iso_utc(text: &str) -> Option<f64> {
    let text = text.strip_suffix('Z')?;
    let (date, time) = text.split_once('T')?;
    let mut date_parts = date.split('-').map(|p| p.parse::<i64>().ok());
    let (year, month, day) = (
        date_parts.next()??,
        date_parts.next()??,
        date_parts.next()??,
    );
    let mut time_parts = time.split(':');
    let hour: i64 = time_parts.next()?.parse().ok()?;
    let minute: i64 = time_parts.next()?.parse().ok()?;
    let second: f64 = time_parts.next()?.parse().ok()?;
    // Days from the civil date (Howard Hinnant's algorithm).
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some((days * 86400 + hour * 3600 + minute * 60) as f64 + second)
}

// -- routing -------------------------------------------------------------------

/// Routes the dashboard owns.
pub fn claims(path: &str) -> bool {
    path == "/" || path == "/index.html" || path.starts_with("/api/")
}

/// Routes served without the bearer token (the static page only).
pub fn is_public(path: &str) -> bool {
    path == "/" || path == "/index.html"
}

const GET_ROUTES: [&str; 11] = [
    "/",
    "/index.html",
    "/api/jobs",
    "/api/status",
    "/api/metrics",
    "/api/requests",
    "/api/installs",
    "/api/benchmarks",
    "/api/logs",
    "/api/models",
    "/api/events",
];
const POST_ROUTES: [&str; 8] = [
    "/api/jobs/dismiss",
    "/api/install",
    "/api/serve",
    "/api/stop",
    "/api/unload",
    "/api/bench",
    "/api/chat",
    "/api/fetch",
];

fn error_reply(status: u16, code: &str, message: &str) -> Reply {
    json_response(
        status,
        &json!({"error": {"message": message, "type": "loom_launch_error", "code": code}}),
    )
}

fn not_found(path: &str) -> Reply {
    error_reply(
        404,
        "NOT_FOUND",
        &format!("{path} is not a dashboard route"),
    )
}

fn conflict(message: &str) -> Reply {
    error_reply(409, "CONFLICT", message)
}

fn invalid(message: &str) -> Reply {
    error_reply(400, "INVALID_ARGUMENT", message)
}

/// Requests that start work must come from the dashboard's own origin: a JSON
/// body (cross-site forms cannot send one without a preflight nobody answers),
/// a matching `Origin`, and a `Host` that is an address or `localhost` (a
/// rebound DNS name pointing at loopback is refused).
fn guard(method: &str, headers: &[(String, String)]) -> Option<Reply> {
    let host = header(headers, "host").unwrap_or("");
    if !host.is_empty() && !local_host(host) {
        return Some(error_reply(
            403,
            "PERMISSION_DENIED",
            "address the dashboard by IP address or localhost",
        ));
    }
    if method != "POST" {
        return None;
    }
    if let Some(origin) = header(headers, "origin").filter(|o| *o != "null") {
        if !host.is_empty() && !same_origin(origin, host) {
            return Some(error_reply(
                403,
                "PERMISSION_DENIED",
                "cross-origin requests are refused",
            ));
        }
    }
    let content_type = header(headers, "content-type").unwrap_or("");
    if !content_type
        .to_ascii_lowercase()
        .starts_with("application/json")
    {
        return Some(error_reply(
            415,
            "INVALID_ARGUMENT",
            "send the request body as application/json",
        ));
    }
    None
}

/// Whether a browser's `Origin` names the listener the request reached: its
/// authority must equal `Host` (an IP address and port on a LAN, `localhost`
/// on this machine), with default ports written or not.
fn same_origin(origin: &str, host: &str) -> bool {
    let (scheme, authority) = origin.split_once("://").unwrap_or(("http", origin));
    let authority = authority.trim_end_matches('/');
    let default_port = match scheme.to_ascii_lowercase().as_str() {
        "https" => ":443",
        _ => ":80",
    };
    let normalize = |text: &str| {
        let lower = text.to_ascii_lowercase();
        lower
            .strip_suffix(default_port)
            .map(str::to_string)
            .unwrap_or(lower)
    };
    normalize(authority) == normalize(host)
}

fn local_host(host: &str) -> bool {
    let name = if let Some(bracketed) = host.strip_prefix('[') {
        bracketed.split(']').next().unwrap_or("")
    } else {
        host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host)
    };
    let name = name.to_ascii_lowercase();
    name.parse::<std::net::IpAddr>().is_ok() || name == "localhost" || name.ends_with(".localhost")
}

/// Answers one dashboard request from the page.
pub fn handle(context: &Arc<Context>, request: tiny_http::Request, method: &str, path: &str) {
    handle_trusted(context, request, method, path, false);
}

/// Answers one dashboard request; `trusted` requests carry the launcher's
/// control token (its own command line), which may load any package
/// directory rather than only the ones the page lists.
pub fn handle_trusted(
    context: &Arc<Context>,
    mut request: tiny_http::Request,
    method: &str,
    path: &str,
    trusted: bool,
) {
    let (target, query) = path.split_once('?').unwrap_or((path, ""));
    let headers = request_headers(&request);
    if target.starts_with("/api/") {
        if let Some(refusal) = guard(method, &headers) {
            let _ = request.respond(refusal);
            return;
        }
    }
    let is_post = POST_ROUTES.contains(&target);
    let is_get = GET_ROUTES.contains(&target)
        || target.starts_with("/api/benchmarks/")
        || (target.starts_with("/api/jobs/") && !is_post);
    match method {
        "GET" | "HEAD" if is_get => {
            if target == "/api/events" {
                events::stream(context.clone(), request);
                return;
            }
            let reply = get_route(context, target, query);
            let _ = request.respond(reply);
        }
        "POST" if is_post => {
            if target == "/api/chat" {
                chat(context, request, &headers);
                return;
            }
            let reply = match read_body(&mut request).and_then(|b| parse_object(&b)) {
                Ok(body) => post_route(context, target, &body, trusted),
                Err(reply) => reply,
            };
            let _ = request.respond(reply);
        }
        _ if is_get || is_post => {
            let allow = if is_get { "GET, HEAD" } else { "POST" };
            let mut reply = error_reply(405, "METHOD_NOT_ALLOWED", "method not allowed");
            if let Ok(header) = tiny_http::Header::from_bytes("Allow", allow) {
                reply.add_header(header);
            }
            let _ = request.respond(reply);
        }
        _ => {
            let _ = request.respond(not_found(target));
        }
    }
}

fn read_body(request: &mut tiny_http::Request) -> std::result::Result<Vec<u8>, Reply> {
    let mut body = Vec::new();
    request
        .as_reader()
        .take(BODY_LIMIT + 1)
        .read_to_end(&mut body)
        .map_err(|error| invalid(&format!("cannot read the request body: {error}")))?;
    if body.len() as u64 > BODY_LIMIT {
        return Err(error_reply(
            413,
            "INVALID_ARGUMENT",
            "the request body is too large",
        ));
    }
    Ok(body)
}

fn parse_object(body: &[u8]) -> std::result::Result<Map<String, Value>, Reply> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(Map::new());
    }
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(invalid("the request body must be a JSON object")),
        Err(error) => Err(invalid(&format!("the request body is not JSON: {error}"))),
    }
}

fn text_field(body: &Map<String, Value>, key: &str) -> Option<String> {
    str_value(get(body, key))
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
}

/// An optional integer field; numeric strings (form inputs) are accepted.
fn int_field(body: &Map<String, Value>, key: &str) -> std::result::Result<Option<i64>, Reply> {
    match get(body, key) {
        Value::Null => Ok(None),
        Value::Number(number) => match number.as_i64() {
            Some(value) => Ok(Some(value)),
            None => Err(invalid(&format!("{key} must be a whole number"))),
        },
        Value::String(text) if text.trim().is_empty() => Ok(None),
        Value::String(text) => text
            .trim()
            .parse::<i64>()
            .map(Some)
            .map_err(|_| invalid(&format!("{key} must be a whole number"))),
        _ => Err(invalid(&format!("{key} must be a whole number"))),
    }
}

/// `owner/name` with hub-safe characters.
fn valid_repo(repo: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && p.len() <= 96
            && !p.starts_with(['.', '-'])
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    };
    match repo.split_once('/') {
        Some((owner, name)) => part(owner) && part(name),
        None => false,
    }
}

fn repo_field(body: &Map<String, Value>) -> std::result::Result<String, Reply> {
    let repo = text_field(body, "repo").ok_or_else(|| invalid("repo is required (owner/name)"))?;
    if !valid_repo(&repo) {
        return Err(invalid(&format!("{repo} is not an owner/name profile")));
    }
    Ok(repo)
}

fn get_route(context: &Arc<Context>, target: &str, query: &str) -> Reply {
    match target {
        "/" | "/index.html" => page::reply(context),
        "/api/status" => json_response(200, &status(context)),
        "/api/metrics" => prometheus(context),
        "/api/requests" => {
            let limit = query_value(query, "limit")
                .and_then(|v| v.parse().ok())
                .unwrap_or(50usize)
                .clamp(1, 1000);
            json_response(200, &requests(context, limit))
        }
        "/api/installs" => json_response(200, &installs(context)),
        "/api/benchmarks" => json_response(200, &benchmarks(context)),
        "/api/logs" => logs(context, query),
        "/api/models" => models(context, query),
        "/api/jobs" => json_response(200, &json!({"jobs": context.jobs.list(40)})),
        _ if target.starts_with("/api/jobs/") => {
            let id = crate::proxy::percent_decode(target.trim_start_matches("/api/jobs/"));
            match context.jobs.get(&id, 400) {
                Some(job) => json_response(200, &job),
                None => error_reply(404, "NOT_FOUND", &format!("no job {id}")),
            }
        }
        _ => {
            let id = target.strip_prefix("/api/benchmarks/").unwrap_or("");
            match BenchmarkStore::new(&context.state_root).load(&crate::proxy::percent_decode(id)) {
                Some(record) if !id.is_empty() => json_response(200, &Value::Object(record)),
                _ => not_found(target),
            }
        }
    }
}

fn post_route(
    context: &Arc<Context>,
    target: &str,
    body: &Map<String, Value>,
    trusted: bool,
) -> Reply {
    let outcome = match target {
        "/api/install" => start_install(context, body),
        "/api/serve" => start_serve(context, body, trusted),
        "/api/stop" | "/api/unload" => stop(context, body),
        "/api/bench" => start_bench(context, body),
        "/api/fetch" => Ok(web_fetch(context, body)),
        "/api/jobs/dismiss" => dismiss(context, body),
        _ => Err(not_found(target)),
    };
    outcome.unwrap_or_else(|reply| reply)
}

// -- read-only routes ------------------------------------------------------------

fn status(context: &Context) -> Value {
    let live = context.live();
    let mut document = Map::new();
    // Launch facts the hosting supervisor reports (`launch`, `settings`).
    for (key, value) in &live.provided {
        document.insert(key.clone(), value.clone());
    }
    let mut launch = context.launch_summary(&live);
    if let (Value::Object(summary), Value::Object(provided)) =
        (&mut launch, get(&live.provided, "launch"))
    {
        for (key, value) in provided {
            summary.entry(key.clone()).or_insert(value.clone());
        }
    }
    let hardware = context.hardware();
    let directory = string(&live.fact("profile_path"));
    let scenario = str_value(&live.fact("kind")) == Some("scenario");
    let settings = context.served_settings(&live);
    let profile = live
        .repo
        .as_deref()
        .and_then(|repo| context.profile_of(repo, directory.as_deref()))
        .map(|p| {
            let local = InstallRegistry::new(&context.state_root)
                .get(&p.repo)
                .is_none_or(|installed| crate::local::is_local_install(&installed));
            profile_summary(&p, local, &settings, scenario)
        });
    let serving = live.kind != Kind::None;
    let busy = context.jobs.running().is_some();
    let web = context.web_access(&live);
    document.insert("mode".into(), json!(context.mode()));
    document.insert("version".into(), json!(env!("CARGO_PKG_VERSION")));
    document.insert(
        "state_directory".into(),
        json!(context.state_root.to_string_lossy()),
    );
    document.insert(
        "hub".into(),
        json!(Hub::new(context.hub_endpoint.as_deref(), Duration::from_secs(1)).endpoint),
    );
    document.insert("launch".into(), launch);
    document.insert("launcher".into(), context.launcher_summary(&live));
    document.insert("proxy".into(), context.metrics_snapshot(&live));
    document.insert("upstream".into(), json!(live.server_endpoint));
    document.insert("current".into(), json!(live.record));
    // A prepared scenario's own deployment and devices (`kind: "scenario"`).
    document.insert("scenario".into(), live.fact("scenario"));
    document.insert("profile".into(), json!(profile));
    document.insert("job".into(), context.jobs.describe(40));
    document.insert("jobs".into(), context.jobs.list(40));
    document.insert(
        "stored_runs".into(),
        json!(BenchmarkStore::new(&context.state_root).list(None).len()),
    );
    document.insert(
        "installs".into(),
        json!(InstallRegistry::new(&context.state_root).list().len()),
    );
    document.insert(
        "hardware".into(),
        json!({
            "gpu": hardware.gpu().map(|g| g.name.clone()),
            "gfx_arch": hardware.gfx_arch(),
            "memory": human_bytes(hardware.total_memory_bytes),
            "default_device_uri": hardware.default_device_uri(),
        }),
    );
    let launcher = context.is_launcher();
    let no_model = (!serving).then_some(if launcher {
        "no model is loaded: load one from the Overview or the Models tab, or run \
         `loom-launch start PROFILE`"
    } else {
        "the launch this page belongs to is not serving"
    });
    let unloading = context
        .jobs
        .active("unload")
        .or(context.jobs.active("stop"));
    document.insert(
        "capabilities".into(),
        json!({
            "serve": launcher && !serving && !busy,
            "stop": serving || context.preparing(),
            "unload": launcher && (serving || context.preparing()),
            "stopping": unloading.is_some(),
            "chat": serving,
            "chat_reason": no_model,
            "bench": serving && !busy,
            "bench_reason": no_model,
            "install": !busy,
            "web_access": web.allowed,
            "web_access_reason": web.reason,
            "web_access_command": web.command,
            // A launcher passes "Allow web access" to the models it loads.
            "launch_allow_web": launcher,
        }),
    );
    document.insert("listener".into(), context.listener.lock().unwrap().clone());
    document.insert("suites".into(), suites());
    Value::Object(document)
}

fn suites() -> Value {
    Value::Array(
        SUITES
            .iter()
            .map(|(name, metrics)| json!({"name": name, "metrics": metrics}))
            .collect(),
    )
}

/// Whether the chat's thinking switch works for `profile`, and when it does
/// not, why: the package fixes it (`thinking: always|never`), or ships no
/// policy at all. `local` packages are mended by copying the document, hub
/// installs by reinstalling.
pub(crate) fn thinking_summary(profile: &Profile, local: bool) -> Value {
    let Some(chat) = &profile.chat else {
        let remedy = if local {
            "copy the package's llm_config.json next to its model.json, or install it from \
             the hub"
                .to_string()
        } else {
            format!(
                "reinstall it with `loom-launch install {}` to fetch the policy",
                profile.repo
            )
        };
        return json!({
            "available": false,
            "mode": "unknown",
            "default": false,
            "locked": null,
            "reason": format!(
                "This package ships no chat policy (llm_config.json), so whether it thinks, \
                 and how to switch that, is unknown; {remedy}."
            ),
        });
    };
    let mode = chat.thinking_mode();
    let (available, locked, reason) = match mode.as_str() {
        "never" => (
            false,
            Some(false),
            Some("This model does not think: its chat policy says thinking: never.".to_string()),
        ),
        "always" => (
            false,
            Some(true),
            Some(
                "This model always thinks: its chat policy says thinking: always, so it \
                 cannot be switched off."
                    .to_string(),
            ),
        ),
        _ if !chat.supports_thinking() => (
            false,
            None,
            Some(format!(
                "Its chat policy says thinking: {mode} but names no think_open delimiter, so \
                 the launcher cannot switch it."
            )),
        ),
        _ => (true, None, None),
    };
    json!({
        "available": available,
        "mode": mode,
        "default": chat.default_thinking(),
        "locked": locked,
        "reason": reason,
    })
}

/// The drafter the policy recommends against the depth the launch serves,
/// with a note when the policy recommends more than is served. The note is
/// informational: a scenario's deployment fixes the depth, and the launcher
/// refuses draft-depth overrides for a scenario.
pub(crate) fn drafter_summary(
    profile: &Profile,
    served_depth: Option<i64>,
    scenario: bool,
) -> Value {
    let drafter = &profile.drafter;
    let Some(depth) = int_value(get(drafter, "depth")) else {
        return Value::Null;
    };
    let kind = str_value(get(drafter, "kind")).unwrap_or("draft");
    let tuned = arr(get(drafter, "tuned"));
    let rate = |wanted: &[&str], at: i64| {
        tuned
            .iter()
            .map(obj)
            .find(|row| {
                int_value(get(row, "depth")) == Some(at)
                    && str_value(get(row, "kind")).is_some_and(|k| wanted.contains(&k))
            })
            .and_then(|row| number_value(get(row, "decode_tps")))
    };
    let served = served_depth.unwrap_or(0);
    let recommended_tps = rate(&[kind], depth);
    let served_tps = if served == 0 {
        rate(&["none"], 0)
    } else {
        rate(&[kind], served)
    };
    let note = (served < depth).then(|| {
        let measured = match (served_tps, recommended_tps) {
            (Some(now), Some(better)) => format!(
                " ({} tok/s against {} tok/s in the publisher's tuning)",
                grouped(now, 1),
                grouped(better, 1)
            ),
            _ => String::new(),
        };
        let fixed = if scenario {
            "This scenario's own deployment fixes the depth, and the launcher refuses \
             draft-depth overrides for a scenario."
        } else {
            "The package's preset sets the depth; a new launch's Draft depth (--draft-depth) \
             can ask for more when the package carries the drafter."
        };
        format!(
            "Served with draft depth {served}; the policy recommends a {kind} drafter at depth \
             {depth}{measured}. {fixed}"
        )
    });
    json!({
        "kind": kind,
        "depth": depth,
        "repo": get(drafter, "repo"),
        "file": get(drafter, "file"),
        "served_depth": served,
        "served_tps": served_tps,
        "recommended_tps": recommended_tps,
        "note": note,
    })
}

fn profile_summary(
    profile: &Profile,
    local: bool,
    settings: &Map<String, Value>,
    scenario: bool,
) -> Value {
    let decode = profile
        .performance
        .as_ref()
        .and_then(|p| p.measurement("decode"))
        .and_then(|m| m.value);
    let execution = profile.execution.clone().unwrap_or_default();
    let served = |key: &str, card: Option<i64>| int_value(get(settings, key)).or(card);
    let draft_depth = served("draft_depth", execution.draft_depth);
    json!({
        "repo": profile.repo,
        "model_id": profile.model_id,
        "name": profile.name,
        "quant": profile.quant,
        "architecture": profile.architecture,
        "context_length": profile.context_length,
        "native_context_length": profile.native_context_length,
        "memory_bytes": profile.memory_bytes,
        "memory": human_bytes(profile.memory_bytes),
        "device_hint": profile.device_hint,
        "decode": decode,
        "chat": profile.chat.as_ref().map(|c| c.describe()),
        "execution": profile.execution.as_ref().map(|e| e.describe()),
        "thinking": thinking_summary(profile, local),
        "served": {
            "preset": execution.preset,
            "draft_depth": draft_depth,
            "rows": served("rows", execution.rows),
            "context_length": served("context_length", execution.context_capacity),
            "max_tokens": served("max_tokens", execution.max_tokens),
            "prefill_capacity": served("prefill_capacity", execution.prefill_capacity),
        },
        "drafter": drafter_summary(profile, draft_depth, scenario),
    })
}

fn prometheus(context: &Context) -> Reply {
    let live = context.live();
    let text = match (context.live_metrics(&live), live.kind) {
        (Some(metrics), _) => metrics.prometheus(),
        (None, Kind::External) => live
            .endpoint
            .as_deref()
            .and_then(|e| Client::new(e, live.token.as_deref(), Duration::from_secs(3)).ok())
            .and_then(|client| client.request("GET", "/loom/metrics", None, None).ok())
            .filter(|(status, _)| *status == 200)
            .map(|(_, body)| String::from_utf8_lossy(&body).into_owned())
            .unwrap_or_else(|| Metrics::default().prometheus()),
        _ => Metrics::default().prometheus(),
    };
    raw_response(200, "text/plain; version=0.0.4", text.into_bytes())
}

/// Recent requests: the live proxy's window, else the latest run's log.
fn requests(context: &Context, limit: usize) -> Value {
    let live = context.live();
    if let Some(metrics) = &live.metrics {
        return json!({"object": "list", "source": "live", "data": metrics.recent(limit)});
    }
    // A launcher with no model: what it carried since it started, else the
    // latest run's log (a fresh launcher has carried nothing yet).
    if let Some(metrics) = context.live_metrics(&live) {
        let recent = metrics.recent(limit);
        if !recent.is_empty() {
            return json!({"object": "list", "source": "live", "data": recent});
        }
    }
    if let (Kind::External, Some(endpoint)) = (live.kind, &live.endpoint) {
        let fetched = Client::new(endpoint, live.token.as_deref(), Duration::from_secs(3))
            .ok()
            .and_then(|c| {
                c.get_json(&format!("/loom/requests?limit={limit}"), None)
                    .ok()
            })
            .filter(|(status, _)| *status == 200);
        if let Some((_, document)) = fetched {
            return json!({"object": "list", "source": "live", "data": get(&document, "data")});
        }
    }
    let data: Vec<Value> = current_run(&context.state_root)
        .map(|run| {
            let text = std::fs::read_to_string(run.request_log()).unwrap_or_default();
            text.lines()
                .rev()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .take(limit)
                .collect()
        })
        .unwrap_or_default();
    json!({"object": "list", "source": "log", "data": data})
}

fn installs(context: &Context) -> Value {
    let live = context.live();
    let ctx = context.cli();
    // A launch is recorded under the package name, the installed name or the
    // model id it serves, depending on how it was started; the package it is
    // serving is the reliable identity, so compare paths first.
    let running_path = string(&live.fact("profile_path"));
    let installs: Vec<Value> = InstallRegistry::new(&context.state_root)
        .list()
        .iter()
        .map(|installed| {
            let mut document = installed.describe();
            document["card_exists"] = json!(installed.card_path().is_file());
            let running = match &running_path {
                Some(serving) => {
                    crate::deployment::absolute(Path::new(serving))
                        == crate::deployment::absolute(Path::new(&installed.path))
                }
                None => live.repo.as_deref() == Some(installed.repo.as_str()),
            };
            document["running"] = json!(running);
            let scenario = crate::scenario::Scenario::detect(Path::new(&installed.path))
                .ok()
                .flatten();
            document["scenario"] = json!(scenario.is_some());
            document["scenario_devices"] = json!(scenario.as_ref().map(|s| s.device_text()));
            if installed.card_path().is_file() {
                if let Ok(profile) = cli::local_profile(&ctx, installed, true) {
                    let execution = profile.execution.clone().unwrap_or_default();
                    document["model_id"] = json!(profile.model_id);
                    document["quant"] = json!(profile.quant);
                    document["context_length"] = json!(profile.context_length);
                    document["native_context_length"] = json!(profile.native_context_length);
                    document["memory_bytes"] = json!(profile.memory_bytes);
                    document["memory"] = json!(human_bytes(profile.memory_bytes));
                    document["max_tokens"] = json!(execution.max_tokens);
                    document["rows"] = json!(execution.rows);
                    document["draft_depth"] = json!(execution.draft_depth);
                    document["thinking"] = json!(profile.chat.as_ref().map(|c| c.thinking_mode()));
                    document["policy"] =
                        thinking_summary(&profile, crate::local::is_local_install(installed));
                    document["drafter"] =
                        drafter_summary(&profile, execution.draft_depth, scenario.is_some());
                }
            }
            document
        })
        .collect();
    json!({"installs": installs})
}

fn benchmarks(context: &Context) -> Value {
    let store = BenchmarkStore::new(&context.state_root);
    let runs: Vec<Value> = store
        .list(None)
        .into_iter()
        .rev()
        .map(|mut entry| {
            let record = str_value(get(&entry, "path"))
                .map(|path| crate::store::read_json(Path::new(path)))
                .unwrap_or_default();
            entry.insert("missing".into(), json!(record.is_empty()));
            entry.insert(
                "elapsed_seconds".into(),
                get(&record, "elapsed_seconds").clone(),
            );
            entry.insert("device".into(), get(&record, "device").clone());
            entry.insert("model".into(), get(&record, "model").clone());
            entry.insert("headline".into(), Value::Array(headline(&record)));
            Value::Object(entry)
        })
        .collect();
    json!({"runs": runs, "suites": suites()})
}

/// Compact rows of a stored record: its comparison, else its measured metrics.
fn headline(record: &Map<String, Value>) -> Vec<Value> {
    let comparison = arr(get(record, "comparison"));
    if !comparison.is_empty() {
        return comparison
            .iter()
            .map(obj)
            .map(|row| {
                let value =
                    number_value(get(row, "value")).or_else(|| number_value(get(row, "measured")));
                json!({
                    "metric": get(row, "metric"),
                    "value": value,
                    "published": get(row, "published"),
                    "ratio": get(row, "ratio"),
                    "roofline_fraction": get(row, "roofline_fraction"),
                })
            })
            .collect();
    }
    child(record, "metrics")
        .iter()
        .filter(|(_, entry)| !get(obj(entry), "value").is_null())
        .map(|(metric, entry)| {
            json!({"metric": metric, "value": get(obj(entry), "value"),
                   "published": null, "ratio": null, "roofline_fraction": null})
        })
        .collect()
}

fn tail(path: &Path, lines: usize) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes);
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn logs(context: &Context, query: &str) -> Reply {
    let lines = query_value(query, "lines")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200usize)
        .clamp(1, 5000);
    let store = RunStore::new(&context.state_root);
    let runs: Vec<String> = store
        .list()
        .into_iter()
        .take(20)
        .map(|r| r.run_id)
        .collect();
    let run = match query_value(query, "run").filter(|r| !r.is_empty()) {
        Some(id) => {
            if !runs.contains(&id) && !(valid_run_id(&id) && store.open(&id).root.is_dir()) {
                return error_reply(404, "NOT_FOUND", &format!("no run {id}"));
            }
            Some(store.open(&id))
        }
        None => current_run(&context.state_root),
    };
    let Some(run) = run else {
        return json_response(
            200,
            &json!({"run": null, "runs": runs, "serve_log": [], "events": [],
                    "message": "no runs recorded"}),
        );
    };
    let size = |path: &Path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    json_response(
        200,
        &json!({
            "run": run.run_id,
            "runs": runs,
            "current": read_current(&context.state_root)
                .and_then(|r| string(get(&r, "run_id"))) == Some(run.run_id.clone()),
            "serve_log": tail(&run.serve_log(), lines),
            "events": tail(&run.event_log(), lines),
            "serve_bytes": size(&run.serve_log()),
            "events_bytes": size(&run.event_log()),
            "metadata": Value::Object(run.read_metadata()),
        }),
    )
}

fn valid_run_id(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('.')
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn models(context: &Context, query: &str) -> Reply {
    let text = query_value(query, "q")
        .unwrap_or_default()
        .trim()
        .to_string();
    let mut offline = query_value(query, "offline")
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"));
    let limit = query_value(query, "limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(20usize)
        .clamp(1, 50);
    let ctx = context.cli();
    let args = SearchArgs {
        query: Some(text.clone()).filter(|q| !q.is_empty()),
        author: None,
        tags: Vec::new(),
        limit,
        no_filter: false,
        no_details: false,
        offline,
        wait: 20.0,
    };
    let hub = ctx.hub(Duration::from_secs(20));
    let cache = ProfileCache::new(&context.state_root);
    let mut warning = None;
    let found: Vec<(HubModel, Option<ProfileDocuments>)> = if offline {
        cli::offline_profiles(&ctx, &args)
    } else {
        let author = text.strip_suffix('/').map(str::to_string);
        let query = if author.is_some() {
            None
        } else {
            args.query.as_deref()
        };
        match hub.profiles(query, author.as_deref(), limit, &[PROFILE_TAG.to_string()]) {
            // Fresh cached documents answer without another round trip.
            Ok(models) => models
                .into_iter()
                .map(|model| {
                    let documents = cache
                        .get(&model.repo)
                        .map(|d| ProfileDocuments::from_cache(&d));
                    (model, documents)
                })
                .collect(),
            Err(error) => {
                warning = Some(format!(
                    "the hub did not answer ({error}); showing local and cached profiles"
                ));
                offline = true;
                cli::offline_profiles(&ctx, &args)
            }
        }
    };
    let hardware = context.hardware().clone();
    let mut rows = cli::details(&ctx, &hub, &hardware, found, true, offline);
    // Local profiles are read from their own files, so they show with or without the hub.
    rows.extend(cli::local_rows(
        &ctx,
        &hardware,
        args.query.as_deref(),
        None,
    ));
    cli::sort_rows(&mut rows);
    let installs = InstallRegistry::new(&context.state_root).list();
    let live = context.live();
    let running_path = string(&live.fact("profile_path"));
    let profiles: Vec<Value> = rows
        .iter()
        .map(|row| {
            search_row(
                row,
                &installs,
                live.repo.as_deref(),
                running_path.as_deref(),
            )
        })
        .collect();
    json_response(
        200,
        &json!({
            "query": text,
            "offline": offline,
            "count": profiles.len(),
            "warning": warning,
            "hardware": {
                "gpu": hardware.gpu().map(|g| g.name.clone()),
                "gfx_arch": hardware.gfx_arch(),
                "memory": human_bytes(hardware.total_memory_bytes),
            },
            "profiles": profiles,
        }),
    )
}

fn search_row(
    row: &cli::SearchRow,
    installs: &[crate::store::InstalledModel],
    running: Option<&str>,
    running_path: Option<&str>,
) -> Value {
    let profile = row.profile.as_ref();
    let path = row.path();
    let running = match (&path, running_path) {
        (Some(path), Some(serving)) => {
            crate::deployment::absolute(Path::new(serving)) == Path::new(path)
        }
        _ => running == Some(row.model.repo.as_str()),
    };
    let decode = profile
        .and_then(|p| p.performance.as_ref())
        .and_then(|p| p.measurement("decode"));
    let compat = row.compat.as_ref();
    json!({
        "repo": row.model.repo,
        "author": row.model.author(),
        "name": row.model.short_name(),
        "revision": row.model.revision,
        "last_modified": row.model.last_modified,
        "downloads": row.model.downloads,
        "likes": row.model.likes,
        "gated": row.model.gated,
        "model_id": profile.map(|p| p.model_id.clone()),
        "quant": profile.and_then(|p| p.quant.clone()),
        "architecture": profile.and_then(|p| p.architecture.clone()),
        "context_length": profile.and_then(|p| p.context_length),
        "native_context_length": profile.and_then(|p| p.native_context_length),
        "memory_bytes": profile.and_then(|p| p.memory_bytes),
        "memory": profile.map(|p| human_bytes(p.memory_bytes)),
        "decode": decode.and_then(|m| m.value),
        "decode_baseline_ratio": decode.and_then(|m| m.baseline_ratio()),
        "device_hint": profile.and_then(|p| p.device_hint.clone()),
        "gfx_arch": profile.and_then(|p| p.gfx_arch.clone()),
        "fit": compat.map(|c| crate::values::round_to(c.score, 3)),
        "fits_memory": compat.and_then(|c| c.fits_memory),
        "device_match": compat.and_then(|c| c.device_match),
        "fit_notes": compat.map(|c| c.notes.clone()).unwrap_or_default(),
        "installed": row.installed(installs),
        "running": running,
        "source": row.source(),
        "path": path,
        "scenario": row.scenario(installs),
        "scenario_devices": row.scenario_devices(installs),
        "error": row.error,
    })
}

// -- routes that start work --------------------------------------------------------

fn accepted(document: Value) -> Reply {
    json_response(202, &document)
}

/// The request a failed job's retry repeats.
fn retry_of(route: &str, body: Value) -> Value {
    json!({"route": route, "body": body})
}

/// What to do after an install failed, by what failed.
fn install_hint(message: &str, repo: &str) -> String {
    if message.contains("loom-fetch was not found") {
        "Install the Loom SDK (pip install \"hrx-loom[hf]\") or start the dashboard with \
         --loom-fetch PATH, then retry."
            .into()
    } else if message.contains("does not understand `huggingface:`") {
        "This loom-fetch predates hub selectors: install a newer Loom SDK, then retry.".into()
    } else if message.contains("did not install") {
        "loom-fetch finished without a model.json; read its output, then retry.".into()
    } else {
        format!(
            "loom-fetch failed; its last lines are in the output. Retry, or run \
             `loom-launch install {repo}` in a terminal to see every prompt (a gated \
             repository needs a hub login)."
        )
    }
}

fn start_install(
    context: &Arc<Context>,
    body: &Map<String, Value>,
) -> std::result::Result<Reply, Reply> {
    if let Some(path) = text_field(body, "path") {
        return start_register(context, body, &path);
    }
    let repo = repo_field(body)?;
    let preset = text_field(body, "preset");
    if preset.as_deref().is_some_and(|p| {
        !p.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
    }) {
        return Err(invalid("preset must be a plain name"));
    }
    let retry = retry_of("/api/install", json!({"repo": repo, "preset": preset}));
    let mut job = context
        .jobs
        .start_with("install", &repo, retry)
        .map_err(|message| conflict(&message))?;
    job.set_fact("source", json!("hub"));
    let id = job.id.clone();
    let worker = context.clone();
    let target = repo.clone();
    std::thread::spawn(move || {
        job.begin();
        let state = worker.state_root.clone();
        let hub = Hub::new(worker.hub_endpoint.as_deref(), Duration::from_secs(5));
        let output = default_models_directory(&state).join(target.replace('/', "-"));
        let options = InstallOptions {
            output: Some(output.clone()),
            preset,
            explicit_fetch: worker.service.as_ref().and_then(|s| s.loom_fetch.clone()),
            hub_endpoint: (!hub.is_default_endpoint()).then(|| hub.endpoint.clone()),
            ..Default::default()
        };
        job.step("card", "Read the card and check access");
        job.line(&format!("installing {target} into {}", output.display()));
        // The lock declares every file and its size: the byte total progress
        // is measured against. Without it, progress counts files.
        let declared = published_lock(&target, None, &hub)
            .as_ref()
            .and_then(declared_bytes);
        match declared {
            Some(bytes) => job.line(&format!(
                "the card's lock declares {} to fetch",
                human_bytes(Some(bytes as i64))
            )),
            None => job.line("the card's lock was not readable first; progress counts files"),
        }
        let tracker = Mutex::new(FetchTracker::new(declared));
        job.set_progress(Some(tracker.lock().unwrap().progress()));
        let observer = job.observer();
        let on_line = |line: &str| {
            observer.line(line);
            let mut tracker = tracker.lock().unwrap();
            if let Some((name, label)) = tracker.observe(line) {
                observer.step(name, label);
            }
            observer.set_progress(Some(tracker.progress()));
        };
        let on_stage = |stage: InstallStage| match stage {
            InstallStage::Documents => {
                observer.step("documents", "Fetch the chat policy and measurements")
            }
            InstallStage::Record => observer.step("record", "Record the install"),
        };
        let outcome =
            install_and_record_observed(&target, Some(&state), &options, &hub, &on_line, &on_stage);
        job.finish(match outcome {
            Ok((result, model)) => {
                job.line(&format!(
                    "installed {} ({})",
                    model.repo,
                    human_bytes(model.size_bytes)
                ));
                if let Some(summary) = result.profile_documents.summary() {
                    job.line(&format!("policy: {summary}"));
                }
                for warning in result.profile_documents.warnings() {
                    job.line(&format!("warning: {warning}"));
                }
                if let Some(summary) = result.documents.summary() {
                    job.line(&format!("scenario: {summary}"));
                }
                if let Some(warning) = &result.documents.warning {
                    job.line(&format!("warning: {warning}"));
                }
                if !result
                    .profile_documents
                    .has(crate::profile::CONFIG_DOCUMENT)
                {
                    job.set_hint(
                        "Installed without llm_config.json: the chat has no policy for this \
                         package (thinking toggle, tool-call format).",
                    );
                }
                Ok(json!({"install": model.describe(), "result": result.describe()}))
            }
            Err(error) => {
                job.set_hint(&install_hint(&error.message, &target));
                Err(error.message)
            }
        });
    });
    Ok(accepted(
        json!({"job": id, "kind": "install", "repo": repo}),
    ))
}

/// `/api/install {path, name?}`: registers a local profile the dashboard
/// lists, in place (`install --from-path`); the page cannot name any other path.
fn start_register(
    context: &Arc<Context>,
    body: &Map<String, Value>,
    path: &str,
) -> std::result::Result<Reply, Reply> {
    let wanted = crate::deployment::absolute(Path::new(path));
    let Some(local) = crate::local::discover(&context.state_root, &context.profile_roots)
        .into_iter()
        .find(|local| local.directory == wanted)
    else {
        return Err(invalid(&format!(
            "{path} is not a local profile this dashboard lists"
        )));
    };
    let name = text_field(body, "name");
    if let Some(name) = &name {
        if !valid_repo(name) {
            return Err(invalid(&format!(
                "{name} is not an owner/name to register under"
            )));
        }
    }
    let name = name.or_else(|| local.registered.as_ref().map(|r| r.repo.clone()));
    let directory = wanted.to_string_lossy().into_owned();
    let retry = retry_of("/api/install", json!({"path": directory, "name": name}));
    let mut job = context
        .jobs
        .start_with("install", &directory, retry)
        .map_err(|message| conflict(&message))?;
    job.set_fact("source", json!("path"));
    let id = job.id.clone();
    let worker = context.clone();
    std::thread::spawn(move || {
        job.begin();
        job.step("read", "Read the package");
        job.set_progress(Some(Progress::of(
            0.0,
            3.0,
            "steps",
            Some("Read the package".into()),
        )));
        let ctx = worker.cli();
        let outcome = cli::register_path(&ctx, &directory, name.as_deref(), false);
        job.finish(match outcome {
            Ok(registered) => {
                let documents = &registered.profile_documents;
                job.end_step(
                    "read",
                    "Read the package",
                    StepState::Done,
                    Some(registered.model.card_path().to_string_lossy().into_owned()),
                );
                job.end_step(
                    "documents",
                    "Add the chat policy and measurements",
                    if documents.has(crate::profile::CONFIG_DOCUMENT) {
                        StepState::Done
                    } else {
                        StepState::Skipped
                    },
                    documents.summary(),
                );
                job.end_step(
                    "record",
                    "Record the install",
                    StepState::Done,
                    Some(format!("as {}", registered.model.repo)),
                );
                job.line(&format!(
                    "registered {} as {} in place",
                    registered.source.display(),
                    registered.model.repo
                ));
                for warning in documents.warnings() {
                    job.line(&format!("warning: {warning}"));
                }
                if let Some(hint) = registered.policy_hint() {
                    job.set_hint(&hint);
                }
                job.set_progress(Some(Progress::of(3.0, 3.0, "steps", None)));
                Ok(registered.describe())
            }
            Err(error) => {
                job.set_hint("Check that the directory holds a readable model.json, then retry.");
                Err(error.message)
            }
        });
    });
    Ok(accepted(
        json!({"job": id, "kind": "install", "path": wanted.to_string_lossy()}),
    ))
}

fn running_message(live: &Live) -> String {
    let repo = live.repo.clone().unwrap_or_else(|| "a profile".into());
    let endpoint = live.endpoint.clone().unwrap_or_default();
    match live.kind {
        Kind::Owned => format!(
            "{repo} is already being served by this launcher at {endpoint}; unload it first \
             (`loom-launch unload`)"
        ),
        _ => {
            let pid = live
                .fact("pid")
                .as_i64()
                .map(|p| format!(" (pid {p})"))
                .unwrap_or_default();
            format!("a launch of {repo} is already running at {endpoint}{pid}; stop it first")
        }
    }
}

fn start_serve(
    context: &Arc<Context>,
    body: &Map<String, Value>,
    trusted: bool,
) -> std::result::Result<Reply, Reply> {
    if context.hosted.is_some() {
        let live = context.live();
        return Err(conflict(&format!(
            "this dashboard is served by the running launch of {} at {}; stop it, then use \
             `loom-launch launcher start` to load models from the page",
            live.repo.as_deref().unwrap_or("a profile"),
            live.endpoint.as_deref().unwrap_or("this address"),
        )));
    }
    let live = context.live();
    if live.kind != Kind::None {
        return Err(conflict(&running_message(&live)));
    }
    // A local profile is named by its directory. The page may only load one
    // the dashboard lists; the launcher's own command line (`start`) may
    // load any package directory it resolved.
    let local_path = text_field(body, "path");
    let repo = match &local_path {
        Some(_) => text_field(body, "repo").unwrap_or_default(),
        None => repo_field(body)?,
    };
    let request = ContextRequest {
        context_length: int_field(body, "context")?,
        max_tokens: int_field(body, "max_tokens")?,
        parallel: int_field(body, "parallel")?,
        draft_depth: int_field(body, "draft_depth")?,
        checkpoint_capacity: int_field(body, "checkpoint_capacity")?,
        session_cache_capacity: int_field(body, "session_cache")?,
        prefix_cache_capacity: int_field(body, "prefix_cache")?,
        prefill_capacity: int_field(body, "prefill_capacity")?,
        ..Default::default()
    };
    let port = match int_field(body, "port")? {
        Some(port) if (1..=65535).contains(&port) => Some(port as u16),
        Some(_) => return Err(invalid("port must be between 1 and 65535")),
        None => None,
    };
    let wait = match get(body, "wait") {
        Value::Null => None,
        value => match number_value(value) {
            Some(seconds) if seconds > 0.0 => Some(seconds),
            _ => return Err(invalid("wait must be a positive number of seconds")),
        },
    };
    let device = text_field(body, "device");
    let allow_web = truthy(get(body, "allow_web"));
    let ctx = context.cli();
    let target = match &local_path {
        Some(path) => {
            let wanted = crate::deployment::absolute(Path::new(path));
            let listed = trusted
                || crate::local::discover(&context.state_root, &context.profile_roots)
                    .into_iter()
                    .any(|local| local.directory == wanted);
            if !listed {
                return Err(invalid(&format!(
                    "{path} is not a local profile this dashboard lists"
                )));
            }
            wanted.to_string_lossy().into_owned()
        }
        None => repo.clone(),
    };
    let resolved = cli::resolve(
        &ctx,
        &ProfileArgs {
            target: Some(target),
            revision: None,
            preset: None,
            offline: true,
            loom_fetch: None,
        },
        false,
    )
    .map_err(|error| conflict(&error.message))?;
    let prepared = cli::prepare(
        &resolved,
        &request,
        device.as_deref(),
        cli::Spelling::Fields,
    )
    .map_err(|error| invalid(&error.message))?;
    let mut retry_body = body.clone();
    retry_body.remove("port");
    let source = if trusted { "start" } else { "dashboard" };
    let load = Load {
        resolved,
        prepared,
        device,
        port,
        allow_web,
        workers: int_field(body, "workers")?,
        pool_backing: text_field(body, "pool_backing"),
        memory_bytes: int_field(body, "memory_bytes")?,
        wait,
        source,
        retry: retry_of("/api/serve", Value::Object(retry_body)),
        on_planned: None,
    };
    let accepted_load = context.load(load).map_err(|message| conflict(&message))?;
    Ok(accepted(accepted_load))
}

impl Context {
    /// Starts loading a model as a `serve` job and answers at once with the
    /// job's id, the repo, the native port and the planned settings; the
    /// job runs until the model serves (or fails). Refused while a model is
    /// loaded or another install, load or benchmark holds the slot.
    pub fn load(self: &Arc<Self>, load: Load) -> std::result::Result<Value, String> {
        if self.hosted.is_some() {
            return Err("a launch's own dashboard does not load models".into());
        }
        let live = self.live();
        if live.kind != Kind::None {
            return Err(running_message(&live));
        }
        let port = match load.port.or_else(|| {
            self.service
                .as_ref()
                .map(|s| s.server_port)
                .filter(|p| *p != 0)
        }) {
            Some(port) => port,
            None => free_port().ok_or("no free loopback port for the server")?,
        };
        let repo = load.resolved.profile.repo.clone();
        let mut job = self.jobs.start_with("serve", &repo, load.retry.clone())?;
        job.set_fact("allow_web", json!(load.allow_web));
        job.set_fact("source", json!(load.source));
        let id = job.id.clone();
        let described = load.prepared.settings().describe();
        let scenario = load.prepared.scenario().map(|s| s.describe());
        let allow_web = load.allow_web;
        let worker = self.clone();
        std::thread::spawn(move || {
            job.begin();
            let outcome = launch(&worker, &mut job, load, port);
            // A cancel meant for this load must not reach the next one.
            worker.cancel_load.store(false, Ordering::SeqCst);
            if let Err(error) = &outcome {
                job.set_hint(&launch_hint(&error.message));
            }
            job.finish(outcome.map_err(|error| error.message));
        });
        Ok(json!({
            "job": id, "kind": "serve", "repo": repo, "port": port, "settings": described,
            "scenario": scenario, "allow_web": allow_web,
        }))
    }
}

/// What to do after a launch failed, by what failed.
fn launch_hint(message: &str) -> String {
    if message.contains(crate::server::CANCELLED) {
        "The load was cancelled by an unload; nothing was left running.".into()
    } else if message.contains("did not become ready within") {
        "The model was still preparing when the wait ended; the serve log shows how far it \
         got. Retry, or launch from a terminal with a longer --wait."
            .into()
    } else if message.contains("cannot run") {
        "loom-serve could not be started: install the Loom SDK or start the dashboard with \
         --loom-serve PATH, then retry."
            .into()
    } else {
        "The server's own words are in serve.log (Logs tab). Nothing was left running; fix \
         the cause, then retry."
            .into()
    }
}

/// Starts one model and waits until it serves (runs on the job's thread),
/// deriving the job's stages from the server's output and events. The model
/// is served behind the launcher's listener: its route points at the model
/// once it is ready and is cleared when it goes away.
fn launch(
    context: &Arc<Context>,
    job: &mut jobs::JobHandle,
    load: Load,
    port: u16,
) -> Result<Value> {
    let Load {
        resolved,
        prepared,
        device,
        allow_web,
        workers,
        pool_backing,
        memory_bytes,
        wait,
        source,
        on_planned,
        ..
    } = load;
    let service = context.service.clone().unwrap_or_default();
    let front = context.front.as_ref().map(|front| {
        let listener = context.listener.lock().unwrap().clone();
        crate::server::Front {
            endpoint: context.endpoint().unwrap_or_default(),
            bind_host: listener
                .get("bind_host")
                .and_then(Value::as_str)
                .unwrap_or("127.0.0.1")
                .to_string(),
            port: listener.get("port").and_then(Value::as_u64).unwrap_or(0) as u16,
            route: front.forwarder.route.clone(),
        }
    });
    let settings = prepared.settings().clone();
    let options = LaunchOptions {
        device,
        port,
        workers,
        pool_backing,
        memory_bytes,
        proxy: false,
        auth_token: None,
        startup_timeout: wait.unwrap_or(service.startup_timeout),
        loom_serve: service.loom_serve.clone(),
        hub_endpoint: context.hub_endpoint.clone(),
        allow_web: context.allow_web || allow_web,
        profile_roots: context.profile_roots.clone(),
        launcher_command: service.command.clone(),
        front,
        ..Default::default()
    };
    let tracker = Arc::new(Mutex::new(LaunchTracker::new(now_seconds())));
    let publish = |observer: &jobs::JobObserver, tracker: &LaunchTracker| {
        observer.set_steps(tracker.steps());
        observer.set_progress(Some(tracker.progress()));
    };
    publish(&job.observer(), &tracker.lock().unwrap());
    let mut metadata = Map::new();
    metadata.insert("repo".into(), json!(resolved.profile.repo));
    metadata.insert("source".into(), json!(source));
    let run = RunStore::new(&context.state_root).create(None, metadata)?;
    let plan = prepared.plan(&resolved, run, &options)?;
    if let Some(on_planned) = on_planned {
        on_planned(&plan);
    }
    if plan.scenario.is_some() {
        job.line(&format!(
            "running the package's own deployment {} across {}",
            plan.deployment_name(),
            plan.device_uri
        ));
    }
    job.line(&format!(
        "launching {} on {} (context {}, max {} output tokens, port {port}{})",
        resolved.profile.repo,
        plan.device_uri,
        settings.context_length.unwrap_or(0),
        settings.max_tokens.unwrap_or(0),
        if options.allow_web {
            ", web access on"
        } else {
            ""
        },
    ));
    job.line(&plan.command.join(" "));
    let (observer, lines) = (job.observer(), tracker.clone());
    let echo = service.echo;
    let supervisor = Supervisor::new(
        plan,
        options,
        &context.state_root,
        Some(Box::new(move |line: &str| {
            observer.line(line);
            if echo {
                use std::io::Write;
                let mut stdout = std::io::stdout().lock();
                let _ = writeln!(stdout, "  | {line}");
                let _ = stdout.flush();
            }
            let mut tracker = lines.lock().unwrap();
            if tracker.line(line, now_seconds()) {
                observer.set_steps(tracker.steps());
                observer.set_progress(Some(tracker.progress()));
            }
        })),
    );
    let (observer, events) = (job.observer(), tracker.clone());
    let (state_root, run_id) = (context.state_root.clone(), supervisor.run().run_id);
    let repo = resolved.profile.repo.clone();
    supervisor.on_event(Box::new(move |event: &Map<String, Value>| {
        if str_value(get(event, "event")) == Some("spawned") {
            crate::launcher::note_model(
                &state_root,
                Some(
                    json!({"pid": get(event, "pid"), "listen": format!("127.0.0.1:{port}"),
                            "run_id": run_id, "repo": repo}),
                ),
            );
        }
        let mut tracker = events.lock().unwrap();
        if tracker.event(event, now_seconds()) {
            observer.set_steps(tracker.steps());
            observer.set_progress(Some(tracker.progress()));
        }
    }));
    *context.preparing.lock().unwrap() = Some(supervisor.clone());
    if context.closed() || context.cancel_load.load(Ordering::SeqCst) {
        // The launcher is shutting down, or an unload came while this load
        // was being planned: nothing starts.
        supervisor.cancel(Duration::from_secs(1));
    }
    let started = supervisor.start();
    if started.is_ok() && supervisor.cancelled() {
        // Unloaded in the moment between ready and recorded.
        supervisor.stop(Duration::from_secs(30));
        context.preparing.lock().unwrap().take();
        crate::launcher::note_model(&context.state_root, None);
        return Err(Error::launch(crate::server::CANCELLED));
    }
    if let Err(error) = started {
        context.preparing.lock().unwrap().take();
        crate::launcher::note_model(&context.state_root, None);
        return Err(error);
    }
    let owned = Arc::new(Owned {
        supervisor: supervisor.clone(),
        allow_web,
        unloading: AtomicBool::new(false),
    });
    // Recorded as loaded before it stops counting as preparing, so an unload
    // in between always finds it.
    *context.owned.lock().unwrap() = Some(owned.clone());
    context.preparing.lock().unwrap().take();
    let watcher = Arc::downgrade(context);
    let handle = std::thread::spawn(move || {
        let code = owned.supervisor.wait_exit();
        if let Some(context) = watcher.upgrade() {
            crate::launcher::note_model(&context.state_root, None);
            let requested = owned.unloading.load(Ordering::SeqCst)
                || owned.supervisor.cancelled()
                || context.closed()
                || signals::stop_requested();
            if !requested {
                context.model_exited(&owned.supervisor, code);
            }
            // Released last: once the page shows no model, the exit is on record.
            {
                let mut slot = context.owned.lock().unwrap();
                if slot.as_ref().is_some_and(|o| Arc::ptr_eq(o, &owned)) {
                    slot.take();
                }
            }
            context.stopping.lock().unwrap().take();
        }
    });
    {
        let mut watchers = context.watchers.lock().unwrap();
        watchers.retain(|watcher| !watcher.is_finished());
        watchers.push(handle);
    }
    let run = supervisor.run();
    {
        let mut tracker = tracker.lock().unwrap();
        tracker.ready(now_seconds(), &supervisor.endpoint());
        publish(&job.observer(), &tracker);
    }
    job.line(&format!("serving at {}", supervisor.endpoint()));
    Ok(json!({
        "repo": supervisor.plan().profile.repo,
        "endpoint": supervisor.endpoint(),
        "server_endpoint": supervisor.server_endpoint(),
        "run": run.run_id,
        "pid": supervisor.pid(),
        "prepare_seconds": get(&run.metadata, "prepare_seconds"),
    }))
}

fn free_port() -> Option<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").ok()?;
    listener.local_addr().ok().map(|a| a.port())
}

/// The exit status a run's supervisor recorded once its server stopped,
/// waiting up to `wait` for it to be written.
fn recorded_exit_code(state_root: &Path, run_id: &str, wait: Duration) -> Option<i64> {
    let run = RunStore::new(state_root).open(run_id);
    let deadline = std::time::Instant::now() + wait;
    loop {
        let metadata = run.read_metadata();
        if str_value(get(&metadata, "state")) == Some("stopped") {
            return strict_int(get(&metadata, "exit_code"));
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// `/api/unload` (and `/api/stop`): unloads the model (SIGTERM, SIGKILL
/// after 30 s) or cancels one still preparing, as a job. A launcher keeps
/// serving afterwards; a hosted page goes down with its launch.
fn stop(context: &Arc<Context>, body: &Map<String, Value>) -> std::result::Result<Reply, Reply> {
    let grace = Duration::from_secs_f64(
        number_value(get(body, "wait"))
            .unwrap_or(30.0)
            .clamp(0.0, 600.0),
    );
    let live = context.live();
    let kind_name = if context.is_launcher() {
        "unload"
    } else {
        "stop"
    };
    if let Some(id) = context
        .jobs
        .active("unload")
        .or_else(|| context.jobs.active("stop"))
    {
        return Err(conflict(&format!(
            "{} is already {} (job {id})",
            live.repo.as_deref().unwrap_or("the model"),
            if context.is_launcher() {
                "unloading"
            } else {
                "stopping"
            }
        )));
    }
    let retry = retry_of("/api/unload", json!({}));
    if live.kind == Kind::None {
        let preparing = context.preparing.lock().unwrap().clone();
        let loading = context
            .jobs
            .running()
            .filter(|(kind, _)| kind == "serve")
            .map(|(_, target)| target);
        if preparing.is_some() || loading.is_some() {
            // A load whose supervisor does not exist yet sees this flag as
            // soon as it does; one already preparing is cancelled directly.
            context.cancel_load.store(true, Ordering::SeqCst);
            let repo = preparing
                .as_ref()
                .map(|s| s.plan().profile.repo.clone())
                .or(loading)
                .unwrap_or_default();
            let mut job = context
                .jobs
                .start_with(kind_name, &repo, retry)
                .map_err(|message| conflict(&message))?;
            let id = job.id.clone();
            let cancelled = repo.clone();
            let worker = context.clone();
            std::thread::spawn(move || {
                job.begin();
                job.step("cancel", "Cancel the preparation");
                job.set_progress(Some(Progress::indeterminate(
                    "steps",
                    Some("Asking the server to stop preparing".into()),
                )));
                // The load's own job ends (as cancelled) once its server is gone.
                let deadline = std::time::Instant::now() + grace + Duration::from_secs(15);
                while worker.jobs.active("serve").is_some() && std::time::Instant::now() < deadline
                {
                    let current = worker.preparing.lock().unwrap().clone();
                    if let Some(supervisor) = current.filter(|s| !s.cancelled()) {
                        supervisor.cancel(grace);
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                job.end_step(
                    "cancel",
                    "Cancel the preparation",
                    StepState::Done,
                    Some("the server was stopped before it served".into()),
                );
                job.finish(Ok(json!({"cancelled": cancelled, "repo": cancelled})));
            });
            return Ok(accepted(
                json!({"stopping": repo, "kind": "preparing", "job": id}),
            ));
        }
        return Err(conflict(if context.is_launcher() {
            "no model is loaded"
        } else {
            "no launch is running"
        }));
    }
    let repo = live.repo.clone().unwrap_or_default();
    {
        let mut stopping = context.stopping.lock().unwrap();
        if stopping.is_some() {
            return Err(conflict(&format!("{repo} is already stopping")));
        }
        *stopping = Some(repo.clone());
    }
    let mut job = match context.jobs.start_with(kind_name, &repo, retry) {
        Ok(job) => job,
        Err(message) => {
            context.stopping.lock().unwrap().take();
            return Err(conflict(&message));
        }
    };
    let id = job.id.clone();
    let pid = strict_int(&live.fact("pid"));
    let run_id = string(&live.fact("run_id"));
    let worker = context.clone();
    let kind = live.kind;
    std::thread::spawn(move || {
        job.begin();
        job.step("drain", "Ask the server to drain");
        job.set_progress(Some(Progress::indeterminate(
            "steps",
            Some("Asking the server to drain".into()),
        )));
        let (code, forced) = match kind {
            Kind::Hosted => {
                // The supervisor of this process sees its server exit and ends
                // the launch; answer the page before that happens.
                std::thread::sleep(Duration::from_millis(250));
                let forced = pid.is_some_and(|pid| stop_pid(pid, grace));
                (None, forced)
            }
            Kind::Owned => {
                let owned = worker.owned.lock().unwrap().clone();
                let code = owned.map(|owned| {
                    owned.unloading.store(true, Ordering::SeqCst);
                    let started = std::time::Instant::now();
                    let code = owned.supervisor.stop(grace);
                    (code, started.elapsed() >= grace && !grace.is_zero())
                });
                worker.owned.lock().unwrap().take();
                match code {
                    Some((code, forced)) => (Some(i64::from(code)), forced),
                    None => (None, false),
                }
            }
            _ => match stop_current(&worker.state_root, grace) {
                crate::server::StopOutcome::Stopped { forced, .. } => (None, forced),
                _ => (None, false),
            },
        };
        job.end_step(
            "drain",
            "Ask the server to drain",
            StepState::Done,
            Some(if forced {
                format!(
                    "it did not exit within {:.0} s, so it was killed",
                    grace.as_secs_f64()
                )
            } else {
                "SIGTERM sent; the server finished its requests".into()
            }),
        );
        job.step("exit", "Wait for the exit");
        let code = code.or_else(|| {
            run_id.as_deref().and_then(|run| {
                recorded_exit_code(&worker.state_root, run, Duration::from_secs(10))
            })
        });
        job.end_step(
            "exit",
            "Stopped",
            StepState::Done,
            Some(match code {
                Some(code) => format!("exit status {code}"),
                None => "the exit status was not recorded".into(),
            }),
        );
        worker.stopping.lock().unwrap().take();
        if worker.service.as_ref().is_some_and(|s| s.echo) {
            println!(
                "  model     unloaded {repo}; the launcher keeps serving at {}",
                worker.endpoint().unwrap_or_default()
            );
        }
        job.finish(Ok(json!({
            "repo": repo, "exit_code": code, "forced": forced, "pid": pid, "run": run_id,
            "launcher": worker.endpoint(),
        })));
    });
    Ok(accepted(
        json!({"stopping": live.repo, "kind": kind.name(), "pid": pid, "job": id}),
    ))
}

/// SIGTERM, then SIGKILL after `grace`; returns whether it had to kill.
fn stop_pid(pid: i64, grace: Duration) -> bool {
    if signals::exited(pid) {
        return false;
    }
    signals::send(pid as i32, signals::SIGTERM);
    let deadline = std::time::Instant::now() + grace;
    while std::time::Instant::now() < deadline {
        if signals::exited(pid) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    signals::send(pid as i32, signals::SIGKILL);
    true
}

fn start_bench(
    context: &Arc<Context>,
    body: &Map<String, Value>,
) -> std::result::Result<Reply, Reply> {
    let suite = text_field(body, "suite").unwrap_or_else(|| "quick".into());
    if suite_metrics(&suite).is_none() {
        let names: Vec<&str> = SUITES.iter().map(|(n, _)| *n).collect();
        return Err(invalid(&format!(
            "unknown suite '{suite}'; choose one of {}",
            names.join(", ")
        )));
    }
    let repeats = int_field(body, "repeats")?.unwrap_or(1);
    if !(1..=9).contains(&repeats) {
        return Err(invalid("repeats must be between 1 and 9"));
    }
    let live = context.live();
    let (Some(endpoint), Some(running)) = (live.endpoint.clone(), live.repo.clone()) else {
        return Err(conflict(
            "no launch is running; launch a profile, then run a suite against it",
        ));
    };
    if let Some(requested) = text_field(body, "repo") {
        if requested != running {
            return Err(conflict(&format!(
                "the running launch serves {running}, not {requested}"
            )));
        }
    }
    let retry = retry_of(
        "/api/bench",
        json!({"suite": suite, "repeats": repeats, "repo": running}),
    );
    let mut job = context
        .jobs
        .start_with("bench", &format!("{running} ({suite})"), retry)
        .map_err(|message| conflict(&message))?;
    job.set_fact("suite", json!(suite));
    let id = job.id.clone();
    let worker = context.clone();
    let repo = running.clone();
    let suite_name = suite.clone();
    std::thread::spawn(move || {
        job.begin();
        let outcome = benchmark(
            &worker,
            &job,
            &live,
            &endpoint,
            &repo,
            &suite_name,
            repeats as usize,
        );
        if outcome.is_err() {
            job.set_hint(
                "The launch may have stopped or refused the requests; check it on the \
                 Overview, then retry.",
            );
        }
        job.finish(outcome.map_err(|error| error.message));
    });
    Ok(accepted(json!({
        "job": id, "kind": "bench", "repo": running, "suite": suite, "repeats": repeats,
    })))
}

/// A suite's step label: `decode, short prompt`, or `warm-up`.
fn bench_step_label(metric: &str) -> String {
    match metric {
        "warm-up" => "Warm up".into(),
        other => {
            let label = crate::profile::metric_label(other);
            let mut chars = label.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        }
    }
}

/// Runs one suite and stores the record where `bench --list` finds it.
fn benchmark(
    context: &Context,
    job: &jobs::JobHandle,
    live: &Live,
    endpoint: &str,
    repo: &str,
    suite: &str,
    repeats: usize,
) -> Result<Value> {
    let ctx = context.cli();
    let client = Client::new(endpoint, live.token.as_deref(), REQUEST_TIMEOUT)?;
    let profile = cli::profile_for_endpoint(
        &ctx,
        &ProfileArgs {
            target: Some(repo.to_string()),
            revision: None,
            preset: None,
            offline: true,
            loom_fetch: None,
        },
        &client,
        live.record.as_ref(),
    );
    let model = cli::served_model(&client, &profile, live.record.as_ref());
    let observer = job.observer();
    let progress = |message: &str| observer.line(message);
    let on_event = |event: &BenchEvent| match event {
        BenchEvent::Begin {
            done,
            total,
            metric,
            repeat,
            repeats,
        } => {
            observer.step(metric, &bench_step_label(metric));
            observer.set_progress(Some(Progress::of(
                *done as f64,
                *total as f64,
                "requests",
                Some(format!("{metric} {repeat}/{repeats}")),
            )));
        }
        BenchEvent::Measured {
            metric,
            value,
            unit,
            note,
        } => {
            let (state, detail) = match value {
                Some(value) if *unit == "fraction" => (StepState::Done, grouped(*value, 3)),
                Some(value) => (StepState::Done, format!("{} {unit}", grouped(*value, 1))),
                None => (
                    StepState::Failed,
                    note.unwrap_or("no usable measurement").to_string(),
                ),
            };
            observer.end_step(metric, &bench_step_label(metric), state, Some(detail));
        }
    };
    let mut runner = Runner::new(&client, &profile);
    if let Some(model) = &model {
        runner.model = model.clone();
    }
    runner.repeats = repeats;
    runner.max_sessions =
        strict_int(&live.fact("max_sessions")).or(profile.execution.as_ref().and_then(|e| e.rows));
    runner.on_progress = Some(&progress);
    runner.on_event = Some(&on_event);
    job.line(&format!(
        "benchmarking {repo} ({suite} suite, {repeats} repeat(s)) at {endpoint}"
    ));
    let mut record = runner.run(suite)?;
    record.insert("endpoint".into(), json!(endpoint));
    job.step("store", "Store the record");
    let path = BenchmarkStore::new(&context.state_root).save(&mut record)?;
    job.line(&format!("stored {}", path.display()));
    Ok(json!({
        "id": get(&record, "id"),
        "path": path.to_string_lossy(),
        "ok": get(&record, "ok"),
        "suite": suite,
        "profile": repo,
        "comparison": get(&record, "comparison"),
    }))
}

/// `/api/jobs/dismiss {id}`: hides a finished job's row.
fn dismiss(context: &Context, body: &Map<String, Value>) -> std::result::Result<Reply, Reply> {
    let id = text_field(body, "id").ok_or_else(|| invalid("id is required"))?;
    match context.jobs.dismiss(&id) {
        Ok(()) => Ok(json_response(200, &json!({"dismissed": id}))),
        Err(message) if message.starts_with("no job") => {
            Err(error_reply(404, "NOT_FOUND", &message))
        }
        Err(message) => Err(conflict(&message)),
    }
}

/// `/api/fetch {url}`: one public page as text for the chat's `web_fetch`
/// tool, when web access is on (see `dashboard/fetch.rs` for the address
/// checks that keep it off this machine's own services). Each fetch is a job
/// row, so the page shows what the model read.
fn web_fetch(context: &Context, body: &Map<String, Value>) -> Reply {
    let live = context.live();
    let access = context.web_access(&live);
    if !access.allowed {
        return error_reply(
            403,
            "PERMISSION_DENIED",
            access.reason.as_deref().unwrap_or("web access is off"),
        );
    }
    let fetcher = fetch::Fetcher {
        #[cfg(test)]
        stubs: context.fetch_stubs.lock().unwrap().clone(),
    };
    let url = str_value(get(body, "url")).map(str::trim).unwrap_or("");
    let mut job = match context
        .jobs
        .start("fetch", if url.is_empty() { "(no URL)" } else { url })
    {
        Ok(job) => job,
        Err(message) => return conflict(&message),
    };
    job.begin();
    job.step("fetch", "Fetch the page for the chat");
    match fetch::fetch_document(&fetcher, Some(url)) {
        Ok(document) => {
            let summary = format!(
                "HTTP {}{}",
                py_status(&document["status"]),
                number_value(&document["bytes"])
                    .map(|bytes| format!(", {}", human_bytes(Some(bytes as i64))))
                    .unwrap_or_default()
            );
            job.end_step(
                "fetch",
                "Fetch the page for the chat",
                StepState::Done,
                Some(summary),
            );
            job.finish(Ok(json!({
                "status": document["status"], "bytes": document["bytes"],
                "title": document["title"], "truncated": document["truncated"],
            })));
            json_response(200, &document)
        }
        Err(failure) => {
            job.set_hint("The model was told why; it may try another address.");
            job.finish(Err(failure.message.clone()));
            failure.reply()
        }
    }
}

fn py_status(value: &Value) -> String {
    crate::values::py_str(value)
}

/// `/api/chat`: the body goes to the native listener's chat route and the
/// answer (server-sent events or JSON) comes back unchanged, counted by the
/// launch's instrumentation like any other request.
fn chat(context: &Arc<Context>, mut request: tiny_http::Request, headers: &[(String, String)]) {
    let body = match read_body(&mut request).and_then(|b| parse_object(&b)) {
        Ok(body) => body,
        Err(reply) => {
            let _ = request.respond(reply);
            return;
        }
    };
    let live = context.live();
    let Some(forwarder) = live.forwarder.clone() else {
        let _ = request.respond(conflict("no launch is running; launch a profile first"));
        return;
    };
    if !matches!(get(&body, "messages"), Value::Array(m) if !m.is_empty()) {
        let _ = request.respond(invalid("messages must be a non-empty array"));
        return;
    }
    let mut payload = body;
    if str_value(get(&payload, "model")).is_none() {
        if let Some(model) = context.served_model(&live) {
            payload.insert("model".into(), json!(model));
        }
    }
    if truthy(get(&payload, "stream")) && !payload.contains_key("stream_options") {
        payload.insert("stream_options".into(), json!({"include_usage": true}));
    }
    let mut outgoing = vec![
        ("content-type".to_string(), "application/json".to_string()),
        (
            "accept".to_string(),
            header(headers, "accept")
                .unwrap_or("text/event-stream, application/json")
                .to_string(),
        ),
    ];
    if let (Kind::External, Some(token)) = (live.kind, &live.token) {
        outgoing.push(("authorization".into(), format!("Bearer {token}")));
    }
    if let Some(session) = header(headers, "x-loom-session") {
        outgoing.push(("x-loom-session".into(), session.to_string()));
    }
    let client = request.remote_addr().map(|a| a.to_string());
    let bytes = serde_json::to_vec(&Value::Object(payload)).expect("JSON values serialize");
    forwarder.forward(
        request,
        "POST",
        "/v1/chat/completions",
        &outgoing,
        bytes,
        client,
    );
}

// -- standalone ------------------------------------------------------------------

/// Prints where a listener on a non-loopback address answers from other
/// machines, and warns when nothing guards it. With an API key the dashboard
/// URLs carry it as `?access_token=`, so they open signed in; `keyed` says
/// whether this listener can require one (`serve --api-key`).
pub fn print_network_access(bind_host: &str, port: u16, token: Option<&str>, keyed: bool) {
    if crate::network::is_loopback_host(bind_host) {
        return;
    }
    let urls = crate::network::reachable_urls(bind_host, port);
    if urls.is_empty() {
        println!("  network   listening on {bind_host}:{port}, but no interface address was found");
    }
    for url in &urls {
        match token {
            Some(token) => println!(
                "  network   {url}?access_token={}",
                crate::proxy::percent_encode(token)
            ),
            None => println!("  network   {url}"),
        }
    }
    match (token, keyed) {
        (Some(_), _) => {}
        (None, true) => println!(
            "  warning   anyone who can reach {bind_host}:{port} can use this endpoint and its \
             dashboard; pass --api-key to require a key"
        ),
        (None, false) => println!(
            "  warning   anyone who can reach {bind_host}:{port} can install, launch, stop and \
             chat from this dashboard"
        ),
    }
}
