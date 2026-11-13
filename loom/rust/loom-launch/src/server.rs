//! Supervision of one native server, its logs and its instrumentation.
//!
//! The supervisor owns one model's process lifetime, readiness and run
//! directory; the native executable owns devices, compilation and inference.
//! Every line the server writes is kept and its JSONL events are indexed
//! separately. Under a launcher service the model is served behind the
//! launcher's listener (a [`Front`]); a launch started for one command (`chat`
//! and `bench` with nothing running) fronts it with a proxy of its own.

use crate::client::Client;
use crate::deployment::{deployment_document, write_deployment, ContextSettings};
use crate::errors::{Error, Result};
use crate::hardware::{detect, Hardware};
use crate::install::find_tool;
use crate::metrics::Metrics;
use crate::profile::Profile;
use crate::proxy::{Proxy, ProxyOptions, Route, StatusProvider, Target, Upstream};
use crate::scenario::Scenario;
use crate::signals;
use crate::store::{read_json, JsonlWriter, RunDirectory};
use crate::values::{arr, get, iso_utc, now_seconds, obj, round_to, str_value, strict_int};
use serde_json::{json, Map, Value};
use std::collections::{BTreeSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const SERVE_BINARY: &str = "loom-serve";
/// Why a launch that was unloaded while preparing did not become ready.
pub const CANCELLED: &str = "the launch was cancelled before the model became ready";
pub const CURRENT_RECORD: &str = "current.json";
const TAIL_LINES: usize = 200;

/// Finds the native server: explicit, `$LOOM_SERVE`, sibling, `PATH`.
pub fn serve_command(explicit: Option<&str>) -> Result<String> {
    find_tool(explicit, "LOOM_SERVE", SERVE_BINARY).ok_or_else(|| {
        Error::launch(
            "loom-serve was not found: install the Loom SDK \
             (`pip install \"hrx-loom[hf]\"`), or pass --loom-serve=PATH",
        )
    })
}

/// The launcher listener a model is served behind instead of a proxy of its
/// own: its address and the route the launcher points at the model.
#[derive(Clone)]
pub struct Front {
    /// The launcher's client-facing address (`http://host:port`).
    pub endpoint: String,
    pub bind_host: String,
    pub port: u16,
    pub route: Arc<Route>,
}

impl std::fmt::Debug for Front {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Front")
            .field("endpoint", &self.endpoint)
            .field("bind_host", &self.bind_host)
            .field("port", &self.port)
            .finish()
    }
}

/// Everything a launch needs beyond the profile itself.
#[derive(Clone, Debug)]
pub struct LaunchOptions {
    pub name: Option<String>,
    pub device: Option<String>,
    /// The native listener's address; the CLI keeps it on loopback so
    /// inference is only reachable through the proxy.
    pub host: String,
    pub port: u16,
    pub workers: Option<i64>,
    pub pool_backing: Option<String>,
    pub memory_bytes: Option<i64>,
    pub proxy: bool,
    /// The address the proxy and its dashboard listen on (`serve --host`).
    pub proxy_host: String,
    pub proxy_port: u16,
    pub auth_token: Option<String>,
    pub startup_timeout: f64,
    pub extra_arguments: Vec<String>,
    pub loom_serve: Option<String>,
    /// Model hub the launch's dashboard searches (default: `$LOOM_HUB_ENDPOINT`).
    pub hub_endpoint: Option<String>,
    /// Let the dashboard fetch public URLs for the chat (`--allow-web`).
    pub allow_web: bool,
    /// Directories the launch's dashboard lists local profiles from.
    pub profile_roots: Vec<PathBuf>,
    /// The `loom-launch` command line that started this launch (an API key
    /// redacted), so a page can show how to restart it with other flags.
    pub launcher_command: Vec<String>,
    /// Serve behind a launcher's listener rather than a proxy of its own.
    pub front: Option<Front>,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        LaunchOptions {
            name: None,
            device: None,
            host: "127.0.0.1".into(),
            port: 8080,
            workers: None,
            pool_backing: None,
            memory_bytes: None,
            proxy: true,
            proxy_host: "127.0.0.1".into(),
            proxy_port: 0,
            auth_token: None,
            startup_timeout: 1800.0,
            extra_arguments: Vec::new(),
            loom_serve: None,
            hub_endpoint: None,
            allow_web: false,
            profile_roots: Vec::new(),
            launcher_command: Vec::new(),
            front: None,
        }
    }
}

/// A command line with the value of `--api-key` replaced by a reference to
/// the variable that can carry it instead.
pub fn redacted_command(arguments: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(arguments.len());
    let mut hide_next = false;
    for argument in arguments {
        if hide_next {
            out.push("\"$LOOM_LAUNCH_API_KEY\"".to_string());
            hide_next = false;
        } else if argument == "--api-key" {
            out.push(argument.clone());
            hide_next = true;
        } else if argument.starts_with("--api-key=") {
            out.push("--api-key=\"$LOOM_LAUNCH_API_KEY\"".to_string());
        } else {
            out.push(argument.clone());
        }
    }
    out
}

impl LaunchOptions {
    pub fn describe(&self) -> Value {
        json!({
            "name": self.name,
            "device": self.device,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "pool_backing": self.pool_backing,
            "memory_bytes": self.memory_bytes,
            "proxy": self.proxy,
            "proxy_host": self.proxy_host,
            "proxy_port": self.proxy_port,
            "auth": self.auth_token.is_some(),
            "allow_web": self.allow_web,
            "startup_timeout": self.startup_timeout,
        })
    }
}

/// A scenario launch: the package's own documents and what the launch changed.
#[derive(Clone, Debug)]
pub struct ScenarioPlan {
    pub scenario: Scenario,
    /// The devices document the server reads: the package's own, or a run
    /// copy carrying `--memory-bytes` / `--pool-backing`.
    pub devices_path: PathBuf,
    /// What the run copy changed (`pool`, `memory_bytes`, `pool_backing`).
    pub overrides: Map<String, Value>,
}

impl ScenarioPlan {
    pub fn describe(&self) -> Value {
        let mut document = self.scenario.describe();
        document["served_devices_path"] = json!(self.devices_path.to_string_lossy());
        document["overrides"] = Value::Object(self.overrides.clone());
        document
    }
}

/// The resolved command, document and addresses of one launch.
#[derive(Clone, Debug)]
pub struct LaunchPlan {
    pub profile: Profile,
    pub run: RunDirectory,
    pub command: Vec<String>,
    /// The deployment the server registers: composed for a profile, the
    /// package's own `deployment.json` for a scenario.
    pub deployment: Value,
    /// The HAL device URI, or the devices a scenario spans (`amdgpu + xdna`).
    pub device_uri: String,
    pub server_host: String,
    pub server_port: u16,
    pub settings: ContextSettings,
    pub scenario: Option<ScenarioPlan>,
}

impl LaunchPlan {
    /// `scenario` for a package's own deployment, else `profile`.
    pub fn kind(&self) -> &'static str {
        if self.scenario.is_some() {
            "scenario"
        } else {
            "profile"
        }
    }

    /// The facts `current.json`, `meta.json` and `/api/status` share.
    pub fn launch_facts(&self) -> Value {
        json!({
            "kind": self.kind(),
            "command": self.command,
            "rows": self.settings.rows,
            "profile_path": self.profile.path,
            "scenario": self.scenario.as_ref().map(ScenarioPlan::describe),
        })
    }

    pub fn deployment_name(&self) -> String {
        str_value(&self.deployment["name"])
            .unwrap_or("")
            .to_string()
    }

    pub fn server_endpoint(&self) -> String {
        format!("http://{}:{}", self.server_host, self.server_port)
    }
}

/// Composes the deployment document and the server command line.
pub fn plan_launch(
    profile: &Profile,
    card_path: &Path,
    settings: &ContextSettings,
    run: RunDirectory,
    options: &LaunchOptions,
    hardware: Option<&Hardware>,
) -> Result<LaunchPlan> {
    let device_uri = match &options.device {
        Some(device) => device.clone(),
        None => match hardware {
            Some(hardware) => hardware.default_device_uri(),
            None => detect().default_device_uri(),
        },
    };
    let document = deployment_document(
        profile,
        card_path,
        settings,
        options.name.as_deref(),
        profile.preset(),
    );
    let deployment_path = write_deployment(&run.deployment_path(), &document)?;
    let mut command = vec![
        serve_command(options.loom_serve.as_deref())?,
        format!("--model={}", deployment_path.display()),
        format!("--listen={}:{}", options.host, options.port),
        format!("--device={device_uri}"),
    ];
    if let Some(workers) = options.workers {
        command.push(format!("--workers={workers}"));
    }
    if let Some(backing) = options.pool_backing.as_deref().filter(|b| !b.is_empty()) {
        command.push(format!("--pool_backing={backing}"));
    }
    // A card records the memory budget it was validated with; use it unless
    // the caller chose one, since the package's own preparation expects it.
    if let Some(memory) = options
        .memory_bytes
        .or(profile.memory_bytes)
        .filter(|m| *m != 0)
    {
        command.push(format!("--memory_bytes={memory}"));
    }
    command.extend(options.extra_arguments.iter().cloned());
    Ok(LaunchPlan {
        profile: profile.clone(),
        run,
        command,
        deployment: document,
        device_uri,
        server_host: options.host.clone(),
        server_port: options.port,
        settings: settings.clone(),
        scenario: None,
    })
}

/// The command line of a scenario: the package's own deployment and devices,
/// no composed document and no `--device`.
pub fn plan_scenario(
    profile: &Profile,
    scenario: &Scenario,
    run: RunDirectory,
    options: &LaunchOptions,
) -> Result<LaunchPlan> {
    let (devices_path, overrides) = scenario.devices_for(
        options.memory_bytes,
        options.pool_backing.as_deref(),
        &run.root,
    )?;
    let mut command = vec![
        serve_command(options.loom_serve.as_deref())?,
        format!("--devices={}", devices_path.display()),
        format!("--model={}", scenario.deployment_path.display()),
        format!("--listen={}:{}", options.host, options.port),
    ];
    if let Some(workers) = options.workers {
        command.push(format!("--workers={workers}"));
    }
    command.extend(options.extra_arguments.iter().cloned());
    Ok(LaunchPlan {
        profile: profile.clone(),
        run,
        command,
        deployment: Value::Object(scenario.deployment.clone()),
        device_uri: scenario.device_text(),
        server_host: options.host.clone(),
        server_port: options.port,
        settings: scenario.settings.clone(),
        scenario: Some(ScenarioPlan {
            scenario: scenario.clone(),
            devices_path,
            overrides,
        }),
    })
}

/// Receives every line the server writes.
pub type OutputCallback = Box<dyn Fn(&str) + Send + Sync>;
/// Receives every event the supervisor records (`launch`, `ready`,
/// `listening`, `registration`, `serving`, `failed`, `stopping`, `stopped`).
pub type EventCallback = Box<dyn Fn(&Map<String, Value>) + Send + Sync>;

struct Inner {
    plan: LaunchPlan,
    options: LaunchOptions,
    state_root: PathBuf,
    run: Mutex<RunDirectory>,
    child: Mutex<Option<Child>>,
    pid: AtomicU32,
    events: Mutex<Option<Arc<JsonlWriter>>>,
    serve_log: Mutex<Option<File>>,
    tail: Mutex<VecDeque<String>>,
    ready_event: AtomicBool,
    server_address: Mutex<Option<String>>,
    failure: Mutex<Option<String>>,
    proxy: Mutex<Option<Proxy>>,
    proxy_address: Mutex<Option<String>>,
    metrics: Arc<Metrics>,
    request_log: Mutex<Option<Arc<JsonlWriter>>>,
    readers: Mutex<Vec<JoinHandle<()>>>,
    on_output: Option<OutputCallback>,
    on_event: Mutex<Option<EventCallback>>,
    started_at: f64,
    exit_code: Mutex<Option<i32>>,
    /// Finishes the run record; a second caller waits for the first.
    shut_down: Once,
    /// Set by [`Supervisor::cancel`]: the launch must not become ready.
    cancelled: AtomicBool,
}

/// Runs one launch: child process, readiness, proxy and shutdown.
#[derive(Clone)]
pub struct Supervisor {
    inner: Arc<Inner>,
}

impl Supervisor {
    pub fn new(
        plan: LaunchPlan,
        options: LaunchOptions,
        state_root: &Path,
        on_output: Option<OutputCallback>,
    ) -> Supervisor {
        let run = plan.run.clone();
        Supervisor {
            inner: Arc::new(Inner {
                plan,
                options,
                state_root: state_root.to_path_buf(),
                run: Mutex::new(run),
                child: Mutex::new(None),
                pid: AtomicU32::new(0),
                events: Mutex::new(None),
                serve_log: Mutex::new(None),
                tail: Mutex::new(VecDeque::new()),
                ready_event: AtomicBool::new(false),
                server_address: Mutex::new(None),
                failure: Mutex::new(None),
                proxy: Mutex::new(None),
                proxy_address: Mutex::new(None),
                metrics: Arc::new(Metrics::default()),
                request_log: Mutex::new(None),
                readers: Mutex::new(Vec::new()),
                on_output,
                on_event: Mutex::new(None),
                started_at: now_seconds(),
                exit_code: Mutex::new(None),
                shut_down: Once::new(),
                cancelled: AtomicBool::new(false),
            }),
        }
    }

    pub fn plan(&self) -> &LaunchPlan {
        &self.inner.plan
    }

    /// Calls `callback` with every event recorded from now on.
    pub fn on_event(&self, callback: EventCallback) {
        *self.inner.on_event.lock().unwrap() = Some(callback);
    }

    /// The options this launch was started with.
    pub fn options(&self) -> &LaunchOptions {
        &self.inner.options
    }

    pub fn run(&self) -> RunDirectory {
        self.inner.run.lock().unwrap().clone()
    }

    pub fn endpoint(&self) -> String {
        if let Some(front) = &self.inner.options.front {
            return front.endpoint.clone();
        }
        self.inner
            .proxy_address
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| self.server_endpoint())
    }

    pub fn server_endpoint(&self) -> String {
        self.inner.plan.server_endpoint()
    }

    /// The client-facing listener: the address it is bound to and its port.
    pub fn listener(&self) -> (String, u16) {
        let inner = &self.inner;
        if let Some(front) = &inner.options.front {
            return (front.bind_host.clone(), front.port);
        }
        match inner.proxy.lock().unwrap().as_ref() {
            Some(proxy) => (inner.options.proxy_host.clone(), proxy.port),
            None => (inner.plan.server_host.clone(), inner.plan.server_port),
        }
    }

    /// URLs other machines reach the client-facing listener at (none when it
    /// is bound to loopback).
    pub fn network_urls(&self) -> Vec<String> {
        let (host, port) = self.listener();
        crate::network::reachable_urls(&host, port)
    }

    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.inner.metrics
    }

    pub fn pid(&self) -> u32 {
        self.inner.pid.load(Ordering::SeqCst)
    }

    /// The request log the proxy writes, once the launch is serving.
    pub fn request_log(&self) -> Option<Arc<JsonlWriter>> {
        self.inner.request_log.lock().unwrap().clone()
    }

    /// True once the server process has exited (or was never started).
    pub fn exited(&self) -> bool {
        self.pid() == 0 || self.child_exit_code().is_some()
    }

    /// The server's exit status, once it exited.
    pub fn exit_status(&self) -> Option<i32> {
        *self.inner.exit_code.lock().unwrap()
    }

    /// Whether [`Supervisor::cancel`] was called.
    pub fn cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Stops a launch that may still be preparing: it will not become ready,
    /// and a server already started is terminated (SIGTERM, then SIGKILL
    /// after `grace`). Safe before, during and after [`Supervisor::start`].
    pub fn cancel(&self, grace: Duration) {
        self.inner.cancelled.store(true, Ordering::SeqCst);
        self.terminate(grace);
    }

    pub fn failure(&self) -> Option<String> {
        self.inner.failure.lock().unwrap().clone()
    }

    pub fn tail(&self, lines: usize) -> Vec<String> {
        let tail = self.inner.tail.lock().unwrap();
        tail.iter()
            .skip(tail.len().saturating_sub(lines))
            .cloned()
            .collect()
    }

    fn update_metadata(&self, fields: Value) {
        let mut run = self.inner.run.lock().unwrap();
        if let Value::Object(fields) = fields {
            for (key, value) in fields {
                run.metadata.insert(key, value);
            }
        }
        let _ = run.save();
    }

    fn record_event(&self, event: Value) {
        record_event(&self.inner, event);
    }

    fn fail(&self, message: &str) {
        {
            let mut failure = self.inner.failure.lock().unwrap();
            if failure.is_none() {
                *failure = Some(message.to_string());
            }
        }
        self.update_metadata(json!({"state": "failed", "error": message}));
        self.record_event(json!({"event": "failed", "error": message}));
    }

    /// Starts the server, waits until the deployment is prepared, starts the
    /// proxy and publishes `current.json`.
    pub fn start(&self) -> Result<()> {
        let inner = &self.inner;
        let run = self.run();
        *inner.events.lock().unwrap() = Some(Arc::new(JsonlWriter::open(&run.event_log())?));
        *inner.serve_log.lock().unwrap() = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(run.serve_log())
            .ok();
        let plan = &inner.plan;
        self.update_metadata(plan.launch_facts());
        self.update_metadata(json!({
            "run_id": run.run_id,
            "repo": plan.profile.repo,
            "revision": plan.profile.revision,
            "model_id": plan.profile.model_id,
            "deployment_name": plan.deployment_name(),
            "command": plan.command,
            "device": plan.device_uri,
            "settings": plan.settings.describe(),
            "deployment": plan.deployment,
            "options": inner.options.describe(),
            "started_at": iso_utc(inner.started_at),
            "launcher_pid": std::process::id(),
            "state": "starting",
        }));
        self.record_event(json!({
            "event": "launch",
            "kind": plan.kind(),
            "repo": plan.profile.repo,
            "command": plan.command,
            "deployment": plan.deployment_name(),
            "device": plan.device_uri,
        }));
        if self.cancelled() {
            let message = CANCELLED.to_string();
            self.cancel_record();
            self.close_streams();
            return Err(Error::launch(message));
        }
        let spawned = crate::install::spawn_retrying(
            Command::new(&plan.command[0])
                .args(&plan.command[1..])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped()),
        );
        let mut child = match spawned {
            Ok(child) => child,
            Err(error) => {
                let message = format!("cannot run {}: {error}", plan.command[0]);
                self.fail(&message);
                self.close_streams();
                return Err(Error::launch(message));
            }
        };
        inner.pid.store(child.id(), Ordering::SeqCst);
        self.record_event(json!({"event": "spawned", "pid": child.id()}));
        let mut readers = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            readers.push(spawn_reader(inner.clone(), Box::new(stdout)));
        }
        if let Some(stderr) = child.stderr.take() {
            readers.push(spawn_reader(inner.clone(), Box::new(stderr)));
        }
        *inner.readers.lock().unwrap() = readers;
        *inner.child.lock().unwrap() = Some(child);
        // A cancel that raced the spawn found no child to stop; stop it now.
        if self.cancelled() {
            self.terminate(Duration::from_secs(10));
        }
        if let Err(error) = self.wait_until_prepared() {
            // A launch that never prepared must not leave its server behind: it
            // owns a device and a port the caller's next launch would find taken.
            self.terminate(Duration::from_secs(10));
            self.close_streams();
            return Err(error);
        }
        if let Err(error) = self.publish(&run) {
            self.fail(&error.message);
            self.terminate(Duration::from_secs(10));
            self.unroute();
            self.close_streams();
            return Err(error);
        }
        Ok(())
    }

    fn cancel_record(&self) {
        {
            let mut failure = self.inner.failure.lock().unwrap();
            if failure.is_none() {
                *failure = Some(CANCELLED.to_string());
            }
        }
        self.update_metadata(json!({"state": "cancelled", "error": CANCELLED}));
        self.record_event(json!({"event": "cancelled"}));
    }

    /// Takes the launcher's route back from this launch, if it still has it.
    fn unroute(&self) {
        if let Some(front) = &self.inner.options.front {
            front.route.clear_if(&self.run().run_id);
        }
    }

    /// After readiness: the proxy, the run record and `current.json`.
    fn publish(&self, run: &RunDirectory) -> Result<()> {
        let inner = &self.inner;
        let plan = &inner.plan;
        if let Some(front) = &inner.options.front {
            // The launcher's listener carries the model from now on: the route
            // is set before current.json announces it, so a client that read
            // the record never meets the "no model" answer.
            let request_log = Arc::new(JsonlWriter::open(&run.request_log())?);
            *inner.request_log.lock().unwrap() = Some(request_log.clone());
            front.route.point(Target {
                upstream: Upstream {
                    host: plan.server_host.clone(),
                    port: plan.server_port,
                },
                request_log: Some(request_log),
                owner: run.run_id.clone(),
            });
        } else if inner.options.proxy {
            self.start_proxy()?;
        }
        let elapsed = now_seconds() - inner.started_at;
        let (bind_host, _) = self.listener();
        let urls = self.network_urls();
        self.update_metadata(json!({
            "state": "serving",
            "endpoint": self.endpoint(),
            "bind_host": bind_host,
            "urls": urls,
            "server_endpoint": self.server_endpoint(),
            "pid": self.pid(),
            "ready_at": iso_utc(now_seconds()),
            "prepare_seconds": round_to(elapsed, 3),
        }));
        let mut record = json!({
                "run_id": run.run_id,
                "run_directory": run.root.to_string_lossy(),
                "endpoint": self.endpoint(),
                "bind_host": bind_host,
                "urls": urls,
                "server_endpoint": self.server_endpoint(),
                "pid": self.pid(),
                "launcher_pid": std::process::id(),
                "repo": plan.profile.repo,
                "deployment": plan.deployment_name(),
                "device": plan.device_uri,
                "started_at": iso_utc(inner.started_at),
                "context_length": plan.settings.context_length,
                "max_tokens": plan.settings.max_tokens,
                "max_sessions": plan.settings.max_sessions.or(plan.settings.rows),
                "auth": inner.options.auth_token.is_some(),
                "allow_web": inner.options.allow_web,
                "launcher_command": inner.options.launcher_command,
        });
        if let Value::Object(facts) = plan.launch_facts() {
            for (key, value) in facts {
                record[key] = value;
            }
        }
        write_current(&inner.state_root, &record)?;
        self.record_event(json!({
            "event": "serving",
            "endpoint": self.endpoint(),
            "server_endpoint": self.server_endpoint(),
        }));
        Ok(())
    }

    fn wait_until_prepared(&self) -> Result<()> {
        let inner = &self.inner;
        let client = Client::new(&self.server_endpoint(), None, Duration::from_secs(5))?;
        let names: BTreeSet<String> = [
            inner.plan.deployment_name(),
            inner.plan.profile.model_id.clone(),
        ]
        .into_iter()
        .filter(|n| !n.is_empty())
        .collect();
        let timeout = inner.options.startup_timeout.max(0.0);
        let deadline = Instant::now() + Duration::from_secs_f64(timeout);
        let mut announced_fallback = false;
        let mut phase: Option<String> = None;
        loop {
            // Checked first: a cancelled server's exit is the cancel's doing.
            if self.cancelled() {
                self.cancel_record();
                return Err(Error::launch(CANCELLED));
            }
            if let Some(code) = self.child_exit_code() {
                // The readers may still hold the server's last lines.
                self.join_readers();
                let tail = self.tail(12).join("\n");
                let mut message =
                    format!("the server exited before the model became ready (exit code {code})");
                if !tail.is_empty() {
                    message.push_str(&format!("; last output:\n{tail}"));
                }
                self.fail(&message);
                return Err(Error::launch(message));
            }
            if signals::stop_requested() {
                let message = "interrupted before the model became ready".to_string();
                self.fail(&message);
                return Err(Error::launch(message));
            }
            if Instant::now() >= deadline {
                let message = format!("the model did not become ready within {timeout:.0}s");
                self.fail(&message);
                return Err(Error::launch(message));
            }
            // The server announces its listener with a ready event; a healthy
            // listener also counts, so a changed event format cannot stall the wait.
            let listening =
                inner.ready_event.load(Ordering::SeqCst) || client.healthy(Duration::from_secs(1));
            if !listening {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            if !inner.ready_event.load(Ordering::SeqCst) && !announced_fallback {
                announced_fallback = true;
                self.record_event(json!({"event": "listening", "detail": "no ready event seen; /healthz answered"}));
            }
            let models = client
                .models_document(Some(Duration::from_secs(5)))
                .unwrap_or_default();
            match registration(&models, &names) {
                Registration::Ready => return Ok(()),
                Registration::Failed(message) => {
                    let message = format!("model preparation failed: {message}");
                    self.fail(&message);
                    return Err(Error::launch(message));
                }
                Registration::Pending => {}
            }
            let reported = registration_phase(&models, &names);
            if reported.is_some() && reported != phase {
                self.record_event(json!({"event": "registration", "state": reported}));
                phase = reported;
            }
            let status = client.server_status(Some(Duration::from_secs(5)));
            if let Registration::Failed(message) = registration(&status, &names) {
                let message = format!("model preparation failed: {message}");
                self.fail(&message);
                return Err(Error::launch(message));
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn start_proxy(&self) -> Result<()> {
        let inner = &self.inner;
        let run = self.run();
        let request_log = Arc::new(JsonlWriter::open(&run.request_log())?);
        *inner.request_log.lock().unwrap() = Some(request_log.clone());
        let mut options = ProxyOptions::new(Upstream {
            host: inner.plan.server_host.clone(),
            port: inner.plan.server_port,
        });
        options.bind_host = inner.options.proxy_host.clone();
        options.bind_port = inner.options.proxy_port;
        options.allow_web = inner.options.allow_web;
        options.auth_token = inner.options.auth_token.clone();
        options.request_log = Some(request_log);
        options.status_provider = Some(self.status_provider());
        options.dashboard_state = Some(inner.state_root.clone());
        options.hub_endpoint = inner.options.hub_endpoint.clone();
        options.profile_roots = inner.options.profile_roots.clone();
        let proxy = Proxy::start(options, inner.metrics.clone())?;
        *inner.proxy_address.lock().unwrap() = Some(proxy.address().to_string());
        *inner.proxy.lock().unwrap() = Some(proxy);
        Ok(())
    }

    /// The launch facts `/loom/status` adds to the proxy's own.
    pub fn status_provider(&self) -> StatusProvider {
        let weak = Arc::downgrade(&self.inner);
        Arc::new(move || {
            let Some(inner) = weak.upgrade() else {
                return Map::new();
            };
            let bind_host = if inner.options.proxy {
                &inner.options.proxy_host
            } else {
                &inner.plan.server_host
            };
            let run = inner.run.lock().unwrap().clone();
            let plan = &inner.plan;
            let mut status = Map::new();
            let mut launch = json!({
                    "repo": plan.profile.repo,
                    "revision": plan.profile.revision,
                    "deployment": plan.deployment_name(),
                    "device": plan.device_uri,
                    "run_directory": run.root.to_string_lossy(),
                    "started_at": get(&run.metadata, "started_at"),
                    "pid": inner.pid.load(Ordering::SeqCst),
                    "server_endpoint": plan.server_endpoint(),
                    "bind_host": bind_host,
                    "context_length": plan.settings.context_length,
                    "max_tokens": plan.settings.max_tokens,
                    "allow_web": inner.options.allow_web,
                    "launcher_command": inner.options.launcher_command,
            });
            let facts = plan.launch_facts();
            for key in ["kind", "command", "rows", "profile_path"] {
                launch[key] = facts[key].clone();
            }
            status.insert("launch".into(), launch);
            status.insert("scenario".into(), facts["scenario"].clone());
            status.insert("settings".into(), plan.settings.describe());
            status
        })
    }

    fn child_exit_code(&self) -> Option<i32> {
        let mut child = self.inner.child.lock().unwrap();
        let process = child.as_mut()?;
        match process.try_wait() {
            Ok(Some(status)) => {
                let code = exit_code(status);
                *self.inner.exit_code.lock().unwrap() = Some(code);
                Some(code)
            }
            _ => None,
        }
    }

    /// Blocks until the server exits, stopping it when SIGINT/SIGTERM arrives.
    pub fn wait(&self) -> i32 {
        signals::install();
        let mut stopping = false;
        loop {
            if let Some(code) = self.child_exit_code() {
                self.shutdown();
                return code;
            }
            if signals::stop_requested() && !stopping {
                stopping = true;
                let supervisor = self.clone();
                std::thread::spawn(move || supervisor.terminate(Duration::from_secs(30)));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Blocks until the server exits on its own or is stopped by someone
    /// else, then finishes the run record; signals are the launcher's to
    /// handle. Returns the exit status.
    pub fn wait_exit(&self) -> i32 {
        loop {
            if let Some(code) = self.child_exit_code() {
                self.shutdown();
                return code;
            }
            if self.inner.shut_down.is_completed() {
                return self.exit_status().unwrap_or(0);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// SIGTERM, a grace period, then SIGKILL; a no-op once the server exited.
    pub fn terminate(&self, grace: Duration) {
        let pid = self.pid() as i32;
        if pid == 0 || self.child_exit_code().is_some() {
            return;
        }
        self.record_event(json!({"event": "stopping"}));
        signals::send(pid, signals::SIGTERM);
        let deadline = Instant::now() + grace;
        while Instant::now() < deadline {
            if self.child_exit_code().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        signals::send(pid, signals::SIGKILL);
        if let Some(child) = self.inner.child.lock().unwrap().as_mut() {
            if let Ok(status) = child.wait() {
                *self.inner.exit_code.lock().unwrap() = Some(exit_code(status));
            }
        }
    }

    /// Stops the server and finishes the run record.
    pub fn stop(&self, grace: Duration) -> i32 {
        self.terminate(grace);
        self.shutdown();
        self.inner.exit_code.lock().unwrap().unwrap_or(0)
    }

    /// Waits until the output readers reach the end of the server's streams,
    /// so the tail and serve.log hold everything it wrote.
    fn join_readers(&self) {
        let readers: Vec<JoinHandle<()>> = self.inner.readers.lock().unwrap().drain(..).collect();
        for reader in readers {
            let _ = reader.join();
        }
    }

    fn close_streams(&self) {
        let inner = &self.inner;
        if let Some(mut proxy) = inner.proxy.lock().unwrap().take() {
            proxy.stop();
        }
        self.join_readers();
        if let Some(log) = inner.request_log.lock().unwrap().take() {
            log.close();
        }
        inner.serve_log.lock().unwrap().take();
    }

    /// Finishes the run record once (idempotent). Every caller returns only
    /// once it is finished, also while another thread is finishing it.
    pub fn shutdown(&self) {
        self.inner.shut_down.call_once(|| {
            let code = *self.inner.exit_code.lock().unwrap();
            self.unroute();
            self.close_streams();
            self.update_metadata(json!({
                "state": "stopped",
                "stopped_at": iso_utc(now_seconds()),
                "exit_code": code,
            }));
            self.record_event(json!({"event": "stopped", "exit_code": code}));
            if let Some(events) = self.inner.events.lock().unwrap().take() {
                events.close();
            }
            let run_id = self.run().run_id;
            clear_current_if(&self.inner.state_root, &run_id);
        });
    }
}

fn exit_code(status: std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
}

fn record_event(inner: &Inner, event: Value) {
    let mut record = Map::new();
    record.insert("time".into(), json!(iso_utc(now_seconds())));
    if let Value::Object(fields) = event {
        for (key, value) in fields {
            record.insert(key, value);
        }
    }
    if let Some(events) = inner.events.lock().unwrap().as_ref() {
        events.write(&Value::Object(record.clone()));
    }
    if let Some(callback) = inner.on_event.lock().unwrap().as_ref() {
        callback(&record);
    }
}

fn spawn_reader(inner: Arc<Inner>, stream: Box<dyn std::io::Read + Send>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut buffer = Vec::new();
        loop {
            buffer.clear();
            match reader.read_until(b'\n', &mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let line = String::from_utf8_lossy(&buffer)
                .trim_end_matches(['\n', '\r'])
                .to_string();
            if let Some(log) = inner.serve_log.lock().unwrap().as_mut() {
                let _ = writeln!(log, "{line}");
                let _ = log.flush();
            }
            {
                let mut tail = inner.tail.lock().unwrap();
                tail.push_back(line.clone());
                while tail.len() > TAIL_LINES {
                    tail.pop_front();
                }
            }
            if let Some(event) = parse_event(&line) {
                if str_value(get(&event, "event")) == Some("ready") {
                    if let Some(address) = str_value(get(&event, "address")) {
                        *inner.server_address.lock().unwrap() = Some(address.to_string());
                    }
                    inner.ready_event.store(true, Ordering::SeqCst);
                }
                // The server's own event: written to the log, not echoed to
                // the event observer, which sees the line as output instead.
                if let Some(events) = inner.events.lock().unwrap().as_ref() {
                    let mut record = Map::new();
                    record.insert("time".into(), json!(iso_utc(now_seconds())));
                    record.extend(event);
                    events.write(&Value::Object(record));
                }
            }
            if let Some(callback) = &inner.on_output {
                callback(&line);
            }
        }
    })
}

/// A server output line that is one JSON object.
pub fn parse_event(line: &str) -> Option<Map<String, Value>> {
    let line = line.trim();
    if !line.starts_with('{') || !line.ends_with('}') {
        return None;
    }
    match serde_json::from_str::<Value>(line) {
        Ok(Value::Object(document)) => Some(document),
        _ => None,
    }
}

/// Where a deployment's registration stands, as the server reports it.
#[derive(Debug, PartialEq)]
pub enum Registration {
    Ready,
    Pending,
    Failed(String),
}

/// Reads the registration of any of `names` from a `/v1/models` or
/// `/v1/status` document. Listed with no phase (or `ready`) is ready; `failed`,
/// `error` and `release_failed` are failures; anything else is still pending.
pub fn registration(document: &Map<String, Value>, names: &BTreeSet<String>) -> Registration {
    for entry in status_entries(document) {
        let reported = str_value(get(&entry, "name")).or_else(|| str_value(get(&entry, "id")));
        if reported.is_some_and(|r| !names.is_empty() && !names.contains(r)) {
            continue;
        }
        if let Some(state) = entry_state(&entry) {
            if matches!(state.as_str(), "failed" | "error" | "release_failed") {
                let text = error_text(&entry)
                    .or_else(|| error_text(obj(get(&entry, "loom"))))
                    .unwrap_or(state);
                return Registration::Failed(text);
            }
        }
    }
    let mut ready = false;
    // A top-level listing entry is ready unless it or its nested view says otherwise.
    for entry in arr(get(document, "data")).iter().map(obj) {
        let reported = str_value(get(entry, "id")).or_else(|| str_value(get(entry, "name")));
        if !reported.is_some_and(|r| names.contains(r)) {
            continue;
        }
        let nested = obj(get(entry, "loom"));
        let state = entry_state(entry).or_else(|| entry_state(nested));
        if matches!(state.as_deref(), None | Some("ready")) {
            ready = true;
        }
    }
    for (name, value) in obj(get(document, "models")) {
        if !names.contains(name) {
            continue;
        }
        let state = entry_state(obj(value));
        if matches!(state.as_deref(), Some("ready")) {
            ready = true;
        }
    }
    if ready {
        Registration::Ready
    } else {
        Registration::Pending
    }
}

/// The phase a still-pending registration of any of `names` reports
/// (`preparing`, `loading`, ...), when the document names one.
pub fn registration_phase(
    document: &Map<String, Value>,
    names: &BTreeSet<String>,
) -> Option<String> {
    status_entries(document)
        .iter()
        .filter(|entry| {
            str_value(get(entry, "name"))
                .or_else(|| str_value(get(entry, "id")))
                .is_some_and(|reported| names.contains(reported))
        })
        .find_map(entry_state)
        .filter(|state| state != "ready")
}

/// Every per-model record a status document carries, however it nests it.
pub fn status_entries(status: &Map<String, Value>) -> Vec<Map<String, Value>> {
    fn visit(
        document: &Map<String, Value>,
        name: Option<String>,
        entries: &mut Vec<Map<String, Value>>,
    ) {
        let mut entry = document.clone();
        let resolved = str_value(get(&entry, "name"))
            .or_else(|| str_value(get(&entry, "id")))
            .or_else(|| str_value(get(&entry, "model")))
            .map(str::to_string)
            .or(name);
        if let Some(resolved) = &resolved {
            if str_value(get(&entry, "name")).is_none() {
                entry.insert("name".into(), json!(resolved));
            }
        }
        let nested = obj(get(&entry, "loom")).clone();
        entries.push(entry);
        if !nested.is_empty() {
            visit(&nested, resolved, entries);
        }
    }
    let mut entries = Vec::new();
    for (name, value) in obj(get(status, "models")) {
        let candidates: Vec<&Value> = match value {
            Value::Array(items) => items.iter().collect(),
            other => vec![other],
        };
        for entry in candidates {
            if let Value::Object(entry) = entry {
                let named = str_value(get(entry, "name"))
                    .map(str::to_string)
                    .or(Some(name.clone()));
                visit(entry, named, &mut entries);
            }
        }
    }
    for key in ["data", "registrations", "deployments"] {
        for entry in arr(get(status, key)) {
            if let Value::Object(entry) = entry {
                visit(entry, None, &mut entries);
            }
        }
    }
    match get(status, "loom") {
        Value::Object(loom) => visit(loom, None, &mut entries),
        Value::Array(items) => {
            for entry in items {
                if let Value::Object(entry) = entry {
                    visit(entry, None, &mut entries);
                }
            }
        }
        _ => {}
    }
    entries
}

/// The lowercase phase an entry reports under `state`, `status` or `registration`.
pub fn entry_state(entry: &Map<String, Value>) -> Option<String> {
    for key in ["state", "status", "registration"] {
        let mut value = get(entry, key);
        if let Value::Object(nested) = value {
            value = nested
                .get("phase")
                .filter(|v| crate::values::truthy(v))
                .or_else(|| nested.get("state"))
                .unwrap_or(&Value::Null);
        }
        if let Value::String(text) = value {
            return Some(text.to_lowercase());
        }
    }
    None
}

/// The message of an error an entry reports, however it is spelled.
pub fn error_text(entry: &Map<String, Value>) -> Option<String> {
    let error = get(entry, "error");
    if let Some(message) = str_value(error) {
        return Some(message.to_string());
    }
    let detail = obj(error);
    str_value(get(detail, "message"))
        .or_else(|| str_value(get(detail, "details")))
        .map(str::to_string)
}

// -- the current-launch record ---------------------------------------------

pub fn current_path(root: &Path) -> PathBuf {
    root.join(CURRENT_RECORD)
}

/// Writes `current.json` with mode 0600 from the moment it exists.
pub fn write_current(root: &Path, record: &Value) -> Result<PathBuf> {
    write_private(root, CURRENT_RECORD, record)
}

/// Writes `root/name` atomically with mode 0600 from the moment it exists.
pub fn write_private(root: &Path, name: &str, record: &Value) -> Result<PathBuf> {
    use std::os::unix::fs::OpenOptionsExt;
    let path = root.join(name);
    let io =
        |error: std::io::Error| Error::launch(format!("cannot write {}: {error}", path.display()));
    std::fs::create_dir_all(root).map_err(io)?;
    let temporary = root.join(format!(
        "{name}.{}.{}.tmp",
        std::process::id(),
        crate::values::random_hex_id()
    ));
    let _ = std::fs::remove_file(&temporary);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary)
        .map_err(io)?;
    file.write_all((crate::values::dumps_sorted(record) + "\n").as_bytes())
        .map_err(io)?;
    drop(file);
    std::fs::rename(&temporary, &path).map_err(io)?;
    Ok(path)
}

pub fn read_current(root: &Path) -> Option<Map<String, Value>> {
    Some(read_json(&current_path(root))).filter(|r| !r.is_empty())
}

pub fn clear_current(root: &Path) {
    let _ = std::fs::remove_file(current_path(root));
}

/// Clears `current.json` only when it still describes `run_id`.
pub fn clear_current_if(root: &Path, run_id: &str) {
    if let Some(record) = read_current(root) {
        if str_value(get(&record, "run_id")).is_none_or(|id| id == run_id) {
            clear_current(root);
        }
    }
}

/// The endpoint of a live launch, when one is running and answering.
pub fn running_endpoint(root: &Path) -> Option<(String, Map<String, Value>)> {
    let record = read_current(root)?;
    if let Some(pid) = strict_int(get(&record, "pid")) {
        if signals::exited(pid) {
            return None;
        }
    }
    let endpoint = str_value(get(&record, "endpoint"))?.to_string();
    let client = Client::new(&endpoint, None, Duration::from_secs(2)).ok()?;
    if !client.healthy(Duration::from_secs(2)) {
        return None;
    }
    Some((endpoint, record))
}

/// What [`stop_current`] did.
#[derive(Clone, Debug, PartialEq)]
pub enum StopOutcome {
    NotRunning,
    NoPid,
    Stopped {
        pid: i64,
        repo: String,
        run_id: String,
        forced: bool,
    },
}

/// Stops the launch recorded in `current.json`: SIGTERM to the server, SIGKILL
/// after `wait`, then a moment for its launcher to finish the run record; a
/// record nobody else will clear is cleared here.
pub fn stop_current(root: &Path, wait: Duration) -> StopOutcome {
    let Some(record) = read_current(root) else {
        return StopOutcome::NotRunning;
    };
    let repo = crate::values::py_str(get(&record, "repo"));
    let run_id = str_value(get(&record, "run_id")).unwrap_or("").to_string();
    let launcher = strict_int(get(&record, "launcher_pid"));
    let Some(pid) = strict_int(get(&record, "pid")) else {
        return StopOutcome::NoPid;
    };
    let mut forced = false;
    if !signals::exited(pid) {
        signals::send(pid as i32, signals::SIGTERM);
        let deadline = Instant::now() + wait;
        while Instant::now() < deadline && !signals::exited(pid) {
            std::thread::sleep(Duration::from_millis(100));
        }
        if !signals::exited(pid) {
            signals::send(pid as i32, signals::SIGKILL);
            forced = true;
        }
    }
    // The supervising launcher finishes the run record and clears current.json;
    // give it a moment, then clear a record nobody else will.
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        let still = read_current(root)
            .is_some_and(|r| str_value(get(&r, "run_id")).unwrap_or("") == run_id);
        let launcher_gone = launcher.is_none_or(signals::exited);
        if !still {
            break;
        }
        if launcher_gone && launcher.is_some() {
            break;
        }
        if launcher.is_none() && Instant::now() + Duration::from_secs(5) > deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    clear_current_if(root, &run_id);
    StopOutcome::Stopped {
        pid,
        repo,
        run_id,
        forced,
    }
}

/// Summarizes the launcher's persistent state for `status --json`.
pub fn describe_state(root: &Path) -> Value {
    json!({
        "state_directory": root.to_string_lossy(),
        "current": read_current(root),
        "launcher": crate::launcher::read_launcher(root).map(|r| crate::launcher::public_record(&r)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::{context_settings, ContextRequest};
    use crate::profile::ProfileDocuments;
    use crate::store::tests::scratch;
    use crate::store::RunStore;
    use crate::testing::{FakeModelConfig, FakeOpenAIServer};

    fn profile() -> Profile {
        let card = json!({
            "schema_version": 1,
            "model": {"id": "test/model", "name": "Test"},
            "services": {"chat": {"kind": "text", "version": 1}},
            "presets": {"chat": {"service": "chat", "execution": {
                "rows": 2, "context_capacity": 8192, "prefill_capacity": 256,
                "max_tokens": 512, "draft_depth": 0}}},
        });
        Profile::from_documents(
            "loom-fun/test-loom",
            &ProfileDocuments {
                model_card: obj(&card).clone(),
                ..Default::default()
            },
            None,
        )
    }

    fn planned(root: &Path, options: LaunchOptions) -> LaunchPlan {
        let run = RunStore::new(root)
            .create(Some("run-1"), Map::new())
            .unwrap();
        let card = root.join("model.json");
        std::fs::write(&card, "{\"model\": {\"id\": \"test/model\"}}").unwrap();
        let parsed = profile();
        let request = ContextRequest {
            context_length: Some(4096),
            ..Default::default()
        };
        plan_launch(
            &parsed,
            &card,
            &context_settings(&parsed, &request).unwrap(),
            run,
            &options,
            None,
        )
        .unwrap()
    }

    fn options() -> LaunchOptions {
        LaunchOptions {
            loom_serve: Some("/bin/true".into()),
            port: 9100,
            device: Some("amdgpu://".into()),
            ..Default::default()
        }
    }

    #[test]
    fn a_plan_writes_the_deployment_and_builds_the_command() {
        let root = scratch("server-plan");
        let plan = planned(&root, options());
        assert_eq!(plan.command[0], "/bin/true");
        assert!(plan
            .command
            .contains(&format!("--model={}", plan.run.deployment_path().display())));
        assert!(plan
            .command
            .contains(&"--listen=127.0.0.1:9100".to_string()));
        assert!(plan.command.contains(&"--device=amdgpu://".to_string()));
        let document = read_json(&plan.run.deployment_path());
        assert!(Path::new(document["model"]["path"].as_str().unwrap()).is_absolute());
        assert_eq!(document["name"], "loom-fun-test-loom");
        assert_eq!(document["model"]["preset"], "chat");
        assert_eq!(document["definition"]["context_capacity"], 4096);
        assert_eq!(document["definition"]["max_tokens"], 512);
    }

    #[test]
    fn launch_options_add_server_resources() {
        let root = scratch("server-options");
        let plan = planned(
            &root,
            LaunchOptions {
                workers: Some(4),
                pool_backing: Some("elastic".into()),
                memory_bytes: Some(1024),
                extra_arguments: vec!["--wait_mode=blocking".into()],
                ..options()
            },
        );
        for flag in [
            "--workers=4",
            "--pool_backing=elastic",
            "--memory_bytes=1024",
            "--wait_mode=blocking",
        ] {
            assert!(plan.command.contains(&flag.to_string()), "{flag}");
        }
    }

    #[test]
    fn the_card_memory_budget_is_used_when_none_is_chosen() {
        let root = scratch("server-memory");
        let run = RunStore::new(&root)
            .create(Some("run-1"), Map::new())
            .unwrap();
        let parsed = Profile::from_documents(
            "loom-fun/test-loom",
            &ProfileDocuments {
                model_card: obj(&json!({"model": {"id": "test/model"}})).clone(),
                info: obj(&json!({"card": {"memory_bytes": 49000000000_i64}})).clone(),
                ..Default::default()
            },
            None,
        );
        let settings = context_settings(&parsed, &ContextRequest::default()).unwrap();
        let plan = plan_launch(
            &parsed,
            &root.join("model.json"),
            &settings,
            run,
            &options(),
            None,
        )
        .unwrap();
        assert!(plan
            .command
            .contains(&"--memory_bytes=49000000000".to_string()));
    }

    #[test]
    fn server_events_are_parsed_only_when_they_are_json_objects() {
        let event = parse_event("{\"event\":\"ready\",\"address\":\"127.0.0.1:8080\"}").unwrap();
        assert_eq!(event["address"], "127.0.0.1:8080");
        assert!(parse_event("plain output").is_none());
        assert!(parse_event("[1,2]").is_none());
        assert!(parse_event("{not json}").is_none());
    }

    #[test]
    fn a_pending_registration_names_its_phase() {
        let names: BTreeSet<String> = ["m".to_string()].into_iter().collect();
        let preparing =
            obj(&json!({"data": [{"id": "m", "loom": {"registration": "loading"}}]})).clone();
        assert_eq!(
            registration_phase(&preparing, &names).as_deref(),
            Some("loading")
        );
        let ready = obj(&json!({"data": [{"id": "m", "loom": {"registration": "ready"}}]})).clone();
        assert_eq!(registration_phase(&ready, &names), None);
        let other =
            obj(&json!({"data": [{"id": "x", "loom": {"registration": "loading"}}]})).clone();
        assert_eq!(registration_phase(&other, &names), None);
    }

    #[test]
    fn an_api_key_never_appears_in_a_recorded_command() {
        let arguments: Vec<String> = [
            "loom-launch",
            "serve",
            "a/b",
            "--api-key",
            "secret",
            "--api-key=two",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let redacted = redacted_command(&arguments).join(" ");
        assert!(
            !redacted.contains("secret") && !redacted.contains("two"),
            "{redacted}"
        );
        assert!(
            redacted.contains("--api-key \"$LOOM_LAUNCH_API_KEY\""),
            "{redacted}"
        );
    }

    #[test]
    fn status_entries_cover_every_shape_the_server_reports() {
        let document = obj(&json!({
            "models": {"scout": {"state": "ready"}},
            "data": [{"name": "other", "status": "Failed"}],
        }))
        .clone();
        let mut names: Vec<String> = status_entries(&document)
            .iter()
            .map(|e| e["name"].as_str().unwrap().to_string())
            .collect();
        names.sort();
        assert_eq!(names, ["other", "scout"]);
        assert_eq!(
            entry_state(obj(&json!({"registration": "READY"}))).as_deref(),
            Some("ready")
        );
        assert_eq!(entry_state(&Map::new()), None);
    }

    #[test]
    fn a_nested_registration_failure_is_reported_not_waited_out() {
        let document = obj(&json!({
            "object": "list",
            "data": [{
                "object": "model",
                "id": "Qwen-Qwen2.5-0.5B-Instruct",
                "loom": {"registration": "failed", "error": {"code": 5, "message": "no such file"}},
            }],
        }))
        .clone();
        let names = BTreeSet::from(["Qwen-Qwen2.5-0.5B-Instruct".to_string()]);
        assert_eq!(
            registration(&document, &names),
            Registration::Failed("no such file".into())
        );
        assert_eq!(
            error_text(obj(&json!({"error": "plain message"}))).as_deref(),
            Some("plain message")
        );
        assert_eq!(error_text(&Map::new()), None);
    }

    #[test]
    fn a_preparing_registration_is_pending_and_a_ready_one_is_ready() {
        let names = BTreeSet::from(["m".to_string()]);
        let preparing =
            obj(&json!({"data": [{"id": "m", "loom": {"registration": "preparing"}}]})).clone();
        assert_eq!(registration(&preparing, &names), Registration::Pending);
        let ready =
            obj(&json!({"data": [{"id": "m", "loom": {"kind": "text", "registration": "ready"}}]}))
                .clone();
        assert_eq!(registration(&ready, &names), Registration::Ready);
        let plain = obj(&json!({"data": [{"id": "m", "loom": {"kind": "text"}}]})).clone();
        assert_eq!(registration(&plain, &names), Registration::Ready);
        let other = obj(&json!({"data": [{"id": "x", "loom": {"kind": "text"}}]})).clone();
        assert_eq!(registration(&other, &names), Registration::Pending);
        let status = obj(&json!({"models": {"m": {"state": "ready", "name": "m"}}})).clone();
        assert_eq!(registration(&status, &names), Registration::Ready);
    }

    #[test]
    fn the_current_record_round_trips_with_private_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let root = scratch("server-current");
        assert!(read_current(&root).is_none());
        write_current(&root, &json!({"pid": std::process::id(), "endpoint": "http://127.0.0.1:1", "repo": "a/b", "run_id": "r"})).unwrap();
        let mode = std::fs::metadata(current_path(&root))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
        assert_eq!(read_current(&root).unwrap()["repo"], "a/b");
        assert_eq!(
            describe_state(&root)["state_directory"],
            root.to_string_lossy().as_ref()
        );
        clear_current_if(&root, "other");
        assert!(read_current(&root).is_some());
        clear_current_if(&root, "r");
        assert!(read_current(&root).is_none());
    }

    #[test]
    fn a_running_endpoint_is_only_reported_while_it_answers() {
        let root = scratch("server-running");
        let server = FakeOpenAIServer::start(FakeModelConfig::default()).unwrap();
        write_current(
            &root,
            &json!({"pid": std::process::id(), "endpoint": server.url, "repo": "a/b"}),
        )
        .unwrap();
        let (endpoint, record) = running_endpoint(&root).unwrap();
        assert_eq!(endpoint, server.url);
        assert_eq!(record["repo"], "a/b");
        write_current(
            &root,
            &json!({"pid": 999999999, "endpoint": server.url, "repo": "a/b"}),
        )
        .unwrap();
        assert!(running_endpoint(&root).is_none());
        write_current(
            &root,
            &json!({"pid": std::process::id(), "endpoint": "http://127.0.0.1:1", "repo": "a/b"}),
        )
        .unwrap();
        assert!(running_endpoint(&root).is_none());
    }
}
