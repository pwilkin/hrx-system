//! The launcher service: one long-lived process per state directory that owns
//! the client-facing listener, the dashboard, the instrumentation (`/loom/*`,
//! `requests.jsonl`), the jobs, and the lifecycle of the model servers it
//! loads as children. It never needs a model: with none loaded, `/v1/*`
//! answers 503 and everything else keeps working.
//!
//! `launcher.json` records the service (pid, endpoints, bind host, start
//! time, whether an API key is required, and a control token its own command
//! line presents); `current.json` keeps describing the loaded model.

use crate::client::Client;
use crate::dashboard::{Context, Service};
use crate::errors::{Error, Result};
use crate::metrics::Metrics;
use crate::proxy::{Proxy, ProxyOptions, Route};
use crate::server::{read_current, stop_current, write_private, StopOutcome};
use crate::signals;
use crate::store::read_json;
use crate::values::{get, iso_utc, now_seconds, str_value, strict_int};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const LAUNCHER_RECORD: &str = "launcher.json";
/// Set in a launcher started with `--detach`: it records itself as detached
/// and ignores the shell it came from.
pub const DETACHED_ENV: &str = "LOOM_LAUNCH_DETACHED";

/// How to start a launcher service.
#[derive(Clone, Debug)]
pub struct LauncherOptions {
    pub service: Service,
    pub api_key: Option<String>,
    /// Started with `--detach` (in its own session, output in `log`).
    pub detached: bool,
    pub log: Option<PathBuf>,
}

/// A running launcher service.
pub struct Launcher {
    proxy: Proxy,
    context: Arc<Context>,
    state_root: PathBuf,
    record: Map<String, Value>,
}

impl Launcher {
    /// Binds the client-facing listener and records the service in
    /// `launcher.json`. Refuses while another launcher of this state
    /// directory is running.
    pub fn start(options: LauncherOptions) -> Result<Launcher> {
        let service = options.service;
        let root = service.state_root.clone();
        if read_launcher(&root).is_some() && running_launcher(&root).is_none() {
            // A launcher that died without cleaning up: stop what it left.
            stop_launcher(&root, Duration::from_secs(1));
        }
        if let Some(record) = running_launcher(&root) {
            return Err(Error::launcher(format!(
                "a launcher is already running at {} (pid {}); `loom-launch launcher stop` ends it, \
                 `loom-launch start PROFILE` loads a model under it",
                crate::values::py_str(get(&record, "endpoint")),
                crate::values::py_str(get(&record, "pid")),
            )));
        }
        signals::install();
        if options.detached {
            signals::ignore_hangup();
        }
        let token = control_token();
        let mut proxy_options = ProxyOptions::routed(Arc::new(Route::default()));
        proxy_options.bind_host = service.host.clone();
        proxy_options.bind_port = service.port;
        proxy_options.auth_token = options.api_key.clone().filter(|k| !k.is_empty());
        proxy_options.control_token = Some(token.clone());
        proxy_options.hub_endpoint = service.hub_endpoint.clone();
        proxy_options.allow_web = service.allow_web;
        proxy_options.profile_roots = service.profile_roots.clone();
        proxy_options.service = Some(service.clone());
        let proxy = Proxy::start(proxy_options, Arc::new(Metrics::default()))?;
        let context = proxy
            .dashboard()
            .cloned()
            .ok_or_else(|| Error::launch("the launcher's dashboard did not start"))?;
        let endpoint = proxy.address().to_string();
        let record = json!({
            "pid": std::process::id(),
            "endpoint": endpoint,
            "dashboard": format!("{endpoint}/"),
            "bind_host": service.host,
            "port": proxy.port,
            "urls": crate::network::reachable_urls(&service.host, proxy.port),
            "server_port": service.server_port,
            "started_at": iso_utc(now_seconds()),
            "auth": proxy_options_auth(&options.api_key),
            "allow_web": service.allow_web,
            "detached": options.detached,
            "log": options.log.as_ref().map(|p| p.to_string_lossy().into_owned()),
            "command": service.command,
            "state_directory": root.to_string_lossy(),
            "version": env!("CARGO_PKG_VERSION"),
            "control_token": token,
        });
        write_private(&root, LAUNCHER_RECORD, &record)?;
        Ok(Launcher {
            proxy,
            context,
            state_root: root,
            record: crate::values::obj(&record).clone(),
        })
    }

    pub fn endpoint(&self) -> &str {
        self.proxy.address()
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// `launcher.json` as written (with the control token).
    pub fn record(&self) -> &Map<String, Value> {
        &self.record
    }

    /// The address the listener is bound to and its port.
    pub fn listener(&self) -> (String, u16) {
        (
            str_value(get(&self.record, "bind_host"))
                .unwrap_or("127.0.0.1")
                .to_string(),
            self.proxy.port,
        )
    }

    /// Blocks until SIGINT or SIGTERM.
    pub fn wait_for_stop(&self) {
        while !signals::stop_requested() {
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    /// Unloads the model (or cancels one preparing), closes the listener and
    /// clears `launcher.json`. Returns what was unloaded.
    pub fn shutdown(mut self, grace: Duration) -> Option<String> {
        self.context.close();
        let unloaded = self.context.stop_owned(grace);
        self.proxy.stop();
        clear_launcher_if(&self.state_root, i64::from(std::process::id()));
        unloaded
    }
}

fn proxy_options_auth(api_key: &Option<String>) -> bool {
    api_key.as_deref().is_some_and(|k| !k.is_empty())
}

/// 32 random bytes as hex, from the kernel.
fn control_token() -> String {
    use std::io::Read;
    let mut bytes = [0u8; 32];
    let read = std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut bytes));
    if read.is_err() {
        return format!(
            "{}{}",
            crate::values::random_hex_id(),
            crate::values::random_hex_id()
        );
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// -- launcher.json ------------------------------------------------------------

pub fn launcher_path(root: &Path) -> PathBuf {
    root.join(LAUNCHER_RECORD)
}

pub fn read_launcher(root: &Path) -> Option<Map<String, Value>> {
    Some(read_json(&launcher_path(root))).filter(|r| !r.is_empty())
}

/// Removes `launcher.json` when it still names `pid`.
pub fn clear_launcher_if(root: &Path, pid: i64) {
    if let Some(record) = read_launcher(root) {
        if strict_int(get(&record, "pid")).is_none_or(|recorded| recorded == pid) {
            let _ = std::fs::remove_file(launcher_path(root));
        }
    }
}

/// The record of a launcher whose process is alive (a dead one's record is
/// stale and ignored).
pub fn running_launcher(root: &Path) -> Option<Map<String, Value>> {
    let record = read_launcher(root)?;
    let pid = strict_int(get(&record, "pid"))?;
    if signals::exited(pid) {
        return None;
    }
    Some(record)
}

/// `launcher.json` without its control token, for `status --json`.
pub fn public_record(record: &Map<String, Value>) -> Value {
    let mut document = record.clone();
    document.remove("control_token");
    Value::Object(document)
}

/// Serializes this process's read-modify-write updates of `launcher.json`.
static RECORD_UPDATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Sets `key` in this process's own `launcher.json` (no-op for another's).
fn update_record(root: &Path, key: &str, value: Value) {
    let _guard = RECORD_UPDATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut record) = read_launcher(root) else {
        return;
    };
    if strict_int(get(&record, "pid")) != Some(i64::from(std::process::id())) {
        return;
    }
    record.insert(key.into(), value);
    let _ = write_private(root, LAUNCHER_RECORD, &Value::Object(record));
}

/// Records the last model that exited on its own (`last_exit`) in this
/// process's `launcher.json`.
pub fn note_exit(root: &Path, exit: &Value) {
    update_record(root, "last_exit", exit.clone());
}

/// Records the model server this launcher runs (`model`: pid, listen
/// address, run, repo) from the moment it is spawned, or clears it; a
/// launcher killed outright leaves this so its server can still be stopped.
pub fn note_model(root: &Path, model: Option<Value>) {
    update_record(root, "model", model.unwrap_or(Value::Null));
}

// -- talking to a running launcher ---------------------------------------------

/// A running launcher, reached through its listener with its control token.
pub struct Remote {
    pub record: Map<String, Value>,
    client: Client,
}

impl Remote {
    /// The launcher of `root`, when one is running and answers.
    pub fn connect(root: &Path) -> Option<Remote> {
        let record = running_launcher(root)?;
        let endpoint = str_value(get(&record, "endpoint"))?.to_string();
        let token = str_value(get(&record, "control_token")).map(str::to_string);
        let client = Client::new(&endpoint, token.as_deref(), Duration::from_secs(30)).ok()?;
        if !client.healthy(Duration::from_secs(3)) {
            return None;
        }
        Some(Remote { record, client })
    }

    pub fn endpoint(&self) -> String {
        self.client.base_url.clone()
    }

    pub fn pid(&self) -> Option<i64> {
        strict_int(get(&self.record, "pid"))
    }

    /// `/api/status` (the launcher, the model, the jobs).
    pub fn status(&self) -> Result<Map<String, Value>> {
        let (status, document) = self
            .client
            .get_json("/api/status", Some(Duration::from_secs(10)))?;
        if status != 200 {
            return Err(Error::launch(format!(
                "the launcher answered /api/status with HTTP {status}"
            )));
        }
        Ok(document)
    }

    /// `/loom/status` (the launcher, the model's launch facts and counters).
    pub fn loom_status(&self) -> Result<Map<String, Value>> {
        let (status, document) = self
            .client
            .get_json("/loom/status", Some(Duration::from_secs(10)))?;
        if status != 200 {
            return Err(Error::launch(format!(
                "the launcher answered /loom/status with HTTP {status}"
            )));
        }
        Ok(document)
    }

    /// Asks the launcher to load a model; answers with the accepted job.
    pub fn load(&self, body: &Value) -> Result<Map<String, Value>> {
        let (status, document) =
            self.client
                .post_json("/api/serve", body, Some(Duration::from_secs(30)))?;
        if status != 202 {
            return Err(Error::launcher(refusal(status, &document)));
        }
        Ok(document)
    }

    /// Asks the launcher to unload its model (or cancel one preparing);
    /// `None` when no model is loaded.
    pub fn unload(&self) -> Result<Option<Map<String, Value>>> {
        self.unload_within(30.0)
    }

    /// [`Remote::unload`] with `wait` seconds of drain before SIGKILL.
    pub fn unload_within(&self, wait: f64) -> Result<Option<Map<String, Value>>> {
        let (status, document) = self.client.post_json(
            "/api/unload",
            &json!({"wait": wait}),
            Some(Duration::from_secs(30)),
        )?;
        match status {
            202 => Ok(Some(document)),
            409 if refusal(status, &document).contains(crate::proxy::NO_MODEL) => Ok(None),
            _ => Err(Error::launcher(refusal(status, &document))),
        }
    }

    pub fn job(&self, id: &str) -> Result<Map<String, Value>> {
        let (status, document) = self.client.get_json(
            &format!("/api/jobs/{}", crate::proxy::percent_encode(id)),
            Some(Duration::from_secs(10)),
        )?;
        if status != 200 {
            return Err(Error::launch(refusal(status, &document)));
        }
        Ok(document)
    }

    /// Follows job `id` until it is done or failed, handing every new
    /// output line to `on_line`; `interrupted` is polled to give up early.
    pub fn follow(
        &self,
        id: &str,
        timeout: Duration,
        on_line: &mut dyn FnMut(&str),
        interrupted: &dyn Fn() -> bool,
    ) -> Result<Map<String, Value>> {
        let deadline = Instant::now() + timeout;
        let mut seen: u64 = 0;
        loop {
            let job = self.job(id)?;
            let count = get(&job, "line_count").as_u64().unwrap_or(0);
            let lines = crate::values::arr(get(&job, "lines"));
            let fresh = (count.saturating_sub(seen) as usize).min(lines.len());
            for line in &lines[lines.len() - fresh..] {
                if let Some(text) = line.as_str() {
                    on_line(text);
                }
            }
            seen = count;
            if matches!(str_value(get(&job, "state")), Some("done" | "failed")) {
                return Ok(job);
            }
            if interrupted() {
                return Err(Error::launch("interrupted"));
            }
            if Instant::now() >= deadline {
                return Err(Error::launch(format!(
                    "job {id} did not finish within {:.0}s",
                    timeout.as_secs_f64()
                )));
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// The error message of a refused dashboard request.
fn refusal(status: u16, document: &Map<String, Value>) -> String {
    str_value(get(crate::values::obj(get(document, "error")), "message"))
        .map(str::to_string)
        .unwrap_or_else(|| format!("HTTP {status}"))
}

// -- stopping ------------------------------------------------------------------

/// What [`stop_launcher`] did.
#[derive(Clone, Debug, PartialEq)]
pub enum LauncherStop {
    /// No launcher was recorded (or its record was stale and was removed).
    NotRunning { stale: bool },
    Stopped {
        pid: i64,
        endpoint: String,
        /// The model it was serving or preparing, if any.
        model: Option<String>,
        /// The launcher did not exit in time and was killed; its children
        /// were stopped here.
        forced: bool,
    },
}

/// Stops the launcher of `root`: SIGTERM (it unloads its model and exits),
/// SIGKILL after `wait`, then any model server it left is stopped and both
/// records are cleared.
pub fn stop_launcher(root: &Path, wait: Duration) -> LauncherStop {
    let Some(record) = read_launcher(root) else {
        return LauncherStop::NotRunning { stale: false };
    };
    let Some(pid) = strict_int(get(&record, "pid")) else {
        let _ = std::fs::remove_file(launcher_path(root));
        return LauncherStop::NotRunning { stale: true };
    };
    let endpoint = crate::values::py_str(get(&record, "endpoint"));
    let current = read_current(root).filter(|c| strict_int(get(c, "launcher_pid")) == Some(pid));
    let mut model = current
        .as_ref()
        .and_then(|c| str_value(get(c, "repo")).map(str::to_string));
    if signals::exited(pid) {
        // A launcher that died without cleaning up: its model may still run.
        stop_recorded_model(&record);
        stop_orphan(root, current.as_ref());
        clear_launcher_if(root, pid);
        return LauncherStop::NotRunning { stale: true };
    }
    if model.is_none() {
        // A model still preparing has no current.json yet; ask the launcher.
        if let Some(remote) = Remote::connect(root) {
            if let Ok(status) = remote.status() {
                model = str_value(get(crate::values::obj(get(&status, "launch")), "repo"))
                    .map(str::to_string);
            }
        }
    }
    let children = children_of(pid);
    signals::send(pid as i32, signals::SIGTERM);
    let deadline = Instant::now() + wait;
    while Instant::now() < deadline && !signals::exited(pid) {
        std::thread::sleep(Duration::from_millis(100));
    }
    let mut forced = false;
    if !signals::exited(pid) {
        forced = true;
        let children = children_of(pid);
        signals::send(pid as i32, signals::SIGKILL);
        for child in children {
            stop_process(child, Duration::from_secs(10));
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !signals::exited(pid) {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    // Whatever it was supervising must not outlive it.
    for child in children {
        if !signals::exited(child) {
            stop_process(child, Duration::from_secs(10));
        }
    }
    if forced {
        stop_recorded_model(&record);
    }
    stop_orphan(root, current.as_ref());
    clear_launcher_if(root, pid);
    LauncherStop::Stopped {
        pid,
        endpoint,
        model,
        forced,
    }
}

/// Stops the model server `launcher.json` names (`model`), when the process
/// with that pid is still that server (its command line carries the
/// recorded `--listen=` address, so a reused pid is left alone).
fn stop_recorded_model(record: &Map<String, Value>) {
    let model = crate::values::obj(get(record, "model"));
    let (Some(pid), Some(listen)) = (
        strict_int(get(model, "pid")),
        str_value(get(model, "listen")),
    ) else {
        return;
    };
    if signals::exited(pid) {
        return;
    }
    let command = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
    let wanted = format!("--listen={listen}");
    if command
        .split(|byte| *byte == 0)
        .any(|word| word == wanted.as_bytes())
    {
        stop_process(pid, Duration::from_secs(10));
    }
}

/// Stops the model a dead launcher recorded and clears its `current.json`.
fn stop_orphan(root: &Path, current: Option<&Map<String, Value>>) {
    let Some(current) = current else {
        return;
    };
    if let Some(model) = strict_int(get(current, "pid")) {
        if !signals::exited(model) {
            stop_process(model, Duration::from_secs(10));
        }
    }
    let run_id = str_value(get(current, "run_id")).unwrap_or("");
    crate::server::clear_current_if(root, run_id);
}

/// SIGTERM, then SIGKILL after `grace`.
fn stop_process(pid: i64, grace: Duration) {
    if signals::exited(pid) {
        return;
    }
    signals::send(pid as i32, signals::SIGTERM);
    let deadline = Instant::now() + grace;
    while Instant::now() < deadline {
        if signals::exited(pid) {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    signals::send(pid as i32, signals::SIGKILL);
}

/// The processes whose parent is `pid` (from `/proc`).
pub fn children_of(pid: i64) -> Vec<i64> {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().to_str()?.parse::<i64>().ok())
        .filter(|candidate| {
            std::fs::read_to_string(format!("/proc/{candidate}/stat"))
                .ok()
                .and_then(|stat| {
                    let (_, rest) = stat.rsplit_once(')')?;
                    rest.split_whitespace().nth(1)?.parse::<i64>().ok()
                })
                == Some(pid)
        })
        .collect()
}

/// Stops a launch recorded by a launcher of before the split (its
/// `current.json` names a supervising process that is not a launcher
/// service): SIGTERM to its server, whose supervisor then exits.
pub fn stop_legacy(root: &Path, wait: Duration) -> Option<StopOutcome> {
    let current = read_current(root)?;
    let launcher = running_launcher(root).and_then(|r| strict_int(get(&r, "pid")));
    let supervisor = strict_int(get(&current, "launcher_pid"));
    if launcher.is_some() && launcher == supervisor {
        return None;
    }
    Some(stop_current(root, wait))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tests::scratch;

    #[test]
    fn the_record_is_private_and_its_token_stays_out_of_status() {
        use std::os::unix::fs::PermissionsExt;
        let root = scratch("launcher-record");
        write_private(
            &root,
            LAUNCHER_RECORD,
            &json!({"pid": std::process::id(), "endpoint": "http://127.0.0.1:1", "control_token": "x"}),
        )
        .unwrap();
        let mode = std::fs::metadata(launcher_path(&root))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600);
        let record = running_launcher(&root).unwrap();
        assert!(public_record(&record).get("control_token").is_none());
        note_exit(&root, &json!({"repo": "a/b", "exit_code": 137}));
        assert_eq!(read_launcher(&root).unwrap()["last_exit"]["exit_code"], 137);
        clear_launcher_if(&root, 1);
        assert!(read_launcher(&root).is_some(), "another pid's record stays");
        clear_launcher_if(&root, i64::from(std::process::id()));
        assert!(read_launcher(&root).is_none());
    }

    #[test]
    fn a_dead_launchers_record_is_stale() {
        let root = scratch("launcher-stale");
        write_private(
            &root,
            LAUNCHER_RECORD,
            &json!({"pid": 999_999_999, "endpoint": "http://127.0.0.1:1"}),
        )
        .unwrap();
        assert!(running_launcher(&root).is_none());
        assert_eq!(
            stop_launcher(&root, Duration::from_secs(1)),
            LauncherStop::NotRunning { stale: true }
        );
        assert!(read_launcher(&root).is_none());
        assert_eq!(
            stop_launcher(&root, Duration::from_secs(1)),
            LauncherStop::NotRunning { stale: false }
        );
    }

    #[test]
    fn children_are_found_by_their_parent() {
        let mut child = std::process::Command::new("sleep")
            .arg("5")
            .spawn()
            .unwrap();
        let found = children_of(i64::from(std::process::id()));
        assert!(found.contains(&(child.id() as i64)), "{found:?}");
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn control_tokens_are_long_and_distinct() {
        let (a, b) = (control_token(), control_token());
        assert_eq!(a.len(), 64);
        assert_ne!(a, b);
    }
}
