//! Acquisition of published profiles through the SDK's own `loom-fetch`.
//!
//! `loom-fetch` owns locks, caches, materialization and gated-repository
//! handling; the launcher adds a default destination, a captured log, streamed
//! progress and the registry later launches resolve names against.

use crate::deployment::absolute;
use crate::errors::{Error, ErrorKind, Result};
use crate::hub::Hub;
use crate::profile::{CONFIG_DOCUMENT, MEASUREMENT_DOCUMENT};
use crate::scenario::{is_scenario, Scenario, DEPLOYMENT_DOCUMENT, DEVICES_DOCUMENT};
use crate::store::{read_json, state_directory, InstallRegistry, InstalledModel};
use crate::values::{arr, child, get, now_seconds, obj, str_value, string};
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const INSTALL_DOCUMENT: &str = "model.json";
pub const FETCH_BINARY: &str = "loom-fetch";

/// Finds a companion executable: explicit, environment, sibling of this
/// executable, then `PATH`.
pub fn find_tool(explicit: Option<&str>, variable: &str, binary: &str) -> Option<String> {
    if let Some(explicit) = explicit.filter(|e| !e.is_empty()) {
        return Some(explicit.to_string());
    }
    if let Ok(value) = std::env::var(variable) {
        if !value.is_empty() {
            return Some(value);
        }
    }
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            let sibling = parent.join(binary);
            if sibling.is_file() {
                return Some(sibling.to_string_lossy().into_owned());
            }
        }
    }
    which(binary).map(|p| p.to_string_lossy().into_owned())
}

/// Spawns `command`, retrying briefly while the executable is busy (ETXTBSY):
/// a script written just before can still be open in a concurrently forked child.
pub fn spawn_retrying(command: &mut Command) -> std::io::Result<std::process::Child> {
    let mut attempts = 0;
    loop {
        match command.spawn() {
            Err(error) if error.raw_os_error() == Some(libc::ETXTBSY) && attempts < 50 => {
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            other => return other,
        }
    }
}

/// Searches `PATH` for an executable file.
pub fn which(binary: &str) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join(binary))
        .find(|candidate| {
            std::fs::metadata(candidate)
                .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false)
        })
}

/// The acquisition command.
pub fn fetch_command(explicit: Option<&str>) -> Result<Vec<String>> {
    find_tool(explicit, "LOOM_FETCH", FETCH_BINARY)
        .map(|path| vec![path])
        .ok_or_else(|| {
            Error::install(
                "loom-fetch was not found: install the Loom SDK \
                 (`pip install \"hrx-loom[hf]\"`), or pass --loom-fetch=PATH",
            )
        })
}

/// One completed acquisition.
#[derive(Clone, Debug)]
pub struct InstallResult {
    pub repo: String,
    pub path: PathBuf,
    pub preset: Option<String>,
    pub revision: Option<String>,
    pub seconds: f64,
    pub command: Vec<String>,
    pub log_path: Option<PathBuf>,
    /// The JSON document `loom-fetch --json` reported.
    pub document: Map<String, Value>,
    /// What became of the profile's own `deployment.json` and `devices.json`.
    pub documents: ScenarioDocuments,
    /// What became of its `llm_config.json` (chat policy) and `info.json`.
    pub profile_documents: ProfileDocumentsStatus,
}

impl InstallResult {
    pub fn card_path(&self) -> PathBuf {
        self.path.join(INSTALL_DOCUMENT)
    }

    pub fn describe(&self) -> Value {
        json!({
            "repo": self.repo,
            "path": self.path.to_string_lossy(),
            "card": self.card_path().to_string_lossy(),
            "preset": self.preset,
            "revision": self.revision,
            "seconds": crate::values::round_to(self.seconds, 3),
            "command": self.command,
            "log": self.log_path.as_ref().map(|p| p.to_string_lossy().into_owned()),
            "scenario_documents": self.documents.describe(),
            "profile_documents": self.profile_documents.describe(),
        })
    }
}

/// Everything one install needs beyond the repository name.
#[derive(Clone, Debug, Default)]
pub struct InstallOptions {
    pub output: Option<PathBuf>,
    pub preset: Option<String>,
    pub revision: Option<String>,
    pub materialize: Option<String>,
    pub explicit_fetch: Option<String>,
    /// Passed to `loom-fetch` as `HF_ENDPOINT` when not the default hub.
    pub hub_endpoint: Option<String>,
    /// Forbid network requests (`loom-fetch --offline`, no hub reads).
    pub offline: bool,
}

/// The fetch selector: `huggingface:<repo>[:<preset>][@<revision>]`.
pub fn selector(repo: &str, preset: Option<&str>, revision: Option<&str>) -> String {
    if repo.starts_with("huggingface:")
        || repo.starts_with("github:")
        || repo.starts_with("http://")
        || repo.starts_with("https://")
    {
        return repo.to_string();
    }
    let mut source = format!("huggingface:{repo}");
    if let Some(preset) = preset.filter(|p| !p.is_empty()) {
        source.push(':');
        source.push_str(preset);
    }
    if let Some(revision) = revision.filter(|r| !r.is_empty()) {
        source.push('@');
        source.push_str(revision);
    }
    source
}

/// Installs one published profile into `output`.
pub fn install(
    repo: &str,
    output: &Path,
    options: &InstallOptions,
    log_path: Option<&Path>,
    on_line: &(dyn Fn(&str) + Sync),
) -> Result<InstallResult> {
    let output = absolute(output);
    let mut command = fetch_command(options.explicit_fetch.as_deref())?;
    command.push(selector(
        repo,
        options.preset.as_deref(),
        options.revision.as_deref(),
    ));
    command.push(format!("--output={}", output.display()));
    command.push(format!(
        "--materialize={}",
        options.materialize.as_deref().unwrap_or("copy")
    ));
    command.push("--no-input".into());
    command.push("--json".into());
    if options.offline {
        command.push("--offline".into());
    }
    let started = Instant::now();
    let log_file = match log_path {
        Some(path) => {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .ok()
        }
        None => None,
    };
    let log_file = Arc::new(Mutex::new(log_file));
    let mut process = Command::new(&command[0]);
    process
        .args(&command[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(endpoint) = options.hub_endpoint.as_deref() {
        process.env("HF_ENDPOINT", endpoint);
    }
    let mut child = spawn_retrying(&mut process)
        .map_err(|error| Error::install(format!("cannot run {}: {error}", command[0])))?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");
    // Progress goes to stderr and the result to stdout; both are logged and
    // shown in arrival order, but only stdout is parsed as the result.
    let combined: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let stdout_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    std::thread::scope(|scope| {
        let pump = |stream: Box<dyn std::io::Read + Send>,
                    keep: Option<Arc<Mutex<Vec<String>>>>| {
            let combined = combined.clone();
            let log_file = log_file.clone();
            move || {
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
                    if let Some(file) = log_file.lock().unwrap().as_mut() {
                        let _ = writeln!(file, "{line}");
                        let _ = file.flush();
                    }
                    on_line(&line);
                    combined.lock().unwrap().push(line.clone());
                    if let Some(keep) = &keep {
                        keep.lock().unwrap().push(line);
                    }
                }
            }
        };
        scope.spawn(pump(Box::new(stdout), Some(stdout_lines.clone())));
        scope.spawn(pump(Box::new(stderr), None));
    });
    let status = child
        .wait()
        .map_err(|error| Error::install(format!("waiting for loom-fetch failed: {error}")))?;
    let elapsed = started.elapsed().as_secs_f64();
    if !status.success() {
        let lines = combined.lock().unwrap();
        let tail = lines[lines.len().saturating_sub(8)..].join("\n");
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".into());
        let mut message = format!("loom-fetch failed for {repo} with exit code {code}:\n{tail}");
        if command[1].starts_with("huggingface:")
            && lines
                .iter()
                .any(|l| l.contains("No such file or directory") && l.contains("huggingface:"))
        {
            message.push_str(
                "\nthis loom-fetch does not understand `huggingface:` selectors (hrx-loom 0.1.0 \
                 predates them): install a newer Loom SDK, or put a checkout's loom/py first on \
                 PYTHONPATH",
            );
        }
        return Err(Error::install(message));
    }
    if !output.join(INSTALL_DOCUMENT).is_file() {
        return Err(Error::install(format!(
            "{repo} did not install a {INSTALL_DOCUMENT} under {}",
            output.display()
        )));
    }
    let document = parse_result(&stdout_lines.lock().unwrap());
    let mut preset = options
        .preset
        .clone()
        .or_else(|| string(get(&document, "preset")));
    if preset.is_none() {
        // The installed card retains the selection fetch made.
        preset = string(get(&read_json(&output.join(INSTALL_DOCUMENT)), "preset"));
    }
    Ok(InstallResult {
        repo: repo.to_string(),
        path: output,
        preset,
        revision: result_revision(&document, repo).or_else(|| options.revision.clone()),
        seconds: elapsed,
        command,
        log_path: log_path.map(Path::to_path_buf),
        document,
        documents: ScenarioDocuments::default(),
        profile_documents: ProfileDocumentsStatus::default(),
    })
}

/// Reads the JSON result `loom-fetch --json` prints: progress may precede it,
/// so every line that opens an object starts a candidate document.
pub fn parse_result(lines: &[String]) -> Map<String, Value> {
    for (index, line) in lines.iter().enumerate() {
        if !line.trim_start().starts_with('{') {
            continue;
        }
        let candidate = lines[index..].join("\n");
        if let Ok(Value::Object(document)) = serde_json::from_str::<Value>(candidate.trim()) {
            return document;
        }
    }
    Map::new()
}

/// The revision a fetch result pinned: a top-level `revision`, else the first
/// hub transport revision of the installed repository.
pub fn result_revision(document: &Map<String, Value>, repo: &str) -> Option<String> {
    if let Some(revision) = string(get(document, "revision")) {
        return Some(revision);
    }
    let files = arr(get(document, "files"));
    let transports = files.iter().map(|file| child(obj(file), "transport"));
    let mut fallback = None;
    for transport in transports {
        let Some(revision) = str_value(get(transport, "revision")) else {
            continue;
        };
        if str_value(get(transport, "repository")) == Some(repo) {
            return Some(revision.to_string());
        }
        fallback.get_or_insert_with(|| revision.to_string());
    }
    fallback
}

/// Where the launcher installs packages unless told otherwise.
pub fn default_models_directory(state: &Path) -> PathBuf {
    state.join("models")
}

/// Installs a profile and records it for later launches.
pub fn install_and_record(
    repo: &str,
    state: Option<&Path>,
    options: &InstallOptions,
    on_line: &(dyn Fn(&str) + Sync),
) -> Result<(InstallResult, InstalledModel)> {
    let hub = Hub::new(options.hub_endpoint.as_deref(), Duration::from_secs(30));
    install_and_record_with(repo, state, options, &hub, on_line)
}

/// The launcher's own stages of an install, after `loom-fetch` has finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallStage {
    /// Reading the documents the card does not declare (chat policy,
    /// measurements, scenario documents).
    Documents,
    /// Writing the install record.
    Record,
}

/// [`install_and_record`] reading the profile's own documents from `hub`.
pub fn install_and_record_with(
    repo: &str,
    state: Option<&Path>,
    options: &InstallOptions,
    hub: &Hub,
    on_line: &(dyn Fn(&str) + Sync),
) -> Result<(InstallResult, InstalledModel)> {
    install_and_record_observed(repo, state, options, hub, on_line, &|_| {})
}

/// [`install_and_record_with`], announcing the stages after the fetch.
pub fn install_and_record_observed(
    repo: &str,
    state: Option<&Path>,
    options: &InstallOptions,
    hub: &Hub,
    on_line: &(dyn Fn(&str) + Sync),
    on_stage: &(dyn Fn(InstallStage) + Sync),
) -> Result<(InstallResult, InstalledModel)> {
    let root = match state {
        Some(state) => absolute(state),
        None => state_directory(None),
    };
    let destination = options
        .output
        .clone()
        .unwrap_or_else(|| default_models_directory(&root).join(repo.replace('/', "-")));
    let mut result = install(
        repo,
        &destination,
        options,
        Some(&root.join("logs").join("fetch.log")),
        on_line,
    )?;
    on_stage(InstallStage::Documents);
    result.profile_documents = acquire_profile_documents(
        &result.path,
        repo,
        result.revision.as_deref(),
        options.offline,
        Some(hub),
        Some(&root),
    );
    result.documents = acquire_scenario_documents(
        &result.path,
        repo,
        result.revision.as_deref(),
        options.offline,
        hub,
    );
    on_stage(InstallStage::Record);
    let model = InstalledModel {
        repo: repo.to_string(),
        path: result.path.to_string_lossy().into_owned(),
        preset: result.preset.clone(),
        revision: result.revision.clone(),
        installed_at: Some(crate::values::iso_utc(now_seconds())),
        size_bytes: Some(directory_size(&result.path) as i64),
        source: None,
        scenario: result.documents.installed().then_some(true),
        scenario_revision: result
            .documents
            .installed()
            .then(|| result.documents.revision.clone())
            .flatten(),
    };
    InstallRegistry::new(&root).record(&model)?;
    Ok((result, model))
}

/// How an install came to carry (or not) the profile's own deployment and
/// device documents, which `loom-fetch` does not install because the card does
/// not declare them.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum DocumentsStatus {
    /// Both were already in the installed directory.
    Present,
    /// Copied from the local hub cache's snapshot of the pinned revision.
    Cache,
    /// Read from the hub at the pinned revision.
    Hub,
    /// The profile publishes neither (the common case).
    #[default]
    Absent,
    /// `--offline` and neither next to the card nor in the local hub cache.
    Offline,
    /// Not a hub profile (a URL or another provider): nothing to look up.
    NotHub,
    /// Only one of the two is published.
    Incomplete,
    /// Both are published but do not describe a servable scenario.
    Invalid,
    /// The hub could not be read; the install itself stands.
    Failed,
    /// Copied from the launcher's own cache of the profile's hub documents,
    /// read earlier at the same revision.
    LauncherCache,
}

/// What became of `deployment.json` and `devices.json` for one install.
#[derive(Clone, Debug, Default)]
pub struct ScenarioDocuments {
    pub status: DocumentsStatus,
    /// The revision the documents were read at (or looked up at).
    pub revision: Option<String>,
    /// Where they came from: a cache directory or the hub URL prefix.
    pub source: Option<String>,
    /// The scenario they describe, once installed.
    pub scenario: Option<Box<Scenario>>,
    /// What the user is told when the documents were expected but not installed.
    pub warning: Option<String>,
}

impl ScenarioDocuments {
    fn with(status: DocumentsStatus, revision: Option<&str>) -> ScenarioDocuments {
        ScenarioDocuments {
            status,
            revision: revision.map(str::to_string),
            ..Default::default()
        }
    }

    /// True when the install now carries both documents.
    pub fn installed(&self) -> bool {
        matches!(
            self.status,
            DocumentsStatus::Present | DocumentsStatus::Cache | DocumentsStatus::Hub
        )
    }

    pub fn status_name(&self) -> &'static str {
        match self.status {
            DocumentsStatus::Present => "present",
            DocumentsStatus::Cache => "cache",
            DocumentsStatus::Hub => "hub",
            DocumentsStatus::Absent => "absent",
            DocumentsStatus::Offline => "offline",
            DocumentsStatus::NotHub => "not-hub",
            DocumentsStatus::Incomplete => "incomplete",
            DocumentsStatus::Invalid => "invalid",
            DocumentsStatus::Failed => "failed",
            DocumentsStatus::LauncherCache => "launcher-cache",
        }
    }

    /// The `install` summary line for an installed scenario.
    pub fn summary(&self) -> Option<String> {
        let scenario = self.scenario.as_ref()?;
        let revision = self.revision.as_deref().unwrap_or("main");
        let origin = match self.status {
            DocumentsStatus::Present => "shipped with the card".to_string(),
            DocumentsStatus::Cache => format!("from the local hub cache at {revision}"),
            DocumentsStatus::Hub => format!("from the hub at {revision}"),
            _ => return None,
        };
        Some(format!(
            "deployment {} on {} ({DEPLOYMENT_DOCUMENT} + {DEVICES_DOCUMENT} {origin})",
            scenario.name(),
            scenario.device_text()
        ))
    }

    pub fn describe(&self) -> Value {
        json!({
            "status": self.status_name(),
            "scenario": self.installed(),
            "revision": self.revision,
            "source": self.source,
            "devices": self.scenario.as_ref().map(|s| s.device_text()),
            "deployment": self.scenario.as_ref().map(|s| s.name()),
            "warning": self.warning,
        })
    }
}

/// The hub repository an install selector names, when it names one.
fn hub_repository(repo: &str) -> Option<String> {
    let bare = match repo.strip_prefix("huggingface:") {
        Some(rest) => rest,
        None if repo.starts_with("github:") || repo.contains("://") => return None,
        None => repo,
    };
    let bare = bare.split('@').next().unwrap_or(bare);
    let mut parts = bare.splitn(3, [':', '/']);
    let (owner, name) = (parts.next()?, parts.next()?);
    (!owner.is_empty() && !name.is_empty()).then(|| format!("{owner}/{name}"))
}

/// The local Hugging Face hub cache (`huggingface_hub`'s own resolution).
fn hub_cache_root() -> Option<PathBuf> {
    let variable = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if let Some(cache) = variable("HF_HUB_CACHE").or_else(|| variable("HUGGINGFACE_HUB_CACHE")) {
        return Some(PathBuf::from(cache));
    }
    if let Some(home) = variable("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    let cache = variable("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| variable("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(cache.join("huggingface").join("hub"))
}

/// The cached snapshot of `repo` at `revision` (a commit, or a ref the cache
/// resolved), when one exists.
fn cached_snapshot(cache: &Path, repo: &str, revision: Option<&str>) -> Option<PathBuf> {
    let root = cache.join(format!("models--{}", repo.replace('/', "--")));
    let revision = revision.unwrap_or("main");
    let commit = if revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit()) {
        revision.to_string()
    } else {
        std::fs::read_to_string(root.join("refs").join(revision))
            .ok()?
            .trim()
            .to_string()
    };
    let snapshot = root.join("snapshots").join(commit);
    snapshot.is_dir().then_some(snapshot)
}

/// Writes both documents into `directory`, then checks they describe a
/// servable scenario; nothing is left behind when they do not.
fn place_documents(directory: &Path, deployment: &[u8], devices: &[u8]) -> Result<Box<Scenario>> {
    for (name, body) in [
        (DEPLOYMENT_DOCUMENT, deployment),
        (DEVICES_DOCUMENT, devices),
    ] {
        if !matches!(serde_json::from_slice::<Value>(body), Ok(Value::Object(_))) {
            return Err(Error::install(format!("{name} is not a JSON object")));
        }
    }
    let remove = || {
        let _ = std::fs::remove_file(directory.join(DEPLOYMENT_DOCUMENT));
        let _ = std::fs::remove_file(directory.join(DEVICES_DOCUMENT));
    };
    for (name, body) in [
        (DEPLOYMENT_DOCUMENT, deployment),
        (DEVICES_DOCUMENT, devices),
    ] {
        let staged = directory.join(format!(".{name}.partial"));
        let written = std::fs::write(&staged, body)
            .and_then(|_| std::fs::rename(&staged, directory.join(name)));
        if let Err(error) = written {
            let _ = std::fs::remove_file(&staged);
            remove();
            return Err(Error::install(format!(
                "cannot write {}: {error}",
                directory.join(name).display()
            )));
        }
    }
    match Scenario::detect(directory) {
        Ok(Some(scenario)) => Ok(Box::new(scenario)),
        Ok(None) => {
            remove();
            Err(Error::install("the documents did not land"))
        }
        Err(error) => {
            remove();
            Err(error)
        }
    }
}

/// Gives an installed profile its own `deployment.json` and `devices.json`
/// when the profile publishes them: from the installed directory itself, the
/// local hub cache's snapshot of the pinned revision, or the hub at that
/// revision (never with `offline`). Nothing here fails the install: a profile
/// without the documents is the common case, and a hub failure is reported in
/// `warning` while the installed package stays as `loom-fetch` left it.
pub fn acquire_scenario_documents(
    directory: &Path,
    repo: &str,
    revision: Option<&str>,
    offline: bool,
    hub: &Hub,
) -> ScenarioDocuments {
    acquire_with_cache(
        directory,
        repo,
        revision,
        offline,
        hub,
        hub_cache_root().as_deref(),
    )
}

fn acquire_with_cache(
    directory: &Path,
    repo: &str,
    revision: Option<&str>,
    offline: bool,
    hub: &Hub,
    cache: Option<&Path>,
) -> ScenarioDocuments {
    if is_scenario(directory) {
        let mut found = ScenarioDocuments::with(DocumentsStatus::Present, revision);
        found.source = Some(directory.to_string_lossy().into_owned());
        match Scenario::detect(directory) {
            Ok(scenario) => found.scenario = scenario.map(Box::new),
            Err(error) => {
                found.status = DocumentsStatus::Invalid;
                found.warning = Some(format!(
                    "{} carries {DEPLOYMENT_DOCUMENT} and {DEVICES_DOCUMENT}, but they do not \
                     describe a servable scenario: {}",
                    directory.display(),
                    error.message
                ));
            }
        }
        return found;
    }
    let Some(repository) = hub_repository(repo) else {
        return ScenarioDocuments::with(DocumentsStatus::NotHub, revision);
    };
    let label = format!("{repository}@{}", revision.unwrap_or("main"));
    if let Some(snapshot) = cache.and_then(|cache| cached_snapshot(cache, &repository, revision)) {
        let read = |name: &str| std::fs::read(snapshot.join(name)).ok();
        if let (Some(deployment), Some(devices)) =
            (read(DEPLOYMENT_DOCUMENT), read(DEVICES_DOCUMENT))
        {
            if let Ok(scenario) = place_documents(directory, &deployment, &devices) {
                let mut found = ScenarioDocuments::with(DocumentsStatus::Cache, revision);
                found.source = Some(snapshot.to_string_lossy().into_owned());
                found.scenario = Some(scenario);
                return found;
            }
        }
    }
    if offline {
        return ScenarioDocuments::with(DocumentsStatus::Offline, revision);
    }
    let failed = |error: Error| {
        let mut failed = ScenarioDocuments::with(DocumentsStatus::Failed, revision);
        failed.warning = Some(format!(
            "could not check {label} for {DEPLOYMENT_DOCUMENT} and {DEVICES_DOCUMENT}: {}. \
             The install itself succeeded, but if the profile is a GPU+NPU (or other \
             multi-device) scenario, `serve` will run a plain single-device deployment until \
             both files are in {}; rerun the install when the hub is reachable, or copy them \
             from the repository at that revision",
            error.message,
            directory.display()
        ));
        failed
    };
    let deployment = match hub.raw(&repository, DEPLOYMENT_DOCUMENT, revision) {
        Ok(body) => Some(body),
        Err(error) if error.kind == ErrorKind::HubNotFound => None,
        Err(error) => return failed(error),
    };
    let devices = match hub.raw(&repository, DEVICES_DOCUMENT, revision) {
        Ok(body) => Some(body),
        Err(error) if error.kind == ErrorKind::HubNotFound => None,
        Err(error) => return failed(error),
    };
    let (deployment, devices) = match (deployment, devices) {
        (Some(deployment), Some(devices)) => (deployment, devices),
        (None, None) => return ScenarioDocuments::with(DocumentsStatus::Absent, revision),
        (deployment, _) => {
            let (has, lacks) = if deployment.is_some() {
                (DEPLOYMENT_DOCUMENT, DEVICES_DOCUMENT)
            } else {
                (DEVICES_DOCUMENT, DEPLOYMENT_DOCUMENT)
            };
            let mut incomplete = ScenarioDocuments::with(DocumentsStatus::Incomplete, revision);
            incomplete.warning = Some(format!(
                "{label} publishes {has} but not {lacks}; installed as an ordinary profile \
                 (a scenario needs both)"
            ));
            return incomplete;
        }
    };
    match place_documents(directory, &deployment, &devices) {
        Ok(scenario) => {
            let mut found = ScenarioDocuments::with(DocumentsStatus::Hub, revision);
            found.source = Some(format!("{}/{repository}", hub.endpoint));
            found.scenario = Some(scenario);
            found
        }
        Err(error) => {
            let mut invalid = ScenarioDocuments::with(DocumentsStatus::Invalid, revision);
            invalid.warning = Some(format!(
                "{label} publishes {DEPLOYMENT_DOCUMENT} and {DEVICES_DOCUMENT}, but they do not \
                 describe a servable scenario ({}); installed as an ordinary profile",
                error.message
            ));
            invalid
        }
    }
}

// -- profile documents ----------------------------------------------------------

/// The documents a package's generator publishes next to its card that
/// `loom-fetch` does not install (the card does not declare them): the
/// generator's configuration, which carries the chat policy (thinking mode and
/// delimiters, tool-call format) and the drafter recommendation, and the
/// measurement record the dashboard compares benchmarks with.
pub const PROFILE_DOCUMENTS: [&str; 2] = [CONFIG_DOCUMENT, MEASUREMENT_DOCUMENT];

/// What became of one profile document.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileDocument {
    pub name: String,
    pub status: DocumentsStatus,
    /// The directory, cache snapshot or hub URL it came from.
    pub source: Option<String>,
    pub warning: Option<String>,
}

impl ProfileDocument {
    fn new(name: &str, status: DocumentsStatus) -> ProfileDocument {
        ProfileDocument {
            name: name.to_string(),
            status,
            source: None,
            warning: None,
        }
    }

    /// True when the installed directory now holds it.
    pub fn installed(&self) -> bool {
        matches!(
            self.status,
            DocumentsStatus::Present
                | DocumentsStatus::Cache
                | DocumentsStatus::Hub
                | DocumentsStatus::LauncherCache
        )
    }

    fn status_name(&self) -> &'static str {
        ScenarioDocuments::with(self.status.clone(), None).status_name()
    }
}

/// What became of `llm_config.json` and `info.json` for one install.
#[derive(Clone, Debug, Default)]
pub struct ProfileDocumentsStatus {
    pub revision: Option<String>,
    pub documents: Vec<ProfileDocument>,
}

impl ProfileDocumentsStatus {
    pub fn get(&self, name: &str) -> Option<&ProfileDocument> {
        self.documents.iter().find(|d| d.name == name)
    }

    /// Whether the install carries `name` now.
    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some_and(ProfileDocument::installed)
    }

    pub fn warnings(&self) -> Vec<String> {
        self.documents
            .iter()
            .filter_map(|d| d.warning.clone())
            .collect()
    }

    /// The `install` summary line: what was added and from where.
    pub fn summary(&self) -> Option<String> {
        let revision = self.revision.as_deref().unwrap_or("main");
        let mut parts = Vec::new();
        for status in [
            DocumentsStatus::Present,
            DocumentsStatus::Cache,
            DocumentsStatus::LauncherCache,
            DocumentsStatus::Hub,
        ] {
            let names: Vec<&str> = self
                .documents
                .iter()
                .filter(|d| d.status == status)
                .map(|d| d.name.as_str())
                .collect();
            if names.is_empty() {
                continue;
            }
            let origin = match status {
                DocumentsStatus::Present => "shipped with the card".to_string(),
                DocumentsStatus::Cache => format!("from the local hub cache at {revision}"),
                DocumentsStatus::LauncherCache => {
                    format!("from the launcher's cache of the hub documents at {revision}")
                }
                _ => format!("from the hub at {revision}"),
            };
            parts.push(format!("{} {origin}", names.join(" + ")));
        }
        let absent: Vec<&str> = self
            .documents
            .iter()
            .filter(|d| !d.installed())
            .map(|d| d.name.as_str())
            .collect();
        if !absent.is_empty() {
            parts.push(format!("no {}", absent.join(" or ")));
        }
        (!parts.is_empty()).then(|| parts.join("; "))
    }

    pub fn describe(&self) -> Value {
        Value::Array(
            self.documents
                .iter()
                .map(|d| {
                    json!({
                        "name": d.name,
                        "status": d.status_name(),
                        "installed": d.installed(),
                        "source": d.source,
                        "warning": d.warning,
                    })
                })
                .collect(),
        )
    }
}

/// The hub repository and revision a package's card was fetched from, as its
/// lock records them (`origins.document.requested.url`, a hub
/// `.../<owner>/<name>/resolve/<revision>/` URL), when it records one.
pub fn document_origin(directory: &Path) -> Option<(String, Option<String>)> {
    let lock = read_json(&directory.join("model.json.lock"));
    let requested = child(child(child(&lock, "origins"), "document"), "requested");
    let url = str_value(get(requested, "url"))?;
    let path = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let mut parts = path.split('/').skip(1).filter(|p| !p.is_empty());
    let (owner, name) = (parts.next()?, parts.next()?);
    let revision = match (parts.next(), parts.next()) {
        (Some("resolve"), Some(revision)) => Some(revision.to_string()),
        _ => None,
    };
    Some((format!("{owner}/{name}"), revision))
}

/// The lock (`model.json.lock`) a hub profile publishes next to its card,
/// which declares every file an install fetches with its size: from the local
/// hub cache's snapshot of `revision` (default `main`), else from the hub.
pub fn published_lock(repo: &str, revision: Option<&str>, hub: &Hub) -> Option<Map<String, Value>> {
    const LOCK: &str = "model.json.lock";
    let repository = hub_repository(repo)?;
    if let Some(snapshot) =
        hub_cache_root().and_then(|cache| cached_snapshot(&cache, &repository, revision))
    {
        let lock = read_json(&snapshot.join(LOCK));
        if !lock.is_empty() {
            return Some(lock);
        }
    }
    match serde_json::from_slice::<Value>(&hub.raw(&repository, LOCK, revision).ok()?) {
        Ok(Value::Object(lock)) => Some(lock),
        _ => None,
    }
}

/// Writes `body` to `directory/name` through a staged file.
fn place_document(directory: &Path, name: &str, body: &[u8]) -> Result<()> {
    let staged = directory.join(format!(".{name}.partial"));
    std::fs::write(&staged, body)
        .and_then(|_| std::fs::rename(&staged, directory.join(name)))
        .map_err(|error| {
            let _ = std::fs::remove_file(&staged);
            Error::install(format!(
                "cannot write {}: {error}",
                directory.join(name).display()
            ))
        })
}

fn json_object(body: &[u8]) -> bool {
    matches!(serde_json::from_slice::<Value>(body), Ok(Value::Object(_)))
}

/// Gives an installed profile its `llm_config.json` and `info.json`: from
/// the installed directory itself, the local hub cache's snapshot of the
/// pinned revision, the launcher's own cache of the hub documents read at that
/// same revision (under `state`), or the hub at that revision (never with
/// `offline`, never without `hub`). Nothing here fails the install: each
/// document is reported as found, absent (the profile publishes none) or not
/// reachable, with a warning saying what the install lacks without it.
pub fn acquire_profile_documents(
    directory: &Path,
    repo: &str,
    revision: Option<&str>,
    offline: bool,
    hub: Option<&Hub>,
    state: Option<&Path>,
) -> ProfileDocumentsStatus {
    let cached = state.and_then(|state| crate::store::ProfileCache::new(state).get_any(repo));
    acquire_profile_with_cache(
        directory,
        repo,
        revision,
        offline,
        hub,
        hub_cache_root().as_deref(),
        cached.as_ref(),
    )
}

/// What one missing document costs the install, for its warning.
fn document_purpose(name: &str) -> &'static str {
    if name == CONFIG_DOCUMENT {
        "chat policy (whether the model thinks, and whether that can be toggled), tool-call \
         format or drafter recommendation"
    } else {
        "publisher measurements to compare benchmarks with"
    }
}

fn acquire_profile_with_cache(
    directory: &Path,
    repo: &str,
    revision: Option<&str>,
    offline: bool,
    hub: Option<&Hub>,
    cache: Option<&Path>,
    launcher_cache: Option<&Map<String, Value>>,
) -> ProfileDocumentsStatus {
    let repository = hub_repository(repo);
    let label = format!(
        "{}@{}",
        repository.as_deref().unwrap_or(repo),
        revision.unwrap_or("main")
    );
    let snapshot = repository
        .as_deref()
        .and_then(|r| cache.and_then(|cache| cached_snapshot(cache, r, revision)));
    // The launcher's cache only stands in for the hub at the same revision.
    let cached = launcher_cache
        .filter(|cached| revision.is_some() && str_value(get(cached, "revision")) == revision);
    // A package directory inside the hub cache is a snapshot: never written to.
    let writable = !cache.is_some_and(|cache| directory.starts_with(cache));
    let mut documents = Vec::new();
    for name in PROFILE_DOCUMENTS {
        let path = directory.join(name);
        if path.is_file() {
            let mut found = ProfileDocument::new(name, DocumentsStatus::Present);
            found.source = Some(directory.to_string_lossy().into_owned());
            if !json_object(&std::fs::read(&path).unwrap_or_default()) {
                found.status = DocumentsStatus::Invalid;
                found.warning = Some(format!(
                    "{} is not a JSON object, so the launcher reads no {} from it",
                    path.display(),
                    document_purpose(name)
                ));
            }
            documents.push(found);
            continue;
        }
        let place = |status: DocumentsStatus, body: &[u8], source: String| {
            let mut placed = ProfileDocument::new(name, status);
            placed.source = Some(source);
            if !writable {
                placed.status = DocumentsStatus::Absent;
                placed.warning = Some(format!(
                    "{} is a hub cache snapshot, so {name} was not written into it",
                    directory.display()
                ));
                placed.source = None;
            } else if let Err(error) = place_document(directory, name, body) {
                placed.status = DocumentsStatus::Failed;
                placed.warning = Some(format!(
                    "{}; the install has no {}",
                    error.message,
                    document_purpose(name)
                ));
            }
            placed
        };
        if let Some(body) = snapshot
            .as_ref()
            .and_then(|s| std::fs::read(s.join(name)).ok())
            .filter(|body| json_object(body))
        {
            let source = snapshot.as_ref().unwrap().to_string_lossy().into_owned();
            documents.push(place(DocumentsStatus::Cache, &body, source));
            continue;
        }
        let key = if name == CONFIG_DOCUMENT {
            "llm_config"
        } else {
            "info"
        };
        if let Some(document) = cached
            .map(|cached| child(cached, key))
            .filter(|document| !document.is_empty())
        {
            let body = serde_json::to_vec_pretty(document).expect("JSON values serialize");
            documents.push(place(
                DocumentsStatus::LauncherCache,
                &body,
                format!("launcher cache of {label}"),
            ));
            continue;
        }
        let (Some(repository), Some(hub)) = (repository.as_deref(), hub) else {
            documents.push(ProfileDocument::new(
                name,
                if repository.is_none() {
                    DocumentsStatus::NotHub
                } else {
                    DocumentsStatus::Offline
                },
            ));
            continue;
        };
        if offline {
            documents.push(ProfileDocument::new(name, DocumentsStatus::Offline));
            continue;
        }
        match hub.raw(repository, name, revision) {
            Ok(body) if json_object(&body) => {
                documents.push(place(
                    DocumentsStatus::Hub,
                    &body,
                    format!("{}/{repository}", hub.endpoint),
                ));
            }
            Ok(_) => {
                let mut invalid = ProfileDocument::new(name, DocumentsStatus::Invalid);
                invalid.warning = Some(format!(
                    "{label} publishes {name}, but it is not a JSON object; the install has no {}",
                    document_purpose(name)
                ));
                documents.push(invalid);
            }
            Err(error) if error.kind == ErrorKind::HubNotFound => {
                documents.push(ProfileDocument::new(name, DocumentsStatus::Absent));
            }
            Err(error) => {
                let mut failed = ProfileDocument::new(name, DocumentsStatus::Failed);
                failed.warning = Some(format!(
                    "could not read {name} of {label}: {}. The install itself succeeded, but \
                     without {name} it has no {}; rerun the install when the hub is reachable",
                    error.message,
                    document_purpose(name)
                ));
                documents.push(failed);
            }
        }
    }
    ProfileDocumentsStatus {
        revision: revision.map(str::to_string),
        documents,
    }
}

/// Sums regular file sizes without following symlinks.
pub fn directory_size(path: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(metadata) = std::fs::symlink_metadata(entry.path()) else {
            continue;
        };
        if metadata.is_dir() {
            total += directory_size(&entry.path());
        } else if metadata.is_file() {
            total += metadata.len();
        } else if metadata.file_type().is_symlink() {
            // os.walk(followlinks=False) still stats file symlinks.
            if let Ok(target) = std::fs::metadata(entry.path()) {
                if target.is_file() {
                    total += target.len();
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tests::scratch;
    use std::os::unix::fs::PermissionsExt;

    const FETCH_SCRIPT: &str = r#"#!/bin/sh
out=""
for arg in "$@"; do
  case "$arg" in --output=*) out="${arg#--output=}";; esac
done
if [ "$FAKE_FETCH_MODE" = "fail" ]; then echo "download failed"; exit 3; fi
if [ "$FAKE_FETCH_MODE" = "old" ]; then echo "loom-fetch: [Errno 2] No such file or directory: '$PWD/$1'" >&2; exit 1; fi
if [ "$FAKE_FETCH_MODE" != "nocreate" ]; then
  mkdir -p "$out"
  printf '{"model": {"id": "a/b"}, "preset": "chat"}' > "$out/model.json"
fi
echo "loom-fetch  Fetch     1/2 files" >&2
echo "fetching 1/2 files"
cat <<'JSON'
{
  "model": "a/b",
  "preset": "chat",
  "files": [
    {"path": "model.json", "transport": {"kind": "hf", "repository": "loom-fun/a-loom", "revision": "abc123"}}
  ]
}
JSON
"#;

    fn fetch_script(root: &Path, mode: &str) -> String {
        let script = root.join(format!("fake-fetch-{mode}"));
        let body = FETCH_SCRIPT.replace("$FAKE_FETCH_MODE", mode);
        std::fs::write(&script, body).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script.to_string_lossy().into_owned()
    }

    #[test]
    fn an_explicit_command_wins_over_discovery() {
        assert_eq!(
            fetch_command(Some("/opt/loom-fetch")).unwrap(),
            ["/opt/loom-fetch"]
        );
    }

    #[test]
    fn selectors_carry_preset_and_revision() {
        assert_eq!(selector("a/b", None, None), "huggingface:a/b");
        assert_eq!(
            selector("a/b", Some("chat"), Some("v1")),
            "huggingface:a/b:chat@v1"
        );
        assert_eq!(selector("https://x/y", Some("chat"), None), "https://x/y");
    }

    #[test]
    fn the_pretty_json_result_is_parsed_after_progress() {
        let lines: Vec<String> = [
            "loom-fetch  Metadata  reading model metadata",
            "{",
            "  \"model\": \"Qwen/Qwen3-4B\",",
            "  \"preset\": \"chat\",",
            "  \"files\": [{\"transport\": {\"repository\": \"loom-fun/q\", \"revision\": \"d9f5\"}}],",
            "  \"output\": \"/x\"",
            "}",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let document = parse_result(&lines);
        assert_eq!(document["preset"], "chat");
        assert_eq!(
            result_revision(&document, "loom-fun/q").as_deref(),
            Some("d9f5")
        );
        assert!(parse_result(&["{ not json".to_string()]).is_empty());
    }

    #[test]
    fn installation_runs_the_fetch_command_and_reads_its_result() {
        let root = scratch("install-run");
        let output = root.join("models").join("a-b");
        let log = root.join("fetch.log");
        let lines = Mutex::new(Vec::new());
        let options = InstallOptions {
            preset: Some("chat".into()),
            explicit_fetch: Some(fetch_script(&root, "ok")),
            ..Default::default()
        };
        let result = install(
            "loom-fun/a-loom",
            &output,
            &options,
            Some(&log),
            &|line: &str| lines.lock().unwrap().push(line.to_string()),
        )
        .unwrap();
        assert_eq!(result.path, output);
        assert!(result.card_path().is_file());
        assert_eq!(result.preset.as_deref(), Some("chat"));
        assert_eq!(result.revision.as_deref(), Some("abc123"));
        assert!(result.command[1].starts_with("huggingface:loom-fun/a-loom"));
        assert!(result.command[1].ends_with(":chat"));
        assert!(!result.command.join(" ").contains("--preset"));
        assert!(lines
            .lock()
            .unwrap()
            .iter()
            .any(|l| l == "fetching 1/2 files"));
        assert!(std::fs::read_to_string(&log)
            .unwrap()
            .contains("fetching 1/2 files"));
    }

    #[test]
    fn a_failed_installation_reports_the_exit_status() {
        let root = scratch("install-fail");
        let options = InstallOptions {
            explicit_fetch: Some(fetch_script(&root, "fail")),
            ..Default::default()
        };
        let error = install(
            "loom-fun/a-loom",
            &root.join("out"),
            &options,
            None,
            &|_: &str| {},
        )
        .unwrap_err();
        assert!(error.message.contains("exit code 3"), "{error}");
        assert!(error.message.contains("download failed"), "{error}");
    }

    #[test]
    fn a_fetch_without_selector_support_is_explained() {
        let root = scratch("install-old");
        let options = InstallOptions {
            explicit_fetch: Some(fetch_script(&root, "old")),
            ..Default::default()
        };
        let error = install(
            "loom-fun/a-loom",
            &root.join("out"),
            &options,
            None,
            &|_: &str| {},
        )
        .unwrap_err();
        assert!(
            error
                .message
                .contains("does not understand `huggingface:` selectors"),
            "{error}"
        );
    }

    #[test]
    fn installation_without_a_card_is_refused() {
        let root = scratch("install-nocard");
        let options = InstallOptions {
            explicit_fetch: Some(fetch_script(&root, "nocreate")),
            ..Default::default()
        };
        let error = install(
            "loom-fun/a-loom",
            &root.join("out"),
            &options,
            None,
            &|_: &str| {},
        )
        .unwrap_err();
        assert!(error.message.contains("did not install"), "{error}");
    }

    use crate::hub::tests::FragmentTransport;
    use crate::hub::Transport;

    /// The hybrid package's three documents, as a hub would serve them.
    fn hybrid_documents() -> (Value, Value, Value) {
        let read = |name: &str| -> Value {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/hybrid-publish")
                .join(name);
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
        };
        (
            read("model.json"),
            read("deployment.json"),
            read("devices.json"),
        )
    }

    /// An installed directory as loom-fetch leaves it: the card only.
    fn fetched(root: &Path) -> PathBuf {
        let directory = root.join("installed");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("model.json"),
            serde_json::to_vec(&hybrid_documents().0).unwrap(),
        )
        .unwrap();
        directory
    }

    fn fragment_hub(routes: Vec<(&str, u16, Value)>) -> (Hub, crate::hub::tests::Calls) {
        let transport = FragmentTransport::new(routes);
        let calls = transport.calls.clone();
        let hub = Hub::with_transport(
            Some("http://hub.test"),
            Some(String::new()),
            Box::new(transport),
        );
        (hub, calls)
    }

    struct Unreachable;

    impl Transport for Unreachable {
        fn get(&self, url: &str, _: &[(String, String)]) -> Result<(u16, Vec<u8>)> {
            Err(Error::new(
                ErrorKind::HubUnreachable,
                format!("cannot reach {url}: connection refused"),
            ))
        }
    }

    const REVISION: &str = "3d5342e3317741e55995fa568ba907e528d811e9";

    #[test]
    fn published_scenario_documents_are_installed_at_the_pinned_revision() {
        let root = scratch("documents-hub");
        let directory = fetched(&root);
        let (_, deployment, devices) = hybrid_documents();
        let (hub, calls) = fragment_hub(vec![
            (
                &format!("/resolve/{REVISION}/deployment.json"),
                200,
                deployment.clone(),
            ),
            (
                &format!("/resolve/{REVISION}/devices.json"),
                200,
                devices.clone(),
            ),
        ]);
        let found = acquire_with_cache(
            &directory,
            "ilintar/hybrid",
            Some(REVISION),
            false,
            &hub,
            Some(&root.join("empty-cache")),
        );
        assert_eq!(found.status, DocumentsStatus::Hub, "{found:?}");
        assert!(found.installed());
        assert!(found.warning.is_none());
        assert_eq!(found.revision.as_deref(), Some(REVISION));
        assert_eq!(
            read_json(&directory.join("deployment.json")),
            *obj(&deployment)
        );
        assert_eq!(read_json(&directory.join("devices.json")), *obj(&devices));
        assert_eq!(
            found.summary().unwrap(),
            format!(
                "deployment qwen3.8-27b on amdgpu + xdna (deployment.json + devices.json from \
                 the hub at {REVISION})"
            )
        );
        let urls: Vec<String> = calls.lock().unwrap().iter().map(|c| c.0.clone()).collect();
        assert_eq!(
            urls,
            [
                format!("http://hub.test/ilintar/hybrid/resolve/{REVISION}/deployment.json"),
                format!("http://hub.test/ilintar/hybrid/resolve/{REVISION}/devices.json"),
            ]
        );
        assert!(!directory.join(".deployment.json.partial").exists());
    }

    #[test]
    fn documents_already_present_or_cached_never_reach_the_hub() {
        let root = scratch("documents-local");
        let (_, deployment, devices) = hybrid_documents();
        let (hub, calls) = fragment_hub(vec![]);
        // Next to the card: used as they are.
        let directory = fetched(&root);
        std::fs::write(directory.join("deployment.json"), deployment.to_string()).unwrap();
        std::fs::write(directory.join("devices.json"), devices.to_string()).unwrap();
        let found = acquire_with_cache(&directory, "a/b", Some(REVISION), false, &hub, None);
        assert_eq!(found.status, DocumentsStatus::Present);
        assert!(found.summary().unwrap().ends_with("shipped with the card)"));
        // In the local hub cache's snapshot of the revision (here via a ref).
        let cache = root.join("hub-cache");
        let snapshot = cache.join("models--a--c").join("snapshots").join(REVISION);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(cache.join("models--a--c").join("refs")).unwrap();
        std::fs::write(cache.join("models--a--c/refs/main"), REVISION).unwrap();
        std::fs::write(snapshot.join("deployment.json"), deployment.to_string()).unwrap();
        std::fs::write(snapshot.join("devices.json"), devices.to_string()).unwrap();
        let second = root.join("second");
        std::fs::create_dir_all(&second).unwrap();
        std::fs::copy(directory.join("model.json"), second.join("model.json")).unwrap();
        let found = acquire_with_cache(&second, "a/c", None, true, &hub, Some(&cache));
        assert_eq!(found.status, DocumentsStatus::Cache, "{found:?}");
        assert!(is_scenario(&second));
        // Offline with nothing local: nothing is fetched and nothing is said.
        let third = root.join("third");
        std::fs::create_dir_all(&third).unwrap();
        let found = acquire_with_cache(&third, "a/d", Some(REVISION), true, &hub, Some(&cache));
        assert_eq!(found.status, DocumentsStatus::Offline);
        assert!(found.warning.is_none());
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn a_profile_without_documents_is_silent_and_a_partial_one_is_explained() {
        let root = scratch("documents-absent");
        let directory = fetched(&root);
        let (hub, calls) = fragment_hub(vec![]);
        let found = acquire_with_cache(&directory, "a/b", Some("v1"), false, &hub, None);
        assert_eq!(found.status, DocumentsStatus::Absent);
        assert!(found.warning.is_none() && found.summary().is_none());
        assert!(!directory.join("deployment.json").exists());
        assert_eq!(calls.lock().unwrap().len(), 2);
        let (_, deployment, _) = hybrid_documents();
        let (hub, _) = fragment_hub(vec![("/deployment.json", 200, deployment)]);
        let found = acquire_with_cache(&directory, "a/b", Some("v1"), false, &hub, None);
        assert_eq!(found.status, DocumentsStatus::Incomplete);
        assert_eq!(
            found.warning.as_deref(),
            Some(
                "a/b@v1 publishes deployment.json but not devices.json; installed as an \
                 ordinary profile (a scenario needs both)"
            )
        );
        assert!(!directory.join("deployment.json").exists());
        // A URL or another provider has no hub repository to read.
        let found = acquire_with_cache(&directory, "https://x/y", None, false, &hub, None);
        assert_eq!(found.status, DocumentsStatus::NotHub);
    }

    #[test]
    fn a_hub_failure_is_reported_without_failing_the_install() {
        let root = scratch("documents-failed");
        let directory = fetched(&root);
        let hub = Hub::with_transport(
            Some("http://hub.test"),
            Some(String::new()),
            Box::new(Unreachable),
        );
        let found = acquire_with_cache(&directory, "a/b", Some(REVISION), false, &hub, None);
        assert_eq!(found.status, DocumentsStatus::Failed);
        let warning = found.warning.unwrap();
        assert!(
            warning.starts_with(&format!(
                "could not check a/b@{REVISION} for deployment.json and devices.json: cannot \
                 reach http://hub.test/a/b/resolve/{REVISION}/deployment.json"
            )),
            "{warning}"
        );
        assert!(
            warning.contains("The install itself succeeded"),
            "{warning}"
        );
        assert!(!is_scenario(&directory));
        // Documents that would not serve are not left behind.
        let (hub, _) = fragment_hub(vec![
            (
                "/deployment.json",
                200,
                json!({"name": "x", "model": {"path": "absent.json"}}),
            ),
            (
                "/devices.json",
                200,
                json!({"devices": {"amdgpu": {"uri": "amdgpu://0"}}}),
            ),
        ]);
        let found = acquire_with_cache(&directory, "a/b", Some(REVISION), false, &hub, None);
        assert_eq!(found.status, DocumentsStatus::Invalid, "{found:?}");
        assert!(!directory.join("deployment.json").exists());
        assert!(!directory.join("devices.json").exists());
    }

    /// The Qwen3.8 hybrid's chat policy, as its `llm_config.json` publishes it.
    fn policy() -> Value {
        json!({"arch": "qwen35", "context_length": 262144,
               "chat": {"thinking": "toggle", "thinking_default": true,
                        "think_open": "<think>\n", "think_close": "\n</think>\n\n"},
               "drafter": {"kind": "dflash", "depth": 3}})
    }

    #[test]
    fn the_chat_policy_and_measurements_are_installed_at_the_pinned_revision() {
        let root = scratch("profile-documents-hub");
        let directory = fetched(&root);
        let info = json!({"schema": "loom-card/2", "performance": {"sessions": []}});
        let (hub, calls) = fragment_hub(vec![
            (
                &format!("/resolve/{REVISION}/llm_config.json"),
                200,
                policy(),
            ),
            (&format!("/resolve/{REVISION}/info.json"), 200, info.clone()),
        ]);
        let found = acquire_profile_with_cache(
            &directory,
            "ilintar/Qwen3.8-27B-hybrid-loom",
            Some(REVISION),
            false,
            Some(&hub),
            Some(&root.join("empty-cache")),
            None,
        );
        assert!(
            found.has("llm_config.json") && found.has("info.json"),
            "{found:?}"
        );
        assert!(found.warnings().is_empty());
        assert_eq!(
            read_json(&directory.join("llm_config.json")),
            *obj(&policy())
        );
        assert_eq!(read_json(&directory.join("info.json")), *obj(&info));
        assert_eq!(
            found.summary().unwrap(),
            format!("llm_config.json + info.json from the hub at {REVISION}")
        );
        let urls: Vec<String> = calls.lock().unwrap().iter().map(|c| c.0.clone()).collect();
        assert_eq!(
            urls,
            [
                format!("http://hub.test/ilintar/Qwen3.8-27B-hybrid-loom/resolve/{REVISION}/llm_config.json"),
                format!("http://hub.test/ilintar/Qwen3.8-27B-hybrid-loom/resolve/{REVISION}/info.json"),
            ]
        );
        // The installed package now reads with its chat policy.
        let profile = crate::profile::Profile::from_local(&directory, None).unwrap();
        let chat = profile.chat.expect("a chat policy");
        assert!(chat.supports_thinking() && chat.default_thinking());
        assert_eq!(found.describe()[0]["status"], "hub");
    }

    #[test]
    fn absent_or_unreachable_profile_documents_leave_the_install_standing() {
        let root = scratch("profile-documents-absent");
        let directory = fetched(&root);
        // Not published: absent, nothing written, nothing to warn about.
        let (hub, _) = fragment_hub(vec![]);
        let found = acquire_profile_with_cache(
            &directory,
            "a/b",
            Some("v1"),
            false,
            Some(&hub),
            None,
            None,
        );
        assert_eq!(
            found.get("llm_config.json").unwrap().status,
            DocumentsStatus::Absent
        );
        assert!(found.warnings().is_empty());
        assert!(!directory.join("llm_config.json").exists());
        // The hub cannot be reached: the warning says what the install lacks.
        let unreachable = Hub::with_transport(
            Some("http://hub.test"),
            Some(String::new()),
            Box::new(Unreachable),
        );
        let found = acquire_profile_with_cache(
            &directory,
            "a/b",
            Some(REVISION),
            false,
            Some(&unreachable),
            None,
            None,
        );
        let config = found.get("llm_config.json").unwrap();
        assert_eq!(config.status, DocumentsStatus::Failed);
        let warning = config.warning.as_deref().unwrap();
        assert!(
            warning.starts_with(&format!(
                "could not read llm_config.json of a/b@{REVISION}: cannot reach"
            )),
            "{warning}"
        );
        assert!(warning.contains("it has no chat policy"), "{warning}");
        assert!(
            warning.contains("The install itself succeeded"),
            "{warning}"
        );
        // Published but not a document: refused, not written.
        let (hub, _) = fragment_hub(vec![("/llm_config.json", 200, json!([1, 2]))]);
        let found =
            acquire_profile_with_cache(&directory, "a/b", None, false, Some(&hub), None, None);
        assert_eq!(
            found.get("llm_config.json").unwrap().status,
            DocumentsStatus::Invalid
        );
        assert!(!directory.join("llm_config.json").exists());
    }

    #[test]
    fn offline_installs_use_only_local_copies_of_the_same_revision() {
        let root = scratch("profile-documents-offline");
        let (hub, calls) = fragment_hub(vec![]);
        // Offline with nothing local: nothing fetched, nothing written.
        let directory = fetched(&root);
        let found = acquire_profile_with_cache(
            &directory,
            "a/b",
            Some(REVISION),
            true,
            Some(&hub),
            None,
            None,
        );
        assert_eq!(
            found.get("info.json").unwrap().status,
            DocumentsStatus::Offline
        );
        // The local hub cache's snapshot of the revision.
        let cache = root.join("hub-cache");
        let snapshot = cache.join("models--a--b").join("snapshots").join(REVISION);
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("llm_config.json"), policy().to_string()).unwrap();
        let found = acquire_profile_with_cache(
            &directory,
            "a/b",
            Some(REVISION),
            true,
            Some(&hub),
            Some(&cache),
            None,
        );
        assert_eq!(
            found.get("llm_config.json").unwrap().status,
            DocumentsStatus::Cache
        );
        assert_eq!(
            found.get("info.json").unwrap().status,
            DocumentsStatus::Offline
        );
        assert_eq!(
            read_json(&directory.join("llm_config.json")),
            *obj(&policy())
        );
        // Already next to the card: kept as it is.
        let found =
            acquire_profile_with_cache(&directory, "a/b", Some(REVISION), true, None, None, None);
        assert_eq!(
            found.get("llm_config.json").unwrap().status,
            DocumentsStatus::Present
        );
        // The launcher's cache of the hub documents counts only at the same revision.
        let other = root.join("other");
        std::fs::create_dir_all(&other).unwrap();
        let launcher_cache = obj(&json!({"revision": REVISION, "llm_config": policy(),
                                         "info": {"schema": "loom-card/2"}}))
        .clone();
        let found = acquire_profile_with_cache(
            &other,
            "a/b",
            Some("0000"),
            true,
            None,
            None,
            Some(&launcher_cache),
        );
        assert!(!found.has("llm_config.json"));
        let found = acquire_profile_with_cache(
            &other,
            "a/b",
            Some(REVISION),
            true,
            None,
            None,
            Some(&launcher_cache),
        );
        assert_eq!(
            found.get("llm_config.json").unwrap().status,
            DocumentsStatus::LauncherCache
        );
        assert!(found.has("info.json"));
        assert!(calls.lock().unwrap().is_empty());
    }

    #[test]
    fn a_lock_names_the_hub_revision_its_card_came_from() {
        let root = scratch("document-origin");
        std::fs::write(
            root.join("model.json.lock"),
            json!({"origins": {"document": {"requested": {"kind": "url",
                "url": format!("https://huggingface.co/ilintar/Qwen3.8-27B-hybrid-loom/resolve/{REVISION}/")}}}})
            .to_string(),
        )
        .unwrap();
        assert_eq!(
            document_origin(&root),
            Some((
                "ilintar/Qwen3.8-27B-hybrid-loom".to_string(),
                Some(REVISION.to_string())
            ))
        );
        assert_eq!(document_origin(&root.join("absent")), None);
    }

    #[test]
    fn install_and_record_keeps_the_installation_for_later_launches() {
        let root = scratch("install-record");
        let options = InstallOptions {
            explicit_fetch: Some(fetch_script(&root, "ok")),
            ..Default::default()
        };
        let (hub, calls) = fragment_hub(vec![]);
        let (result, model) = install_and_record_with(
            "loom-fun/a-loom",
            Some(&root),
            &options,
            &hub,
            &|_: &str| {},
        )
        .unwrap();
        // The profile publishes no scenario or profile documents: the hub was
        // asked for them at the pinned revision, and the record is as before.
        assert_eq!(result.documents.status, DocumentsStatus::Absent);
        let asked: Vec<String> = calls.lock().unwrap().iter().map(|c| c.0.clone()).collect();
        for name in ["llm_config.json", "info.json", "deployment.json"] {
            assert!(
                asked
                    .iter()
                    .any(|url| url.ends_with(&format!("/resolve/abc123/{name}"))),
                "{asked:?}"
            );
        }
        assert!(!result.profile_documents.has("llm_config.json"));
        assert_eq!(
            result.profile_documents.summary().as_deref(),
            Some("no llm_config.json or info.json")
        );
        assert!(!result.path.join("deployment.json").exists());
        assert_eq!(
            PathBuf::from(&model.path),
            default_models_directory(&root).join("loom-fun-a-loom")
        );
        assert!(model.size_bytes.unwrap() > 0);
        assert!(model.installed_at.is_some());
        assert_eq!(model.preset.as_deref(), Some("chat"));
        assert_eq!(model.revision.as_deref(), Some("abc123"));
        assert_eq!(
            InstallRegistry::new(&root)
                .get("loom-fun/a-loom")
                .unwrap()
                .path,
            model.path
        );
        assert!(result.card_path().is_file());
        let registry = read_json(&root.join("installs.json"));
        let entry = &registry["installs"]["loom-fun/a-loom"];
        let mut keys: Vec<_> = entry.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "installed_at",
                "path",
                "preset",
                "repo",
                "revision",
                "size_bytes"
            ]
        );
    }
}
