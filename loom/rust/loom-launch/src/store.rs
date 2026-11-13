//! Where the launcher keeps its runs, installs, cached documents and results.
//!
//! Everything persistent lives under one state directory (`--state`, else
//! `$LOOM_LAUNCH_STATE`, else `~/.local/state/loom-launch`). The layout and the
//! JSON shapes are shared with the Python launcher, so either can read what the
//! other wrote.

use crate::deployment::{absolute, expand_user};
use crate::errors::{Error, Result};
use crate::values::{
    child, dumps_compact, dumps_sorted, get, human_bytes, now_seconds, obj, str_value, strict_int,
    string,
};
use serde_json::{json, Map, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Resolves the launcher's state directory (absolute), creating nothing.
pub fn state_directory(explicit: Option<&str>) -> PathBuf {
    if let Some(explicit) = explicit.filter(|e| !e.is_empty()) {
        return absolute(Path::new(explicit));
    }
    if let Ok(from_environment) = std::env::var("LOOM_LAUNCH_STATE") {
        if !from_environment.is_empty() {
            return absolute(Path::new(&from_environment));
        }
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".local").join("state").join("loom-launch")
}

/// Publishes a JSON document whole (temporary file + rename), sorted keys,
/// one-space indentation and a trailing newline, like the Python launcher.
pub fn write_json(path: &Path, document: &Value) -> Result<PathBuf> {
    let io = |error: std::io::Error| {
        Error::launcher(format!("cannot write {}: {error}", path.display()))
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temporary = path.with_file_name(format!("{file_name}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, dumps_sorted(document) + "\n").map_err(io)?;
    std::fs::rename(&temporary, path).map_err(io)?;
    Ok(path.to_path_buf())
}

/// Reads a JSON object, answering an empty object when absent or invalid.
pub fn read_json(path: &Path) -> Map<String, Value> {
    match std::fs::read(path) {
        Ok(bytes) => crate::values::parse_object(&bytes),
        Err(_) => Map::new(),
    }
}

/// An append-only event log; one record per line, flushed as it is written.
pub struct JsonlWriter {
    path: PathBuf,
    file: Mutex<Option<File>>,
}

impl JsonlWriter {
    pub fn open(path: &Path) -> Result<JsonlWriter> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::launcher(format!("cannot create {}: {e}", parent.display())))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| Error::launcher(format!("cannot open {}: {e}", path.display())))?;
        Ok(JsonlWriter {
            path: path.to_path_buf(),
            file: Mutex::new(Some(file)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn write(&self, record: &Value) {
        let line = dumps_compact(record) + "\n";
        if let Some(file) = self.file.lock().unwrap().as_mut() {
            let _ = file.write_all(line.as_bytes());
            let _ = file.flush();
        }
    }

    pub fn close(&self) {
        self.file.lock().unwrap().take();
    }
}

/// `YYYYmmddTHHMMSS-<pid>` in UTC.
pub fn new_run_id() -> String {
    format!(
        "{}-{}",
        crate::values::compact_utc(now_seconds()),
        std::process::id()
    )
}

/// One `serve` run: its command, logs, events and requests.
#[derive(Clone, Debug)]
pub struct RunDirectory {
    pub root: PathBuf,
    pub run_id: String,
    pub metadata: Map<String, Value>,
}

impl RunDirectory {
    pub fn serve_log(&self) -> PathBuf {
        self.root.join("serve.log")
    }

    pub fn event_log(&self) -> PathBuf {
        self.root.join("events.jsonl")
    }

    pub fn request_log(&self) -> PathBuf {
        self.root.join("requests.jsonl")
    }

    pub fn deployment_path(&self) -> PathBuf {
        self.root.join("serve.json")
    }

    pub fn metadata_path(&self) -> PathBuf {
        self.root.join("meta.json")
    }

    pub fn save(&self) -> Result<()> {
        write_json(&self.metadata_path(), &Value::Object(self.metadata.clone())).map(|_| ())
    }

    pub fn read_metadata(&self) -> Map<String, Value> {
        read_json(&self.metadata_path())
    }

    pub fn describe(&self) -> Value {
        json!({
            "id": self.run_id,
            "directory": self.root.to_string_lossy(),
            "serve_log": self.serve_log().to_string_lossy(),
            "event_log": self.event_log().to_string_lossy(),
            "request_log": self.request_log().to_string_lossy(),
            "metadata": self.metadata,
        })
    }
}

/// Creates and enumerates run directories under one state directory.
pub struct RunStore {
    pub root: PathBuf,
}

impl RunStore {
    pub fn new(root: &Path) -> RunStore {
        RunStore {
            root: root.to_path_buf(),
        }
    }

    pub fn create(
        &self,
        run_id: Option<&str>,
        metadata: Map<String, Value>,
    ) -> Result<RunDirectory> {
        let run_id = run_id.map(str::to_string).unwrap_or_else(new_run_id);
        let directory = RunDirectory {
            root: self.root.join("runs").join(&run_id),
            run_id,
            metadata,
        };
        std::fs::create_dir_all(&directory.root).map_err(|e| {
            Error::launcher(format!("cannot create {}: {e}", directory.root.display()))
        })?;
        directory.save()?;
        Ok(directory)
    }

    pub fn open(&self, run_id: &str) -> RunDirectory {
        RunDirectory {
            root: self.root.join("runs").join(run_id),
            run_id: run_id.to_string(),
            metadata: Map::new(),
        }
    }

    /// Run directories, newest first.
    pub fn list(&self) -> Vec<RunDirectory> {
        let mut names: Vec<String> = match std::fs::read_dir(self.root.join("runs")) {
            Ok(entries) => entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect(),
            Err(_) => Vec::new(),
        };
        names.sort();
        names.reverse();
        names.iter().map(|name| self.open(name)).collect()
    }

    pub fn latest(&self) -> Option<RunDirectory> {
        self.list().into_iter().next()
    }
}

/// One installed package the launcher can serve without the hub.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InstalledModel {
    pub repo: String,
    pub path: String,
    pub preset: Option<String>,
    pub revision: Option<String>,
    pub installed_at: Option<String>,
    pub size_bytes: Option<i64>,
    /// `local` for a package registered with `install --from-path`; absent
    /// for a hub install (the Python launcher's records have no source).
    pub source: Option<String>,
    /// Set when the installed directory carries its own `deployment.json` and
    /// `devices.json`; absent for an ordinary profile.
    pub scenario: Option<bool>,
    /// The hub revision those two documents were read at.
    pub scenario_revision: Option<String>,
}

impl InstalledModel {
    pub fn card_path(&self) -> PathBuf {
        PathBuf::from(&self.path).join("model.json")
    }

    pub fn describe(&self) -> Value {
        json!({
            "repo": self.repo,
            "path": self.path,
            "card": self.card_path().to_string_lossy(),
            "preset": self.preset,
            "revision": self.revision,
            "installed_at": self.installed_at,
            "size_bytes": self.size_bytes,
            "size": human_bytes(self.size_bytes),
            "source": self.source.as_deref().unwrap_or("hub"),
            "scenario": self.scenario.unwrap_or(false),
            "scenario_revision": self.scenario_revision,
        })
    }
}

/// The launcher's index of installed packages (`installs.json`).
pub struct InstallRegistry {
    pub root: PathBuf,
    pub path: PathBuf,
}

impl InstallRegistry {
    pub fn new(root: &Path) -> InstallRegistry {
        InstallRegistry {
            root: root.to_path_buf(),
            path: root.join("installs.json"),
        }
    }

    fn document(&self) -> Map<String, Value> {
        let mut document = read_json(&self.path);
        if !document.get("installs").is_some_and(Value::is_object) {
            document.insert("installs".into(), json!({}));
        }
        document
    }

    pub fn list(&self) -> Vec<InstalledModel> {
        let document = self.document();
        let installs = child(&document, "installs");
        let mut repos: Vec<&String> = installs.keys().collect();
        repos.sort();
        let mut models = Vec::new();
        for repo in repos {
            let entry = obj(&installs[repo]);
            let Some(recorded) = str_value(get(entry, "path")) else {
                continue;
            };
            models.push(InstalledModel {
                repo: string(get(entry, "repo")).unwrap_or_else(|| repo.clone()),
                path: self.recorded_path(recorded),
                preset: string(get(entry, "preset")),
                revision: string(get(entry, "revision")),
                installed_at: string(get(entry, "installed_at")),
                size_bytes: strict_int(get(entry, "size_bytes")),
                source: string(get(entry, "source")),
                scenario: get(entry, "scenario").as_bool(),
                scenario_revision: string(get(entry, "scenario_revision")),
            });
        }
        models
    }

    /// Resolves a recorded path, tolerating a relative one written before the
    /// state directory was made absolute: it is tried against the working
    /// directory and then every ancestor of the state root.
    fn recorded_path(&self, path: &str) -> String {
        let candidate = expand_user(Path::new(path));
        if candidate.is_absolute() {
            return candidate.to_string_lossy().into_owned();
        }
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let root = absolute(&self.root);
        let mut bases = vec![cwd.clone()];
        bases.extend(root.ancestors().skip(1).map(Path::to_path_buf));
        for base in bases {
            let resolved = base.join(&candidate);
            if resolved.is_dir() {
                return resolved.to_string_lossy().into_owned();
            }
        }
        cwd.join(candidate).to_string_lossy().into_owned()
    }

    pub fn get(&self, repo: &str) -> Option<InstalledModel> {
        self.list().into_iter().find(|m| m.repo == repo)
    }

    pub fn find_by_path(&self, path: &Path) -> Option<InstalledModel> {
        let resolved = canonical(path);
        self.list()
            .into_iter()
            .find(|m| canonical(Path::new(&m.path)) == resolved)
    }

    /// Records one install, keeping every other entry and top-level key.
    pub fn record(&self, model: &InstalledModel) -> Result<()> {
        let mut document = self.document();
        let installs = document
            .get_mut("installs")
            .and_then(Value::as_object_mut)
            .expect("document() guarantees an installs object");
        let mut entry = json!({
            "repo": model.repo,
            "path": model.path,
            "preset": model.preset,
            "revision": model.revision,
            "installed_at": model.installed_at,
            "size_bytes": model.size_bytes,
        });
        if let Some(source) = &model.source {
            entry["source"] = json!(source);
        }
        if let Some(scenario) = model.scenario {
            entry["scenario"] = json!(scenario);
        }
        if let Some(revision) = &model.scenario_revision {
            entry["scenario_revision"] = json!(revision);
        }
        installs.insert(model.repo.clone(), entry);
        write_json(&self.path, &Value::Object(document)).map(|_| ())
    }

    pub fn forget(&self, repo: &str) -> Result<bool> {
        let mut document = self.document();
        let installs = document
            .get_mut("installs")
            .and_then(Value::as_object_mut)
            .expect("document() guarantees an installs object");
        if installs.shift_remove(repo).is_none() {
            return Ok(false);
        }
        write_json(&self.path, &Value::Object(document))?;
        Ok(true)
    }

    /// The most recently installed package whose card still exists.
    pub fn latest(&self) -> Option<InstalledModel> {
        let mut models: Vec<InstalledModel> = self
            .list()
            .into_iter()
            .filter(|m| m.card_path().is_file())
            .collect();
        models.sort_by(|a, b| {
            a.installed_at
                .clone()
                .unwrap_or_default()
                .cmp(&b.installed_at.clone().unwrap_or_default())
        });
        models.pop()
    }
}

fn canonical(path: &Path) -> PathBuf {
    let expanded = expand_user(path);
    std::fs::canonicalize(&expanded).unwrap_or_else(|_| absolute(&expanded))
}

/// Cached profile documents, so listing and showing work offline.
pub struct ProfileCache {
    pub root: PathBuf,
    pub max_age_seconds: f64,
}

impl ProfileCache {
    pub fn new(state: &Path) -> ProfileCache {
        ProfileCache {
            root: state.join("cache").join("profiles"),
            max_age_seconds: 6.0 * 3600.0,
        }
    }

    fn path(&self, repo: &str) -> PathBuf {
        self.root.join(format!("{}.json", repo.replace('/', "-")))
    }

    /// A cached entry younger than `max_age_seconds`.
    pub fn get(&self, repo: &str) -> Option<Map<String, Value>> {
        let path = self.path(repo);
        let modified = std::fs::metadata(&path).ok()?.modified().ok()?;
        let age = std::time::SystemTime::now()
            .duration_since(modified)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        if age > self.max_age_seconds {
            return None;
        }
        Some(read_json(&path)).filter(|d| !d.is_empty())
    }

    /// A cached entry of any age (offline use: stale beats nothing).
    pub fn get_any(&self, repo: &str) -> Option<Map<String, Value>> {
        Some(read_json(&self.path(repo))).filter(|d| !d.is_empty())
    }

    pub fn put(&self, repo: &str, document: &Value) -> Result<()> {
        write_json(&self.path(repo), document).map(|_| ())
    }

    /// Every cached entry with its file stem (`<owner>-<name>`).
    pub fn entries(&self) -> Vec<(String, Map<String, Value>)> {
        let mut found = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let stem = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let document = read_json(&path);
                if !document.is_empty() {
                    found.push((stem, document));
                }
            }
        }
        found.sort_by(|a, b| a.0.cmp(&b.0));
        found
    }

    pub fn clear(&self) -> usize {
        self.entries()
            .iter()
            .filter(|(stem, _)| {
                std::fs::remove_file(self.root.join(format!("{stem}.json"))).is_ok()
            })
            .count()
    }
}

/// Stored local benchmark runs, one JSON record each, plus an index.
pub struct BenchmarkStore {
    pub root: PathBuf,
    pub index_path: PathBuf,
}

impl BenchmarkStore {
    pub fn new(state: &Path) -> BenchmarkStore {
        let root = state.join("benchmarks");
        BenchmarkStore {
            index_path: root.join("index.jsonl"),
            root,
        }
    }

    pub fn save(&self, record: &mut Map<String, Value>) -> Result<PathBuf> {
        let slug = str_value(get(record, "profile"))
            .unwrap_or("unknown")
            .replace('/', "-");
        let run_id = string(get(record, "id")).unwrap_or_else(new_run_id);
        record.insert("id".into(), json!(run_id));
        let path = self.root.join(&slug).join(format!("{run_id}.json"));
        write_json(&path, &Value::Object(record.clone()))?;
        let line = json!({
            "id": run_id,
            "profile": get(record, "profile"),
            "suite": get(record, "suite"),
            "started_at": get(record, "started_at"),
            "path": path.to_string_lossy(),
            "ok": get(record, "ok"),
        });
        let index = JsonlWriter::open(&self.index_path)?;
        index.write(&line);
        Ok(path)
    }

    pub fn index(&self) -> Vec<Map<String, Value>> {
        let Ok(text) = std::fs::read_to_string(&self.index_path) else {
            return Vec::new();
        };
        text.lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .filter_map(|line| match serde_json::from_str::<Value>(line) {
                Ok(Value::Object(map)) => Some(map),
                _ => None,
            })
            .collect()
    }

    pub fn list(&self, profile: Option<&str>) -> Vec<Map<String, Value>> {
        self.index()
            .into_iter()
            .filter(|r| str_value(get(r, "path")).is_some())
            .filter(|r| profile.is_none_or(|p| str_value(get(r, "profile")) == Some(p)))
            .collect()
    }

    pub fn latest(&self, profile: Option<&str>) -> Option<Map<String, Value>> {
        let records = self.list(profile);
        let last = records.last()?;
        Some(read_json(Path::new(str_value(get(last, "path"))?))).filter(|d| !d.is_empty())
    }

    pub fn load(&self, run_id: &str) -> Option<Map<String, Value>> {
        for record in self.index() {
            if str_value(get(&record, "id")) == Some(run_id) {
                if let Some(path) = str_value(get(&record, "path")) {
                    return Some(read_json(Path::new(path))).filter(|d| !d.is_empty());
                }
            }
        }
        None
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh directory under `target/test-scratch` (never the RAM-backed /tmp).
    pub(crate) fn scratch(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-scratch")
            .join(format!(
                "{name}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn json_documents_round_trip_whole() {
        let root = scratch("store-json");
        let path = write_json(&root.join("a").join("b.json"), &json!({"value": 1})).unwrap();
        assert_eq!(Value::Object(read_json(&path)), json!({"value": 1}));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\n \"value\": 1\n}\n"
        );
        assert!(read_json(&root.join("absent.json")).is_empty());
        std::fs::write(root.join("broken.json"), "{").unwrap();
        assert!(read_json(&root.join("broken.json")).is_empty());
    }

    #[test]
    fn jsonl_logs_append_one_record_per_line() {
        let root = scratch("store-jsonl");
        let writer = JsonlWriter::open(&root.join("events.jsonl")).unwrap();
        writer.write(&json!({"event": "ready"}));
        writer.write(&json!({"event": "serving", "endpoint": "http://x"}));
        writer.close();
        let text = std::fs::read_to_string(root.join("events.jsonl")).unwrap();
        assert_eq!(
            text,
            "{\"event\":\"ready\"}\n{\"event\":\"serving\",\"endpoint\":\"http://x\"}\n"
        );
    }

    #[test]
    fn run_directories_are_created_and_enumerated() {
        let root = scratch("store-runs");
        let store = RunStore::new(&root);
        let mut metadata = Map::new();
        metadata.insert("repo".into(), json!("a/b"));
        let first = store.create(Some("20260101T000000-1"), metadata).unwrap();
        store.create(Some("20260102T000000-1"), Map::new()).unwrap();
        assert_eq!(first.serve_log().file_name().unwrap(), "serve.log");
        assert_eq!(first.deployment_path().file_name().unwrap(), "serve.json");
        assert_eq!(
            store.open("20260101T000000-1").read_metadata()["repo"],
            "a/b"
        );
        let ids: Vec<_> = store.list().into_iter().map(|r| r.run_id).collect();
        assert_eq!(ids, ["20260102T000000-1", "20260101T000000-1"]);
        assert_eq!(store.latest().unwrap().run_id, "20260102T000000-1");
    }

    #[test]
    fn installs_are_recorded_and_can_be_forgotten() {
        let root = scratch("store-installs");
        std::fs::write(
            root.join("installs.json"),
            "{\"installs\": {\"other/x\": {\"repo\": \"other/x\", \"path\": \"/nowhere\"}}, \"foreign\": 7}",
        )
        .unwrap();
        let registry = InstallRegistry::new(&root);
        let model = InstalledModel {
            repo: "loom-fun/a-loom".into(),
            path: root.join("models").join("a").to_string_lossy().into_owned(),
            preset: Some("chat".into()),
            revision: Some("abc".into()),
            installed_at: Some("2026-11-12T00:00:00Z".into()),
            size_bytes: Some(1024),
            ..Default::default()
        };
        std::fs::create_dir_all(root.join("models").join("a")).unwrap();
        registry.record(&model).unwrap();
        assert_eq!(
            registry.get("loom-fun/a-loom").unwrap().preset.as_deref(),
            Some("chat")
        );
        assert_eq!(
            registry
                .find_by_path(&root.join("models").join("a"))
                .unwrap()
                .repo,
            "loom-fun/a-loom"
        );
        assert_eq!(
            registry.get("loom-fun/a-loom").unwrap().describe()["size"],
            "1.0 KiB"
        );
        let raw = read_json(&root.join("installs.json"));
        assert_eq!(raw["foreign"], 7);
        assert!(raw["installs"]["other/x"].is_object());
        assert!(registry.forget("loom-fun/a-loom").unwrap());
        assert!(!registry.forget("loom-fun/a-loom").unwrap());
        assert!(registry.get("loom-fun/a-loom").is_none());
    }

    #[test]
    fn a_legacy_relative_install_path_still_resolves() {
        let base = scratch("store-legacy");
        let state = base.join("state");
        let card = state.join("models").join("a-loom").join("model.json");
        std::fs::create_dir_all(card.parent().unwrap()).unwrap();
        std::fs::write(&card, "{}").unwrap();
        write_json(
            &state.join("installs.json"),
            &json!({"installs": {"a/loom": {"path": "state/models/a-loom"}}}),
        )
        .unwrap();
        let model = InstallRegistry::new(&state).get("a/loom").unwrap();
        assert_eq!(
            PathBuf::from(&model.path),
            state.join("models").join("a-loom")
        );
        assert_eq!(model.card_path(), card);
    }

    #[test]
    fn profile_cache_round_trips_and_clears() {
        let root = scratch("store-cache");
        let cache = ProfileCache::new(&root);
        assert!(cache.get("loom-fun/a-loom").is_none());
        cache
            .put(
                "loom-fun/a-loom",
                &json!({"model_card": {"model": {"id": "a"}}}),
            )
            .unwrap();
        assert_eq!(
            cache.get("loom-fun/a-loom").unwrap()["model_card"]["model"]["id"],
            "a"
        );
        assert_eq!(cache.entries().len(), 1);
        assert_eq!(cache.clear(), 1);
        assert!(cache.get_any("loom-fun/a-loom").is_none());
    }

    #[test]
    fn benchmark_records_are_stored_indexed_and_loadable() {
        let root = scratch("store-bench");
        let store = BenchmarkStore::new(&root);
        assert!(store.list(None).is_empty());
        let mut record = obj(&json!({
            "id": "run-1",
            "profile": "loom-fun/a-loom",
            "suite": "quick",
            "started_at": "2026-11-12T00:00:00Z",
            "ok": true,
            "metrics": {"decode": {"value": 10.0}},
        }))
        .clone();
        let path = store.save(&mut record).unwrap();
        assert!(path.is_file());
        assert!(path.ends_with("benchmarks/loom-fun-a-loom/run-1.json"));
        assert_eq!(store.index()[0]["id"], "run-1");
        assert_eq!(store.list(Some("loom-fun/a-loom")).len(), 1);
        assert!(store.list(Some("other")).is_empty());
        assert_eq!(
            store.latest(Some("loom-fun/a-loom")).unwrap()["metrics"]["decode"]["value"],
            10.0
        );
        assert_eq!(store.load("run-1").unwrap()["suite"], "quick");
        assert!(store.load("absent").is_none());
    }

    #[test]
    fn a_relative_state_directory_resolves_absolutely() {
        assert!(state_directory(Some("relative/state")).is_absolute());
    }
}
