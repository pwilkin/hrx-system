//! Prepared scenarios: packages that carry their own deployment and machine.
//!
//! A package directory holding both `deployment.json` and `devices.json` was
//! prepared for one machine shape (for example a GPU plus an XDNA NPU sharing an
//! exchange pool). The launcher serves it exactly as prepared:
//! `loom-serve --devices=<devices.json> --model=<deployment.json>`, without a
//! composed deployment or a `--device`. The scenario's documents own its
//! capacities and devices, so launcher options that would change them are
//! refused rather than silently dropped.

use crate::deployment::{absolute, ContextSettings};
use crate::errors::{Error, Result};
use crate::profile::{Execution, CARD_DOCUMENT};
use crate::store::write_json;
use crate::values::{child, get, int_value, obj, str_value, string};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const DEPLOYMENT_DOCUMENT: &str = "deployment.json";
pub const DEVICES_DOCUMENT: &str = "devices.json";

/// One package prepared with its own deployment and device documents.
#[derive(Clone, Debug)]
pub struct Scenario {
    pub directory: PathBuf,
    pub deployment_path: PathBuf,
    pub devices_path: PathBuf,
    pub deployment: Map<String, Value>,
    pub devices: Map<String, Value>,
    /// The capacities the deployment declares over its card's preset.
    pub settings: ContextSettings,
}

/// True when `directory` holds both scenario documents.
pub fn is_scenario(directory: &Path) -> bool {
    directory.join(DEPLOYMENT_DOCUMENT).is_file() && directory.join(DEVICES_DOCUMENT).is_file()
}

fn read_document(path: &Path) -> Result<Map<String, Value>> {
    let text = std::fs::read(path)
        .map_err(|error| Error::launch(format!("{} is unreadable: {error}", path.display())))?;
    match serde_json::from_slice::<Value>(&text) {
        Ok(Value::Object(document)) => Ok(document),
        Ok(_) => Err(Error::launch(format!(
            "{} is not a JSON object",
            path.display()
        ))),
        Err(error) => Err(Error::launch(format!(
            "{} is not JSON: {error}",
            path.display()
        ))),
    }
}

impl Scenario {
    /// Reads the scenario in `directory` (a package directory or its card),
    /// `None` when the directory is an ordinary profile.
    pub fn detect(directory: &Path) -> Result<Option<Scenario>> {
        let directory = if directory.is_file() {
            directory.parent().unwrap_or(Path::new("."))
        } else {
            directory
        };
        if !is_scenario(directory) {
            return Ok(None);
        }
        let directory = absolute(directory);
        let deployment_path = directory.join(DEPLOYMENT_DOCUMENT);
        let devices_path = directory.join(DEVICES_DOCUMENT);
        let deployment = read_document(&deployment_path)?;
        let devices = read_document(&devices_path)?;
        if child(&devices, "devices").is_empty() {
            return Err(Error::launch(format!(
                "{} names no devices",
                devices_path.display()
            )));
        }
        let settings = declared_settings(&directory, &deployment)?;
        Ok(Some(Scenario {
            directory,
            deployment_path,
            devices_path,
            deployment,
            devices,
            settings,
        }))
    }

    /// The deployment's registered name.
    pub fn name(&self) -> String {
        string(get(&self.deployment, "name")).unwrap_or_else(|| {
            self.directory
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    }

    /// Device names in the order `devices.json` lists them.
    pub fn device_names(&self) -> Vec<String> {
        child(&self.devices, "devices").keys().cloned().collect()
    }

    /// `name -> uri` for every device the scenario binds.
    pub fn device_uris(&self) -> Map<String, Value> {
        child(&self.devices, "devices")
            .iter()
            .map(|(name, entry)| (name.clone(), get(obj(entry), "uri").clone()))
            .collect()
    }

    /// `amdgpu + xdna`.
    pub fn device_text(&self) -> String {
        self.device_names().join(" + ")
    }

    /// The pool the deployment places its model in.
    pub fn pool(&self) -> Option<String> {
        string(get(&self.deployment, "pool"))
    }

    pub fn pool_names(&self) -> Vec<String> {
        child(&self.devices, "pools").keys().cloned().collect()
    }

    /// The machine-readable record kept in `current.json`, `meta.json` and `/api/status`.
    pub fn describe(&self) -> Value {
        json!({
            "kind": "scenario",
            "directory": self.directory.to_string_lossy(),
            "deployment": self.name(),
            "deployment_path": self.deployment_path.to_string_lossy(),
            "devices_path": self.devices_path.to_string_lossy(),
            "devices": self.device_text(),
            "device_uris": self.device_uris(),
            "pool": self.pool(),
            "pools": self.pool_names(),
            "preset": get(child(&self.deployment, "model"), "preset"),
            "context_length": self.settings.context_length,
            "rows": self.settings.rows,
            "max_tokens": self.settings.max_tokens,
            "prefill_capacity": self.settings.prefill_capacity,
            "draft_depth": self.settings.draft_depth,
        })
    }

    /// Refuses launcher options that would change what the scenario declares.
    /// `options` names each offending option as the caller spells it.
    pub fn refuse(&self, options: &[String]) -> Result<()> {
        if options.is_empty() {
            return Ok(());
        }
        let dash = |v: Option<i64>| v.map(|n| n.to_string()).unwrap_or_else(|| "-".into());
        Err(Error::launch(format!(
            "{} is a prepared scenario: it owns its capacities and devices, so {} cannot \
             change them. Its capacities come from {} (deployment {}: context {}, max tokens {}, \
             rows {}, draft depth {}) and its devices from {} ({}); drop {} or edit those files",
            self.directory.display(),
            options.join(", "),
            self.deployment_path.display(),
            self.name(),
            dash(self.settings.context_length),
            dash(self.settings.max_tokens),
            dash(self.settings.rows),
            dash(self.settings.draft_depth),
            self.devices_path.display(),
            self.device_text(),
            if options.len() == 1 { "it" } else { "them" },
        )))
    }

    /// The devices document the server reads. `loom-serve` refuses
    /// `--memory_bytes` and `--pool_backing` next to `--devices`, so a launch
    /// that sets them gets a copy of `devices.json` in its run directory with
    /// the deployment's pool changed; otherwise the package's own file is used.
    pub fn devices_for(
        &self,
        memory_bytes: Option<i64>,
        pool_backing: Option<&str>,
        run_directory: &Path,
    ) -> Result<(PathBuf, Map<String, Value>)> {
        let mut overrides = Map::new();
        if let Some(memory) = memory_bytes {
            if memory < 0 {
                return Err(Error::launch("memory bytes must not be negative"));
            }
            overrides.insert("memory_bytes".into(), json!(memory));
        }
        if let Some(backing) = pool_backing.filter(|b| !b.is_empty()) {
            if !matches!(backing, "elastic" | "fixed") {
                return Err(Error::launch("pool backing must be elastic or fixed"));
            }
            overrides.insert("pool_backing".into(), json!(backing));
        }
        if overrides.is_empty() {
            return Ok((self.devices_path.clone(), overrides));
        }
        let pools = self.pool_names();
        let pool = match self.pool() {
            Some(pool) => pool,
            None if pools.len() == 1 => pools[0].clone(),
            None => {
                return Err(Error::launch(format!(
                    "{} names no pool and {} has {} pools, so --memory-bytes and \
                     --pool-backing cannot tell which to change; edit the pool in {}",
                    self.deployment_path.display(),
                    self.devices_path.display(),
                    pools.len(),
                    self.devices_path.display()
                )))
            }
        };
        let mut devices = self.devices.clone();
        let entry = devices
            .get_mut("pools")
            .and_then(Value::as_object_mut)
            .and_then(|pools| pools.get_mut(&pool))
            .and_then(Value::as_object_mut)
            .ok_or_else(|| {
                Error::launch(format!(
                    "{} places the model in pool {pool}, which {} does not declare",
                    self.deployment_path.display(),
                    self.devices_path.display()
                ))
            })?;
        for (key, value) in &overrides {
            entry.insert(key.clone(), value.clone());
        }
        let fixed = str_value(get(entry, "pool_backing")) == Some("fixed");
        if fixed && int_value(get(entry, "memory_bytes")).is_some_and(|m| m != 0) {
            return Err(Error::launch(format!(
                "pool {pool} would be fixed with a memory budget; loom-serve requires elastic \
                 backing for a physical budget (drop --memory-bytes or --pool-backing fixed)"
            )));
        }
        overrides.insert("pool".into(), json!(pool));
        let path = write_json(
            &run_directory.join(DEVICES_DOCUMENT),
            &Value::Object(devices),
        )?;
        Ok((path, overrides))
    }
}

/// The capacities a scenario serves: its card's preset execution, overlaid
/// with whatever the deployment's `definition` sets, as `loom-serve` resolves them.
fn declared_settings(directory: &Path, deployment: &Map<String, Value>) -> Result<ContextSettings> {
    let model = child(deployment, "model");
    let card_path = directory.join(str_value(get(model, "path")).unwrap_or(CARD_DOCUMENT));
    let card = if card_path.is_file() {
        read_document(&card_path)?
    } else {
        return Err(Error::launch(format!(
            "{} loads {}, which does not exist",
            directory.join(DEPLOYMENT_DOCUMENT).display(),
            card_path.display()
        )));
    };
    let execution =
        Execution::from_card(&card, str_value(get(model, "preset"))).unwrap_or_default();
    let definition = child(deployment, "definition");
    let field = |key: &str, declared: Option<i64>| int_value(get(definition, key)).or(declared);
    Ok(ContextSettings {
        context_length: field("context_capacity", execution.context_capacity),
        max_tokens: field("max_tokens", execution.max_tokens),
        rows: field("rows", execution.rows),
        max_sessions: field("max_sessions", None),
        prefill_capacity: field("prefill_capacity", execution.prefill_capacity),
        checkpoint_capacity: field("checkpoint_capacity", None),
        session_cache_capacity: field("session_cache_capacity", None),
        prefix_cache_capacity: field("prefix_cache_capacity", None),
        cache_target_bytes: field("cache_target_bytes", None),
        draft_depth: field("draft_depth", execution.draft_depth),
        continuation_epochs: field("continuation_epochs", None),
        pending_requests: field("pending_requests", None),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::store::tests::scratch;

    /// A copy of the GPU/NPU hybrid package's documents (no payloads).
    pub(crate) fn hybrid_package(root: &Path) -> PathBuf {
        let directory = root.join("hybrid");
        std::fs::create_dir_all(&directory).unwrap();
        let write = |name: &str, value: Value| {
            std::fs::write(
                directory.join(name),
                serde_json::to_string_pretty(&value).unwrap(),
            )
            .unwrap()
        };
        write(
            "model.json",
            json!({
                "schema_version": 2,
                "model": {"id": "qwen3.8-27b", "name": "Qwen3.8 27B proven native hybrid"},
                "programs": {"source": {
                    "device_placement": {"*": "amdgpu", "resident_lifetime": "xdna"},
                    "exchange_pool": "exchange"}},
                "services": {"chat": {"kind": "text", "version": 1}},
                "presets": {"chat": {"service": "chat", "variant": "gguf", "programs": "source",
                    "execution": {"rows": 1, "context_capacity": 40960, "prefill_capacity": 1024,
                                  "max_tokens": 4096, "draft_depth": 0, "group_mode": "two_bank"}}},
            }),
        );
        write(
            "deployment.json",
            json!({"name": "qwen3.8-27b", "model": {"path": "model.json", "preset": "chat"},
                   "definition": {"kind": "text"}, "device": "amdgpu", "pool": "model"}),
        );
        write(
            "devices.json",
            json!({"devices": {"amdgpu": {"uri": "amdgpu://0"}, "xdna": {"uri": "xdna://0?columns=8"}},
                   "pools": {"model": {"devices": ["amdgpu"], "memory_bytes": 52000000000_i64},
                             "exchange": {"devices": ["amdgpu", "xdna"], "uncached": true,
                                          "uncached_devices": ["amdgpu"], "pool_backing": "fixed"}}}),
        );
        write(
            "llm_config.json",
            json!({"arch": "qwen35", "gguf": "Qwen3.8-27B-Q4_K_M.gguf", "context_length": 262144}),
        );
        directory
    }

    #[test]
    fn a_package_with_both_documents_is_a_scenario() {
        let root = scratch("scenario-detect");
        let directory = hybrid_package(&root);
        let scenario = Scenario::detect(&directory).unwrap().unwrap();
        assert_eq!(scenario.name(), "qwen3.8-27b");
        assert_eq!(scenario.device_text(), "amdgpu + xdna");
        assert_eq!(scenario.device_uris()["xdna"], "xdna://0?columns=8");
        assert_eq!(scenario.pool().as_deref(), Some("model"));
        assert_eq!(scenario.settings.context_length, Some(40960));
        assert_eq!(scenario.settings.max_tokens, Some(4096));
        assert_eq!(scenario.settings.rows, Some(1));
        assert_eq!(scenario.settings.draft_depth, Some(0));
        let described = scenario.describe();
        assert_eq!(described["kind"], "scenario");
        assert_eq!(described["devices"], "amdgpu + xdna");
        // The card path also detects it; one document alone does not.
        assert!(Scenario::detect(&directory.join("model.json"))
            .unwrap()
            .is_some());
        std::fs::remove_file(directory.join("devices.json")).unwrap();
        assert!(Scenario::detect(&directory).unwrap().is_none());
    }

    #[test]
    fn a_deployment_definition_overrides_the_preset() {
        let root = scratch("scenario-definition");
        let directory = hybrid_package(&root);
        std::fs::write(
            directory.join("deployment.json"),
            r#"{"name": "n", "model": {"path": "model.json", "preset": "chat"},
                "definition": {"kind": "text", "context_capacity": 8192, "rows": 2}, "pool": "model"}"#,
        )
        .unwrap();
        let scenario = Scenario::detect(&directory).unwrap().unwrap();
        assert_eq!(scenario.settings.context_length, Some(8192));
        assert_eq!(scenario.settings.rows, Some(2));
        assert_eq!(scenario.settings.max_tokens, Some(4096));
    }

    #[test]
    fn contradicting_options_are_refused_naming_the_files() {
        let root = scratch("scenario-refuse");
        let scenario = Scenario::detect(&hybrid_package(&root)).unwrap().unwrap();
        assert!(scenario.refuse(&[]).is_ok());
        let error = scenario
            .refuse(&["--context".into(), "--parallel".into()])
            .unwrap_err();
        let message = error.message;
        assert!(message.contains("prepared scenario"), "{message}");
        assert!(message.contains("owns its capacities"), "{message}");
        assert!(message.contains("--context, --parallel"), "{message}");
        assert!(message.contains("deployment.json"), "{message}");
        assert!(message.contains("devices.json"), "{message}");
        assert!(message.contains("context 40960"), "{message}");
        assert!(message.contains("amdgpu + xdna"), "{message}");
    }

    #[test]
    fn pool_overrides_go_to_a_run_copy_of_the_devices() {
        let root = scratch("scenario-pools");
        let scenario = Scenario::detect(&hybrid_package(&root)).unwrap().unwrap();
        let run = root.join("run");
        std::fs::create_dir_all(&run).unwrap();
        let (path, overrides) = scenario.devices_for(None, None, &run).unwrap();
        assert_eq!(path, scenario.devices_path);
        assert!(overrides.is_empty());
        let (path, overrides) = scenario
            .devices_for(Some(48000000000), Some("elastic"), &run)
            .unwrap();
        assert_eq!(path, run.join("devices.json"));
        assert_eq!(overrides["pool"], "model");
        let written = crate::store::read_json(&path);
        assert_eq!(written["pools"]["model"]["memory_bytes"], 48000000000_i64);
        assert_eq!(written["pools"]["model"]["pool_backing"], "elastic");
        assert_eq!(written["pools"]["exchange"]["pool_backing"], "fixed");
        // The package's own file is untouched.
        let original = crate::store::read_json(&scenario.devices_path);
        assert_eq!(original["pools"]["model"]["memory_bytes"], 52000000000_i64);
        assert!(scenario
            .devices_for(None, Some("fixed"), &run)
            .unwrap_err()
            .message
            .contains("requires elastic"));
        assert!(scenario.devices_for(None, Some("huge"), &run).is_err());
    }
}
