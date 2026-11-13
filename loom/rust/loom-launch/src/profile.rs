//! Published Loom profiles: cards, measurements, chat policy and requirements.
//!
//! A profile merges the documents a Loom publisher ships: the portable model
//! card (`model.json`, the runtime contract), the measurement record
//! (`info.json`, schema `loom-card/2`), the generator's configuration
//! (`llm_config.json`, which owns the chat policy) and the human README. Each
//! document is read tolerantly.

use crate::errors::{Error, ErrorKind, Result};
use crate::hub::{Hub, HubModel};
use crate::values::{
    arr, bool_value, child, get, int_value, number_value, obj, opt_f64, str_value, string, truthy,
};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const CARD_DOCUMENT: &str = "model.json";
pub const MEASUREMENT_DOCUMENT: &str = "info.json";
pub const CONFIG_DOCUMENT: &str = "llm_config.json";
pub const README_DOCUMENT: &str = "README.md";

/// Metrics a Loom measurement record publishes, in publishing order.
pub const METRIC_ORDER: [&str; 10] = [
    "decode",
    "decode_2k",
    "decode_32k",
    "prefill_1k",
    "prefill_2k",
    "prefill_8k",
    "prefill_16k",
    "prefill_32k",
    "streams4",
    "spec_acceptance",
];

pub fn metric_label(metric: &str) -> &str {
    match metric {
        "decode" => "decode, short prompt",
        "decode_2k" => "decode, 2k context",
        "decode_32k" => "decode, 32k context",
        "prefill_1k" => "prefill, 1k document",
        "prefill_2k" => "prefill, 2k document",
        "prefill_8k" => "prefill, 8k document",
        "prefill_16k" => "prefill, 16k document",
        "prefill_32k" => "prefill, 32k document",
        "streams4" => "decode, 4 streams",
        "spec_acceptance" => "draft acceptance",
        other => other,
    }
}

/// Metrics measured as a fraction rather than a rate.
pub fn is_fraction_metric(metric: &str) -> bool {
    metric == "spec_acceptance"
}

/// One capability the publisher's evidence established.
#[derive(Clone, Debug)]
pub struct Feature {
    pub name: String,
    pub status: String,
    pub detail: Map<String, Value>,
}

impl Feature {
    pub fn enabled(&self) -> bool {
        self.status == "yes"
    }

    pub fn describe(&self) -> Value {
        json!({"status": self.status, "detail": self.detail})
    }
}

/// One correctness gate the publisher ran against the package.
#[derive(Clone, Debug)]
pub struct Gate {
    pub name: String,
    pub status: String,
    pub level: Option<String>,
    pub date: Option<String>,
    pub source: Option<String>,
}

impl Gate {
    pub fn passed(&self) -> bool {
        self.status == "pass"
    }

    pub fn failed(&self) -> bool {
        matches!(self.status.as_str(), "fail" | "failed" | "error")
    }

    pub fn describe(&self) -> Value {
        json!({
            "status": self.status,
            "level": self.level,
            "date": self.date,
            "source": self.source,
        })
    }
}

/// One measured metric: this engine, the baseline and the hardware ceiling.
#[derive(Clone, Debug, Default)]
pub struct Measurement {
    pub metric: String,
    pub value: Option<f64>,
    pub baseline: Option<f64>,
    pub acceptance: Option<f64>,
    pub ceiling: Option<f64>,
    pub date: Option<String>,
}

impl Measurement {
    pub fn label(&self) -> &str {
        metric_label(&self.metric)
    }

    pub fn unit(&self) -> &'static str {
        if is_fraction_metric(&self.metric) {
            "fraction"
        } else {
            "tok/s"
        }
    }

    pub fn roofline_fraction(&self) -> Option<f64> {
        match (self.value, self.ceiling) {
            (Some(value), Some(ceiling)) if ceiling != 0.0 => Some(value / ceiling),
            _ => None,
        }
    }

    pub fn baseline_ratio(&self) -> Option<f64> {
        match (self.value, self.baseline) {
            (Some(value), Some(baseline)) if baseline != 0.0 => Some(value / baseline),
            _ => None,
        }
    }

    pub fn describe(&self) -> Value {
        json!({
            "metric": self.metric,
            "label": self.label(),
            "unit": self.unit(),
            "value": opt_f64(self.value),
            "baseline": opt_f64(self.baseline),
            "acceptance": opt_f64(self.acceptance),
            "ceiling": opt_f64(self.ceiling),
            "roofline_fraction": opt_f64(self.roofline_fraction()),
            "baseline_ratio": opt_f64(self.baseline_ratio()),
            "date": self.date,
        })
    }
}

/// The measurements a card publishes, with their baseline and roofline.
#[derive(Clone, Debug, Default)]
pub struct Performance {
    pub measurements: Vec<Measurement>,
    pub baseline_name: Option<String>,
    pub baseline_version: Option<String>,
    pub baseline_args: Option<String>,
    pub device: Option<String>,
    /// Other engines the publisher compared against: name -> metric -> value.
    pub extra_baselines: BTreeMap<String, BTreeMap<String, f64>>,
    pub session_count: usize,
}

impl Performance {
    /// Reads `info.json`; the last session that measured a metric wins.
    pub fn from_measurement_document(info: &Map<String, Value>) -> Option<Performance> {
        let performance = child(info, "performance");
        let sessions: Vec<&Map<String, Value>> =
            arr(get(performance, "sessions")).iter().map(obj).collect();
        let roofline = child(info, "roofline");
        let roofline_metrics = child(roofline, "metrics");
        if sessions.is_empty() && roofline_metrics.is_empty() {
            return None;
        }
        let mut latest: BTreeMap<String, (&Map<String, Value>, Option<String>)> = BTreeMap::new();
        for session in &sessions {
            let date = string(get(session, "date"));
            for (name, entry) in child(session, "metrics") {
                let entry = obj(entry);
                if number_value(get(entry, "loom")).is_none() {
                    continue;
                }
                latest.insert(name.clone(), (entry, date.clone()));
            }
        }
        let mut ordered: Vec<String> = METRIC_ORDER
            .iter()
            .filter(|name| latest.contains_key(**name))
            .map(|name| name.to_string())
            .collect();
        ordered.extend(
            latest
                .keys()
                .filter(|name| !METRIC_ORDER.contains(&name.as_str()))
                .cloned(),
        );
        let measurements = ordered
            .iter()
            .map(|name| {
                let (entry, date) = &latest[name];
                Measurement {
                    metric: name.clone(),
                    value: number_value(get(entry, "loom")),
                    baseline: number_value(get(entry, "baseline")),
                    acceptance: number_value(get(entry, "acceptance")),
                    ceiling: number_value(get(child(roofline_metrics, name), "ceiling")),
                    date: date.clone(),
                }
            })
            .collect();
        let mut extra = BTreeMap::new();
        for (name, entry) in child(performance, "baselines") {
            let session = child(obj(entry), "session");
            let mut values = BTreeMap::new();
            for (metric, row) in child(session, "metrics") {
                if let Some(value) = number_value(get(obj(row), "baseline")) {
                    values.insert(metric.clone(), value);
                }
            }
            if !values.is_empty() {
                extra.insert(name.clone(), values);
            }
        }
        let baseline = child(performance, "baseline");
        let device = child(roofline, "device");
        Some(Performance {
            measurements,
            baseline_name: string(get(baseline, "name")),
            baseline_version: string(get(baseline, "version")),
            baseline_args: string(get(baseline, "args")),
            device: string(get(device, "name")),
            extra_baselines: extra,
            session_count: sessions.len(),
        })
    }

    pub fn measurement(&self, metric: &str) -> Option<&Measurement> {
        self.measurements.iter().rev().find(|m| m.metric == metric)
    }

    pub fn has_measurements(&self) -> bool {
        self.measurements.iter().any(|m| m.value.is_some())
    }
}

/// The capacities a card's preset declares for its service.
#[derive(Clone, Debug, Default)]
pub struct Execution {
    pub preset: Option<String>,
    pub service: Option<String>,
    pub service_kind: Option<String>,
    pub rows: Option<i64>,
    pub context_capacity: Option<i64>,
    pub pool_capacity: Option<i64>,
    pub prefill_capacity: Option<i64>,
    pub max_tokens: Option<i64>,
    pub draft_depth: Option<i64>,
}

impl Execution {
    pub fn from_card(
        model_card: &Map<String, Value>,
        preferred: Option<&str>,
    ) -> Option<Execution> {
        let presets = child(model_card, "presets");
        let declared = preferred.or_else(|| str_value(get(model_card, "preset")));
        let selected = choose_preset(presets, declared)?;
        let entry = child(presets, &selected);
        let execution = child(entry, "execution");
        let service_name = string(get(entry, "service"));
        let services = child(model_card, "services");
        let service = match &service_name {
            Some(name) => child(services, name),
            None => obj(&Value::Null),
        };
        Some(Execution {
            preset: Some(selected),
            service: service_name.clone(),
            service_kind: string(get(service, "kind")),
            rows: int_value(get(execution, "rows")),
            context_capacity: int_value(get(execution, "context_capacity")),
            pool_capacity: int_value(get(execution, "pool_capacity")),
            prefill_capacity: int_value(get(execution, "prefill_capacity")),
            max_tokens: int_value(get(execution, "max_tokens")),
            draft_depth: int_value(get(execution, "draft_depth")),
        })
    }

    pub fn describe(&self) -> Value {
        json!({
            "preset": self.preset,
            "service": self.service,
            "service_kind": self.service_kind,
            "rows": self.rows,
            "context_capacity": self.context_capacity,
            "pool_capacity": self.pool_capacity,
            "prefill_capacity": self.prefill_capacity,
            "max_tokens": self.max_tokens,
            "draft_depth": self.draft_depth,
        })
    }
}

/// The chat rendering policy the package's generator emitted.
///
/// The served model renders prompts in its own programs; this policy is what
/// the generator recorded, which lets the launcher preview a rendered prompt and
/// explain which template controls a package consumes.
#[derive(Clone, Debug, Default)]
pub struct ChatPolicy {
    pub policy: Map<String, Value>,
}

impl ChatPolicy {
    pub fn from_config(llm_config: &Map<String, Value>) -> Option<ChatPolicy> {
        let chat = child(llm_config, "chat");
        if chat.is_empty() {
            None
        } else {
            Some(ChatPolicy {
                policy: chat.clone(),
            })
        }
    }

    fn field(&self, name: &str) -> String {
        self.field_or(name, "")
    }

    fn field_or(&self, name: &str, default: &str) -> String {
        str_value(get(&self.policy, name))
            .unwrap_or(default)
            .to_string()
    }

    pub fn thinking_mode(&self) -> String {
        self.field_or("thinking", "never")
    }

    pub fn supports_thinking(&self) -> bool {
        self.thinking_mode() != "never" && !self.field("think_open").is_empty()
    }

    pub fn default_thinking(&self) -> bool {
        bool_value(get(&self.policy, "thinking_default")).unwrap_or(false)
    }

    pub fn terminal(&self) -> String {
        self.field("terminal")
    }

    pub fn tool_call_format(&self) -> Option<String> {
        string(get(&self.policy, "call_format"))
    }

    pub fn default_system(&self) -> String {
        self.field("default_system")
    }

    pub fn describe(&self) -> Value {
        json!({
            "thinking": self.thinking_mode(),
            "supports_thinking": self.supports_thinking(),
            "default_thinking": self.default_thinking(),
            "terminal": self.terminal(),
            "tool_calls": self.tool_call_format(),
            "default_system": self.default_system(),
        })
    }

    /// Renders a conversation the way this package frames prompts.
    pub fn render(
        &self,
        messages: &[Value],
        thinking: Option<bool>,
        tools: Option<&[Value]>,
    ) -> String {
        let thinking = thinking.unwrap_or_else(|| self.default_thinking());
        let mut system_text: Option<String> = None;
        let mut turns: Vec<(String, String)> = Vec::new();
        for message in messages {
            let message = obj(message);
            let role = str_value(get(message, "role"))
                .unwrap_or("user")
                .to_string();
            let text = message_text(get(message, "content"));
            if role == "system" && system_text.is_none() {
                system_text = Some(text);
                continue;
            }
            turns.push((role, text));
        }
        let system_text = system_text.unwrap_or_else(|| self.default_system());
        let mut parts = String::new();
        if !system_text.is_empty() {
            parts.push_str(&self.system_text(&system_text));
        }
        if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
            parts.push_str(&self.tools_section(tools));
        }
        for (role, text) in &turns {
            parts.push_str(&self.turn_text(role, text));
        }
        parts.push_str(&self.generation_prompt(thinking));
        parts
    }

    fn system_text(&self, text: &str) -> String {
        format!(
            "{}{}{}",
            self.field("system_prefix"),
            text,
            self.field("suffix")
        )
    }

    fn turn_text(&self, role: &str, text: &str) -> String {
        match role {
            "assistant" => format!(
                "{}{}{}",
                self.field("assistant_prefix"),
                self.history_text(text),
                self.field("suffix")
            ),
            "tool" => format!(
                "{}{}{}{}{}",
                self.field("tool_turn_prefix"),
                self.field("tool_item_prefix"),
                text,
                self.field("tool_item_suffix"),
                self.field("tool_turn_suffix")
            ),
            "system" => self.system_text(text),
            _ => format!(
                "{}{}{}",
                self.field("user_prefix"),
                text,
                self.field("suffix")
            ),
        }
    }

    fn history_text(&self, text: &str) -> String {
        let open = self.field("think_open");
        let close = self.field("think_close");
        if self.field("history_think") != "never" || open.is_empty() || close.is_empty() {
            return text.to_string();
        }
        strip_delimited(text, &open, &close)
    }

    fn tools_section(&self, tools: &[Value]) -> String {
        let prefix = self.field("tools_prefix");
        if prefix.is_empty() {
            return String::new();
        }
        let entries: Vec<Value> = tools
            .iter()
            .map(|tool| {
                let tool_object = obj(tool);
                let function = get(tool_object, "function");
                if truthy(function) {
                    function.clone()
                } else {
                    Value::Object(tool_object.clone())
                }
            })
            .collect();
        let body = serde_json::to_string(&entries).expect("JSON values always serialize");
        format!(
            "{}{}{}{}",
            prefix,
            body,
            self.field("tools_suffix"),
            self.field("tools_system_separator")
        )
    }

    fn generation_prompt(&self, thinking: bool) -> String {
        let assistant = self.field("assistant_prefix");
        let generation = self.field_or("generation_prompt", &assistant);
        if thinking {
            generation
        } else {
            self.field_or("generation_prompt_plain", &generation)
        }
    }
}

/// Removes every `open ... close` span (lazy, leftmost), like `re.sub` with
/// `re.escape(open) + ".*?" + re.escape(close)` and DOTALL.
fn strip_delimited(text: &str, open: &str, close: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    loop {
        let Some(start) = rest.find(open) else {
            out.push_str(rest);
            break;
        };
        let after_open = &rest[start + open.len()..];
        let Some(end) = after_open.find(close) else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        rest = &after_open[end + close.len()..];
    }
    out
}

/// One published Loom package, seen through its own documents.
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub repo: String,
    pub model_id: String,
    pub name: String,
    pub revision: Option<String>,
    /// Local installation directory, when the profile came from one.
    pub path: Option<String>,
    pub tags: Vec<String>,
    pub license: Option<String>,
    pub quant: Option<String>,
    pub architecture: Option<String>,
    /// The context the card's preset serves (the headline figure).
    pub context_length: Option<i64>,
    /// The model's own context ceiling, which a launch may not exceed.
    pub native_context_length: Option<i64>,
    pub memory_bytes: Option<i64>,
    pub device_hint: Option<String>,
    pub gfx_arch: Option<String>,
    pub features: Vec<Feature>,
    pub gates: Vec<Gate>,
    pub performance: Option<Performance>,
    pub execution: Option<Execution>,
    pub chat: Option<ChatPolicy>,
    pub drafter: Map<String, Value>,
}

/// The documents a profile is read from, as they are cached on disk.
#[derive(Clone, Debug, Default)]
pub struct ProfileDocuments {
    pub model_card: Map<String, Value>,
    pub info: Map<String, Value>,
    pub llm_config: Map<String, Value>,
    pub readme: Option<String>,
    pub tags: Vec<String>,
    pub revision: Option<String>,
}

impl ProfileDocuments {
    /// The cache document (`cache/profiles/<slug>.json`), readable by the Python launcher.
    pub fn to_cache(&self, repo: &str) -> Value {
        json!({
            "repo": repo,
            "model_card": self.model_card,
            "info": self.info,
            "llm_config": self.llm_config,
            "readme": self.readme.clone().unwrap_or_default(),
            "tags": self.tags,
            "revision": self.revision,
        })
    }

    pub fn from_cache(document: &Map<String, Value>) -> ProfileDocuments {
        ProfileDocuments {
            model_card: child(document, "model_card").clone(),
            info: child(document, "info").clone(),
            llm_config: child(document, "llm_config").clone(),
            readme: string(get(document, "readme")),
            tags: arr(get(document, "tags"))
                .iter()
                .filter_map(|t| t.as_str().map(str::to_string))
                .collect(),
            revision: string(get(document, "revision")),
        }
    }
}

impl Profile {
    pub fn from_documents(
        repo: &str,
        documents: &ProfileDocuments,
        path: Option<String>,
    ) -> Profile {
        let model_card = &documents.model_card;
        let info = &documents.info;
        let llm_config = &documents.llm_config;
        let model = child(model_card, "model");
        let card = child(info, "card");
        let features: Vec<Feature> = child(card, "features")
            .iter()
            .map(|(name, entry)| {
                let entry = obj(entry);
                Feature {
                    name: name.clone(),
                    status: string(get(entry, "status")).unwrap_or_else(|| "unknown".into()),
                    detail: child(entry, "value").clone(),
                }
            })
            .collect();
        let gates: Vec<Gate> = child(info, "gates")
            .iter()
            .map(|(name, entry)| {
                let entry = obj(entry);
                Gate {
                    name: name.clone(),
                    status: string(get(entry, "status")).unwrap_or_else(|| "unknown".into()),
                    level: string(get(entry, "level")),
                    date: string(get(entry, "date")),
                    source: string(get(entry, "source")),
                }
            })
            .collect();
        let execution = Execution::from_card(model_card, None);
        let performance = Performance::from_measurement_document(info);
        let memory_bytes = int_value(get(card, "memory_bytes"))
            .or_else(|| preset_memory(model_card, execution.as_ref()));
        let context_feature = features
            .iter()
            .find(|f| f.name == "context")
            .map(|f| &f.detail);
        let (context_length, native_context_length) = served_and_native_context(
            int_value(get(llm_config, "context_length")),
            execution.as_ref().and_then(|e| e.context_capacity),
            context_feature,
        );
        let device_hint = performance.as_ref().and_then(|p| p.device.clone());
        let model_name = string(get(model, "name"));
        let readme_title = readme_title(documents.readme.as_deref());
        let quant = quantization(&[
            str_value(get(llm_config, "gguf")),
            str_value(get(child(info, "roofline"), "checkpoint")),
            model_name.as_deref(),
            readme_title.as_deref(),
            Some(repo),
        ]);
        let gfx_text = [device_hint.as_deref(), Some(repo)]
            .iter()
            .flatten()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        Profile {
            repo: repo.to_string(),
            model_id: string(get(model, "id")).unwrap_or_else(|| repo.to_string()),
            name: model_name.unwrap_or_else(|| model_id_of(repo).to_string()),
            revision: documents.revision.clone(),
            path,
            tags: documents.tags.clone(),
            license: license(&documents.tags),
            quant,
            architecture: string(get(llm_config, "arch")),
            context_length,
            native_context_length,
            memory_bytes,
            device_hint,
            gfx_arch: gfx_arch(&gfx_text),
            features,
            gates,
            performance,
            execution,
            chat: ChatPolicy::from_config(llm_config),
            drafter: child(llm_config, "drafter").clone(),
        }
    }

    /// Reads the documents of a published profile, pinned to one revision.
    pub fn documents_from_hub(
        hub: &Hub,
        repo: &str,
        revision: Option<&str>,
        model: Option<HubModel>,
    ) -> Result<ProfileDocuments> {
        let model = match model {
            Some(model) => model,
            None => hub.model(repo)?,
        };
        let revision = revision
            .map(str::to_string)
            .or_else(|| model.revision.clone());
        let model_card = hub.json_optional(repo, CARD_DOCUMENT, revision.as_deref())?;
        if model_card.is_empty() {
            return Err(Error::profile(format!(
                "{repo} does not publish a root {CARD_DOCUMENT}; pass an explicit model.json path"
            )));
        }
        let info = hub.json_optional(repo, MEASUREMENT_DOCUMENT, revision.as_deref())?;
        let llm_config = hub.json_optional(repo, CONFIG_DOCUMENT, revision.as_deref())?;
        let readme = hub.raw_optional(repo, README_DOCUMENT, revision.as_deref())?;
        Ok(ProfileDocuments {
            model_card,
            info,
            llm_config,
            readme: Some(String::from_utf8_lossy(&readme).into_owned()),
            tags: model.tags,
            revision,
        })
    }

    pub fn from_hub(
        hub: &Hub,
        repo: &str,
        revision: Option<&str>,
    ) -> Result<(Profile, ProfileDocuments)> {
        let documents = Self::documents_from_hub(hub, repo, revision, None)?;
        Ok((Profile::from_documents(repo, &documents, None), documents))
    }

    /// Reads the documents of an installed package; sidecars are optional.
    pub fn local_documents(directory: &Path) -> Result<(PathBuf, ProfileDocuments)> {
        let directory = if directory.is_file() {
            directory.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            directory.to_path_buf()
        };
        let card_path = directory.join(CARD_DOCUMENT);
        if !card_path.is_file() {
            return Err(Error::profile(format!(
                "{} is not an installed model card",
                card_path.display()
            )));
        }
        let text = std::fs::read(&card_path).map_err(|error| {
            Error::profile(format!("{} is unreadable: {error}", card_path.display()))
        })?;
        let model_card = match serde_json::from_slice::<Value>(&text) {
            Ok(Value::Object(map)) => map,
            Ok(_) => Map::new(),
            Err(error) => {
                return Err(Error::profile(format!(
                    "{} is unreadable: {error}",
                    card_path.display()
                )))
            }
        };
        Ok((
            directory.clone(),
            ProfileDocuments {
                model_card,
                info: optional_json(&directory.join(MEASUREMENT_DOCUMENT)),
                llm_config: optional_json(&directory.join(CONFIG_DOCUMENT)),
                readme: std::fs::read_to_string(directory.join(README_DOCUMENT)).ok(),
                tags: Vec::new(),
                revision: None,
            },
        ))
    }

    /// Reads an installed package. Its repository is `repo` when the caller
    /// knows it (an install record), else the card's model id, else the
    /// directory name.
    pub fn from_local(directory: &Path, repo: Option<&str>) -> Result<Profile> {
        let (directory, documents) = Self::local_documents(directory)?;
        let repo = repo
            .map(str::to_string)
            .or_else(|| string(get(child(&documents.model_card, "model"), "id")))
            .unwrap_or_else(|| {
                directory
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
        Ok(Profile::from_documents(
            &repo,
            &documents,
            Some(directory.to_string_lossy().into_owned()),
        ))
    }

    /// A placeholder for an endpoint whose package is unknown locally.
    pub fn named(name: &str) -> Profile {
        Profile {
            repo: name.to_string(),
            model_id: name.to_string(),
            name: name.to_string(),
            ..Default::default()
        }
    }

    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}", self.repo)
    }

    pub fn slug(&self) -> String {
        self.repo.replace('/', "-")
    }

    pub fn feature(&self, name: &str) -> Option<&Feature> {
        self.features.iter().find(|f| f.name == name)
    }

    pub fn enabled(&self, name: &str) -> bool {
        self.feature(name).is_some_and(Feature::enabled)
    }

    pub fn passed_gates(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .gates
            .iter()
            .filter(|g| g.passed())
            .map(|g| g.name.clone())
            .collect();
        names.sort();
        names
    }

    pub fn failed_gates(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .gates
            .iter()
            .filter(|g| g.failed())
            .map(|g| g.name.clone())
            .collect();
        names.sort();
        names
    }

    pub fn preset(&self) -> Option<&str> {
        self.execution.as_ref().and_then(|e| e.preset.as_deref())
    }

    pub fn render(
        &self,
        messages: &[Value],
        thinking: Option<bool>,
        tools: Option<&[Value]>,
    ) -> Result<String> {
        match &self.chat {
            Some(chat) => Ok(chat.render(messages, thinking, tools)),
            None => Err(Error::new(
                ErrorKind::Profile,
                format!("{} publishes no chat policy", self.repo),
            )),
        }
    }

    pub fn benchmark_rows(&self) -> Vec<Value> {
        match &self.performance {
            None => Vec::new(),
            Some(performance) => performance
                .measurements
                .iter()
                .filter(|m| m.value.is_some())
                .map(Measurement::describe)
                .collect(),
        }
    }

    pub fn describe(&self) -> Value {
        let features: Map<String, Value> = self
            .features
            .iter()
            .map(|f| (f.name.clone(), f.describe()))
            .collect();
        let gates: Map<String, Value> = self
            .gates
            .iter()
            .map(|g| (g.name.clone(), g.describe()))
            .collect();
        json!({
            "repo": self.repo,
            "revision": self.revision,
            "model_id": self.model_id,
            "name": self.name,
            "path": self.path,
            "url": self.url(),
            "tags": self.tags,
            "license": self.license,
            "quant": self.quant,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "native_context_length": self.native_context_length,
            "memory_bytes": self.memory_bytes,
            "device": self.device_hint,
            "gfx_arch": self.gfx_arch,
            "features": features,
            "gates": gates,
            "execution": self.execution.as_ref().map(Execution::describe),
            "chat": self.chat.as_ref().map(ChatPolicy::describe),
            "drafter": self.drafter,
            "benchmarks": self.benchmark_rows(),
            "performance": self.performance.as_ref().map(|p| json!({
                "baseline": p.baseline_name,
                "baseline_version": p.baseline_version,
                "device": p.device,
                "sessions": p.session_count,
            })),
        })
    }

    /// `40960 tokens (model native 262144)`, or `262144 tokens` when the
    /// preset serves the whole native context.
    pub fn context_text(&self) -> String {
        match (
            self.context_length.filter(|c| *c != 0),
            self.native_context_length.filter(|c| *c != 0),
        ) {
            (Some(served), Some(native)) if served != native => {
                format!("{served} tokens (model native {native})")
            }
            (Some(served), _) => format!("{served} tokens"),
            (None, Some(native)) => format!("{native} tokens"),
            (None, None) => "unknown".to_string(),
        }
    }

    pub fn summary_line(&self) -> String {
        let mut fields = vec![self.repo.clone()];
        if let Some(quant) = &self.quant {
            fields.push(quant.clone());
        }
        if let Some(context) = self.context_length.filter(|c| *c != 0) {
            fields.push(format!("ctx {context}"));
        }
        if let Some(decode) = self
            .performance
            .as_ref()
            .and_then(|p| p.measurement("decode"))
            .and_then(|m| m.value)
        {
            fields.push(format!("decode {decode:.1} tok/s"));
        }
        fields.join("  ")
    }
}

/// The context a profile serves and the model's own ceiling.
///
/// The served context is the preset's `context_capacity`, else the measured
/// `context.tokens` feature, else the native context. The native context is
/// the generator's `context_length`, else the `context.native` feature, else
/// the served context: a launch may narrow the served context, and may widen
/// it up to the native one.
fn served_and_native_context(
    config_context: Option<i64>,
    preset_context: Option<i64>,
    feature: Option<&Map<String, Value>>,
) -> (Option<i64>, Option<i64>) {
    let feature_value = |key: &str| {
        feature
            .and_then(|detail| int_value(get(detail, key)))
            .filter(|v| *v != 0)
    };
    let native = config_context
        .filter(|v| *v != 0)
        .or_else(|| feature_value("native"));
    let served = preset_context
        .filter(|v| *v != 0)
        .or_else(|| feature_value("tokens"))
        .or(native);
    (served, native.or(served))
}

/// The trailing repository name, used when a card omits a model name.
pub fn model_id_of(repo: &str) -> &str {
    match repo.split_once('/') {
        Some((_, name)) => name,
        None => repo,
    }
}

fn choose_preset(presets: &Map<String, Value>, preferred: Option<&str>) -> Option<String> {
    if presets.is_empty() {
        return None;
    }
    if let Some(preferred) = preferred.filter(|p| presets.contains_key(*p)) {
        return Some(preferred.to_string());
    }
    if presets.len() == 1 {
        return presets.keys().next().cloned();
    }
    for name in ["chat", "default", "interactive", "serve"] {
        if presets.contains_key(name) {
            return Some(name.to_string());
        }
    }
    let mut names: Vec<&String> = presets.keys().collect();
    names.sort();
    names.first().map(|n| n.to_string())
}

fn preset_memory(model_card: &Map<String, Value>, execution: Option<&Execution>) -> Option<i64> {
    let preset = execution?.preset.as_deref()?;
    let memory = child(child(child(model_card, "presets"), preset), "memory");
    int_value(get(memory, "weights_bytes"))
}

fn license(tags: &[String]) -> Option<String> {
    tags.iter()
        .find_map(|tag| tag.strip_prefix("license:").map(str::to_string))
}

fn readme_title(readme: Option<&str>) -> Option<String> {
    for line in readme?.lines() {
        let stripped = line.trim();
        if let Some(title) = stripped.strip_prefix("# ") {
            let title = title.trim();
            return if title.is_empty() {
                None
            } else {
                Some(title.to_string())
            };
        }
    }
    None
}

fn message_text(content: &Value) -> String {
    if let Some(text) = str_value(content) {
        return text.to_string();
    }
    arr(content)
        .iter()
        .map(obj)
        .filter(|part| str_value(get(part, "type")) == Some("text"))
        .map(|part| str_value(get(part, "text")).unwrap_or("").to_string())
        .collect()
}

fn optional_json(path: &Path) -> Map<String, Value> {
    match std::fs::read(path) {
        Ok(bytes) if !bytes.is_empty() => crate::values::parse_object(&bytes),
        _ => Map::new(),
    }
}

// -- pattern matching (the Python launcher's two regular expressions) ---------

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// One element of a fixed-shape pattern.
#[derive(Clone, Copy)]
enum Atom {
    Char(char),
    Range(char, char),
    /// `[A-Z0-9]`
    UpperAlnum,
    /// `[0-9a-f]`
    LowerHex,
    /// `[SML]`
    SizeLetter,
}

impl Atom {
    fn matches(self, c: char) -> bool {
        match self {
            Atom::Char(expected) => c == expected,
            Atom::Range(low, high) => (low..=high).contains(&c),
            Atom::UpperAlnum => c.is_ascii_uppercase() || c.is_ascii_digit(),
            Atom::LowerHex => c.is_ascii_digit() || ('a'..='f').contains(&c),
            Atom::SizeLetter => matches!(c, 'S' | 'M' | 'L'),
        }
    }
}

/// A fixed prefix followed by `repeat` (min..=max, greedy) of one atom.
struct Shape {
    prefix: Vec<Atom>,
    repeat: Option<(Atom, usize, usize)>,
}

fn literal(text: &str) -> Vec<Atom> {
    text.chars().map(Atom::Char).collect()
}

/// Leftmost-first search for `\b(alt1|alt2|...)\b` over fixed shapes.
fn search_shapes(text: &str, shapes: &[Shape]) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    let boundary_after = |end: usize| end == chars.len() || !is_word(chars[end]);
    for start in 0..chars.len() {
        if start > 0 && is_word(chars[start - 1]) {
            continue;
        }
        for shape in shapes {
            let prefix_end = start + shape.prefix.len();
            if prefix_end > chars.len()
                || !shape
                    .prefix
                    .iter()
                    .zip(&chars[start..prefix_end])
                    .all(|(atom, c)| atom.matches(*c))
            {
                continue;
            }
            match shape.repeat {
                None => {
                    if boundary_after(prefix_end) {
                        return Some(chars[start..prefix_end].iter().collect());
                    }
                }
                Some((atom, min, max)) => {
                    let mut available = 0;
                    while available < max
                        && prefix_end + available < chars.len()
                        && atom.matches(chars[prefix_end + available])
                    {
                        available += 1;
                    }
                    let mut count = available;
                    while count >= min {
                        if boundary_after(prefix_end + count) {
                            return Some(chars[start..prefix_end + count].iter().collect());
                        }
                        if count == 0 {
                            break;
                        }
                        count -= 1;
                    }
                }
            }
        }
    }
    None
}

fn quantization_shapes() -> Vec<Shape> {
    let digit = Atom::Range('2', '8');
    vec![
        Shape {
            prefix: vec![
                Atom::Char('I'),
                Atom::Char('Q'),
                Atom::Range('1', '4'),
                Atom::Char('_'),
            ],
            repeat: Some((Atom::UpperAlnum, 2, 4)),
        },
        Shape {
            prefix: vec![
                Atom::Char('Q'),
                digit,
                Atom::Char('_'),
                Atom::Char('K'),
                Atom::Char('_'),
                Atom::SizeLetter,
            ],
            repeat: None,
        },
        Shape {
            prefix: vec![Atom::Char('Q'), digit, Atom::Char('_'), Atom::Char('K')],
            repeat: None,
        },
        Shape {
            prefix: vec![
                Atom::Char('Q'),
                digit,
                Atom::Char('_'),
                Atom::Range('0', '1'),
            ],
            repeat: None,
        },
        Shape {
            prefix: vec![Atom::Char('Q'), digit],
            repeat: None,
        },
        Shape {
            prefix: literal("BF16"),
            repeat: None,
        },
        Shape {
            prefix: literal("F16"),
            repeat: None,
        },
        Shape {
            prefix: literal("F32"),
            repeat: None,
        },
        Shape {
            prefix: literal("MXFP4"),
            repeat: None,
        },
        Shape {
            prefix: literal("MXFP8"),
            repeat: None,
        },
    ]
}

/// The first quantization name found in the given texts, in order.
pub fn quantization(texts: &[Option<&str>]) -> Option<String> {
    let shapes = quantization_shapes();
    texts
        .iter()
        .flatten()
        .filter(|text| !text.is_empty())
        .find_map(|text| search_shapes(text, &shapes))
}

/// The first `gfxNNNN` architecture name in `text`.
pub fn gfx_arch(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    let shapes = [Shape {
        prefix: literal("gfx"),
        repeat: Some((Atom::LowerHex, 3, 6)),
    }];
    search_shapes(text, &shapes)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn chat_policy() -> Value {
        json!({
            "system_prefix": "<|im_start|>system\n",
            "user_prefix": "<|im_start|>user\n",
            "assistant_prefix": "<|im_start|>assistant\n",
            "suffix": "<|im_end|>\n",
            "default_system": "",
            "think_open": "<think>",
            "think_close": "</think>",
            "history_think": "never",
            "generation_prompt": "<|im_start|>assistant\n",
            "generation_prompt_plain": "<|im_start|>assistant\n",
            "thinking": "toggle",
            "thinking_default": true,
            "terminal": "<|im_end|>",
            "call_format": "hermes",
            "tools_prefix": "TOOLS:",
            "tools_suffix": ":END",
            "tools_system_separator": "\n",
        })
    }

    pub(crate) fn card(presets: Option<Value>) -> Value {
        json!({
            "schema_version": 1,
            "model": {"id": "Qwen/Qwen3-4B", "name": "Qwen3-4B (Loom port, GGUF)"},
            "services": {"chat": {"kind": "text", "version": 1}},
            "presets": presets.unwrap_or_else(|| json!({
                "chat": {
                    "service": "chat",
                    "variant": "gguf",
                    "programs": "source",
                    "execution": {
                        "rows": 4,
                        "context_capacity": 262144,
                        "pool_capacity": 327680,
                        "prefill_capacity": 1024,
                        "max_tokens": 4096,
                        "draft_depth": 3,
                    },
                }
            })),
        })
    }

    pub(crate) fn measurements() -> Value {
        json!({
            "schema": "loom-card/2",
            "card": {
                "memory_bytes": 63000000000_i64,
                "features": {
                    "context": {"status": "yes", "value": {"native": 262144}},
                    "tool_calling": {"status": "yes", "value": null},
                    "audio": {"status": "n/a", "value": null},
                },
            },
            "gates": {
                "compare": {"status": "pass", "level": "full", "date": "2026-11-08"},
                "vision": {"status": "n/a"},
            },
            "performance": {
                "baseline": {"name": "llama.cpp", "version": "b1368", "args": "-ngl 99"},
                "sessions": [
                    {"date": "2026-11-08", "metrics": {"decode": {"loom": 70.0}}},
                    {
                        "date": "2026-11-11",
                        "metrics": {
                            "decode": {"loom": 76.1, "baseline": 51.7},
                            "prefill_1k": {"loom": 2891.8},
                            "spec_acceptance": {"loom": 0.44},
                        },
                    },
                ],
                "baselines": {
                    "vulkan": {"session": {"metrics": {"decode": {"baseline": 72.6}}}}
                },
            },
            "roofline": {
                "device": {
                    "name": "AMD Ryzen AI Max+ 395 \"Strix Halo\" (gfx1151)",
                    "bandwidth_bytes_per_s": 216000000000.0,
                    "peak_flops": 59000000000000.0,
                },
                "metrics": {
                    "decode": {"ceiling": 123.6},
                    "prefill_1k": {"ceiling": 7802.6},
                },
            },
        })
    }

    pub(crate) fn config() -> Value {
        json!({
            "arch": "qwen3",
            "gguf": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            "context_length": 262144,
            "chat": chat_policy(),
            "drafter": {"kind": "dflash", "depth": 3},
        })
    }

    pub(crate) fn documents(model_card: Value, info: Value, llm_config: Value) -> ProfileDocuments {
        ProfileDocuments {
            model_card: obj(&model_card).clone(),
            info: obj(&info).clone(),
            llm_config: obj(&llm_config).clone(),
            readme: Some("# Qwen3-4B (Q4_K_M) for AMD Loom\n".into()),
            tags: vec!["loom".into(), "license:apache-2.0".into()],
            revision: Some("abc123".into()),
        }
    }

    pub(crate) fn profile() -> Profile {
        Profile::from_documents(
            "loom-fun/qwen3-4b-loom",
            &documents(card(None), measurements(), config()),
            None,
        )
    }

    fn messages(items: &[(&str, &str)]) -> Vec<Value> {
        items
            .iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect()
    }

    #[test]
    fn profile_reads_identity_capacities_and_features() {
        let parsed = profile();
        assert_eq!(parsed.model_id, "Qwen/Qwen3-4B");
        assert_eq!(parsed.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(parsed.architecture.as_deref(), Some("qwen3"));
        assert_eq!(parsed.license.as_deref(), Some("apache-2.0"));
        assert_eq!(parsed.context_length, Some(262144));
        assert_eq!(parsed.memory_bytes, Some(63000000000));
        assert_eq!(parsed.gfx_arch.as_deref(), Some("gfx1151"));
        assert!(parsed.enabled("context"));
        assert!(parsed.enabled("tool_calling"));
        assert!(!parsed.enabled("audio"));
        let execution = parsed.execution.as_ref().unwrap();
        assert_eq!(execution.preset.as_deref(), Some("chat"));
        assert_eq!(execution.service_kind.as_deref(), Some("text"));
        assert_eq!(execution.max_tokens, Some(4096));
        assert_eq!(parsed.passed_gates(), vec!["compare".to_string()]);
        assert!(parsed.failed_gates().is_empty());
        assert_eq!(parsed.drafter["kind"], "dflash");
    }

    #[test]
    fn measurements_take_the_last_session_that_measured_each_metric() {
        let parsed = profile();
        let performance = parsed.performance.as_ref().unwrap();
        let metrics: Vec<_> = performance
            .measurements
            .iter()
            .map(|m| m.metric.as_str())
            .collect();
        assert_eq!(metrics, ["decode", "prefill_1k", "spec_acceptance"]);
        let decode = performance.measurement("decode").unwrap();
        assert_eq!(decode.value, Some(76.1));
        assert_eq!(decode.baseline, Some(51.7));
        assert_eq!(decode.date.as_deref(), Some("2026-11-11"));
        assert!((decode.roofline_fraction().unwrap() - 76.1 / 123.6).abs() < 1e-9);
        assert_eq!(
            performance.measurement("spec_acceptance").unwrap().unit(),
            "fraction"
        );
        assert_eq!(performance.extra_baselines["vulkan"]["decode"], 72.6);
        assert_eq!(performance.extra_baselines.len(), 1);
        assert_eq!(performance.baseline_name.as_deref(), Some("llama.cpp"));
    }

    #[test]
    fn an_older_card_without_measurements_still_reads() {
        let documents = ProfileDocuments {
            model_card: obj(&card(None)).clone(),
            ..Default::default()
        };
        let parsed = Profile::from_documents("loom-fun/x-Q4_K_M-loom", &documents, None);
        assert_eq!(parsed.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(parsed.context_length, Some(262144));
        assert!(parsed.features.is_empty());
        assert!(parsed.gates.is_empty());
        assert!(parsed.performance.is_none());
        assert!(parsed.chat.is_none());
        assert_eq!(parsed.memory_bytes, None);
        assert!(parsed.benchmark_rows().is_empty());
    }

    #[test]
    fn missing_documents_fall_back_to_the_feature_context() {
        let info = json!({"card": {"features": {"context": {"status": "yes", "value": {"native": 40960}}}}});
        let documents = ProfileDocuments {
            model_card: obj(&json!({"model": {"id": "a/b"}})).clone(),
            info: obj(&info).clone(),
            ..Default::default()
        };
        let parsed = Profile::from_documents("x/y", &documents, None);
        assert_eq!(parsed.context_length, Some(40960));
        assert!(parsed.execution.is_none());
        assert_eq!(parsed.name, "y");
    }

    #[test]
    fn chat_policy_renders_system_user_and_generation_prompt() {
        let rendered = profile()
            .render(
                &messages(&[("system", "be terse"), ("user", "2+2?")]),
                Some(false),
                None,
            )
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nbe terse<|im_end|>\n\
             <|im_start|>user\n2+2?<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn chat_policy_defaults_to_the_package_thinking_mode() {
        let parsed = profile();
        let chat = parsed.chat.as_ref().unwrap();
        assert!(chat.supports_thinking());
        assert!(chat.default_thinking());
        assert!(parsed
            .render(&messages(&[("user", "hi")]), None, None)
            .unwrap()
            .ends_with("<|im_start|>assistant\n"));
        assert_eq!(chat.thinking_mode(), "toggle");
    }

    #[test]
    fn chat_policy_strips_reasoning_from_history_when_it_says_never() {
        let rendered = profile()
            .render(
                &messages(&[("assistant", "x<think>hidden</think>y"), ("user", "hi")]),
                Some(false),
                None,
            )
            .unwrap();
        assert!(!rendered.contains("<think>"));
        assert!(!rendered.contains("x y"));
        assert!(rendered.contains("xy<|im_end|>"));
    }

    #[test]
    fn chat_policy_renders_tool_declarations_after_the_system_turn() {
        let tools = vec![json!({"type": "function", "function": {"name": "read"}})];
        let rendered = profile()
            .render(&messages(&[("user", "hi")]), Some(false), Some(&tools))
            .unwrap();
        assert!(
            rendered.contains("TOOLS:[{\"name\":\"read\"}]:END"),
            "{rendered}"
        );
    }

    #[test]
    fn chat_policy_renders_tool_results_as_a_tool_turn() {
        let mut parsed = profile();
        let chat = parsed.chat.as_mut().unwrap();
        for (key, value) in [
            ("tool_turn_prefix", "<|im_start|>user"),
            ("tool_item_prefix", "\n<tool_response>\n"),
            ("tool_item_suffix", "\n</tool_response>"),
            ("tool_turn_suffix", "<|im_end|>\n"),
        ] {
            chat.policy.insert(key.into(), json!(value));
        }
        let rendered = parsed
            .render(&messages(&[("tool", "42")]), Some(false), None)
            .unwrap();
        assert!(rendered.contains("<tool_response>"));
        assert!(rendered.contains("42"));
        assert!(rendered.contains("</tool_response>"));
        assert_eq!(
            rendered,
            "<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn plain_generation_prompt_and_default_system_are_honoured() {
        let mut policy = obj(&chat_policy()).clone();
        policy.insert(
            "generation_prompt_plain".into(),
            json!("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        );
        policy.insert("default_system".into(), json!("You are helpful."));
        let chat = ChatPolicy { policy };
        let rendered = chat.render(&messages(&[("user", "hi")]), Some(false), None);
        assert_eq!(
            rendered,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let thinking = chat.render(&messages(&[("user", "hi")]), Some(true), None);
        assert!(thinking.ends_with("<|im_end|>\n<|im_start|>assistant\n"));
        // An explicit empty system message suppresses the default system prompt.
        let rendered = chat.render(
            &messages(&[("system", ""), ("user", "hi")]),
            Some(true),
            None,
        );
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn execution_prefers_a_named_chat_preset() {
        let presets = json!({
            "alpha": {"service": "chat", "execution": {"context_capacity": 4096}},
            "chat": {"service": "chat", "execution": {"context_capacity": 8192}},
        });
        let parsed = Profile::from_documents(
            "loom-fun/qwen3-4b-loom",
            &documents(card(Some(presets)), measurements(), config()),
            None,
        );
        let execution = parsed.execution.unwrap();
        assert_eq!(execution.preset.as_deref(), Some("chat"));
        assert_eq!(execution.context_capacity, Some(8192));
    }

    #[test]
    fn quantization_follows_the_python_expression() {
        let q = |text: &str| quantization(&[Some(text)]);
        assert_eq!(
            q("Qwen3-4B-Instruct-2507-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(q("model-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(q("model-Q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(q("model-Q4_K_XL.gguf"), None);
        assert_eq!(q("gemma-BF16-loom").as_deref(), Some("BF16"));
        assert_eq!(q("Qwen3.5-2B"), None);
        assert_eq!(q("x-Q4_K-loom").as_deref(), Some("Q4_K"));
        assert_eq!(q("x-MXFP4").as_deref(), Some("MXFP4"));
        assert_eq!(gfx_arch("AMD (gfx1151)").as_deref(), Some("gfx1151"));
        assert_eq!(gfx_arch("gfx90a:sramecc+").as_deref(), Some("gfx90a"));
        assert_eq!(gfx_arch("xgfx1100"), None);
    }

    /// The publication profile of the Qwen3.8-27B GPU/NPU hybrid.
    pub(crate) fn hybrid_profile() -> Profile {
        let parse = |text: &str| obj(&serde_json::from_str::<Value>(text).unwrap()).clone();
        Profile::from_documents(
            "qwen3.8-27b",
            &ProfileDocuments {
                model_card: parse(include_str!("../tests/fixtures/hybrid-publish/model.json")),
                info: parse(include_str!("../tests/fixtures/hybrid-publish/info.json")),
                llm_config: parse(include_str!(
                    "../tests/fixtures/hybrid-publish/llm_config.json"
                )),
                ..Default::default()
            },
            None,
        )
    }

    #[test]
    fn the_served_context_is_the_preset_and_the_native_one_is_the_ceiling() {
        let hybrid = hybrid_profile();
        assert_eq!(hybrid.context_length, Some(40960));
        assert_eq!(hybrid.native_context_length, Some(262144));
        assert_eq!(hybrid.context_text(), "40960 tokens (model native 262144)");
        assert_eq!(hybrid.describe()["context_length"], 40960);
        assert_eq!(hybrid.describe()["native_context_length"], 262144);
        assert!(hybrid.summary_line().contains("ctx 40960"));
        // A preset serving the whole native context reads as one figure.
        let whole = profile();
        assert_eq!(whole.context_text(), "262144 tokens");
        // Without a preset, the measured served tokens, then the native context.
        let served = served_and_native_context(
            None,
            None,
            Some(obj(&json!({"native": 262144, "tokens": 32768}))),
        );
        assert_eq!(served, (Some(32768), Some(262144)));
        assert_eq!(served_and_native_context(None, None, None), (None, None));
        assert_eq!(
            served_and_native_context(None, Some(8192), None),
            (Some(8192), Some(8192))
        );
    }

    #[test]
    fn describe_lists_the_python_keys_in_order() {
        let document = profile().describe();
        let keys: Vec<_> = document.as_object().unwrap().keys().cloned().collect();
        assert_eq!(
            keys,
            [
                "repo",
                "revision",
                "model_id",
                "name",
                "path",
                "url",
                "tags",
                "license",
                "quant",
                "architecture",
                "context_length",
                "native_context_length",
                "memory_bytes",
                "device",
                "gfx_arch",
                "features",
                "gates",
                "execution",
                "chat",
                "drafter",
                "benchmarks",
                "performance"
            ]
        );
    }
}
