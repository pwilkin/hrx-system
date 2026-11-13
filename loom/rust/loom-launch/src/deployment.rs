//! Deployment documents composed from a profile and local launcher settings.
//!
//! The server owns the document's meaning; the launcher only chooses values it
//! can justify from the card. A card's preset fixes the model's own context and
//! output capacity, so a launcher setting may only narrow what the package
//! declares.

use crate::errors::{Error, Result};
use crate::profile::Profile;
use crate::store::write_json;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// The server's own default when a deployment does not choose a value.
pub const DEFAULT_MAX_TOKENS: i64 = 512;
/// Context used when neither the caller nor the card names one.
pub const DEFAULT_CONTEXT: i64 = 16384;

/// The capacities one launch asks the text service to provision.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContextSettings {
    pub context_length: Option<i64>,
    pub max_tokens: Option<i64>,
    pub rows: Option<i64>,
    pub max_sessions: Option<i64>,
    pub prefill_capacity: Option<i64>,
    pub checkpoint_capacity: Option<i64>,
    pub session_cache_capacity: Option<i64>,
    pub prefix_cache_capacity: Option<i64>,
    pub cache_target_bytes: Option<i64>,
    pub draft_depth: Option<i64>,
    pub continuation_epochs: Option<i64>,
    pub pending_requests: Option<i64>,
}

impl ContextSettings {
    pub fn describe(&self) -> Value {
        json!({
            "context_length": self.context_length,
            "max_tokens": self.max_tokens,
            "rows": self.rows,
            "max_sessions": self.max_sessions,
            "prefill_capacity": self.prefill_capacity,
            "checkpoint_capacity": self.checkpoint_capacity,
            "session_cache_capacity": self.session_cache_capacity,
            "prefix_cache_capacity": self.prefix_cache_capacity,
            "cache_target_bytes": self.cache_target_bytes,
            "draft_depth": self.draft_depth,
            "continuation_epochs": self.continuation_epochs,
            "pending_requests": self.pending_requests,
        })
    }

    /// The `definition` object a deployment document carries.
    pub fn definition(&self) -> Map<String, Value> {
        let mut definition = Map::new();
        definition.insert("kind".into(), json!("text"));
        let fields = [
            ("rows", self.rows),
            ("max_sessions", self.max_sessions),
            ("context_capacity", self.context_length),
            ("prefill_capacity", self.prefill_capacity),
            ("checkpoint_capacity", self.checkpoint_capacity),
            ("session_cache_capacity", self.session_cache_capacity),
            ("prefix_cache_capacity", self.prefix_cache_capacity),
            ("cache_target_bytes", self.cache_target_bytes),
            ("max_tokens", self.max_tokens),
            ("draft_depth", self.draft_depth),
            ("continuation_epochs", self.continuation_epochs),
            ("pending_requests", self.pending_requests),
        ];
        for (name, value) in fields {
            if let Some(value) = value {
                definition.insert(name.into(), json!(value));
            }
        }
        definition
    }
}

/// What one launch asks for; every field narrows the card's own declaration.
#[derive(Clone, Debug, Default)]
pub struct ContextRequest {
    pub context_length: Option<i64>,
    pub max_tokens: Option<i64>,
    pub parallel: Option<i64>,
    pub draft_depth: Option<i64>,
    pub checkpoint_capacity: Option<i64>,
    pub session_cache_capacity: Option<i64>,
    pub prefix_cache_capacity: Option<i64>,
    pub cache_target_bytes: Option<i64>,
    pub prefill_capacity: Option<i64>,
    pub continuation_epochs: Option<i64>,
    pub pending_requests: Option<i64>,
}

/// Resolves one launch's capacities against the profile's own limits.
pub fn context_settings(profile: &Profile, request: &ContextRequest) -> Result<ContextSettings> {
    let execution = profile.execution.as_ref();
    let declared_max_tokens = execution.and_then(|e| e.max_tokens);
    let declared_rows = execution.and_then(|e| e.rows);
    let declared_prefill = execution.and_then(|e| e.prefill_capacity);
    let declared_draft = execution.and_then(|e| e.draft_depth);
    // The preset's served context is the default; the model's native context
    // is the ceiling a request may widen it to.
    let served_context = profile.context_length.filter(|c| *c != 0);
    let native_context = profile
        .native_context_length
        .filter(|c| *c != 0)
        .or(served_context);
    let resolved_context = request
        .context_length
        .or(served_context)
        .unwrap_or(DEFAULT_CONTEXT);
    if resolved_context < 1 {
        return Err(Error::launch(
            "context length must be a positive token count",
        ));
    }
    if let Some(native) = native_context {
        if resolved_context > native {
            return Err(Error::launch(format!(
                "context {resolved_context} exceeds the model's {native} tokens; \
                 the package cannot serve more than it was prepared for"
            )));
        }
    }
    let resolved_max_tokens = request
        .max_tokens
        .or(declared_max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if resolved_max_tokens < 1 {
        return Err(Error::launch("max tokens must be a positive token count"));
    }
    if resolved_max_tokens >= resolved_context {
        return Err(Error::launch(format!(
            "max tokens {resolved_max_tokens} leaves no room in a {resolved_context}-token context"
        )));
    }
    let mut rows = declared_rows;
    let mut sessions = None;
    if let Some(parallel) = request.parallel {
        if parallel < 1 {
            return Err(Error::launch("parallel requests must be at least one"));
        }
        rows = Some(parallel);
        sessions = Some(parallel);
    }
    let draft = request.draft_depth.or(declared_draft);
    if let Some(depth) = draft {
        if !(0..=7).contains(&depth) {
            return Err(Error::launch("draft depth must be between 0 and 7"));
        }
    }
    Ok(ContextSettings {
        context_length: Some(resolved_context),
        max_tokens: Some(resolved_max_tokens),
        rows,
        max_sessions: sessions,
        prefill_capacity: request.prefill_capacity.or(declared_prefill),
        checkpoint_capacity: request.checkpoint_capacity,
        session_cache_capacity: request.session_cache_capacity,
        prefix_cache_capacity: request.prefix_cache_capacity,
        cache_target_bytes: request.cache_target_bytes,
        draft_depth: draft,
        continuation_epochs: request.continuation_epochs,
        pending_requests: request.pending_requests,
    })
}

/// Composes the deployment document one launch registers.
pub fn deployment_document(
    profile: &Profile,
    card_path: &Path,
    settings: &ContextSettings,
    name: Option<&str>,
    preset: Option<&str>,
) -> Value {
    let mut model = Map::new();
    model.insert(
        "path".into(),
        json!(absolute(card_path).to_string_lossy().into_owned()),
    );
    if let Some(selected) = preset.or_else(|| profile.preset()) {
        model.insert("preset".into(), json!(selected));
    }
    let mut document = Map::new();
    document.insert(
        "name".into(),
        json!(name.map(str::to_string).unwrap_or_else(|| profile.slug())),
    );
    document.insert("definition".into(), Value::Object(settings.definition()));
    document.insert("model".into(), Value::Object(model));
    Value::Object(document)
}

/// An absolute, `~`-expanded path. `loom-serve` resolves `model.path` relative
/// to the deployment document's own directory, so a relative path in a run
/// directory could never be what the caller meant.
pub fn absolute(path: &Path) -> PathBuf {
    let expanded = expand_user(path);
    let joined = if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&expanded))
            .unwrap_or(expanded)
    };
    normalize(&joined)
}

/// Expands a leading `~` to `$HOME`.
pub fn expand_user(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text == "~" || text.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(text.trim_start_matches('~').trim_start_matches('/'));
        }
    }
    path.to_path_buf()
}

/// Lexical normalization like `os.path.abspath` (no symlink resolution).
fn normalize(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Writes a deployment document next to a run's logs.
pub fn write_deployment(path: &Path, document: &Value) -> Result<PathBuf> {
    write_json(path, document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::tests::profile;

    #[test]
    fn context_settings_default_to_the_card() {
        let settings = context_settings(&profile(), &ContextRequest::default()).unwrap();
        assert_eq!(settings.context_length, Some(262144));
        assert_eq!(settings.max_tokens, Some(4096));
        assert_eq!(settings.rows, Some(4));
        assert_eq!(settings.max_sessions, None);
        assert_eq!(settings.draft_depth, Some(3));
        let definition = settings.definition();
        assert_eq!(definition["kind"], "text");
        assert_eq!(definition["context_capacity"], 262144);
        assert_eq!(definition["max_tokens"], 4096);
        assert!(!definition.contains_key("session_cache_capacity"));
    }

    #[test]
    fn narrower_context_and_output_are_accepted() {
        let request = ContextRequest {
            context_length: Some(8192),
            max_tokens: Some(512),
            ..Default::default()
        };
        let settings = context_settings(&profile(), &request).unwrap();
        assert_eq!(
            (settings.context_length, settings.max_tokens),
            (Some(8192), Some(512))
        );
    }

    #[test]
    fn context_beyond_the_model_is_rejected() {
        let request = ContextRequest {
            context_length: Some(300000),
            ..Default::default()
        };
        let error = context_settings(&profile(), &request).unwrap_err();
        assert!(
            error.message.contains("exceeds the model's 262144 tokens"),
            "{error}"
        );
    }

    #[test]
    fn the_served_context_is_the_default_and_the_native_one_the_ceiling() {
        let hybrid = crate::profile::tests::hybrid_profile();
        let settings = context_settings(&hybrid, &ContextRequest::default()).unwrap();
        assert_eq!(settings.context_length, Some(40960));
        assert_eq!(settings.max_tokens, Some(4096));
        assert_eq!(settings.rows, Some(1));
        // Widening toward the native context is allowed; beyond it is not.
        let wider = ContextRequest {
            context_length: Some(50000),
            ..Default::default()
        };
        assert_eq!(
            context_settings(&hybrid, &wider).unwrap().context_length,
            Some(50000)
        );
        let beyond = ContextRequest {
            context_length: Some(262145),
            ..Default::default()
        };
        assert!(context_settings(&hybrid, &beyond)
            .unwrap_err()
            .message
            .contains("exceeds the model's 262144 tokens"));
    }

    #[test]
    fn output_limit_must_leave_room_in_the_context() {
        let request = ContextRequest {
            context_length: Some(100),
            max_tokens: Some(100),
            ..Default::default()
        };
        let error = context_settings(&profile(), &request).unwrap_err();
        assert!(error.message.contains("leaves no room"), "{error}");
    }

    #[test]
    fn parallel_sets_rows_and_sessions_together() {
        let request = ContextRequest {
            parallel: Some(2),
            ..Default::default()
        };
        let settings = context_settings(&profile(), &request).unwrap();
        assert_eq!((settings.rows, settings.max_sessions), (Some(2), Some(2)));
        let zero = ContextRequest {
            parallel: Some(0),
            ..Default::default()
        };
        assert!(context_settings(&profile(), &zero).is_err());
        let deep = ContextRequest {
            draft_depth: Some(8),
            ..Default::default()
        };
        assert!(context_settings(&profile(), &deep)
            .unwrap_err()
            .message
            .contains("draft depth"));
    }

    #[test]
    fn deployment_document_references_the_card_and_preset() {
        let parsed = profile();
        let request = ContextRequest {
            context_length: Some(8192),
            session_cache_capacity: Some(4),
            prefix_cache_capacity: Some(2),
            checkpoint_capacity: Some(1),
            parallel: Some(4),
            ..Default::default()
        };
        let settings = context_settings(&parsed, &request).unwrap();
        let document = deployment_document(
            &parsed,
            Path::new("/models/qwen3/model.json"),
            &settings,
            None,
            None,
        );
        assert_eq!(document["name"], "loom-fun-qwen3-4b-loom");
        assert_eq!(
            document["model"],
            json!({"path": "/models/qwen3/model.json", "preset": "chat"})
        );
        assert_eq!(
            document["definition"],
            json!({
                "kind": "text", "rows": 4, "max_sessions": 4, "context_capacity": 8192,
                "prefill_capacity": 1024, "checkpoint_capacity": 1, "session_cache_capacity": 4,
                "prefix_cache_capacity": 2, "max_tokens": 4096, "draft_depth": 3
            })
        );
    }

    #[test]
    fn the_model_path_is_always_absolute() {
        let parsed = profile();
        let settings = context_settings(&parsed, &ContextRequest::default()).unwrap();
        let document = deployment_document(
            &parsed,
            Path::new("relative/../models/x/model.json"),
            &settings,
            Some("n"),
            None,
        );
        let path = document["model"]["path"].as_str().unwrap();
        assert!(Path::new(path).is_absolute(), "{path}");
        assert!(path.ends_with("/models/x/model.json"), "{path}");
        assert!(!path.contains(".."), "{path}");
        assert_eq!(document["name"], "n");
    }
}
