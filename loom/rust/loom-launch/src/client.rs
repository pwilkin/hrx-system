//! A small OpenAI-shaped client with exact client-side timing.
//!
//! The chat and benchmark paths need the same primitive: send one completion,
//! stream the deltas as they arrive, and know when the first token arrived.

use crate::errors::{Error, Result};
use crate::values::{arr, get, obj, opt_f64, round_opt, str_value, strict_int};
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Read};
use std::time::{Duration, Instant};

/// Token usage a server reported for one completion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
}

impl Usage {
    pub fn from_document(document: &Value) -> Option<Usage> {
        let usage = obj(document);
        if usage.is_empty() {
            return None;
        }
        let details = obj(get(usage, "prompt_tokens_details"));
        Some(Usage {
            prompt_tokens: strict_int(get(usage, "prompt_tokens")),
            completion_tokens: strict_int(get(usage, "completion_tokens")),
            total_tokens: strict_int(get(usage, "total_tokens")),
            cached_tokens: strict_int(get(details, "cached_tokens")),
        })
    }

    pub fn describe(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
        })
    }
}

/// One completion as the client observed it.
#[derive(Clone, Debug, Default)]
pub struct ChatResult {
    pub status: u16,
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<Value>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub ttft_seconds: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub chunks: usize,
    pub error: Option<String>,
    pub document: Map<String, Value>,
}

impl ChatResult {
    /// Decode rate over the interval after the first token arrived.
    pub fn decode_tokens_per_second(&self) -> Option<f64> {
        let completion = self.usage.as_ref()?.completion_tokens?;
        if completion < 2 {
            return None;
        }
        let interval = self.duration_seconds? - self.ttft_seconds?;
        if interval <= 0.0 {
            return None;
        }
        Some((completion - 1) as f64 / interval)
    }

    pub fn describe(&self) -> Value {
        json!({
            "status": self.status,
            "text": self.text,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": self.usage.as_ref().map(Usage::describe),
            "ttft_seconds": opt_f64(round_opt(self.ttft_seconds, 4)),
            "duration_seconds": opt_f64(round_opt(self.duration_seconds, 4)),
            "decode_tokens_per_second": opt_f64(round_opt(self.decode_tokens_per_second(), 4)),
            "chunks": self.chunks,
            "error": self.error,
        })
    }
}

/// An OpenAI-compatible endpoint client bound to one base URL.
#[derive(Clone)]
pub struct Client {
    pub base_url: String,
    token: Option<String>,
    timeout: Duration,
    agent: ureq::Agent,
}

/// A status and complete body, whatever the status was.
pub type RawResponse = (u16, Vec<u8>);

impl Client {
    pub fn new(base_url: &str, token: Option<&str>, timeout: Duration) -> Result<Client> {
        let base = base_url.trim_end_matches('/');
        if let Some((scheme, _)) = base.split_once("://") {
            if scheme != "http" {
                return Err(Error::launch(format!(
                    "unsupported endpoint scheme in {base_url}"
                )));
            }
        }
        let base = if base.contains("://") {
            base.to_string()
        } else {
            format!("http://{base}")
        };
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10).min(timeout))
            .timeout_read(timeout)
            .timeout_write(timeout)
            .redirects(0)
            .user_agent("loom-launch/1")
            .build();
        Ok(Client {
            base_url: base,
            token: token.filter(|t| !t.is_empty()).map(str::to_string),
            timeout,
            agent,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn prepare(
        &self,
        method: &str,
        path: &str,
        headers: &[(String, String)],
        timeout: Option<Duration>,
    ) -> ureq::Request {
        let mut request = self
            .agent
            .request(method, &self.url(path))
            .set("Accept", "application/json");
        if let Some(token) = &self.token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        for (name, value) in headers {
            request = request.set(name, value);
        }
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        request
    }

    /// Sends one request and reads the whole body.
    pub fn request(
        &self,
        method: &str,
        path: &str,
        document: Option<&Value>,
        timeout: Option<Duration>,
    ) -> Result<RawResponse> {
        let request = self.prepare(method, path, &[], timeout);
        let outcome = match document {
            Some(document) => request
                .set("Content-Type", "application/json")
                .send_bytes(&serde_json::to_vec(document).expect("JSON values serialize")),
            None => request.call(),
        };
        let response = match outcome {
            Ok(response) => response,
            Err(ureq::Error::Status(_, response)) => response,
            Err(ureq::Error::Transport(error)) => {
                return Err(Error::launch(format!("{method} {path} failed: {error}")))
            }
        };
        let status = response.status();
        let mut body = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut body)
            .map_err(|error| Error::launch(format!("{method} {path} failed: {error}")))?;
        Ok((status, body))
    }

    pub fn get_json(
        &self,
        path: &str,
        timeout: Option<Duration>,
    ) -> Result<(u16, Map<String, Value>)> {
        let (status, body) = self.request("GET", path, None, timeout)?;
        Ok((status, crate::values::parse_object(&body)))
    }

    pub fn post_json(
        &self,
        path: &str,
        document: &Value,
        timeout: Option<Duration>,
    ) -> Result<(u16, Map<String, Value>)> {
        let (status, body) = self.request("POST", path, Some(document), timeout)?;
        Ok((status, crate::values::parse_object(&body)))
    }

    pub fn healthy(&self, timeout: Duration) -> bool {
        matches!(self.get_json("/healthz", Some(timeout)), Ok((200, _)))
    }

    /// The `/v1/models` document (empty when it does not answer 200).
    pub fn models_document(&self, timeout: Option<Duration>) -> Result<Map<String, Value>> {
        let (status, document) = self.get_json("/v1/models", timeout)?;
        Ok(if status == 200 { document } else { Map::new() })
    }

    pub fn models(&self, timeout: Option<Duration>) -> Result<Vec<Map<String, Value>>> {
        Ok(arr(get(&self.models_document(timeout)?, "data"))
            .iter()
            .filter_map(|e| e.as_object().cloned())
            .collect())
    }

    pub fn model_names(&self, timeout: Option<Duration>) -> Vec<String> {
        self.models(timeout)
            .unwrap_or_default()
            .iter()
            .filter_map(|e| str_value(get(e, "id")).map(str::to_string))
            .collect()
    }

    pub fn server_status(&self, timeout: Option<Duration>) -> Map<String, Value> {
        match self.get_json("/v1/status", timeout) {
            Ok((200, document)) => document,
            _ => Map::new(),
        }
    }

    /// Sends one chat turn. `fields` are added to the payload when not null; a
    /// streamed request asks for usage in the final frame.
    pub fn chat(
        &self,
        messages: &[Value],
        model: Option<&str>,
        stream: bool,
        headers: &[(String, String)],
        fields: &Map<String, Value>,
        on_delta: &mut dyn FnMut(&Map<String, Value>),
    ) -> ChatResult {
        let mut payload = Map::new();
        payload.insert("messages".into(), Value::Array(messages.to_vec()));
        payload.insert("stream".into(), json!(stream));
        if let Some(model) = model.filter(|m| !m.is_empty()) {
            payload.insert("model".into(), json!(model));
        }
        for (name, value) in fields {
            if !value.is_null() {
                payload.insert(name.clone(), value.clone());
            }
        }
        if stream && !payload.contains_key("stream_options") {
            payload.insert("stream_options".into(), json!({"include_usage": true}));
        }
        self.complete(&payload, headers, on_delta)
    }

    /// Sends one completion request and measures it from the client side.
    pub fn complete(
        &self,
        payload: &Map<String, Value>,
        headers: &[(String, String)],
        on_delta: &mut dyn FnMut(&Map<String, Value>),
    ) -> ChatResult {
        let started = Instant::now();
        let mut result = ChatResult::default();
        let body = serde_json::to_vec(payload).expect("JSON values serialize");
        let request = self
            .prepare("POST", "/v1/chat/completions", headers, None)
            .set("Content-Type", "application/json");
        let response = match request.send_bytes(&body) {
            Ok(response) => response,
            Err(ureq::Error::Status(_, response)) => response,
            Err(ureq::Error::Transport(error)) => {
                result.error = Some(format!("POST /v1/chat/completions failed: {error}"));
                result.duration_seconds = Some(started.elapsed().as_secs_f64());
                return result;
            }
        };
        result.status = response.status();
        let streaming = response
            .header("Content-Type")
            .unwrap_or("")
            .starts_with("text/event-stream");
        if !streaming {
            let mut data = Vec::new();
            if let Err(error) = response.into_reader().read_to_end(&mut data) {
                result.error = Some(format!("reading the completion failed: {error}"));
            }
            let document = crate::values::parse_object(&data);
            let message = first_message(&document);
            result.text = str_value(get(&message, "content"))
                .unwrap_or("")
                .to_string();
            result.reasoning = str_value(get(&message, "reasoning_content"))
                .unwrap_or("")
                .to_string();
            result.tool_calls = arr(get(&message, "tool_calls"))
                .iter()
                .filter(|e| e.is_object())
                .cloned()
                .collect();
            result.finish_reason = finish_reason(&document);
            result.usage = Usage::from_document(get(&document, "usage"));
            let elapsed = started.elapsed().as_secs_f64();
            result.duration_seconds = Some(elapsed);
            result.ttft_seconds = Some(elapsed);
            if result.status >= 400 {
                result.error = Some(
                    error_message(&document).unwrap_or_else(|| format!("HTTP {}", result.status)),
                );
            }
            result.document = document;
            return result;
        }
        let mut reader = BufReader::new(response.into_reader());
        let mut line = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(error) => {
                    result.error = Some(format!("the stream ended abnormally: {error}"));
                    break;
                }
            }
            let Some(data) = line.strip_prefix(b"data:") else {
                continue;
            };
            let data = data.trim_ascii();
            if data.is_empty() {
                continue;
            }
            if data == b"[DONE]" {
                break;
            }
            let Ok(Value::Object(event)) = serde_json::from_slice::<Value>(data) else {
                continue;
            };
            result.chunks += 1;
            let (text, reasoning, calls) = delta_text(&event);
            if result.ttft_seconds.is_none()
                && (!text.is_empty() || !reasoning.is_empty() || !calls.is_empty())
            {
                result.ttft_seconds = Some(started.elapsed().as_secs_f64());
            }
            result.text.push_str(&text);
            result.reasoning.push_str(&reasoning);
            result.tool_calls.extend(calls);
            if let Some(finish) = finish_reason(&event) {
                result.finish_reason = Some(finish);
            }
            if let Some(usage) = Usage::from_document(get(&event, "usage")) {
                result.usage = Some(usage);
            }
            if let Some(message) = error_message(&event) {
                result.error = Some(message);
            }
            on_delta(&event);
        }
        let elapsed = started.elapsed().as_secs_f64();
        result.duration_seconds = Some(elapsed);
        if result.ttft_seconds.is_none() {
            result.ttft_seconds = Some(elapsed);
        }
        result
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

fn first_message(document: &Map<String, Value>) -> Map<String, Value> {
    for choice in arr(get(document, "choices")) {
        let message = obj(get(obj(choice), "message"));
        if !message.is_empty() {
            return message.clone();
        }
    }
    Map::new()
}

fn finish_reason(document: &Map<String, Value>) -> Option<String> {
    arr(get(document, "choices"))
        .iter()
        .find_map(|choice| str_value(get(obj(choice), "finish_reason")).map(str::to_string))
}

/// Text, reasoning and tool calls carried by one streamed event.
pub fn delta_text(event: &Map<String, Value>) -> (String, String, Vec<Value>) {
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    for choice in arr(get(event, "choices")) {
        let choice = obj(choice);
        let mut delta = obj(get(choice, "delta"));
        if delta.is_empty() {
            delta = obj(get(choice, "message"));
        }
        text.push_str(str_value(get(delta, "content")).unwrap_or(""));
        reasoning.push_str(str_value(get(delta, "reasoning_content")).unwrap_or(""));
        calls.extend(
            arr(get(delta, "tool_calls"))
                .iter()
                .filter(|c| c.is_object())
                .cloned(),
        );
    }
    (text, reasoning, calls)
}

fn error_message(document: &Map<String, Value>) -> Option<String> {
    str_value(get(obj(get(document, "error")), "message")).map(str::to_string)
}
