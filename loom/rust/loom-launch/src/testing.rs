//! A local stand-in for a served Loom model, for tests and GPU-free rehearsals.
//!
//! It answers the endpoints the launcher reads and the ones it forwards, with
//! configurable latency and token counts, so proxy accounting, client timing,
//! chat turns, benchmark rows and supervision can be exercised without a device.
//! The hidden `fake-serve` verb runs it with `loom-serve`'s command-line shape.
//!
//! A request that offers `tools` can ask for a tool call: a user message
//! `/js CODE` is answered with one `run_javascript` call, `/fetch URL` with one
//! `web_fetch` call, and `/loop` with a `run_javascript` call every round; once
//! a tool message follows, the answer quotes what the tool returned.

use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How the stand-in model behaves.
#[derive(Clone, Debug)]
pub struct FakeModelConfig {
    pub model: String,
    pub completion_tokens: usize,
    /// Delay before the first content token.
    pub first_token_delay: Duration,
    /// Delay between content tokens.
    pub token_delay: Duration,
    /// Status to fail every completion with, when nonzero.
    pub fail_status: u16,
    /// Emit a reasoning delta before the answer.
    pub reasoning: bool,
    /// Report this many prompt tokens instead of counting words.
    pub prompt_tokens: Option<usize>,
    /// Report the registration as failed with this message.
    pub registration_error: Option<String>,
    /// Report the registration as `preparing` for this long after start.
    pub preparation: Duration,
}

impl Default for FakeModelConfig {
    fn default() -> Self {
        FakeModelConfig {
            model: "test/model".into(),
            completion_tokens: 8,
            first_token_delay: Duration::ZERO,
            token_delay: Duration::ZERO,
            fail_status: 0,
            reasoning: false,
            prompt_tokens: None,
            registration_error: None,
            preparation: Duration::ZERO,
        }
    }
}

/// One request the stand-in received.
#[derive(Clone, Debug)]
pub struct SeenRequest {
    pub method: String,
    pub path: String,
    pub payload: Map<String, Value>,
    pub headers: Vec<(String, String)>,
}

/// An HTTP server impersonating one prepared text deployment.
pub struct FakeOpenAIServer {
    pub url: String,
    pub address: String,
    pub requests: Arc<Mutex<Vec<SeenRequest>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl FakeOpenAIServer {
    /// Starts on an ephemeral loopback port.
    pub fn start(config: FakeModelConfig) -> std::io::Result<FakeOpenAIServer> {
        Self::start_on("127.0.0.1:0", config)
    }

    pub fn start_on(listen: &str, config: FakeModelConfig) -> std::io::Result<FakeOpenAIServer> {
        let listener = TcpListener::bind(listen)?;
        let local = listener.local_addr()?;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let started = Instant::now();
        let config = Arc::new(config);
        let thread = {
            let requests = requests.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let config = config.clone();
                    let requests = requests.clone();
                    std::thread::spawn(move || {
                        let _ = serve_connection(stream, &config, &requests, started);
                    });
                }
            })
        };
        Ok(FakeOpenAIServer {
            url: format!("http://{local}"),
            address: local.to_string(),
            requests,
            stop,
            thread: Some(thread),
        })
    }

    /// Chat-completion requests received so far.
    pub fn completions(&self) -> Vec<SeenRequest> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.path == "/v1/chat/completions")
            .cloned()
            .collect()
    }

    /// Blocks the calling thread while the server runs.
    pub fn join(mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(&self.address);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for FakeOpenAIServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn serve_connection(
    stream: TcpStream,
    config: &FakeModelConfig,
    requests: &Mutex<Vec<SeenRequest>>,
    started: Instant,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    let length = headers
        .iter()
        .find(|(n, _)| n == "content-length")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    let path = target.split('?').next().unwrap_or("").to_string();
    let payload = match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    };
    let registration = if let Some(message) = &config.registration_error {
        json!({"registration": "failed", "error": {"code": 5, "message": message}})
    } else if started.elapsed() < config.preparation {
        json!({"registration": "preparing"})
    } else {
        json!({"registration": "ready"})
    };
    match (method.as_str(), path.as_str()) {
        ("GET", "/healthz") => write_json(&mut writer, 200, &json!({"status": "ready"})),
        ("GET", "/v1/models") => {
            let mut loom = json!({"kind": "text", "endpoint": "/v1/chat/completions"});
            for (key, value) in registration.as_object().unwrap() {
                loom[key] = value.clone();
            }
            write_json(
                &mut writer,
                200,
                &json!({
                    "object": "list",
                    "data": [{"id": config.model, "object": "model", "owned_by": "local", "loom": loom}],
                }),
            )
        }
        ("GET", "/v1/status") => {
            let state = registration["registration"].clone();
            write_json(
                &mut writer,
                200,
                &json!({"models": {config.model.clone(): {"state": state, "name": config.model}}}),
            )
        }
        ("POST", _) | ("DELETE", _) => {
            requests.lock().unwrap().push(SeenRequest {
                method: method.clone(),
                path: path.clone(),
                payload: payload.clone(),
                headers: headers.clone(),
            });
            if path != "/v1/chat/completions" {
                return write_json(
                    &mut writer,
                    404,
                    &json!({"error": {"message": "not found", "code": "NOT_FOUND"}}),
                );
            }
            if config.fail_status != 0 {
                return write_json(
                    &mut writer,
                    config.fail_status,
                    &json!({"error": {
                        "message": "the model refused the request",
                        "type": "loom_request_error",
                        "code": "FAILED_PRECONDITION",
                    }}),
                );
            }
            let prompt_tokens = config.prompt_tokens.unwrap_or_else(|| {
                payload
                    .get("messages")
                    .and_then(Value::as_array)
                    .map(|messages| {
                        messages
                            .iter()
                            .map(|m| {
                                m.get("content")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .split_whitespace()
                                    .count()
                            })
                            .sum()
                    })
                    .unwrap_or(0)
            });
            let completion_tokens = payload
                .get("max_tokens")
                .and_then(Value::as_u64)
                .map(|limit| (limit as usize).min(config.completion_tokens))
                .unwrap_or(config.completion_tokens);
            let stream = payload
                .get("stream")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if let Some(plan) = tool_plan(&payload) {
                return answer_with_tools(&mut writer, config, &plan, stream, prompt_tokens);
            }
            if stream {
                stream_completion(&mut writer, config, prompt_tokens, completion_tokens)
            } else {
                write_json(
                    &mut writer,
                    200,
                    &completion_document(config, prompt_tokens, completion_tokens),
                )
            }
        }
        _ => write_json(
            &mut writer,
            404,
            &json!({"error": {"message": "not found", "code": "NOT_FOUND"}}),
        ),
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "Status",
    }
}

fn write_json(writer: &mut TcpStream, status: u16, document: &Value) -> std::io::Result<()> {
    let payload = serde_json::to_vec(document).unwrap();
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason(status),
        payload.len()
    );
    writer.write_all(head.as_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}

fn delta_chunk(
    config: &FakeModelConfig,
    role: Option<&str>,
    content: &str,
    reasoning: &str,
) -> Value {
    let mut delta = Map::new();
    if let Some(role) = role {
        delta.insert("role".into(), json!(role));
    }
    if !reasoning.is_empty() {
        delta.insert("reasoning_content".into(), json!(reasoning));
    }
    if !content.is_empty() {
        delta.insert("content".into(), json!(content));
    }
    json!({
        "object": "chat.completion.chunk",
        "model": config.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
    })
}

fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
    })
}

fn stream_completion(
    writer: &mut TcpStream,
    config: &FakeModelConfig,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> std::io::Result<()> {
    writer.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
          Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )?;
    let mut event = |document: &Value| -> std::io::Result<()> {
        let mut data = b"data: ".to_vec();
        data.extend(serde_json::to_vec(document).unwrap());
        data.extend(b"\n\n");
        let mut frame = format!("{:x}\r\n", data.len()).into_bytes();
        frame.extend(&data);
        frame.extend(b"\r\n");
        writer.write_all(&frame)?;
        writer.flush()
    };
    event(&delta_chunk(config, Some("assistant"), "", ""))?;
    if config.reasoning {
        std::thread::sleep(config.first_token_delay);
        event(&delta_chunk(config, None, "", "Considering the request."))?;
    }
    for index in 0..completion_tokens {
        if index == 0 {
            std::thread::sleep(config.first_token_delay);
        }
        std::thread::sleep(config.token_delay);
        event(&delta_chunk(config, None, &format!(" token{index}"), ""))?;
    }
    event(&json!({
        "object": "chat.completion.chunk",
        "model": config.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }))?;
    event(&json!({
        "object": "chat.completion.chunk",
        "model": config.model,
        "choices": [],
        "usage": usage(prompt_tokens, completion_tokens),
    }))?;
    let done = b"data: [DONE]\n\n";
    let mut frame = format!("{:x}\r\n", done.len()).into_bytes();
    frame.extend(done);
    frame.extend(b"\r\n0\r\n\r\n");
    writer.write_all(&frame)?;
    writer.flush()
}

/// What a conversation that offers tools asks the stand-in for.
#[derive(Clone, Debug, PartialEq)]
enum ToolPlan {
    /// One call of this tool with these arguments.
    Call(String, Value),
    /// A plain answer (after a tool replied).
    Reply(String),
}

fn tool_plan(payload: &Map<String, Value>) -> Option<ToolPlan> {
    let tools = payload.get("tools")?.as_array()?;
    let offered = |name: &str| {
        tools
            .iter()
            .any(|tool| tool["function"]["name"].as_str() == Some(name))
    };
    let messages = payload.get("messages")?.as_array()?;
    let last = messages.last()?;
    let asked = messages
        .iter()
        .rev()
        .find(|m| m["role"] == "user")?
        .get("content")?
        .as_str()?
        .trim();
    let looping = asked.starts_with("/loop") && offered("run_javascript");
    if last["role"] == "tool" {
        if looping {
            return Some(ToolPlan::Call(
                "run_javascript".into(),
                json!({"code": "return 1 + 1"}),
            ));
        }
        let said: String = last["content"]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(300)
            .collect();
        return Some(ToolPlan::Reply(format!("The tool answered: {said}")));
    }
    if last["role"] != "user" {
        return None;
    }
    if let Some(code) = asked
        .strip_prefix("/js ")
        .filter(|_| offered("run_javascript"))
    {
        return Some(ToolPlan::Call(
            "run_javascript".into(),
            json!({ "code": code }),
        ));
    }
    if let Some(url) = asked
        .strip_prefix("/fetch ")
        .filter(|_| offered("web_fetch"))
    {
        return Some(ToolPlan::Call(
            "web_fetch".into(),
            json!({ "url": url.trim() }),
        ));
    }
    looping.then(|| ToolPlan::Call("run_javascript".into(), json!({"code": "return 1 + 1"})))
}

/// A tool call (streamed in pieces, as OpenAI servers do) or a plain reply.
fn answer_with_tools(
    writer: &mut TcpStream,
    config: &FakeModelConfig,
    plan: &ToolPlan,
    stream: bool,
    prompt_tokens: usize,
) -> std::io::Result<()> {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let chunk = |delta: Value, finish: Value| {
        json!({
            "object": "chat.completion.chunk",
            "model": config.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })
    };
    let (events, message, finish, completion_tokens) = match plan {
        ToolPlan::Call(name, arguments) => {
            let id = format!("call_{}", CALLS.fetch_add(1, Ordering::SeqCst));
            let text = arguments.to_string();
            let cut = text
                .char_indices()
                .map(|(i, _)| i)
                .find(|&i| i >= text.len() / 2)
                .unwrap_or(text.len());
            let mut events = vec![chunk(
                json!({"tool_calls": [{"index": 0, "id": id, "type": "function",
                                       "function": {"name": name, "arguments": ""}}]}),
                Value::Null,
            )];
            for piece in [&text[..cut], &text[cut..]] {
                events.push(chunk(
                    json!({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}),
                    Value::Null,
                ));
            }
            let message = json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": id, "type": "function", "function": {"name": name, "arguments": text}}]});
            (events, message, "tool_calls", 12)
        }
        ToolPlan::Reply(text) => {
            let events = text
                .split_inclusive(' ')
                .map(|word| chunk(json!({ "content": word }), Value::Null))
                .collect::<Vec<_>>();
            let words = events.len();
            let message = json!({"role": "assistant", "content": text});
            (events, message, "stop", words)
        }
    };
    if !stream {
        return write_json(
            writer,
            200,
            &json!({
                "object": "chat.completion",
                "model": config.model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": usage(prompt_tokens, completion_tokens),
            }),
        );
    }
    writer.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
          Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )?;
    let mut send = |data: &[u8]| -> std::io::Result<()> {
        let mut frame = format!("{:x}\r\n", data.len()).into_bytes();
        frame.extend(data);
        frame.extend(b"\r\n");
        writer.write_all(&frame)?;
        writer.flush()
    };
    let mut documents = vec![chunk(json!({"role": "assistant"}), Value::Null)];
    documents.extend(events);
    documents.push(chunk(json!({}), json!(finish)));
    documents.push(json!({
        "object": "chat.completion.chunk",
        "model": config.model,
        "choices": [],
        "usage": usage(prompt_tokens, completion_tokens),
    }));
    for document in documents {
        std::thread::sleep(config.token_delay);
        let mut data = b"data: ".to_vec();
        data.extend(serde_json::to_vec(&document).unwrap());
        data.extend(b"\n\n");
        send(&data)?;
    }
    send(b"data: [DONE]\n\n")?;
    writer.write_all(b"0\r\n\r\n")?;
    writer.flush()
}

fn completion_document(
    config: &FakeModelConfig,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Value {
    let content: String = (0..completion_tokens)
        .map(|i| format!(" token{i}"))
        .collect();
    json!({
        "object": "chat.completion",
        "model": config.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": usage(prompt_tokens, completion_tokens),
    })
}

/// `loom-launch fake-serve`: `loom-serve`'s command-line shape over the
/// stand-in. Reads the deployment name from `--model=<serve.json>`, binds
/// `--listen`, prints the ready event, and serves until killed. Like
/// `loom-serve`, it needs `--device` or `--devices` and refuses `--devices`
/// next to the single-device flags.
///
/// Environment hooks: `LOOM_FAKE_SERVE_FAIL` (registration error message),
/// `LOOM_FAKE_SERVE_PREPARE_MS` (time spent "preparing"),
/// `LOOM_FAKE_SERVE_TOKENS` (completion length), `LOOM_FAKE_SERVE_TOKEN_MS`,
/// `LOOM_FAKE_SERVE_PANEL_MS` (print `loom-serve`'s i8-panel derivation lines,
/// `deriving`, then `derived` and the loader's `validation_seconds` this long
/// after; `0` prints `reused` at once).
pub fn fake_serve_main(arguments: &[String]) -> i32 {
    let mut listen = "127.0.0.1:8080".to_string();
    let mut model = "test/model".to_string();
    // loom-serve's own rule: a devices document names every device and pool,
    // so it cannot be combined with the single-device flags.
    let flag = |name: &str| {
        arguments
            .iter()
            .any(|a| a.starts_with(&format!("--{name}=")))
    };
    if flag("devices") {
        let single = [
            "device",
            "pool_backing",
            "memory_bytes",
            "weight_slab_bytes",
        ];
        if single.iter().any(|name| flag(name)) {
            eprintln!(
                "fake-serve: --devices cannot be combined with --device or single-device \
                 backing, slab, memory, or idle flags"
            );
            return 2;
        }
        let path = arguments
            .iter()
            .find_map(|a| a.strip_prefix("--devices="))
            .unwrap_or("");
        let devices = crate::store::read_json(std::path::Path::new(path));
        if !devices.get("devices").is_some_and(Value::is_object) {
            eprintln!("fake-serve: {path} is not a devices document");
            return 2;
        }
    } else if !flag("device") {
        eprintln!("fake-serve: --device is required, or provide --devices=path.json");
        return 2;
    }
    for argument in arguments {
        if let Some(value) = argument.strip_prefix("--listen=") {
            listen = value.to_string();
        } else if let Some(value) = argument.strip_prefix("--model=") {
            let document = crate::store::read_json(std::path::Path::new(value));
            if let Some(name) = document.get("name").and_then(Value::as_str) {
                model = name.to_string();
            }
        }
    }
    let number = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<u64>().ok());
    let config = FakeModelConfig {
        model,
        completion_tokens: number("LOOM_FAKE_SERVE_TOKENS").unwrap_or(64) as usize,
        token_delay: Duration::from_millis(number("LOOM_FAKE_SERVE_TOKEN_MS").unwrap_or(1)),
        registration_error: std::env::var("LOOM_FAKE_SERVE_FAIL")
            .ok()
            .filter(|v| !v.is_empty()),
        preparation: Duration::from_millis(number("LOOM_FAKE_SERVE_PREPARE_MS").unwrap_or(0)),
        ..Default::default()
    };
    eprintln!("fake-serve: serving {} (no device is used)", config.model);
    match FakeOpenAIServer::start_on(&listen, config) {
        Ok(server) => {
            eprintln!(
                "{}",
                json!({"event": "ready", "address": server.address, "models": 1})
            );
            if let Some(panel) = number("LOOM_FAKE_SERVE_PANEL_MS") {
                std::thread::spawn(move || {
                    let directory = "parameter/derived-cache-raw/fake";
                    if panel == 0 {
                        eprintln!("i8-panel reused {directory}");
                    } else {
                        eprintln!("i8-panel deriving {directory}");
                        std::thread::sleep(Duration::from_millis(panel));
                        eprintln!("i8-panel derived {directory}");
                    }
                    eprintln!(
                        "i8-panel validation_seconds={:.6} source_hashes=1 payload_hashes=0 verify=0",
                        panel as f64 / 1000.0
                    );
                });
            }
            server.join();
            0
        }
        Err(error) => {
            eprintln!("fake-serve: cannot listen on {listen}: {error}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(payload: Value) -> Option<ToolPlan> {
        tool_plan(payload.as_object().unwrap())
    }

    #[test]
    fn tool_requests_are_planned_from_the_conversation() {
        let tools = json!([{"type": "function", "function": {"name": "run_javascript"}},
                           {"type": "function", "function": {"name": "web_fetch"}}]);
        let user = |text: &str| json!({"role": "user", "content": text});
        assert_eq!(
            plan(json!({"tools": tools, "messages": [user("/js 6 * 7")]})),
            Some(ToolPlan::Call(
                "run_javascript".into(),
                json!({"code": "6 * 7"})
            ))
        );
        assert_eq!(
            plan(json!({"tools": tools, "messages": [user("/fetch https://example.com/ ")]})),
            Some(ToolPlan::Call(
                "web_fetch".into(),
                json!({"url": "https://example.com/"})
            ))
        );
        let answered = json!([user("/js 6 * 7"),
            {"role": "assistant", "content": null, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_0", "content": "{\"result\":42}"}]);
        assert_eq!(
            plan(json!({"tools": tools, "messages": answered})),
            Some(ToolPlan::Reply("The tool answered: {\"result\":42}".into()))
        );
        let looping = json!([user("/loop"), {"role": "tool", "content": "2"}]);
        assert!(matches!(
            plan(json!({"tools": tools, "messages": looping})),
            Some(ToolPlan::Call(..))
        ));
        assert_eq!(plan(json!({"messages": [user("/js 1")]})), None);
        assert_eq!(
            plan(json!({"tools": tools, "messages": [user("hello")]})),
            None
        );
        let only_js = json!([{"type": "function", "function": {"name": "run_javascript"}}]);
        assert_eq!(
            plan(json!({"tools": only_js, "messages": [user("/fetch https://example.com/")]})),
            None
        );
    }
}
