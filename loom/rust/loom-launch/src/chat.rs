//! A terminal conversation against a launched profile.
//!
//! The chat client is a normal OpenAI client with launcher additions: it knows
//! the profile's template policy (so it can toggle thinking, preview a rendered
//! prompt and report how much of the served context a turn used) and it keeps
//! the full conversation history a retained-session server expects.

use crate::client::{ChatResult, Client};
use crate::profile::Profile;
use crate::values::{arr, get, iso_utc, now_seconds, obj, str_value};
use serde_json::{json, Map, Value};
use std::io::{BufRead, Write};

/// What a chat turn is sent through; the client, or a stub in tests.
pub trait ChatBackend {
    fn chat(
        &self,
        messages: &[Value],
        model: Option<&str>,
        headers: &[(String, String)],
        fields: &Map<String, Value>,
        on_delta: &mut dyn FnMut(&Map<String, Value>),
    ) -> ChatResult;
}

impl ChatBackend for Client {
    fn chat(
        &self,
        messages: &[Value],
        model: Option<&str>,
        headers: &[(String, String)],
        fields: &Map<String, Value>,
        on_delta: &mut dyn FnMut(&Map<String, Value>),
    ) -> ChatResult {
        Client::chat(self, messages, model, true, headers, fields, on_delta)
    }
}

/// The endpoint rejected or failed one turn.
#[derive(Debug)]
pub struct ChatFailure(pub String);

/// One conversation bound to an endpoint and a model name.
pub struct ChatSession<'a> {
    pub backend: &'a dyn ChatBackend,
    pub profile: Profile,
    pub model: Option<String>,
    pub thinking: Option<bool>,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub session: Option<String>,
    /// The served context, when it is narrower than the package's own limit.
    pub context_length: Option<i64>,
    /// Extra controls passed to the package's chat policy unchanged.
    pub template_kwargs: Map<String, Value>,
    pub messages: Vec<Value>,
    pub results: Vec<ChatResult>,
}

impl<'a> ChatSession<'a> {
    pub fn new(backend: &'a dyn ChatBackend, profile: Profile) -> ChatSession<'a> {
        let model = Some(profile.model_id.clone());
        let thinking = profile.chat.as_ref().map(|c| c.default_thinking());
        ChatSession {
            backend,
            profile,
            model,
            thinking,
            max_tokens: None,
            temperature: None,
            session: None,
            context_length: None,
            template_kwargs: Map::new(),
            messages: Vec::new(),
            results: Vec::new(),
        }
    }

    pub fn set_system(&mut self, system: Option<&str>) {
        if let Some(system) = system {
            self.messages
                .push(json!({"role": "system", "content": system}));
        }
    }

    pub fn reset(&mut self) {
        self.messages
            .retain(|m| str_value(get(obj(m), "role")) == Some("system"));
        self.results.clear();
    }

    pub fn send(
        &mut self,
        text: &str,
        on_text: &mut dyn FnMut(&str),
        on_reasoning: &mut dyn FnMut(&str),
    ) -> Result<&ChatResult, ChatFailure> {
        self.messages.push(json!({"role": "user", "content": text}));
        let mut fields = Map::new();
        if let Some(max_tokens) = self.max_tokens {
            fields.insert("max_tokens".into(), json!(max_tokens));
        }
        if let Some(temperature) = self.temperature {
            fields.insert("temperature".into(), json!(temperature));
        }
        if let Some(thinking) = self.thinking {
            fields.insert("enable_thinking".into(), json!(thinking));
        }
        if !self.template_kwargs.is_empty() {
            fields.insert(
                "chat_template_kwargs".into(),
                Value::Object(self.template_kwargs.clone()),
            );
        }
        let headers: Vec<(String, String)> = self
            .session
            .iter()
            .map(|s| ("X-Loom-Session".to_string(), s.clone()))
            .collect();
        let mut on_delta = |event: &Map<String, Value>| {
            for choice in arr(get(event, "choices")) {
                let delta = obj(get(obj(choice), "delta"));
                if let Some(reasoning) = str_value(get(delta, "reasoning_content")) {
                    on_reasoning(reasoning);
                }
                if let Some(content) = str_value(get(delta, "content")) {
                    on_text(content);
                }
            }
        };
        let result = self.backend.chat(
            &self.messages,
            self.model.as_deref(),
            &headers,
            &fields,
            &mut on_delta,
        );
        if let Some(error) = &result.error {
            // The user message stays in history only if the turn completed.
            self.messages.pop();
            return Err(ChatFailure(error.clone()));
        }
        let mut assistant = json!({"role": "assistant", "content": result.text});
        if !result.tool_calls.is_empty() {
            assistant["tool_calls"] = Value::Array(result.tool_calls.clone());
        }
        self.messages.push(assistant);
        self.results.push(result);
        Ok(self.results.last().unwrap())
    }

    /// The context a request is actually held to.
    pub fn context_budget(&self) -> Option<i64> {
        self.context_length
            .filter(|c| *c != 0)
            .or(self.profile.context_length)
    }

    pub fn last_prompt_tokens(&self) -> Option<i64> {
        self.results.last()?.usage.as_ref()?.prompt_tokens
    }

    pub fn transcript(&self) -> Value {
        json!({
            "model": self.model,
            "profile": self.profile.repo,
            "thinking": self.thinking,
            "messages": self.messages,
            "usage": self.results.iter().map(|r| r.usage.as_ref().map(|u| u.describe())).collect::<Vec<_>>(),
        })
    }

    /// The rendered prompt the next turn would produce, per the package policy.
    pub fn preview(&self) -> crate::errors::Result<String> {
        let mut messages = self.messages.clone();
        messages.push(json!({"role": "user", "content": "<next turn>"}));
        self.profile.render(&messages, self.thinking, None)
    }
}

pub const HELP: &str = "commands
  /help                 show this help
  /reset                forget the conversation, keep the system prompt
  /think on|off         request or suppress the package's reasoning block
  /system TEXT          replace the system prompt and reset
  /prompt               print the rendered prompt for the next turn
  /usage                show the last turn's token usage and rate
  /save PATH            write the transcript as JSON
  /exit                 leave (Ctrl-D works too)";

/// The usage line: `[12 prompt, 4 completion, 15.0 tok/s, ttft 50 ms]  (0% of 4096)`.
pub fn usage_line(session: &ChatSession<'_>, result: &ChatResult) -> Option<String> {
    let usage = result.usage.as_ref()?;
    let mut parts = Vec::new();
    if let Some(prompt) = usage.prompt_tokens.filter(|v| *v != 0) {
        parts.push(format!("{prompt} prompt"));
    }
    if let Some(completion) = usage.completion_tokens.filter(|v| *v != 0) {
        parts.push(format!("{completion} completion"));
    }
    if let Some(rate) = result.decode_tokens_per_second().filter(|r| *r != 0.0) {
        parts.push(format!("{} tok/s", crate::values::grouped(rate, 1)));
    }
    if let Some(ttft) = result.ttft_seconds {
        parts.push(format!("ttft {:.0} ms", ttft * 1000.0));
    }
    if let Some(cached) = usage.cached_tokens.filter(|v| *v != 0) {
        parts.push(format!("{cached} cached"));
    }
    if parts.is_empty() {
        return None;
    }
    let mut suffix = String::new();
    if let (Some(prompt), Some(context)) = (
        usage.prompt_tokens.filter(|v| *v != 0),
        session.context_budget().filter(|c| *c != 0),
    ) {
        suffix = format!("  ({}% of {context})", prompt * 100 / context);
    }
    Some(format!("[{}]{suffix}", parts.join(", ")))
}

/// A small interactive loop; returns a process exit code.
pub fn run_repl(
    session: &mut ChatSession<'_>,
    input: &mut dyn FnMut(&str) -> Option<String>,
    write: &mut dyn FnMut(&str),
) -> i32 {
    loop {
        let Some(line) = input("you> ") else {
            write("\n");
            return 0;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            if command(session, line, write) {
                return 0;
            }
            continue;
        }
        turn(session, line, write);
    }
}

fn turn(session: &mut ChatSession<'_>, text: &str, write: &mut dyn FnMut(&str)) {
    write("\nassistant> ");
    let shown_reasoning = std::cell::Cell::new(false);
    let sink = std::cell::RefCell::new(write);
    let outcome = {
        let mut on_reasoning = |chunk: &str| {
            let mut guard = sink.borrow_mut();
            let out: &mut dyn FnMut(&str) = &mut **guard;
            if !shown_reasoning.get() {
                out("\n[reasoning] ");
                shown_reasoning.set(true);
            }
            out(chunk);
        };
        let mut on_text = |chunk: &str| {
            let mut guard = sink.borrow_mut();
            let out: &mut dyn FnMut(&str) = &mut **guard;
            if shown_reasoning.get() {
                out("\n\n");
                shown_reasoning.set(false);
            }
            out(chunk);
        };
        session.send(text, &mut on_text, &mut on_reasoning).cloned()
    };
    let write = sink.into_inner();
    match outcome {
        Err(ChatFailure(message)) => write(&format!("\nerror: {message}\n")),
        Ok(result) => {
            write("\n");
            if !result.tool_calls.is_empty() {
                write(&format!(
                    "[{} tool call(s) requested]\n",
                    result.tool_calls.len()
                ));
                write(
                    &(crate::values::dumps_pretty(&Value::Array(result.tool_calls.clone())) + "\n"),
                );
            }
            if let Some(line) = usage_line(session, &result) {
                write(&(line + "\n"));
            }
        }
    }
}

/// Handles one `/command`; true ends the loop.
fn command(session: &mut ChatSession<'_>, line: &str, write: &mut dyn FnMut(&str)) -> bool {
    let (name, argument) = line.split_once(' ').unwrap_or((line, ""));
    let argument = argument.trim();
    match name {
        "/exit" | "/quit" => return true,
        "/help" => write(&format!("{HELP}\n")),
        "/reset" => {
            session.reset();
            write("conversation reset\n");
        }
        "/think" => {
            if argument == "on" || argument == "off" {
                session.thinking = Some(argument == "on");
                write(&format!("thinking {argument}\n"));
            } else {
                let state = match session.thinking {
                    Some(true) => "True",
                    Some(false) => "False",
                    None => "None",
                };
                write(&format!("thinking is {state}\n"));
            }
        }
        "/system" => {
            session.messages = vec![json!({"role": "system", "content": argument})];
            session.results.clear();
            write("system prompt replaced; conversation reset\n");
        }
        "/prompt" | "/prompt-preview" => match session.preview() {
            Ok(rendered) => write(&format!("{rendered}\n")),
            Err(error) => write(&format!("cannot render a prompt: {error}\n")),
        },
        "/usage" => match session.results.last() {
            None => write("no turns yet\n"),
            Some(result) => write(&(crate::values::dumps_pretty(&result.describe()) + "\n")),
        },
        "/save" => {
            if argument.is_empty() {
                write("usage: /save PATH\n");
            } else {
                match std::fs::write(argument, crate::values::dumps_pretty(&session.transcript())) {
                    Ok(()) => write(&format!("wrote {argument}\n")),
                    Err(error) => write(&format!("cannot write {argument}: {error}\n")),
                }
            }
        }
        other => write(&format!("unknown command {other}; /help lists them\n")),
    }
    false
}

/// Sends one prompt, streaming the answer to stdout and usage to stderr.
pub fn one_shot(
    session: &mut ChatSession<'_>,
    text: &str,
    write: &mut dyn FnMut(&str),
) -> Result<ChatResult, ChatFailure> {
    let mut on_reasoning = |chunk: &str| {
        let mut stderr = std::io::stderr();
        let _ = stderr.write_all(chunk.as_bytes());
        let _ = stderr.flush();
    };
    let result = session.send(text, write, &mut on_reasoning)?.clone();
    write("\n");
    if !result.tool_calls.is_empty() {
        eprintln!(
            "{}",
            crate::values::dumps_pretty(&Value::Array(result.tool_calls.clone()))
        );
    }
    if let Some(line) = usage_line(session, &result) {
        eprintln!("{line}");
    }
    Ok(result)
}

/// The JSON form `chat --json` prints for one turn.
pub fn describe_turn(result: &ChatResult) -> Value {
    let mut document = result.describe();
    document["time"] = json!(iso_utc(now_seconds()));
    document
}

/// Reads one line from stdin after printing `prompt`; `None` at end of input.
pub fn stdin_line(prompt: &str) -> Option<String> {
    let mut stdout = std::io::stdout();
    let _ = stdout.write_all(prompt.as_bytes());
    let _ = stdout.flush();
    let mut line = String::new();
    match std::io::stdin().lock().read_line(&mut line) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(line),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::Usage;
    use crate::profile::{Profile, ProfileDocuments};
    use std::cell::RefCell;

    type Call = (
        Vec<Value>,
        Map<String, Value>,
        Vec<(String, String)>,
        Option<String>,
    );

    struct StubClient {
        replies: RefCell<Vec<Result<Value, ChatResult>>>,
        calls: RefCell<Vec<Call>>,
    }

    impl StubClient {
        fn new(replies: Vec<Result<Value, ChatResult>>) -> StubClient {
            StubClient {
                replies: RefCell::new(replies),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl ChatBackend for StubClient {
        fn chat(
            &self,
            messages: &[Value],
            model: Option<&str>,
            headers: &[(String, String)],
            fields: &Map<String, Value>,
            on_delta: &mut dyn FnMut(&Map<String, Value>),
        ) -> ChatResult {
            self.calls.borrow_mut().push((
                messages.to_vec(),
                fields.clone(),
                headers.to_vec(),
                model.map(str::to_string),
            ));
            let reply = self.replies.borrow_mut().remove(0);
            let reply = match reply {
                Err(result) => return result,
                Ok(reply) => reply,
            };
            let chunks: Vec<String> = reply
                .get("chunks")
                .and_then(Value::as_array)
                .map(|c| {
                    c.iter()
                        .filter_map(|s| s.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            for chunk in &chunks {
                let event = json!({"choices": [{"delta": {"content": chunk}}]});
                on_delta(event.as_object().unwrap());
            }
            ChatResult {
                status: 200,
                text: reply
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| chunks.concat()),
                usage: Some(Usage {
                    prompt_tokens: Some(
                        reply
                            .get("prompt_tokens")
                            .and_then(Value::as_i64)
                            .unwrap_or(12),
                    ),
                    completion_tokens: Some(
                        reply
                            .get("completion_tokens")
                            .and_then(Value::as_i64)
                            .unwrap_or(4),
                    ),
                    total_tokens: Some(16),
                    cached_tokens: None,
                }),
                ttft_seconds: Some(0.05),
                duration_seconds: Some(0.25),
                ..Default::default()
            }
        }
    }

    fn profile() -> Profile {
        let config = json!({
            "context_length": 4096,
            "chat": {
                "system_prefix": "<|im_start|>system\n",
                "user_prefix": "<|im_start|>user\n",
                "assistant_prefix": "<|im_start|>assistant\n",
                "suffix": "<|im_end|>\n",
                "generation_prompt": "<|im_start|>assistant\n",
                "generation_prompt_plain": "<|im_start|>assistant\n",
                "thinking": "toggle",
                "thinking_default": true,
                "terminal": "<|im_end|>",
            },
        });
        Profile::from_documents(
            "loom-fun/test-loom",
            &ProfileDocuments {
                model_card: obj(&json!({"model": {"id": "test/model", "name": "Test"}})).clone(),
                llm_config: obj(&config).clone(),
                ..Default::default()
            },
            None,
        )
    }

    fn feed(lines: &[&str]) -> impl FnMut(&str) -> Option<String> {
        let mut remaining: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        move |_| {
            if remaining.is_empty() {
                None
            } else {
                Some(remaining.remove(0))
            }
        }
    }

    #[test]
    fn a_turn_appends_history_and_reports_usage() {
        let client = StubClient::new(vec![Ok(json!({"chunks": ["he", "llo"]}))]);
        let mut chat = ChatSession::new(&client, profile());
        let streamed = RefCell::new(String::new());
        let text = chat
            .send(
                "hi",
                &mut |c| streamed.borrow_mut().push_str(c),
                &mut |_| {},
            )
            .unwrap()
            .text
            .clone();
        assert_eq!(text, "hello");
        assert_eq!(*streamed.borrow(), "hello");
        let roles: Vec<_> = chat
            .messages
            .iter()
            .map(|m| m["role"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(roles, ["user", "assistant"]);
        assert_eq!(
            client.calls.borrow()[0].0,
            vec![json!({"role": "user", "content": "hi"})]
        );
        assert_eq!(client.calls.borrow()[0].3.as_deref(), Some("test/model"));
        assert_eq!(chat.last_prompt_tokens(), Some(12));
        assert_eq!(chat.context_budget(), Some(4096));
        chat.context_length = Some(1024);
        assert_eq!(chat.context_budget(), Some(1024));
    }

    #[test]
    fn template_controls_and_sessions_are_sent() {
        let client = StubClient::new(vec![
            Ok(json!({"text": "ok"})),
            Ok(json!({"text": "again"})),
        ]);
        let mut chat = ChatSession::new(&client, profile());
        chat.thinking = Some(false);
        chat.session = Some("history".into());
        chat.max_tokens = Some(64);
        chat.template_kwargs
            .insert("preserve_thinking".into(), json!(true));
        chat.send("hi", &mut |_| {}, &mut |_| {}).unwrap();
        let (_, fields, headers, _) = client.calls.borrow()[0].clone();
        assert_eq!(fields["enable_thinking"], false);
        assert_eq!(fields["max_tokens"], 64);
        assert_eq!(
            fields["chat_template_kwargs"],
            json!({"preserve_thinking": true})
        );
        assert_eq!(
            headers,
            vec![("X-Loom-Session".to_string(), "history".to_string())]
        );
        chat.send("more", &mut |_| {}, &mut |_| {}).unwrap();
        assert_eq!(client.calls.borrow()[1].0.len(), 3);
    }

    #[test]
    fn a_failed_turn_leaves_history_unchanged() {
        let failure = ChatResult {
            status: 500,
            error: Some("model failed".into()),
            ..Default::default()
        };
        let client = StubClient::new(vec![Err(failure)]);
        let mut chat = ChatSession::new(&client, profile());
        let error = chat.send("hi", &mut |_| {}, &mut |_| {}).unwrap_err();
        assert!(error.0.contains("model failed"));
        assert!(chat.messages.is_empty());
        assert!(chat.results.is_empty());
    }

    #[test]
    fn reset_keeps_the_system_prompt() {
        let client = StubClient::new(vec![Ok(json!({"text": "ok"}))]);
        let mut chat = ChatSession::new(&client, profile());
        chat.set_system(Some("be terse"));
        chat.send("hi", &mut |_| {}, &mut |_| {}).unwrap();
        chat.reset();
        assert_eq!(
            chat.messages,
            vec![json!({"role": "system", "content": "be terse"})]
        );
        assert!(chat.results.is_empty());
    }

    #[test]
    fn tool_calls_are_kept_in_history() {
        let result = ChatResult {
            status: 200,
            tool_calls: vec![json!({"id": "1", "function": {"name": "read", "arguments": "{}"}})],
            ..Default::default()
        };
        let client = StubClient::new(vec![Err(result)]);
        let mut chat = ChatSession::new(&client, profile());
        chat.send("read a file", &mut |_| {}, &mut |_| {}).unwrap();
        assert_eq!(
            chat.messages.last().unwrap()["tool_calls"][0]["function"]["name"],
            "read"
        );
    }

    #[test]
    fn repl_commands_change_session_state() {
        let client = StubClient::new(vec![Ok(json!({"text": "ok"}))]);
        let mut chat = ChatSession::new(&client, profile());
        let mut written = String::new();
        let code = run_repl(
            &mut chat,
            &mut feed(&[
                "/help",
                "/think off",
                "/system be nice",
                "/usage",
                "/unknown",
                "/exit",
            ]),
            &mut |t| written.push_str(t),
        );
        assert_eq!(code, 0);
        assert!(written.contains("commands"));
        assert!(written.contains("thinking off"));
        assert!(written.contains("system prompt replaced"));
        assert!(written.contains("no turns yet"));
        assert!(written.contains("unknown command"));
        assert_eq!(chat.thinking, Some(false));
        assert_eq!(
            chat.messages,
            vec![json!({"role": "system", "content": "be nice"})]
        );
    }

    #[test]
    fn repl_prints_answers_and_usage() {
        let client = StubClient::new(vec![Ok(json!({"chunks": ["answer"]}))]);
        let mut chat = ChatSession::new(&client, profile());
        let mut written = String::new();
        run_repl(&mut chat, &mut feed(&["hello", "/exit"]), &mut |t| {
            written.push_str(t)
        });
        assert!(written.contains("answer"));
        assert!(
            written.contains("[12 prompt, 4 completion, 15.0 tok/s, ttft 50 ms]  (0% of 4096)"),
            "{written}"
        );
    }

    #[test]
    fn repl_previews_the_rendered_prompt() {
        let client = StubClient::new(vec![]);
        let mut chat = ChatSession::new(&client, profile());
        chat.thinking = Some(false);
        let mut written = String::new();
        run_repl(&mut chat, &mut feed(&["/prompt"]), &mut |t| {
            written.push_str(t)
        });
        assert!(
            written.contains("<|im_start|>user\n<next turn><|im_end|>\n<|im_start|>assistant\n")
        );
    }

    #[test]
    fn repl_saves_the_transcript() {
        let root = crate::store::tests::scratch("chat-save");
        let target = root.join("transcript.json");
        let client = StubClient::new(vec![Ok(json!({"text": "ok"}))]);
        let mut chat = ChatSession::new(&client, profile());
        let save = format!("/save {}", target.display());
        run_repl(&mut chat, &mut feed(&["hi", &save, "/exit"]), &mut |_| {});
        let document: Value =
            serde_json::from_str(&std::fs::read_to_string(&target).unwrap()).unwrap();
        assert_eq!(document["profile"], "loom-fun/test-loom");
        assert_eq!(
            document["messages"].as_array().unwrap().last().unwrap()["content"],
            "ok"
        );
    }
}
