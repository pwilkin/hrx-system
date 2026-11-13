//! Dashboard routes over real HTTP: a launcher's page on a listener of the
//! test's own, a page hosted by a launch's proxy, and reading a launch another
//! process recorded. The model is the in-process fake server.

use super::*;
use crate::hardware::Hardware;
use crate::proxy::{Proxy, ProxyOptions};
use crate::server::write_current;
use crate::store::tests::scratch;
use crate::store::{write_json, InstalledModel};
use crate::testing::{FakeModelConfig, FakeOpenAIServer};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Instant;

const REPO: &str = "loom-fun/test-loom";

fn card_documents() -> ProfileDocuments {
    let card = json!({
        "schema_version": 1,
        "model": {"id": "test/model", "name": "Test model"},
        "services": {"chat": {"kind": "text", "version": 1}},
        "presets": {"chat": {"service": "chat", "execution": {
            "rows": 2, "context_capacity": 8192, "prefill_capacity": 1024,
            "max_tokens": 512, "draft_depth": 0}}},
    });
    let info = json!({
        "card": {"memory_bytes": 1000000},
        "performance": {
            "baseline": {"name": "llama.cpp", "version": "b1", "args": "-ngl 99"},
            "sessions": [{"date": "2026-11-11", "metrics": {
                "decode": {"loom": 10.0, "baseline": 5.0},
                "prefill_1k": {"loom": 200.0}}}],
        },
        "roofline": {"device": {"name": "gfx1151"}, "metrics": {"decode": {"ceiling": 100.0}}},
    });
    let config = json!({
        "arch": "test", "gguf": "test-Q4_K_M.gguf", "context_length": 8192,
        "chat": {"user_prefix": "<u>", "assistant_prefix": "<a>", "suffix": "</s>", "thinking": "toggle",
                 "think_open": "<think>", "think_close": "</think>", "thinking_default": true},
    });
    ProfileDocuments {
        model_card: obj(&card).clone(),
        info: obj(&info).clone(),
        llm_config: obj(&config).clone(),
        readme: None,
        tags: vec!["loom-serve".into()],
        revision: Some("abc".into()),
    }
}

/// A state directory with one installed card, one stored run and one launch log.
fn fixture(name: &str) -> PathBuf {
    let state = scratch(name);
    let documents = card_documents();
    let card = state.join("models").join("loom-fun-test-loom");
    write_json(
        &card.join("model.json"),
        &Value::Object(documents.model_card.clone()),
    )
    .unwrap();
    write_json(
        &card.join("info.json"),
        &Value::Object(documents.info.clone()),
    )
    .unwrap();
    write_json(
        &card.join("llm_config.json"),
        &Value::Object(documents.llm_config.clone()),
    )
    .unwrap();
    InstallRegistry::new(&state)
        .record(&InstalledModel {
            repo: REPO.into(),
            path: card.to_string_lossy().into_owned(),
            preset: Some("chat".into()),
            revision: Some("abc".into()),
            installed_at: Some("2026-11-12T00:00:00Z".into()),
            size_bytes: Some(4096),
            ..Default::default()
        })
        .unwrap();
    ProfileCache::new(&state)
        .put(REPO, &documents.to_cache(REPO))
        .unwrap();
    let mut record = obj(&json!({
        "id": "20261112T000000-1",
        "kind": "loom-launch-benchmark",
        "profile": REPO,
        "suite": "quick",
        "started_at": "2026-11-12T00:00:00Z",
        "elapsed_seconds": 3.5,
        "ok": true,
        "harness": {"workload": "loom-launch/text-v1", "repeats": 1, "thinking": false},
        "metrics": {"decode": {"value": 12.0, "unit": "tok/s", "label": "decode, short prompt"}},
        "comparison": [{"metric": "decode", "label": "decode, short prompt", "measured": 12.0,
                        "value": 12.0, "published": 10.0, "ratio": 1.2, "ceiling": 100.0,
                        "roofline_fraction": 0.12}],
        "notes": ["a note"],
    }))
    .clone();
    BenchmarkStore::new(&state).save(&mut record).unwrap();
    let run = RunStore::new(&state)
        .create(Some("20261112T000000-1"), Map::new())
        .unwrap();
    std::fs::write(run.serve_log(), "server starting\nserver ready\n").unwrap();
    std::fs::write(run.event_log(), "{\"event\":\"ready\"}\n").unwrap();
    std::fs::write(
        run.request_log(),
        "{\"id\":\"r1\",\"route\":\"chat_completions\",\"status\":200}\n",
    )
    .unwrap();
    state
}

fn machine() -> Hardware {
    Hardware {
        system: "Linux".into(),
        machine: "x86_64".into(),
        cpu: None,
        total_memory_bytes: Some(64 << 30),
        devices: Vec::new(),
        npu: false,
    }
}

/// A dashboard on its own loopback listener, for one test.
struct Served {
    context: Arc<Context>,
    server: Arc<tiny_http::Server>,
    base: String,
    port: u16,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Served {
    fn drop(&mut self) {
        self.context.close();
        self.server.unblock();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve(context: Arc<Context>) -> Served {
    context.set_hardware(machine());
    let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").unwrap());
    let port = server.server_addr().to_ip().unwrap().port();
    let accept = server.clone();
    let shared = context.clone();
    let thread = std::thread::spawn(move || {
        for request in accept.incoming_requests() {
            let context = shared.clone();
            std::thread::spawn(move || {
                let method = request.method().as_str().to_uppercase();
                let path = request.url().to_string();
                handle(&context, request, &method, &path);
            });
        }
    });
    Served {
        context,
        server,
        base: format!("http://127.0.0.1:{port}"),
        port,
        thread: Some(thread),
    }
}

fn standalone(state: &Path, loom_fetch: Option<String>) -> Served {
    standalone_with(state, loom_fetch, false)
}

fn standalone_with(state: &Path, loom_fetch: Option<String>, allow_web: bool) -> Served {
    serve(Context::standalone(&Service {
        state_root: state.to_path_buf(),
        hub_endpoint: Some("http://127.0.0.1:9".into()),
        loom_fetch,
        allow_web,
        ..Default::default()
    }))
}

struct Answer {
    status: u16,
    content_type: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Answer {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("{e}: {}", self.body))
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

fn call(method: &str, url: &str, body: Option<&Value>) -> Answer {
    let request = ureq::request(method, url).timeout(Duration::from_secs(30));
    let outcome = match body {
        Some(body) => request
            .set("Content-Type", "application/json")
            .send_string(&body.to_string()),
        None => request.call(),
    };
    let response = match outcome {
        Ok(response) | Err(ureq::Error::Status(_, response)) => response,
        Err(error) => panic!("{method} {url}: {error}"),
    };
    let headers = response
        .headers_names()
        .into_iter()
        .map(|name| {
            let value = response.header(&name).unwrap_or("").to_string();
            (name, value)
        })
        .collect();
    Answer {
        status: response.status(),
        content_type: response.header("Content-Type").unwrap_or("").to_string(),
        headers,
        body: response.into_string().unwrap(),
    }
}

/// A raw HTTP/1.1 exchange, for headers ureq will not send.
fn raw(port: u16, request: &str) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.read_line(&mut status).unwrap();
    status
}

fn wait_for(what: &str, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !ready() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn the_page_carries_every_panel_and_loads_nothing_from_elsewhere() {
    let state = fixture("dashboard-page");
    let served = standalone(&state, None);
    let page = call("GET", &format!("{}/", served.base), None);
    assert_eq!(page.status, 200);
    assert!(page.content_type.starts_with("text/html"));
    for id in [
        "topbar",
        "launch-repo",
        "launch-endpoint",
        "chip-device",
        "chip-context",
        "chip-sessions",
        "upstream-pill",
        "theme-toggle",
        "activity",
        "activity-list",
        "activity-empty",
        "activity-pulse",
        "launch-card",
        "policy-card",
        "policy-thinking",
        "policy-web",
        "policy-drafter",
        "traffic",
        "model-q",
        "model-results",
        "installed",
        "launch-form",
        "launch-allow-web",
        "launch-error",
        "chat-log",
        "chat-thinking",
        "thinking-note",
        "web-note",
        "web-command",
        "suites",
        "bench-error",
        "runs-body",
        "compare-body",
        "log-view",
        "log-follow",
    ] {
        assert!(page.body.contains(&format!("id=\"{id}\"")), "missing #{id}");
    }
    // Five tabs, each controlling its own panel.
    assert!(page.body.contains("role=\"tablist\""));
    for panel in ["overview", "models", "chat", "benchmarks", "logs"] {
        assert!(
            page.body
                .contains(&format!("id=\"tab-{panel}\" class=\"tab\" role=\"tab\"")),
            "missing the {panel} tab"
        );
        assert!(page.body.contains(&format!("aria-controls=\"{panel}\"")));
        assert!(
            page.body.contains(&format!(
                "id=\"{panel}\" class=\"panel\" role=\"tabpanel\" aria-labelledby=\"tab-{panel}\""
            )),
            "missing the {panel} panel"
        );
    }
    // A determinate bar for measured progress, an indeterminate one otherwise,
    // and both still when motion is reduced.
    assert!(page.body.contains("data-indeterminate"));
    assert!(page.body.contains("role: 'progressbar'"));
    for external in ["http://", "https://", "//cdn", "@import", "url("] {
        assert!(
            !page.body.contains(external),
            "the page references {external}"
        );
    }
    assert!(!page.body.contains("/*STYLE*/") && !page.body.contains("/*SCRIPT*/"));
    assert!(page.body.contains("'use strict'"));
    assert!(page.body.contains("prefers-color-scheme"));
    assert!(page.body.contains("prefers-reduced-motion"));
    // Without scripts: the static summary names what the state directory holds.
    assert!(page.body.contains("<noscript>"));
    // Overriding noscript's display would show its markup as text when scripts run.
    assert!(!page.body.contains("noscript { display"));
    assert!(page.body.contains("Static summary"));
    assert!(page.body.contains("<h3>Activity</h3>"));
    assert!(page.body.contains("Web access: web access is off"));
    assert!(page.body.contains("/api/benchmarks/20261112T000000-1"));
    assert!(page.body.contains("12.0 tok/s"));
    let policy = page.header("Content-Security-Policy").unwrap();
    assert!(policy.contains("default-src 'none'"));
    assert!(policy.contains("connect-src 'self'"));
    // The JavaScript tool runs in a Blob worker; nothing else may be a worker.
    assert!(policy.contains("worker-src blob:"));
    assert!(!policy.contains("unsafe-eval"));
    for id in ["chat-tool-js", "chat-tool-web"] {
        assert!(page.body.contains(&format!("id=\"{id}\"")), "missing #{id}");
    }
    for name in [
        "run_javascript",
        "web_fetch",
        "no DOM, no network and a 2 s limit",
    ] {
        assert!(
            page.body.contains(name),
            "the page does not describe {name}"
        );
    }
    assert_eq!(
        call("GET", &format!("{}/index.html", served.base), None).status,
        200
    );
}

#[test]
fn json_routes_answer_their_documented_shapes() {
    let state = fixture("dashboard-json");
    let served = standalone(&state, None);
    let get = |path: &str| call("GET", &format!("{}{path}", served.base), None);

    let status = get("/api/status").json();
    assert_eq!(status["mode"], "launcher");
    assert_eq!(status["launch"]["state"], "none");
    assert_eq!(status["launch"]["kind"], "none");
    assert_eq!(status["stored_runs"], 1);
    assert_eq!(status["installs"], 1);
    assert_eq!(status["capabilities"]["serve"], true);
    assert_eq!(status["capabilities"]["chat"], false);
    assert_eq!(status["suites"].as_array().unwrap().len(), 3);
    assert_eq!(status["hardware"]["memory"], "64.0 GiB");
    assert!(status["job"].is_null());
    // A launcher counts what it carries whether or not a model is loaded.
    assert_eq!(status["proxy"]["requests"], 0);
    assert_eq!(status["launcher"]["model"], "none");
    assert_eq!(status["launcher"]["counters"]["no_model"], 0);

    let installs = get("/api/installs").json();
    let first = &installs["installs"][0];
    assert_eq!(first["repo"], REPO);
    assert_eq!(first["quant"], "Q4_K_M");
    assert_eq!(first["context_length"], 8192);
    assert_eq!(first["card_exists"], true);
    assert_eq!(first["running"], false);
    assert_eq!(first["rows"], 2);
    assert_eq!(first["thinking"], "toggle");

    let runs = get("/api/benchmarks").json();
    let run = &runs["runs"][0];
    assert_eq!(run["id"], "20261112T000000-1");
    assert_eq!(run["suite"], "quick");
    assert_eq!(run["missing"], false);
    assert_eq!(run["headline"][0]["metric"], "decode");
    assert_eq!(run["headline"][0]["value"], 12.0);
    assert_eq!(run["headline"][0]["ratio"], 1.2);
    assert_eq!(runs["suites"][0]["name"], "quick");

    let record = get("/api/benchmarks/20261112T000000-1").json();
    assert_eq!(record["comparison"][0]["published"], 10.0);
    assert_eq!(get("/api/benchmarks/absent").status, 404);

    let logs = get("/api/logs?lines=1").json();
    assert_eq!(logs["run"], "20261112T000000-1");
    assert_eq!(logs["serve_log"], json!(["server ready"]));
    assert_eq!(logs["events"], json!(["{\"event\":\"ready\"}"]));
    assert_eq!(logs["current"], false);
    assert_eq!(get("/api/logs?run=absent").status, 404);

    let requests = get("/api/requests?limit=5").json();
    assert_eq!(requests["source"], "log");
    assert_eq!(requests["data"][0]["id"], "r1");

    let metrics = get("/api/metrics");
    assert_eq!(metrics.status, 200);
    assert!(metrics.body.contains("loom_launch_requests_total"));

    let models = get("/api/models?q=test&offline=1").json();
    assert_eq!(models["offline"], true);
    assert_eq!(models["count"], 1);
    let profile = &models["profiles"][0];
    assert_eq!(profile["repo"], REPO);
    assert_eq!(profile["quant"], "Q4_K_M");
    assert_eq!(profile["context_length"], 8192);
    assert_eq!(profile["decode"], 10.0);
    assert_eq!(profile["installed"], true);
    assert!(profile["fit"].as_f64().unwrap() > 0.0);
    assert!(!profile["fit_notes"].as_array().unwrap().is_empty());
    assert_eq!(
        get("/api/models?q=nothing-like-it&offline=1").json()["count"],
        0
    );
    // The hub at :9 refuses connections, so an online search falls back to the cache.
    let fallback = get("/api/models?q=test").json();
    assert_eq!(fallback["offline"], true);
    assert!(fallback["warning"].as_str().unwrap().contains("cached"));
    assert_eq!(fallback["count"], 1);
}

#[test]
fn methods_origins_and_bodies_are_checked() {
    let state = fixture("dashboard-guard");
    let served = standalone(&state, None);
    let url = |path: &str| format!("{}{path}", served.base);
    assert_eq!(call("GET", &url("/api/install"), None).status, 405);
    assert_eq!(
        call("POST", &url("/api/status"), Some(&json!({}))).status,
        405
    );
    assert_eq!(call("GET", &url("/api/absent"), None).status, 404);
    let stopped = call("POST", &url("/api/stop"), Some(&json!({})));
    assert_eq!(stopped.status, 409);
    assert!(stopped.json()["error"]["message"]
        .as_str()
        .unwrap()
        .contains("no model is loaded"));
    let not_json = raw(
        served.port,
        "POST /api/stop HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\n{}",
    );
    assert!(not_json.contains(" 415 "), "{not_json}");
    let cross_site = raw(
        served.port,
        "POST /api/stop HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: http://elsewhere.example\r\n\
         Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}",
    );
    assert!(cross_site.contains(" 403 "), "{cross_site}");
    let rebound = raw(
        served.port,
        "GET /api/status HTTP/1.1\r\nHost: rebound.example:80\r\n\r\n",
    );
    assert!(rebound.contains(" 403 "), "{rebound}");
    let localhost = raw(
        served.port,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert!(localhost.contains(" 200 "), "{localhost}");
    let broken = raw(
        served.port,
        "POST /api/install HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 1\r\n\r\n{",
    );
    assert!(broken.contains(" 400 "), "{broken}");
    for body in [
        json!({}),
        json!({"repo": "no-slash"}),
        json!({"repo": "a/../b"}),
    ] {
        assert_eq!(call("POST", &url("/api/install"), Some(&body)).status, 400);
    }
    let suite = call("POST", &url("/api/bench"), Some(&json!({"suite": "huge"})));
    assert_eq!(suite.status, 400);
    let no_launch = call("POST", &url("/api/bench"), Some(&json!({"suite": "quick"})));
    assert_eq!(no_launch.status, 409);
    let chat = call(
        "POST",
        &url("/api/chat"),
        Some(&json!({"messages": [{"role": "user", "content": "hi"}]})),
    );
    assert_eq!(chat.status, 409);
}

#[test]
fn serve_validates_before_starting_and_refuses_while_a_launch_is_recorded() {
    let state = fixture("dashboard-serve");
    let served = standalone(&state, None);
    let url = format!("{}/api/serve", served.base);
    let absent = call("POST", &url, Some(&json!({"repo": "loom-fun/absent-loom"})));
    assert_eq!(absent.status, 409);
    assert!(absent.body.contains("not installed"), "{}", absent.body);
    let too_long = call(
        "POST",
        &url,
        Some(&json!({"repo": REPO, "context": 1_000_000})),
    );
    assert_eq!(too_long.status, 400, "{}", too_long.body);
    let not_a_number = call(
        "POST",
        &url,
        Some(&json!({"repo": REPO, "parallel": "two"})),
    );
    assert_eq!(not_a_number.status, 400);
    assert!(served.context.jobs().running().is_none());

    // Another process's launch, recorded in current.json and alive (our pid).
    let model = FakeOpenAIServer::start(FakeModelConfig::default()).unwrap();
    write_current(
        &state,
        &json!({"pid": std::process::id(), "endpoint": model.url, "server_endpoint": model.url,
                "repo": REPO, "run_id": "20261112T000000-1", "context_length": 4096,
                "device": "task", "started_at": "2026-11-12T00:00:00Z"}),
    )
    .unwrap();
    let refused = call("POST", &url, Some(&json!({"repo": REPO})));
    assert_eq!(refused.status, 409);
    let message = refused.json()["error"]["message"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(message.contains("already running"), "{message}");
    assert!(message.contains(REPO), "{message}");
    let status = call("GET", &format!("{}/api/status", served.base), None).json();
    assert_eq!(status["launch"]["kind"], "external");
    assert_eq!(status["launch"]["state"], "serving");
    assert_eq!(status["launch"]["context_length"], 4096);
    assert!(status["launch"]["uptime_seconds"].as_f64().unwrap() > 0.0);
    assert_eq!(status["capabilities"]["serve"], false);
    assert_eq!(status["capabilities"]["chat"], true);
    assert_eq!(status["profile"]["chat"]["supports_thinking"], true);

    // A dashboard hosted by a launch's proxy never starts another launch.
    let proxy = hosted_proxy(&state, &model);
    let hosted = call(
        "POST",
        &format!("{}/api/serve", proxy.address()),
        Some(&json!({"repo": REPO})),
    );
    assert_eq!(hosted.status, 409);
    assert!(
        hosted.body.contains("served by the running launch"),
        "{}",
        hosted.body
    );
}

fn hosted_proxy(state: &Path, model: &FakeOpenAIServer) -> Proxy {
    let port: u16 = model.address.rsplit(':').next().unwrap().parse().unwrap();
    let mut options = ProxyOptions::new(Upstream {
        host: "127.0.0.1".into(),
        port,
    });
    options.dashboard_state = Some(state.to_path_buf());
    options.hub_endpoint = Some("http://127.0.0.1:9".into());
    options.status_provider = Some(Arc::new(|| {
        let mut status = Map::new();
        status.insert(
            "launch".into(),
            json!({"repo": REPO, "pid": 0, "context_length": 4096, "device": "task"}),
        );
        status
    }));
    Proxy::start(options, Arc::new(Metrics::default())).unwrap()
}

fn user(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

#[test]
fn chat_relays_the_model_stream_and_counts_it() {
    let state = fixture("dashboard-chat");
    let model = FakeOpenAIServer::start(FakeModelConfig {
        completion_tokens: 6,
        token_delay: Duration::from_millis(20),
        reasoning: true,
        ..Default::default()
    })
    .unwrap();
    let proxy = hosted_proxy(&state, &model);
    let url = format!("{}/api/chat", proxy.address());
    let streamed = call(
        "POST",
        &url,
        Some(&json!({"messages": user("hello there"), "stream": true, "enable_thinking": true})),
    );
    assert_eq!(streamed.status, 200, "{}", streamed.body);
    assert!(streamed.content_type.starts_with("text/event-stream"));
    let frames: Vec<&str> = streamed
        .body
        .lines()
        .filter(|l| l.starts_with("data: "))
        .collect();
    assert!(frames.len() >= 8, "{}", streamed.body);
    assert!(streamed
        .body
        .contains("\"reasoning_content\":\"Considering the request.\""));
    assert!(streamed.body.contains("\" token0\""));
    assert!(streamed.body.contains("\"usage\""));
    assert_eq!(*frames.last().unwrap(), "data: [DONE]");
    // The served name and the usage request were filled in; the rest passed through.
    let seen = model.completions();
    assert_eq!(seen[0].payload["model"], "test/model");
    assert_eq!(seen[0].payload["stream_options"]["include_usage"], true);
    assert_eq!(seen[0].payload["enable_thinking"], true);
    // The turn is counted like any other request through the proxy.
    let deadline = Instant::now() + Duration::from_secs(5);
    while proxy.metrics().snapshot()["requests"] != 1 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let snapshot = proxy.metrics().snapshot();
    assert_eq!(snapshot["by_route"]["chat_completions|200|stream"], 1);
    assert_eq!(snapshot["completion_tokens"], 6);

    let complete = call(
        "POST",
        &url,
        Some(&json!({"messages": user("again"), "stream": false, "model": "test/model"})),
    );
    assert_eq!(complete.status, 200);
    assert!(complete.content_type.starts_with("application/json"));
    assert_eq!(complete.json()["usage"]["completion_tokens"], 6);
    assert_eq!(
        call("POST", &url, Some(&json!({"messages": []}))).status,
        400
    );

    // Hosted status reads the proxy's counters and the supervisor's facts.
    let metrics = proxy.metrics().clone();
    wait_for("both request records", || {
        metrics.snapshot()["requests"] == 2
    });
    let status = call("GET", &format!("{}/api/status", proxy.address()), None).json();
    assert_eq!(status["mode"], "hosted");
    assert_eq!(status["launch"]["kind"], "hosted");
    assert_eq!(status["launch"]["repo"], REPO);
    assert_eq!(status["proxy"]["requests"], 2);
    assert_eq!(status["capabilities"]["serve"], false);
    let requests = call("GET", &format!("{}/api/requests", proxy.address()), None).json();
    assert_eq!(requests["source"], "live");
    assert_eq!(requests["data"].as_array().unwrap().len(), 2);
}

#[test]
fn chat_reaches_a_launch_recorded_by_another_process() {
    let state = fixture("dashboard-external");
    let model = FakeOpenAIServer::start(FakeModelConfig::default()).unwrap();
    write_current(
        &state,
        &json!({"pid": std::process::id(), "endpoint": model.url, "server_endpoint": model.url,
                "repo": REPO, "run_id": "20261112T000000-1", "deployment": "test/model"}),
    )
    .unwrap();
    let served = standalone(&state, None);
    let streamed = call(
        "POST",
        &format!("{}/api/chat", served.base),
        Some(&json!({"messages": user("hi"), "stream": true})),
    );
    assert_eq!(streamed.status, 200);
    assert!(streamed.body.contains("data: [DONE]"));
    assert_eq!(model.completions()[0].payload["model"], "test/model");
}

/// One open `/api/events` stream, read event by event.
struct EventReader {
    reader: BufReader<TcpStream>,
}

impl EventReader {
    fn open(port: u16) -> EventReader {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream
            .write_all(
                b"GET /api/events HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: text/event-stream\r\n\r\n",
            )
            .unwrap();
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert!(line.contains(" 200 "), "{line}");
        let mut content_type = None;
        loop {
            line.clear();
            reader.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                break;
            }
            if let Some(value) = line.strip_prefix("Content-Type: ") {
                content_type = Some(value.trim().to_string());
            }
        }
        assert_eq!(content_type.as_deref(), Some("text/event-stream"));
        EventReader { reader }
    }

    fn next(&mut self) -> Value {
        let mut line = String::new();
        loop {
            line.clear();
            assert!(
                self.reader.read_line(&mut line).unwrap() > 0,
                "the stream ended"
            );
            if let Some(data) = line.strip_prefix("data: ") {
                return serde_json::from_str(data.trim()).expect("one JSON document per event");
            }
        }
    }
}

fn read_events(port: u16, count: usize) -> Vec<Value> {
    let mut reader = EventReader::open(port);
    (0..count).map(|_| reader.next()).collect()
}

#[test]
fn events_stream_snapshots_with_samples_jobs_and_log_lines() {
    let state = fixture("dashboard-events");
    let served = standalone(&state, None);
    let events = read_events(served.port, 1);
    let first = &events[0];
    assert_eq!(first["seq"], 0);
    assert_eq!(first["mode"], "launcher");
    assert_eq!(first["launch"]["state"], "none");
    assert_eq!(first["metrics"]["requests"], 0);
    assert_eq!(first["launcher"]["state"], "running");
    assert_eq!(first["in_flight"], 0);
    assert!(first["last_request"].is_null());
    assert!(first["job"].is_null());
    assert!(first["samples"].is_array());
    assert_eq!(first["log"]["run"], "20261112T000000-1");
    assert_eq!(first["log"]["serve"], json!([]));

    // A hosted dashboard streams the proxy's counters and appended log lines.
    let model = FakeOpenAIServer::start(FakeModelConfig::default()).unwrap();
    let proxy = hosted_proxy(&state, &model);
    let client = Client::new(proxy.address(), None, Duration::from_secs(10)).unwrap();
    let messages = user("hello").as_array().unwrap().clone();
    client.chat(&messages, None, true, &[], &Map::new(), &mut |_| {});
    let metrics = proxy.metrics().clone();
    wait_for("the request record", || metrics.snapshot()["requests"] == 1);
    let mut reader = EventReader::open(proxy.port);
    let mut events = vec![reader.next()];
    let run = RunStore::new(&state).open("20261112T000000-1");
    let mut log = std::fs::OpenOptions::new()
        .append(true)
        .open(run.serve_log())
        .unwrap();
    writeln!(log, "a fresh line").unwrap();
    events.push(reader.next());
    events.push(reader.next());
    assert_eq!(events[0]["launch"]["kind"], "hosted");
    assert_eq!(events[0]["metrics"]["requests"], 1);
    assert_eq!(events[0]["last_request"]["route"], "chat_completions");
    // The sampler ticks once a second, so under load its first sample can
    // trail the first event by a tick; it arrives in one of the three.
    let sampled = events
        .iter()
        .find(|e| !e["samples"].as_array().unwrap().is_empty())
        .expect("no event carried a counter sample");
    assert_eq!(sampled["samples"][0]["key"], proxy.address());
    assert_eq!(events[2]["seq"], 2);
    let appended: Vec<&Value> = events
        .iter()
        .flat_map(|e| e["log"]["serve"].as_array().unwrap())
        .collect();
    assert_eq!(appended, [&json!("a fresh line")]);
}

const FETCH_SCRIPT: &str = r#"#!/bin/sh
out=""
for arg in "$@"; do
  case "$arg" in --output=*) out="${arg#--output=}";; esac
done
sleep 0.5
mkdir -p "$out"
printf '{"model": {"id": "a/b"}, "preset": "chat", "presets": {"chat": {"execution": {"context_capacity": 4096}}}}' > "$out/model.json"
echo "loom-fetch  Fetch     1/2 files" >&2
echo "fetching 1/2 files"
cat <<'JSON'
{"model": "a/b", "preset": "chat",
 "files": [{"path": "model.json", "transport": {"kind": "hf", "repository": "loom-fun/a-loom", "revision": "abc123"}}]}
JSON
"#;

#[test]
fn install_runs_in_the_background_and_records_the_profile() {
    use std::os::unix::fs::PermissionsExt;
    let state = fixture("dashboard-install");
    let script = state.join("fake-fetch");
    std::fs::write(&script, FETCH_SCRIPT).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let served = standalone(&state, Some(script.to_string_lossy().into_owned()));
    let url = format!("{}/api/install", served.base);
    let started = call("POST", &url, Some(&json!({"repo": "loom-fun/a-loom"})));
    assert_eq!(started.status, 202, "{}", started.body);
    let job = started.json()["job"].as_str().unwrap().to_string();
    let busy = call("POST", &url, Some(&json!({"repo": "loom-fun/other-loom"})));
    assert_eq!(busy.status, 409);
    assert!(busy.body.contains("still running"));
    let jobs = served.context.jobs().clone();
    wait_for("the install", || {
        !matches!(
            jobs.state_of(&job),
            Some(JobState::Running | JobState::Queued)
        )
    });
    let described = jobs.describe(50);
    assert_eq!(described["state"], "done", "{described}");
    assert!(described["lines"]
        .as_array()
        .unwrap()
        .iter()
        .any(|l| l == "fetching 1/2 files"));
    assert_eq!(described["result"]["install"]["revision"], "abc123");
    let installed = InstallRegistry::new(&state).get("loom-fun/a-loom").unwrap();
    assert!(installed.card_path().is_file());
    // The finished job stays visible on the event stream.
    let events = read_events(served.port, 1);
    assert_eq!(events[0]["job"]["id"], job);
    assert_eq!(events[0]["job"]["kind"], "install");
}

#[test]
fn bench_runs_a_suite_and_stores_it_where_the_cli_lists_it() {
    let state = fixture("dashboard-bench");
    let model = FakeOpenAIServer::start(FakeModelConfig {
        completion_tokens: 160,
        first_token_delay: Duration::from_millis(1),
        ..Default::default()
    })
    .unwrap();
    let proxy = hosted_proxy(&state, &model);
    let url = format!("{}/api/bench", proxy.address());
    let wrong = call(
        "POST",
        &url,
        Some(&json!({"suite": "quick", "repo": "other/x"})),
    );
    assert_eq!(wrong.status, 409);
    let started = call("POST", &url, Some(&json!({"suite": "quick", "repo": REPO})));
    assert_eq!(started.status, 202, "{}", started.body);
    let job = started.json()["job"].as_str().unwrap().to_string();
    let deadline = Instant::now() + Duration::from_secs(60);
    let finished = loop {
        let status = call("GET", &format!("{}/api/status", proxy.address()), None).json();
        if status["job"]["id"] == job.as_str() && status["job"]["state"] != "running" {
            break status["job"].clone();
        }
        assert!(Instant::now() < deadline, "the benchmark did not finish");
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(finished["state"], "done", "{finished}");
    assert_eq!(finished["result"]["ok"], true);
    let id = finished["result"]["id"].as_str().unwrap().to_string();
    let store = BenchmarkStore::new(&state);
    let listed: Vec<String> = store
        .list(None)
        .iter()
        .map(|r| str_value(get(r, "id")).unwrap().to_string())
        .collect();
    assert!(listed.contains(&id), "{listed:?}");
    let record = store.load(&id).unwrap();
    assert_eq!(record["suite"], "quick");
    assert_eq!(record["profile"], REPO);
    assert_eq!(record["comparison"][0]["metric"], "decode");
    assert_eq!(record["comparison"][0]["published"], 10.0);
    assert!(finished["lines"]
        .as_array()
        .unwrap()
        .iter()
        .any(|l| l.as_str().unwrap().starts_with("decode:")));
    let runs = call("GET", &format!("{}/api/benchmarks", proxy.address()), None).json();
    assert_eq!(runs["runs"][0]["id"], id.as_str());
    // The benchmark went through the proxy, so the traffic card saw it.
    assert!(proxy.metrics().snapshot()["requests"].as_u64().unwrap() >= 4);
}

#[test]
fn helpers_parse_what_the_routes_receive() {
    assert_eq!(parse_iso_utc("1970-01-01T00:00:00Z"), Some(0.0));
    assert_eq!(parse_iso_utc("2026-11-12T03:43:47Z"), Some(1794455027.0));
    assert_eq!(
        parse_iso_utc("2026-11-12T03:43:47.500Z"),
        Some(1794455027.5)
    );
    assert_eq!(parse_iso_utc("yesterday"), None);
    let upstream = upstream_of("http://127.0.0.1:40361").unwrap();
    assert_eq!(
        (upstream.host.as_str(), upstream.port),
        ("127.0.0.1", 40361)
    );
    let upstream = upstream_of("http://[::1]:8080/").unwrap();
    assert_eq!((upstream.host.as_str(), upstream.port), ("::1", 8080));
    assert!(upstream_of("http://nowhere").is_none());
    assert!(valid_repo("loom-fun/unsloth-Qwen3-4B-Instruct-2507-loom"));
    assert!(!valid_repo("loom-fun"));
    assert!(!valid_repo("../x"));
    assert!(!valid_repo("a/b/c"));
    assert!(!valid_repo("a/.hidden"));
    assert!(local_host("127.0.0.1:8090"));
    assert!(local_host("[::1]:8090"));
    assert!(local_host("localhost:1"));
    assert!(local_host("10.1.2.3"));
    assert!(!local_host("evil.example:8090"));
    assert_eq!(
        page::escape("<a href=\"x\">&'"),
        "&lt;a href=&quot;x&quot;&gt;&amp;&#39;"
    );
}

/// A raw request with the headers a browser on the LAN sends.
fn raw_with(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: &str) -> String {
    let mut request = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
    for (name, value) in headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str(&format!("Content-Length: {}\r\n\r\n{body}", body.len()));
    raw(port, &request)
}

#[test]
fn a_lan_address_passes_the_guard_and_a_foreign_origin_does_not() {
    let state = fixture("dashboard-lan");
    let served = standalone(&state, None);
    let lan = "192.168.0.200:8099";
    let status = raw_with(served.port, "GET", "/api/status", &[("Host", lan)], "");
    assert!(status.contains(" 200 "), "{status}");
    // Same origin over the LAN address reaches the route (409: nothing runs).
    let json = ("Content-Type", "application/json");
    let same = raw_with(
        served.port,
        "POST",
        "/api/stop",
        &[("Host", lan), ("Origin", "http://192.168.0.200:8099"), json],
        "{}",
    );
    assert!(same.contains(" 409 "), "{same}");
    let foreign = raw_with(
        served.port,
        "POST",
        "/api/stop",
        &[("Host", lan), ("Origin", "http://evil.example"), json],
        "{}",
    );
    assert!(foreign.contains(" 403 "), "{foreign}");
    // Another machine on the LAN is another origin too.
    let neighbour = raw_with(
        served.port,
        "POST",
        "/api/stop",
        &[("Host", lan), ("Origin", "http://192.168.0.201:8099"), json],
        "{}",
    );
    assert!(neighbour.contains(" 403 "), "{neighbour}");
    let named = raw_with(
        served.port,
        "GET",
        "/api/status",
        &[("Host", "nas.lan:8099")],
        "",
    );
    assert!(named.contains(" 403 "), "{named}");
    assert!(same_origin(
        "http://192.168.0.200:8099",
        "192.168.0.200:8099"
    ));
    assert!(same_origin("http://192.168.0.200", "192.168.0.200:80"));
    assert!(same_origin("http://LOCALHOST:8090", "localhost:8090"));
    assert!(same_origin("http://[::1]:8090", "[::1]:8090"));
    assert!(!same_origin(
        "http://192.168.0.200:8098",
        "192.168.0.200:8099"
    ));
    assert!(!same_origin("http://evil.example", "192.168.0.200:8099"));
}

/// A stand-in for the internet: answers a few paths and counts requests.
struct Stub {
    address: std::net::SocketAddr,
    seen: Arc<Mutex<Vec<String>>>,
    server: Arc<tiny_http::Server>,
}

impl Drop for Stub {
    fn drop(&mut self) {
        self.server.unblock();
    }
}

fn stub() -> Stub {
    let server = Arc::new(tiny_http::Server::http("127.0.0.1:0").unwrap());
    let address = server.server_addr().to_ip().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (accept, log) = (server.clone(), seen.clone());
    std::thread::spawn(move || {
        for request in accept.incoming_requests() {
            let path = request.url().to_string();
            let agent = request
                .headers()
                .iter()
                .find(|h| h.field.equiv("User-Agent"))
                .map(|h| h.value.to_string())
                .unwrap_or_default();
            log.lock().unwrap().push(format!("{path} {agent}"));
            let reply = match path.as_str() {
                "/page" => raw_response(
                    200,
                    "text/html; charset=utf-8",
                    b"<html><head><title>Stub page</title><script>secret()</script></head>\
                      <body><h1>Facts</h1><p>The answer is&nbsp;42.</p></body></html>"
                        .to_vec(),
                ),
                "/big" => raw_response(200, "text/plain", vec![b'x'; 2 * 1024 * 1024]),
                "/moved" => {
                    let mut reply = raw_response(302, "text/html", Vec::new());
                    reply.add_header(tiny_http::Header::from_bytes("Location", "/page").unwrap());
                    reply
                }
                _ => raw_response(404, "text/plain", b"absent".to_vec()),
            };
            let _ = request.respond(reply);
        }
    });
    Stub {
        address,
        seen,
        server,
    }
}

fn fetch_answer(served: &Served, url: &str) -> Answer {
    call(
        "POST",
        &format!("{}/api/fetch", served.base),
        Some(&json!({ "url": url })),
    )
}

#[test]
fn web_fetch_is_refused_unless_the_launcher_allows_it() {
    let state = fixture("dashboard-fetch-off");
    let served = standalone(&state, None);
    let status = call("GET", &format!("{}/api/status", served.base), None).json();
    assert_eq!(status["capabilities"]["web_access"], false);
    let refused = fetch_answer(&served, "https://example.com/");
    assert_eq!(refused.status, 403);
    let message = refused.json()["error"]["message"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(message.contains("--allow-web"), "{message}");
}

#[test]
fn web_fetch_reads_public_pages_and_refuses_this_machine() {
    let state = fixture("dashboard-fetch");
    let served = standalone_with(&state, None, true);
    let status = call("GET", &format!("{}/api/status", served.base), None).json();
    assert_eq!(status["capabilities"]["web_access"], true);
    let stub = stub();
    served
        .context
        .stub_public_host("public.example", stub.address);

    // This machine, its LAN and other schemes are refused before any request.
    for (url, needle) in [
        ("file:///etc/passwd", "only http and https"),
        ("http://localhost:8080/api/status", "this machine"),
        (
            &format!("http://127.0.0.1:{}/api/stop", served.port),
            "loopback",
        ),
        (
            &format!("http://127.0.0.1:{}/page", stub.address.port()),
            "loopback",
        ),
        ("http://192.168.0.200:8099/", "private"),
        ("http://10.1.2.3/", "private"),
        ("http://[::1]/", "loopback"),
        ("http://169.254.169.254/latest/meta-data/", "link-local"),
    ] {
        let answer = fetch_answer(&served, url);
        assert_eq!(answer.status, 403, "{url}: {}", answer.body);
        let message = answer.json()["error"]["message"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(message.contains(needle), "{url}: {message}");
    }
    assert_eq!(fetch_answer(&served, "not a url").status, 400);
    let missing = call(
        "POST",
        &format!("{}/api/fetch", served.base),
        Some(&json!({})),
    );
    assert_eq!(missing.status, 400);
    assert!(
        stub.seen.lock().unwrap().is_empty(),
        "a refused URL was fetched"
    );

    // A public page comes back as text.
    let page = fetch_answer(&served, "http://public.example/page");
    assert_eq!(page.status, 200, "{}", page.body);
    let document = page.json();
    assert_eq!(document["url"], "http://public.example/page");
    assert_eq!(document["status"], 200);
    assert_eq!(document["content_type"], "text/html; charset=utf-8");
    assert_eq!(document["title"], "Stub page");
    assert_eq!(document["text"], "Facts\nThe answer is 42.");
    assert_eq!(document["truncated"], false);
    assert!(document["bytes"].as_u64().unwrap() > 100);
    let seen = stub.seen.lock().unwrap().clone();
    assert!(seen[0].starts_with("/page Mozilla/5.0"), "{seen:?}");

    // The body is read up to 1 MiB and the text handed back is at most 8 KiB.
    let big = fetch_answer(&served, "http://public.example/big").json();
    assert_eq!(big["bytes"], 1024 * 1024);
    assert_eq!(big["truncated"], true);
    assert_eq!(big["text"].as_str().unwrap().len(), 8 * 1024);

    // Redirects are reported, not followed.
    let moved = fetch_answer(&served, "http://public.example/moved").json();
    assert_eq!(moved["status"], 302);
    assert_eq!(moved["location"], "http://public.example/page");
    assert!(moved["text"].as_str().unwrap().contains("fetch that URL"));
    let pages = stub
        .seen
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.starts_with("/page"))
        .count();
    assert_eq!(pages, 1, "the redirect was followed");
    let absent = fetch_answer(&served, "http://public.example/absent").json();
    assert_eq!(absent["status"], 404);
    assert_eq!(absent["text"], "absent");

    // A host that does not answer is a readable 502.
    let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let dead = closed.local_addr().unwrap();
    drop(closed);
    served.context.stub_public_host("down.example", dead);
    let down = fetch_answer(&served, "http://down.example/");
    assert_eq!(down.status, 502);
    assert!(down.json()["error"]["message"]
        .as_str()
        .unwrap()
        .contains("could not fetch"));

    // The route sits behind the same guard as the other /api/* routes.
    let foreign = raw_with(
        served.port,
        "POST",
        "/api/fetch",
        &[
            ("Host", "127.0.0.1"),
            ("Origin", "http://evil.example"),
            ("Content-Type", "application/json"),
        ],
        "{\"url\": \"http://public.example/page\"}",
    );
    assert!(foreign.contains(" 403 "), "{foreign}");
    let form = raw_with(
        served.port,
        "POST",
        "/api/fetch",
        &[("Host", "127.0.0.1"), ("Content-Type", "text/plain")],
        "{\"url\": \"http://public.example/page\"}",
    );
    assert!(form.contains(" 415 "), "{form}");
}

#[test]
fn chat_passes_tools_through_and_relays_tool_calls() {
    let state = fixture("dashboard-chat-tools");
    let model = FakeOpenAIServer::start(FakeModelConfig::default()).unwrap();
    let proxy = hosted_proxy(&state, &model);
    let url = format!("{}/api/chat", proxy.address());
    let tools = json!([{"type": "function", "function": {
        "name": "run_javascript", "description": "Run JavaScript",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]);
    let streamed = call(
        "POST",
        &url,
        Some(&json!({"messages": user("/js 6 * 7"), "stream": true, "tools": tools})),
    );
    assert_eq!(streamed.status, 200, "{}", streamed.body);
    assert!(
        streamed.body.contains("\"tool_calls\""),
        "{}",
        streamed.body
    );
    assert!(streamed.body.contains("\"name\":\"run_javascript\""));
    assert!(streamed.body.contains("\"finish_reason\":\"tool_calls\""));
    assert_eq!(model.completions()[0].payload["tools"], tools);
    // The answer to the tool round goes back with the call and its result.
    let follow_up = json!([
        {"role": "user", "content": "/js 6 * 7"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "call_0", "type": "function",
            "function": {"name": "run_javascript", "arguments": "{\"code\":\"6 * 7\"}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"ok\":true,\"result\":42}"},
    ]);
    let answered = call(
        "POST",
        &url,
        Some(&json!({"messages": follow_up, "stream": false, "tools": tools})),
    );
    assert_eq!(answered.status, 200);
    let content = answered.json()["choices"][0]["message"]["content"].clone();
    assert!(content.as_str().unwrap().contains("42"), "{content}");
    assert_eq!(
        model.completions()[1].payload["messages"][2]["tool_call_id"],
        "call_0"
    );
    // The listener reports where it is bound.
    let status = call("GET", &format!("{}/api/status", proxy.address()), None).json();
    assert_eq!(status["listener"]["bind_host"], "127.0.0.1");
    assert_eq!(status["listener"]["urls"], json!([]));
    assert_eq!(status["capabilities"]["web_access"], false);
}

/// A `loom-fetch` that prints its console progress the way the real one does.
const FETCH_PROGRESS_SCRIPT: &str = r#"#!/bin/sh
out=""
for arg in "$@"; do
  case "$arg" in --output=*) out="${arg#--output=}";; esac
done
if [ "$FAKE_FETCH_MODE" = "fail" ]; then
  echo "loom-fetch  Metadata  reading model metadata" >&2
  echo "loom-fetch: 401 Client Error: access to loom-fun/a-loom is restricted" >&2
  exit 4
fi
echo "loom-fetch  Metadata  reading model metadata" >&2
echo "loom-fetch  Fetch     0/2 files" >&2
sleep 0.3
echo "loom-fetch  Download  [######......]  50%  1.0 KiB / 2.0 KiB  ETA 1s  parameter/a.gguf  0/2 files" >&2
sleep 0.3
mkdir -p "$out"
printf '{"model": {"id": "a/b"}, "preset": "chat", "presets": {"chat": {"execution": {"context_capacity": 4096}}}}' > "$out/model.json"
echo "loom-fetch  Install   1/2 files" >&2
echo "loom-fetch  Complete  2/2 files" >&2
cat <<'JSON'
{"model": "a/b", "preset": "chat",
 "files": [{"path": "model.json", "transport": {"kind": "hf", "repository": "loom-fun/a-loom", "revision": "abc123"}}]}
JSON
"#;

fn fetch_script(state: &Path, mode: &str) -> String {
    use std::os::unix::fs::PermissionsExt;
    let script = state.join(format!("fake-fetch-{mode}"));
    std::fs::write(
        &script,
        FETCH_PROGRESS_SCRIPT.replace("$FAKE_FETCH_MODE", mode),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script.to_string_lossy().into_owned()
}

fn wait_for_job(served: &Served, id: &str) -> Value {
    let jobs = served.context.jobs().clone();
    wait_for(id, || {
        !matches!(
            jobs.state_of(id),
            Some(JobState::Running | JobState::Queued)
        )
    });
    call("GET", &format!("{}/api/jobs/{id}", served.base), None).json()
}

#[test]
fn an_install_is_a_job_row_with_steps_progress_and_a_terminal_state() {
    let state = fixture("dashboard-job-install");
    let served = standalone(&state, Some(fetch_script(&state, "ok")));
    let url = format!("{}/api/install", served.base);
    let started = call("POST", &url, Some(&json!({"repo": "loom-fun/a-loom"})));
    assert_eq!(started.status, 202, "{}", started.body);
    let id = started.json()["job"].as_str().unwrap().to_string();
    // A double click is refused, and the refusal says why.
    let again = call("POST", &url, Some(&json!({"repo": "loom-fun/a-loom"})));
    assert_eq!(again.status, 409);
    assert!(again.json()["error"]["message"]
        .as_str()
        .unwrap()
        .contains("install of loom-fun/a-loom is still running"));
    // Mid-way, the row already carries its step and a measured progress.
    let jobs = served.context.jobs().clone();
    wait_for("the download line", || {
        jobs.get(&id, 50).is_some_and(|job| {
            job["progress"]["label"]
                .as_str()
                .is_some_and(|l| l.starts_with("Download parameter/a.gguf"))
        })
    });
    let running = jobs.get(&id, 50).unwrap();
    assert_eq!(running["state"], "running");
    assert_eq!(running["step"], "Fetch and verify files");
    assert_eq!(running["progress"]["unit"], "files");
    assert_eq!(running["progress"]["total"], 2.0);
    assert_eq!(running["progress"]["bytes"]["done"], 1024);
    let finished = wait_for_job(&served, &id);
    assert_eq!(finished["state"], "done", "{finished}");
    assert_eq!(finished["source"], "hub");
    let steps: Vec<&str> = finished["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert_eq!(steps, ["card", "files", "place", "documents", "record"]);
    assert!(finished["steps"]
        .as_array()
        .unwrap()
        .iter()
        .all(|s| s["state"] == "done" && s["elapsed_seconds"].is_number()));
    assert_eq!(
        finished["progress"]["files"],
        json!({"done": 2, "total": 2})
    );
    assert_eq!(finished["progress"]["fraction"], 1.0);
    assert_eq!(finished["retry"]["route"], "/api/install");
    assert_eq!(finished["retry"]["body"]["repo"], "loom-fun/a-loom");
    // The hub at :9 refuses: the install stands, and says it lacks a policy.
    assert!(finished["hint"]
        .as_str()
        .unwrap()
        .contains("without llm_config.json"));
    assert!(finished["lines"].as_array().unwrap().iter().any(|l| l
        .as_str()
        .unwrap()
        .starts_with("warning: could not read llm_config.json")));
    // The row survives a reload: /api/status and the event stream carry it.
    let status = call("GET", &format!("{}/api/status", served.base), None).json();
    assert_eq!(status["jobs"][0]["id"], id.as_str());
    assert_eq!(status["job"]["id"], id.as_str());
    let events = read_events(served.port, 1);
    assert_eq!(events[0]["jobs"][0]["id"], id.as_str());
    assert_eq!(events[0]["jobs"][0]["state"], "done");
    // Dismissed, it leaves the list; unknown ids are 404.
    let dismiss = format!("{}/api/jobs/dismiss", served.base);
    assert_eq!(call("POST", &dismiss, Some(&json!({"id": id}))).status, 200);
    let listed = call("GET", &format!("{}/api/jobs", served.base), None).json();
    assert!(listed["jobs"].as_array().unwrap().is_empty());
    assert_eq!(
        call("POST", &dismiss, Some(&json!({"id": "absent-1"}))).status,
        404
    );
    assert_eq!(
        call("GET", &format!("{}/api/jobs/absent-1", served.base), None).status,
        404
    );
}

#[test]
fn a_failed_install_leaves_a_row_saying_what_failed_and_how_to_retry() {
    let state = fixture("dashboard-job-failed");
    let served = standalone(&state, Some(fetch_script(&state, "fail")));
    let started = call(
        "POST",
        &format!("{}/api/install", served.base),
        Some(&json!({"repo": "loom-fun/a-loom"})),
    );
    let id = started.json()["job"].as_str().unwrap().to_string();
    let failed = wait_for_job(&served, &id);
    assert_eq!(failed["state"], "failed");
    let error = failed["error"].as_str().unwrap();
    assert!(error.contains("exit code 4"), "{error}");
    assert!(
        error.contains("access to loom-fun/a-loom is restricted"),
        "{error}"
    );
    assert!(failed["hint"].as_str().unwrap().contains("Retry"));
    assert_eq!(failed["steps"][0]["state"], "failed");
    assert_eq!(failed["retry"]["body"]["repo"], "loom-fun/a-loom");
    // Running jobs cannot be dismissed; failed ones can.
    let dismissed = call(
        "POST",
        &format!("{}/api/jobs/dismiss", served.base),
        Some(&json!({"id": id})),
    );
    assert_eq!(dismissed.status, 200);
}

#[test]
fn a_listed_local_profile_is_registered_from_its_path_with_its_policy() {
    const ORIGIN: &str = "loom-fun/pkg-loom";
    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
    let state = scratch("dashboard-register");
    let roots = state.join("profiles");
    let package = roots.join("pkg");
    let documents = card_documents();
    write_json(
        &package.join("model.json"),
        &Value::Object(documents.model_card.clone()),
    )
    .unwrap();
    write_json(
        &package.join("model.json.lock"),
        &json!({"origins": {"document": {"requested": {"kind": "url",
            "url": format!("https://huggingface.co/{ORIGIN}/resolve/{REVISION}/")}}}}),
    )
    .unwrap();
    // The launcher read this profile's hub documents earlier, at that revision.
    let mut cached = documents.clone();
    cached.revision = Some(REVISION.into());
    ProfileCache::new(&state)
        .put(ORIGIN, &cached.to_cache(ORIGIN))
        .unwrap();
    let served = serve(Context::standalone(&Service {
        state_root: state.clone(),
        hub_endpoint: Some("http://127.0.0.1:9".into()),
        profile_roots: vec![roots.clone()],
        ..Default::default()
    }));
    let url = format!("{}/api/install", served.base);
    let elsewhere = call("POST", &url, Some(&json!({"path": "/etc"})));
    assert_eq!(elsewhere.status, 400);
    assert!(elsewhere
        .body
        .contains("not a local profile this dashboard lists"));
    let started = call(
        "POST",
        &url,
        Some(&json!({"path": package.to_string_lossy()})),
    );
    assert_eq!(started.status, 202, "{}", started.body);
    let id = started.json()["job"].as_str().unwrap().to_string();
    let finished = wait_for_job(&served, &id);
    assert_eq!(finished["state"], "done", "{finished}");
    assert_eq!(finished["source"], "path");
    assert_eq!(finished["steps"][1]["name"], "documents");
    assert_eq!(finished["steps"][1]["state"], "done");
    assert_eq!(
        finished["result"]["profile_documents"][0]["status"],
        "launcher-cache"
    );
    assert!(package.join("llm_config.json").is_file());
    assert_eq!(
        finished["retry"]["body"]["path"],
        package.to_string_lossy().as_ref()
    );
    let installed = InstallRegistry::new(&state).get("local/pkg").unwrap();
    assert_eq!(installed.source.as_deref(), Some("local"));
    let installs = call("GET", &format!("{}/api/installs", served.base), None).json();
    let row = installs["installs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["repo"] == "local/pkg")
        .unwrap()
        .clone();
    assert_eq!(row["policy"]["available"], true);
    assert_eq!(row["policy"]["default"], true);
}

#[test]
fn web_access_says_why_it_is_off_and_how_to_turn_it_on() {
    let state = fixture("dashboard-web-reason");
    let served = standalone(&state, None);
    served.context.set_listener("127.0.0.1", 8090);
    let status = call("GET", &format!("{}/api/status", served.base), None).json();
    let capabilities = &status["capabilities"];
    assert_eq!(capabilities["web_access"], false);
    assert_eq!(
        capabilities["web_access_reason"],
        "web access is off: tick \"Allow web access\" when you launch, or restart the \
         launcher with --allow-web"
    );
    assert_eq!(
        capabilities["web_access_command"],
        "loom-launch launcher start --proxy-port 8090 --allow-web"
    );
    assert_eq!(capabilities["launch_allow_web"], true);
    let refused = fetch_answer(&served, "https://example.com/");
    assert_eq!(refused.status, 403);
    assert_eq!(
        refused.json()["error"]["message"],
        capabilities["web_access_reason"]
    );

    // A launch's own dashboard names the command that restarts that launch.
    let model = FakeOpenAIServer::start(FakeModelConfig::default()).unwrap();
    let proxy = hosted_proxy(&state, &model);
    let status = call("GET", &format!("{}/api/status", proxy.address()), None).json();
    let capabilities = &status["capabilities"];
    assert_eq!(
        capabilities["web_access_reason"],
        "web access is off for this launch: restart it with --allow-web"
    );
    assert_eq!(
        capabilities["web_access_command"],
        format!(
            "loom-launch serve {REPO} --proxy-port {} --allow-web",
            proxy.port
        )
    );
    assert_eq!(capabilities["launch_allow_web"], false);

    // The recorded command line is preferred, its key left as a reference.
    let recorded = obj(
        &json!({"launch": {"repo": REPO, "pid": 0, "launcher_command":
        ["/opt/bin/loom-launch", "serve", REPO, "--api-key", "\"$LOOM_LAUNCH_API_KEY\""]}}),
    )
    .clone();
    let context = Context::hosted(Hosted {
        state_root: state.clone(),
        hub_endpoint: None,
        forwarder: Arc::new(Forwarder::new(
            Upstream {
                host: "127.0.0.1".into(),
                port: 9,
            },
            Arc::new(Metrics::default()),
            None,
            Duration::from_secs(1),
        )),
        status_provider: Some(Arc::new(move || recorded.clone())),
        endpoint: "http://127.0.0.1:8099".into(),
        auth_token: None,
        allow_web: false,
        profile_roots: Vec::new(),
    });
    let live = context.live();
    assert_eq!(
        context.web_access(&live).command.as_deref(),
        Some(
            "/opt/bin/loom-launch serve loom-fun/test-loom --api-key \"$LOOM_LAUNCH_API_KEY\" \
             --allow-web"
        )
    );
    context.close();

    // With --allow-web there is nothing to explain.
    let open = standalone_with(&state, None, true);
    let status = call("GET", &format!("{}/api/status", open.base), None).json();
    assert_eq!(status["capabilities"]["web_access"], true);
    assert!(status["capabilities"]["web_access_reason"].is_null());
}

#[test]
fn the_thinking_switch_and_the_drafter_explain_themselves() {
    let with_policy = |chat: Value, drafter: Value| {
        let mut documents = card_documents();
        let config = documents.llm_config.entry("chat").or_insert(Value::Null);
        *config = chat;
        documents.llm_config.insert("drafter".into(), drafter);
        Profile::from_documents(REPO, &documents, None)
    };
    let toggle = with_policy(
        json!({"thinking": "toggle", "thinking_default": true, "think_open": "<think>\n"}),
        Value::Null,
    );
    let summary = thinking_summary(&toggle, false);
    assert_eq!(summary["available"], true);
    assert_eq!(summary["default"], true);
    assert!(summary["reason"].is_null());
    let always = with_policy(
        json!({"thinking": "always", "think_open": "<think>"}),
        Value::Null,
    );
    let summary = thinking_summary(&always, false);
    assert_eq!(summary["available"], false);
    assert_eq!(summary["locked"], true);
    assert!(summary["reason"]
        .as_str()
        .unwrap()
        .contains("always thinks"));
    let never = with_policy(json!({"thinking": "never"}), Value::Null);
    assert_eq!(thinking_summary(&never, false)["locked"], false);
    // No policy at all: the remedy depends on where the package came from.
    let mut bare = card_documents();
    bare.llm_config.remove("chat");
    let bare = Profile::from_documents("ilintar/Qwen3.8-27B-hybrid-loom", &bare, None);
    let hub = thinking_summary(&bare, false);
    assert_eq!(hub["mode"], "unknown");
    let reason = hub["reason"].as_str().unwrap();
    assert!(
        reason.contains("ships no chat policy (llm_config.json)"),
        "{reason}"
    );
    assert!(
        reason.contains("`loom-launch install ilintar/Qwen3.8-27B-hybrid-loom`"),
        "{reason}"
    );
    let local = thinking_summary(&bare, true)["reason"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        local.contains("copy the package's llm_config.json"),
        "{local}"
    );

    // The Qwen3.8 hybrid: served at depth 0, its policy recommends dflash at 3.
    let hybrid = with_policy(
        json!({"thinking": "toggle", "think_open": "<think>\n"}),
        json!({"kind": "dflash", "depth": 3, "tuned": [
            {"kind": "none", "depth": 0, "decode_tps": 12.6},
            {"kind": "mtp", "depth": 3, "decode_tps": 32.36},
            {"kind": "dflash", "depth": 3, "decode_tps": 35.44}]}),
    );
    let drafter = drafter_summary(&hybrid, Some(0), true);
    assert_eq!(drafter["served_tps"], 12.6);
    assert_eq!(drafter["recommended_tps"], 35.44);
    assert_eq!(
        drafter["note"],
        "Served with draft depth 0; the policy recommends a dflash drafter at depth 3 (12.6 \
         tok/s against 35.4 tok/s in the publisher's tuning). This scenario's own deployment \
         fixes the depth, and the launcher refuses draft-depth overrides for a scenario."
    );
    assert!(drafter_summary(&hybrid, Some(3), true)["note"].is_null());
    assert!(drafter_summary(&toggle, Some(0), false).is_null());
}
