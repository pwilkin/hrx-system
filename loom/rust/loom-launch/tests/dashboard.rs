//! `loom-launch dashboard --port` end to end, against the GPU-free `fake-serve`.
//!
//! The real binary serves the page standalone, starts a launch from
//! `POST /api/serve`, streams a chat through it, benchmarks it into the store
//! `bench --list` reads, stops it, and on SIGINT stops a launch it still owns.
//! Scratch data lives under Cargo's target directory, never in /tmp.

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

mod common;
use common::free_port;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_loom-launch")
}

fn workspace(name: &str) -> PathBuf {
    let directory =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A state directory whose registry knows one installed card.
fn state_with_card(root: &Path) -> PathBuf {
    let state = root.join("state");
    let card = root.join("card");
    std::fs::create_dir_all(&card).unwrap();
    std::fs::write(
        card.join("model.json"),
        r#"{
  "schema_version": 1,
  "model": {"id": "test/model", "name": "Test model"},
  "services": {"chat": {"kind": "text", "version": 1}},
  "presets": {"chat": {"service": "chat", "execution": {"rows": 2, "context_capacity": 8192, "max_tokens": 512}}}
}"#,
    )
    .unwrap();
    std::fs::write(
        card.join("llm_config.json"),
        r#"{"arch": "test", "gguf": "test-Q4_K_M.gguf", "context_length": 8192,
  "chat": {"user_prefix": "<u>", "assistant_prefix": "<a>", "suffix": "</s>", "thinking": "never"}}"#,
    )
    .unwrap();
    std::fs::create_dir_all(&state).unwrap();
    std::fs::write(
        state.join("installs.json"),
        serde_json::to_string(&json!({"installs": {"loom-fun/test-loom": {
            "repo": "loom-fun/test-loom", "path": card, "preset": "chat",
            "installed_at": "2026-11-12T00:00:00Z", "size_bytes": 1}}}))
        .unwrap(),
    )
    .unwrap();
    state
}

struct Dashboard {
    child: Child,
    base: String,
}

impl Dashboard {
    fn start(root: &Path, state: &Path) -> Dashboard {
        let shim = root.join("fake-loom-serve");
        std::fs::write(
            &shim,
            format!("#!/bin/sh\nexec '{}' fake-serve \"$@\"\n", binary()),
        )
        .unwrap();
        std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
        let port = free_port();
        let log = std::fs::File::create(root.join("dashboard.log")).unwrap();
        let child = Command::new(binary())
            .arg("--state")
            .arg(state)
            .args(["--hub", "http://127.0.0.1:9", "dashboard", "--port"])
            .arg(port.to_string())
            .arg("--loom-serve")
            .arg(&shim)
            .env_remove("LOOM_LAUNCH_API_KEY")
            .env_remove("LOOM_SERVE")
            .env("LOOM_FAKE_SERVE_TOKENS", "200")
            .env("LOOM_FAKE_SERVE_TOKEN_MS", "0")
            .env("LOOM_FAKE_SERVE_PREPARE_MS", "600")
            .env("LOOM_FAKE_SERVE_PANEL_MS", "300")
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap();
        let dashboard = Dashboard {
            child,
            base: format!("http://127.0.0.1:{port}"),
        };
        dashboard.wait_for("the listener", |d| d.get("/").0 == 200);
        dashboard
    }

    fn call(&self, method: &str, path: &str, body: Option<Value>) -> (u16, String) {
        let request =
            ureq::request(method, &format!("{}{path}", self.base)).timeout(Duration::from_secs(60));
        let outcome = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        match outcome {
            Ok(response) | Err(ureq::Error::Status(_, response)) => {
                (response.status(), response.into_string().unwrap())
            }
            Err(_) => (0, String::new()),
        }
    }

    fn get(&self, path: &str) -> (u16, String) {
        self.call("GET", path, None)
    }

    fn json(&self, path: &str) -> Value {
        serde_json::from_str(&self.get(path).1).unwrap_or(Value::Null)
    }

    fn post(&self, path: &str, body: Value) -> (u16, Value) {
        let (status, text) = self.call("POST", path, Some(body));
        (status, serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    fn wait_for(&self, what: &str, ready: impl Fn(&Dashboard) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !ready(self) {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn launch_state(&self) -> String {
        self.json("/api/status")["launch"]["state"]
            .as_str()
            .unwrap_or("")
            .to_string()
    }

    fn first_event(&self) -> Value {
        let address = self.base.trim_start_matches("http://");
        let mut stream = std::net::TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        write!(
            stream,
            "GET /api/events HTTP/1.1\r\nHost: {address}\r\nAccept: text/event-stream\r\n\r\n"
        )
        .unwrap();
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        loop {
            line.clear();
            assert!(reader.read_line(&mut line).unwrap() > 0);
            if let Some(data) = line.strip_prefix("data: ") {
                return serde_json::from_str(data.trim()).unwrap();
            }
        }
    }
}

impl Drop for Dashboard {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            // SAFETY: kill(2) has no memory-safety preconditions.
            unsafe {
                libc::kill(self.child.id() as i32, libc::SIGINT);
            }
            let deadline = Instant::now() + Duration::from_secs(40);
            while Instant::now() < deadline {
                if let Ok(Some(_)) = self.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn alive(pid: i64) -> bool {
    // SAFETY: kill(2) with signal 0 only probes the process.
    pid > 0 && unsafe { libc::kill(pid as i32, 0) } == 0
}

fn current(state: &Path) -> Value {
    std::fs::read_to_string(state.join("current.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null)
}

#[test]
fn a_standalone_dashboard_launches_chats_benchmarks_and_stops() {
    let root = workspace("dashboard-e2e");
    let state = state_with_card(&root);
    let dashboard = Dashboard::start(&root, &state);
    let status = dashboard.json("/api/status");
    assert_eq!(status["mode"], "launcher");
    assert_eq!(status["launch"]["state"], "none");
    assert_eq!(status["capabilities"]["serve"], true);
    // Web access is off until a launch asks for it, and the page is told why.
    assert_eq!(status["capabilities"]["web_access"], false);
    assert!(status["capabilities"]["web_access_reason"]
        .as_str()
        .unwrap()
        .contains("tick \"Allow web access\""));
    let (page_status, page) = dashboard.get("/");
    assert_eq!(page_status, 200);
    assert!(page.contains("id=\"traffic\"") && page.contains("Static summary"));

    let (accepted, started) = dashboard.post(
        "/api/serve",
        json!({"repo": "loom-fun/test-loom", "context": 4096, "parallel": 2, "allow_web": true}),
    );
    assert_eq!(accepted, 202, "{started}");
    assert_eq!(started["settings"]["context_length"], 4096);
    let serve_job = started["job"].as_str().unwrap().to_string();
    // While it prepares, the row shows stages, not a made-up percentage.
    dashboard.wait_for("a preparing stage", |d| {
        let job = d.json(&format!("/api/jobs/{serve_job}"));
        job["state"] == "running" && job["steps"].as_array().is_some_and(|s| !s.is_empty())
    });
    let preparing = dashboard.json(&format!("/api/jobs/{serve_job}"));
    assert!(preparing["progress"]["total"].is_null(), "{preparing}");
    assert_eq!(preparing["progress"]["unit"], "stages");
    let (twice, refused) = dashboard.post("/api/serve", json!({"repo": "loom-fun/test-loom"}));
    assert_eq!(twice, 409, "{refused}");
    dashboard.wait_for("the launch to serve", |d| d.launch_state() == "serving");
    let launched = dashboard.json(&format!("/api/jobs/{serve_job}"));
    assert_eq!(launched["state"], "done", "{launched}");
    let stages: Vec<&str> = launched["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        stages,
        ["start", "listen", "prepare", "panels", "ready"],
        "{launched}"
    );
    let panels = &launched["steps"][3];
    assert_eq!(panels["state"], "done");
    assert_eq!(
        panels["detail"],
        "derived and cached; the loader took 0.3 s"
    );
    assert!(launched["steps"][2]["detail"]
        .as_str()
        .unwrap()
        .starts_with("registration: preparing"));
    assert_eq!(launched["retry"]["body"]["repo"], "loom-fun/test-loom");
    // The launch asked for web access; its policy fixes thinking off.
    let status = dashboard.json("/api/status");
    assert_eq!(status["capabilities"]["web_access"], true);
    assert_eq!(status["profile"]["thinking"]["locked"], false);
    assert!(status["profile"]["thinking"]["reason"]
        .as_str()
        .unwrap()
        .contains("does not think"));
    let status = dashboard.json("/api/status");
    assert_eq!(status["launch"]["kind"], "owned");
    assert_eq!(status["launch"]["context_length"], 4096);
    assert_eq!(status["launch"]["max_sessions"], 2);
    assert_eq!(status["job"]["state"], "done");
    let pid = status["launch"]["pid"].as_i64().unwrap();
    assert!(alive(pid));
    assert_eq!(current(&state)["repo"], "loom-fun/test-loom");

    let (refused, body) = dashboard.post("/api/serve", json!({"repo": "loom-fun/test-loom"}));
    assert_eq!(refused, 409);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("already being served"));

    let (chat_status, stream) = dashboard.call(
        "POST",
        "/api/chat",
        Some(json!({"messages": [{"role": "user", "content": "hello"}], "stream": true})),
    );
    assert_eq!(chat_status, 200);
    assert!(stream.contains("\" token0\"") && stream.contains("data: [DONE]"));
    dashboard.wait_for("the chat in the counters", |d| {
        d.first_event()["metrics"]["requests"].as_u64().unwrap_or(0) >= 1
    });
    dashboard.wait_for("the upstream probe", |d| {
        d.first_event()["launch"]["up"] == true
    });
    let event = dashboard.first_event();
    assert_eq!(event["launch"]["kind"], "owned");
    assert_eq!(event["last_request"]["route"], "chat_completions");
    // The dashboard's chat is in the launch's own request log.
    let run_directory = PathBuf::from(current(&state)["run_directory"].as_str().unwrap());
    let requests = std::fs::read_to_string(run_directory.join("requests.jsonl")).unwrap();
    assert!(requests.contains("chat_completions"));

    let (accepted, job) = dashboard.post("/api/bench", json!({"suite": "quick"}));
    assert_eq!(accepted, 202, "{job}");
    dashboard.wait_for("the benchmark", |d| {
        let job = &d.json("/api/status")["job"];
        job["kind"] == "bench" && job["state"] != "running"
    });
    let finished = dashboard.json("/api/status")["job"].clone();
    assert_eq!(finished["state"], "done", "{finished}");
    // Per-request progress: warm-up plus one request per row.
    assert_eq!(finished["progress"]["unit"], "requests");
    assert_eq!(finished["progress"]["total"], 4.0);
    assert_eq!(finished["progress"]["done"], 4.0);
    let decode = finished["steps"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "decode")
        .unwrap()
        .clone();
    assert_eq!(decode["label"], "Decode, short prompt");
    assert!(
        decode["detail"].as_str().unwrap().ends_with("tok/s"),
        "{decode}"
    );
    let listed = Command::new(binary())
        .arg("--state")
        .arg(&state)
        .args(["--json", "bench", "--list"])
        .output()
        .unwrap();
    let listed: Value = serde_json::from_slice(&listed.stdout).unwrap();
    let runs = listed["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["id"], finished["result"]["id"]);
    assert_eq!(runs[0]["suite"], "quick");

    let (stopping, stop) = dashboard.post("/api/stop", json!({}));
    assert_eq!(stopping, 202);
    let stop_job = stop["job"].as_str().unwrap().to_string();
    let (again, _) = dashboard.post("/api/stop", json!({}));
    assert_eq!(again, 409);
    dashboard.wait_for("the launch to stop", |d| d.launch_state() == "none");
    dashboard.wait_for("the stop row to finish", |d| {
        d.json(&format!("/api/jobs/{stop_job}"))["state"] == "done"
    });
    let stopped = dashboard.json(&format!("/api/jobs/{stop_job}"));
    assert_eq!(stopped["steps"][0]["name"], "drain");
    assert_eq!(stopped["steps"][1]["label"], "Stopped");
    assert!(
        stopped["steps"][1]["detail"]
            .as_str()
            .unwrap()
            .starts_with("exit status"),
        "{stopped}"
    );
    // Every action of the session is still a row.
    let kinds: Vec<String> = dashboard.json("/api/jobs")["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["kind"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(kinds, ["unload", "bench", "serve"]);
    let deadline = Instant::now() + Duration::from_secs(20);
    while alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(!alive(pid), "the server outlived its stop");
    dashboard.wait_for("current.json to clear", |_| current(&state).is_null());
    let (none, _) = dashboard.post("/api/stop", json!({}));
    assert_eq!(none, 409);
}

#[test]
fn interrupting_the_dashboard_stops_the_launch_it_started() {
    let root = workspace("dashboard-sigint");
    let state = state_with_card(&root);
    let mut dashboard = Dashboard::start(&root, &state);
    let (accepted, _) = dashboard.post("/api/serve", json!({"repo": "loom-fun/test-loom"}));
    assert_eq!(accepted, 202);
    dashboard.wait_for("the launch to serve", |d| d.launch_state() == "serving");
    let pid = dashboard.json("/api/status")["launch"]["pid"]
        .as_i64()
        .unwrap();
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe {
        libc::kill(dashboard.child.id() as i32, libc::SIGINT);
    }
    let deadline = Instant::now() + Duration::from_secs(40);
    let code = loop {
        if let Ok(Some(status)) = dashboard.child.try_wait() {
            break status.code();
        }
        assert!(Instant::now() < deadline, "the dashboard did not exit");
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(code, Some(0));
    assert!(!alive(pid), "the launch outlived its dashboard");
    assert!(
        current(&state).is_null(),
        "the dashboard exited before clearing current.json"
    );
}

#[test]
fn a_dashboard_on_every_interface_prints_its_urls() {
    let root = workspace("dashboard-lan");
    let state = state_with_card(&root);
    let port = free_port();
    let log_path = root.join("dashboard.log");
    let log = std::fs::File::create(&log_path).unwrap();
    let mut child = Command::new(binary())
        .arg("--state")
        .arg(&state)
        .args([
            "--hub",
            "http://127.0.0.1:9",
            "dashboard",
            "--host",
            "0.0.0.0",
            "--allow-web",
            "--port",
        ])
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(log.try_clone().unwrap())
        .stderr(log)
        .spawn()
        .unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + Duration::from_secs(30);
    let status: Value = loop {
        if let Ok(response) = ureq::get(&format!("{base}/api/status")).call() {
            break response.into_json().unwrap();
        }
        assert!(Instant::now() < deadline, "the dashboard did not answer");
        std::thread::sleep(Duration::from_millis(100));
    };
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe {
        libc::kill(child.id() as i32, libc::SIGINT);
    }
    let _ = child.wait();
    let printed = std::fs::read_to_string(&log_path).unwrap();
    assert!(
        printed.contains(&format!("dashboard at {base}/")),
        "{printed}"
    );
    let expected = loom_launch::network::reachable_urls("0.0.0.0", port);
    for url in &expected {
        assert!(
            printed.contains(&format!("network   {url}")),
            "{url} missing:\n{printed}"
        );
    }
    assert!(
        printed.contains("warning   anyone who can reach 0.0.0.0"),
        "{printed}"
    );
    assert!(printed.contains("--allow-web"), "{printed}");
    assert_eq!(status["listener"]["bind_host"], "0.0.0.0");
    assert_eq!(status["listener"]["urls"], json!(expected));
    assert_eq!(status["capabilities"]["web_access"], true);
}
