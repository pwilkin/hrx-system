//! End-to-end supervision against the GPU-free `fake-serve` stand-in.
//!
//! Each test runs the real `loom-launch` binary with `--loom-serve` pointing at a
//! shim that executes `loom-launch fake-serve`, which takes `loom-serve`'s
//! arguments, prints the ready event and answers the OpenAI routes. Scratch data
//! lives under Cargo's target directory, never in /tmp.

use serde_json::Value;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

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

fn shim(directory: &Path, body: &str) -> PathBuf {
    let path = directory.join("fake-loom-serve");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn fake_shim(directory: &Path) -> PathBuf {
    shim(directory, &format!("exec '{}' fake-serve \"$@\"", binary()))
}

fn card(directory: &Path) -> PathBuf {
    let card = directory.join("card");
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
    card
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn port_free(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

fn launcher(state: &Path, arguments: &[&str]) -> Command {
    let mut command = Command::new(binary());
    command
        .arg("--state")
        .arg(state)
        .arg("--hub")
        .arg("http://127.0.0.1:9")
        .args(arguments)
        .env_remove("LOOM_LAUNCH_API_KEY")
        .env_remove("LOOM_SERVE")
        .env("LOOM_FAKE_SERVE_TOKENS", "200")
        .env("LOOM_FAKE_SERVE_TOKEN_MS", "0");
    command
}

fn run(command: &mut Command) -> Output {
    let output = command.output().unwrap();
    eprintln!(
        "--- {:?}\nstatus {:?}\nstdout:\n{}\nstderr:\n{}",
        command.get_args().collect::<Vec<_>>(),
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).expect("stdout is one JSON document")
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn a_detached_launch_serves_chats_benchmarks_and_stops() {
    let root = workspace("detached");
    let state = root.join("state");
    let serve = fake_shim(&root);
    let card = card(&root);
    let port = free_port();
    let port_text = port.to_string();
    let card_text = card.to_str().unwrap();
    let output = run(launcher(
        &state,
        &[
            "serve",
            card_text,
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port_text,
            "--device",
            "task",
            "--detach",
            "--wait",
            "60",
            "--parallel",
            "2",
        ],
    )
    .env("LOOM_FAKE_SERVE_PREPARE_MS", "1200"));
    assert_eq!(output.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("serving test/model at http://127.0.0.1:")
    );

    let current_path = state.join("current.json");
    let mode = std::fs::metadata(&current_path)
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600);
    let current = read(&current_path);
    let endpoint = current["endpoint"].as_str().unwrap().to_string();
    assert_eq!(
        current["server_endpoint"],
        format!("http://127.0.0.1:{port}")
    );
    assert_eq!(current["context_length"], 8192);
    assert_eq!(current["max_sessions"], 2);
    let run_directory = PathBuf::from(current["run_directory"].as_str().unwrap());
    let deployment = read(&run_directory.join("serve.json"));
    assert_eq!(
        deployment["model"]["path"],
        format!("{card_text}/model.json")
    );
    assert_eq!(deployment["definition"]["rows"], 2);
    assert_eq!(deployment["definition"]["max_sessions"], 2);

    let output = run(&mut launcher(&state, &["status"]));
    assert_eq!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stdout).contains("traffic"));

    let output = run(&mut launcher(
        &state,
        &[
            "--json",
            "chat",
            card_text,
            "--prompt",
            "hello there",
            "--max-tokens",
            "16",
        ],
    ));
    assert_eq!(output.status.code(), Some(0));
    let turn = json(&output);
    assert!(turn["text"].as_str().unwrap().starts_with(" token0"));
    assert_eq!(turn["usage"]["completion_tokens"], 16);
    assert!(String::from_utf8_lossy(&output.stderr).contains("% of 8192"));

    let output = run(&mut launcher(
        &state,
        &["--json", "bench", card_text, "--suite", "quick"],
    ));
    assert_eq!(output.status.code(), Some(0));
    let record = json(&output);
    assert_eq!(record["ok"], true);
    assert_eq!(record["endpoint"], endpoint);
    assert_eq!(record["harness"]["workload"], "loom-launch/text-v1");
    assert_eq!(record["harness"]["parallel"], 2);
    assert!(record["metrics"]["decode"]["value"].as_f64().unwrap() > 0.0);
    let stored = PathBuf::from(record["path"].as_str().unwrap());
    assert!(stored.is_file());
    let id = record["id"].as_str().unwrap().to_string();
    let output = run(&mut launcher(&state, &["bench", "--list"]));
    assert!(String::from_utf8_lossy(&output.stdout).contains(&id));
    let output = run(&mut launcher(&state, &["bench", "--show", &id]));
    assert_eq!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("metric                      value  published"));

    let metrics = ureq::get(&format!("{endpoint}/loom/metrics"))
        .call()
        .unwrap()
        .into_string()
        .unwrap();
    assert!(metrics.contains("loom_launch_completion_tokens_total"));
    assert!(metrics.contains("loom_launch_upstream_up 1"));
    let requests: Vec<Value> = std::fs::read_to_string(run_directory.join("requests.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(
        requests
            .iter()
            .filter(|r| r["route"] == "chat_completions")
            .count()
            >= 5
    );

    // `stop` unloads the model and says the launcher keeps running.
    let launcher_record = read(&state.join("launcher.json"));
    let launcher_pid = launcher_record["pid"].as_i64().unwrap();
    assert_eq!(launcher_record["endpoint"], endpoint);
    let output = run(&mut launcher(&state, &["stop"]));
    assert_eq!(output.status.code(), Some(0));
    let said = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(said.contains("unloaded test/model"), "{said}");
    assert!(
        said.contains("the launcher is still running")
            && said.contains("loom-launch launcher stop"),
        "{said}"
    );
    assert!(!current_path.exists());
    assert_eq!(read(&state.join("launcher.json"))["pid"], launcher_pid);
    let meta = read(&run_directory.join("meta.json"));
    assert_eq!(meta["state"], "stopped");
    assert_eq!(meta["source"], "serve");
    let events: Vec<String> = std::fs::read_to_string(run_directory.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["event"]
                .as_str()
                .unwrap_or("")
                .to_string()
        })
        .collect();
    for expected in ["launch", "ready", "serving", "stopped"] {
        assert!(
            events.iter().any(|e| e == expected),
            "{expected} missing from {events:?}"
        );
    }
    assert!(std::fs::read_to_string(run_directory.join("serve.log"))
        .unwrap()
        .contains("\"event\":\"ready\""));
    assert!(port_free(port));
    let output = run(&mut launcher(&state, &["stop"]));
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stdout).contains("no model is loaded"));
    let output = run(&mut launcher(&state, &["launcher", "stop"]));
    assert_eq!(output.status.code(), Some(0));
    assert!(!state.join("launcher.json").exists());
    let output = run(&mut launcher(&state, &["stop"]));
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn a_failed_registration_is_reported_without_waiting_out_the_timeout() {
    let root = workspace("failed");
    let state = root.join("state");
    let serve = fake_shim(&root);
    let card = card(&root);
    let port = free_port();
    let started = Instant::now();
    let output = run(launcher(
        &state,
        &[
            "serve",
            card.to_str().unwrap(),
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--device",
            "task",
            "--wait",
            "300",
        ],
    )
    .env("LOOM_FAKE_SERVE_FAIL", "no such file: parameter/model.gguf"));
    assert_eq!(output.status.code(), Some(1));
    assert!(started.elapsed() < Duration::from_secs(60));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("model preparation failed: no such file: parameter/model.gguf"),
        "{stderr}"
    );
    assert!(!state.join("current.json").exists());
    let deadline = Instant::now() + Duration::from_secs(5);
    while !port_free(port) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(port_free(port), "the failed server was left running");
    let run_directory = std::fs::read_dir(state.join("runs"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let meta = read(&run_directory.join("meta.json"));
    assert!(meta["error"].as_str().unwrap().contains("no such file"));
}

#[test]
fn a_server_that_exits_early_is_reported_with_its_output() {
    let root = workspace("early-exit");
    let state = root.join("state");
    let serve = shim(&root, "echo 'cannot open device: boom' >&2; exit 3");
    let card = card(&root);
    let output = run(&mut launcher(
        &state,
        &[
            "serve",
            card.to_str().unwrap(),
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &free_port().to_string(),
            "--device",
            "task",
            "--wait",
            "60",
        ],
    ));
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("exit code 3"), "{stderr}");
    assert!(stderr.contains("cannot open device: boom"), "{stderr}");
}

#[test]
fn an_ephemeral_benchmark_launch_is_stopped_afterwards() {
    let root = workspace("ephemeral");
    let state = root.join("state");
    let serve = fake_shim(&root);
    let card = card(&root);
    let port = free_port();
    let output = run(&mut launcher(
        &state,
        &[
            "--json",
            "bench",
            card.to_str().unwrap(),
            "--suite",
            "quick",
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--device",
            "task",
            "--wait",
            "60",
        ],
    ));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(json(&output)["ok"], true);
    assert!(!state.join("current.json").exists());
    assert!(port_free(port));
    let run_directory = std::fs::read_dir(state.join("runs"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let meta = read(&run_directory.join("meta.json"));
    assert_eq!(meta["source"], "ephemeral");
    assert_eq!(meta["state"], "stopped");
}

#[test]
fn a_proxy_that_cannot_bind_stops_the_prepared_server() {
    let root = workspace("proxy-taken");
    let state = root.join("state");
    let serve = fake_shim(&root);
    let card = card(&root);
    let port = free_port();
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let taken_port = taken.local_addr().unwrap().port().to_string();
    let output = run(&mut launcher(
        &state,
        &[
            "serve",
            card.to_str().unwrap(),
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--proxy-port",
            &taken_port,
            "--device",
            "task",
            "--wait",
            "60",
        ],
    ));
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("cannot bind the proxy"));
    assert!(!state.join("current.json").exists());
    let deadline = Instant::now() + Duration::from_secs(5);
    while !port_free(port) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(port_free(port), "the prepared server was left running");
    drop(taken);
}

#[test]
fn a_launch_on_every_interface_reports_its_urls_and_keeps_inference_on_loopback() {
    let root = workspace("lan");
    let state = root.join("state");
    let serve = fake_shim(&root);
    let card = card(&root);
    let port = free_port();
    let proxy_port = free_port();
    let key = "lan-key-7f3a9";
    let output = run(launcher(
        &state,
        &[
            "serve",
            card.to_str().unwrap(),
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port.to_string(),
            "--host",
            "0.0.0.0",
            "--proxy-port",
            &proxy_port.to_string(),
            "--allow-web",
            "--device",
            "task",
            "--detach",
            "--wait",
            "60",
        ],
    )
    .env("LOOM_LAUNCH_API_KEY", key));
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let expected = loom_launch::network::reachable_urls("0.0.0.0", proxy_port);
    for url in &expected {
        assert!(
            stdout.contains(&format!("network    {url}")),
            "{url} missing:\n{stdout}"
        );
    }
    // Output that is not a terminal never carries the key.
    assert!(!stdout.contains(key), "{stdout}");
    assert!(
        stdout.contains("?access_token=<your --api-key>"),
        "{stdout}"
    );

    let current = read(&state.join("current.json"));
    assert_eq!(current["bind_host"], "0.0.0.0");
    assert_eq!(current["urls"], serde_json::json!(expected));
    assert_eq!(
        current["endpoint"],
        format!("http://127.0.0.1:{proxy_port}")
    );
    assert_eq!(
        current["server_endpoint"],
        format!("http://127.0.0.1:{port}")
    );

    let get = |base: &str, path: &str| {
        ureq::get(&format!("{base}{path}"))
            .set("Authorization", &format!("Bearer {key}"))
            .timeout(Duration::from_secs(10))
            .call()
            .map(|r| r.into_json::<Value>().unwrap())
            .map_err(|error| error.to_string())
    };
    let local = format!("http://127.0.0.1:{proxy_port}");
    let status = get(&local, "/api/status").unwrap();
    assert_eq!(status["listener"]["bind_host"], "0.0.0.0");
    assert_eq!(status["launch"]["bind_host"], "0.0.0.0");
    assert_eq!(status["capabilities"]["web_access"], true);
    assert!(
        ureq::get(&format!("{local}/api/status")).call().is_err(),
        "the key is required"
    );
    // From an interface address the proxy answers (with the key) and the
    // native listener does not answer at all.
    if let Some(address) = loom_launch::network::interface_ipv4_addresses().first() {
        let lan = format!("http://{address}:{proxy_port}");
        let status = get(&lan, "/api/status").unwrap();
        assert_eq!(status["launch"]["state"], "serving");
        let native = std::net::TcpStream::connect_timeout(
            &format!("{address}:{port}").parse().unwrap(),
            Duration::from_secs(2),
        );
        assert!(
            native.is_err(),
            "the native listener is reachable on {address}"
        );
    }

    let record = std::fs::read_to_string(state.join("launcher.json")).unwrap();
    assert!(!record.contains(key), "the key reached launcher.json");
    assert!(record.contains("\"auth\": true"), "{record}");
    let output = run(&mut launcher(&state, &["stop", "--all"]));
    assert_eq!(output.status.code(), Some(0));
    assert!(!state.join("launcher.json").exists());
    let mut logs = String::new();
    for entry in std::fs::read_dir(state.join("runs")).unwrap().flatten() {
        for name in ["serve.log", "events.jsonl", "meta.json"] {
            logs.push_str(&std::fs::read_to_string(entry.path().join(name)).unwrap_or_default());
        }
    }
    logs.push_str(&std::fs::read_to_string(state.join("logs/detached.log")).unwrap_or_default());
    assert!(!logs.is_empty());
    assert!(!logs.contains(key), "the key reached a log file");
    assert!(
        logs.contains("\"bind_host\""),
        "the run record lacks bind_host"
    );
}

#[test]
fn serving_the_native_listener_beyond_loopback_is_refused() {
    let root = workspace("no-proxy-lan");
    let state = root.join("state");
    let serve = fake_shim(&root);
    let card = card(&root);
    let output = run(&mut launcher(
        &state,
        &[
            "serve",
            card.to_str().unwrap(),
            "--loom-serve",
            serve.to_str().unwrap(),
            "--host",
            "0.0.0.0",
            "--no-proxy",
        ],
    ));
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("drop --no-proxy"));
    assert!(!state.join("current.json").exists());
}
