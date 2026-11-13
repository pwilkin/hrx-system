//! The launcher service end to end, against the GPU-free `fake-serve`.
//!
//! A launcher runs with no model, serves its page and routes, answers `/v1/*`
//! with 503 until a model is loaded, keeps its pid and endpoint while models
//! are loaded, unloaded and lost, and `launcher stop` ends everything. Every
//! test ends its launcher; scratch data lives under Cargo's target directory.

use serde_json::{json, Value};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
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

fn port_free(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

fn alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: kill(2) with signal 0 only probes the process.
    if unsafe { libc::kill(pid as i32, 0) } != 0 {
        return false;
    }
    // A zombie waiting for its parent has exited.
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .map(|stat| {
            !stat
                .rsplit_once(')')
                .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
        })
        .unwrap_or(false)
}

fn wait_until(what: &str, seconds: u64, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while !ready() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// A package directory, the `fake-serve` shim and a state directory.
struct Rig {
    root: PathBuf,
    state: PathBuf,
    card: PathBuf,
    shim: PathBuf,
    port: u16,
}

impl Rig {
    fn new(name: &str) -> Rig {
        let root = workspace(name);
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
        let shim = root.join("fake-loom-serve");
        std::fs::write(
            &shim,
            format!("#!/bin/sh\nexec '{}' fake-serve \"$@\"\n", binary()),
        )
        .unwrap();
        std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
        Rig {
            state: root.join("state"),
            root,
            card,
            shim,
            port: free_port(),
        }
    }

    fn command(&self, arguments: &[&str]) -> Command {
        let mut command = Command::new(binary());
        command
            .arg("--state")
            .arg(&self.state)
            .args(["--hub", "http://127.0.0.1:9"])
            .args(arguments)
            .env_remove("LOOM_LAUNCH_API_KEY")
            .env_remove("LOOM_SERVE")
            .env("LOOM_FAKE_SERVE_TOKENS", "16")
            .env("LOOM_FAKE_SERVE_TOKEN_MS", "0")
            .env("LOOM_FAKE_SERVE_PREPARE_MS", "300");
        command
    }

    fn run(&self, arguments: &[&str]) -> Output {
        self.run_with(arguments, &[])
    }

    fn run_with(&self, arguments: &[&str], environment: &[(&str, &str)]) -> Output {
        let mut command = self.command(arguments);
        for (name, value) in environment {
            command.env(name, value);
        }
        let output = command.output().unwrap();
        eprintln!(
            "--- {arguments:?}\nstatus {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    /// `launcher start --detach` with the shim as loom-serve.
    fn start_launcher(&self) -> Launcher {
        let port = self.port.to_string();
        let output = self.run(&[
            "launcher",
            "start",
            "--detach",
            "--port",
            &port,
            "--loom-serve",
            self.shim.to_str().unwrap(),
        ]);
        assert_eq!(output.status.code(), Some(0));
        let record = self.launcher_record();
        Launcher {
            pid: record["pid"].as_i64().unwrap(),
            endpoint: record["endpoint"].as_str().unwrap().to_string(),
            state: self.state.clone(),
        }
    }

    fn launcher_record(&self) -> Value {
        read(&self.state.join("launcher.json"))
    }

    fn current(&self) -> Value {
        read(&self.state.join("current.json"))
    }

    fn load(&self, extra: &[&str]) -> Output {
        let mut arguments = vec![
            "start",
            self.card.to_str().unwrap(),
            "--device",
            "task",
            "--wait",
            "60",
        ];
        arguments.extend_from_slice(extra);
        self.run(&arguments)
    }
}

fn read(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null)
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// A launcher a test started; dropping it stops whatever is left.
struct Launcher {
    pid: i64,
    endpoint: String,
    state: PathBuf,
}

impl Launcher {
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
        let request = ureq::request(method, &format!("{}{path}", self.endpoint))
            .timeout(Duration::from_secs(30));
        let outcome = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        match outcome {
            Ok(response) | Err(ureq::Error::Status(_, response)) => {
                let status = response.status();
                let text = response.into_string().unwrap_or_default();
                (status, serde_json::from_str(&text).unwrap_or(Value::Null))
            }
            Err(error) => panic!("{method} {path}: {error}"),
        }
    }

    fn get(&self, path: &str) -> (u16, Value) {
        self.call("GET", path, None)
    }

    fn status(&self) -> Value {
        self.get("/api/status").1
    }

    fn jobs(&self) -> Vec<Value> {
        self.get("/api/jobs").1["jobs"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    }
}

impl Drop for Launcher {
    fn drop(&mut self) {
        if alive(self.pid) {
            let _ = Command::new(binary())
                .arg("--state")
                .arg(&self.state)
                .args(["launcher", "stop", "--wait", "20"])
                .output();
        }
    }
}

#[test]
fn a_launcher_runs_without_a_model_and_serves_its_page_and_routes() {
    let rig = Rig::new("launcher-empty");
    let launcher = rig.start_launcher();
    let record = rig.launcher_record();
    let mode = std::fs::metadata(rig.state.join("launcher.json"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600);
    for key in [
        "pid",
        "endpoint",
        "dashboard",
        "bind_host",
        "port",
        "started_at",
        "auth",
    ] {
        assert!(
            !record[key].is_null(),
            "launcher.json lacks {key}: {record}"
        );
    }
    assert_eq!(record["auth"], false);
    assert_eq!(record["detached"], true);
    assert!(!rig.state.join("current.json").exists());

    let page = ureq::get(&format!("{}/", launcher.endpoint))
        .call()
        .unwrap()
        .into_string()
        .unwrap();
    assert!(
        page.contains("id=\"launcher-card\""),
        "the page has no launcher card"
    );
    let status = launcher.status();
    assert_eq!(status["mode"], "launcher");
    assert_eq!(status["launch"]["state"], "none");
    assert_eq!(status["launcher"]["state"], "running");
    assert_eq!(status["launcher"]["pid"], launcher.pid);
    assert_eq!(status["launcher"]["model"], "none");
    assert_eq!(status["capabilities"]["serve"], true);
    assert_eq!(status["capabilities"]["chat"], false);
    assert_eq!(status["capabilities"]["bench"], false);
    assert!(status["capabilities"]["bench_reason"]
        .as_str()
        .unwrap()
        .contains("no model is loaded"));
    let (code, health) = launcher.get("/healthz");
    assert_eq!(code, 200);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["model"], false);
    let (code, loom) = launcher.get("/loom/status");
    assert_eq!(code, 200);
    assert!(loom["launch"].is_null());
    assert_eq!(loom["launcher"]["pid"], launcher.pid);
    assert!(loom["proxy"]["upstream"].is_null());
    let metrics = ureq::get(&format!("{}/loom/metrics", launcher.endpoint))
        .call()
        .unwrap()
        .into_string()
        .unwrap();
    assert!(metrics.contains("loom_launch_upstream_up 0"), "{metrics}");

    // `/v1/*` answers 503 with an OpenAI-shaped error.
    for (method, path, body) in [
        ("GET", "/v1/models", None),
        (
            "POST",
            "/v1/chat/completions",
            Some(json!({"messages": [{"role": "user", "content": "hi"}]})),
        ),
    ] {
        let (code, document) = launcher.call(method, path, body);
        assert_eq!(code, 503, "{path}");
        assert_eq!(
            document,
            json!({"error": {"message": "no model is loaded", "type": "loom_launch_error",
                             "code": "UNAVAILABLE"}}),
            "{path}"
        );
    }
    let (code, refused) = launcher.call(
        "POST",
        "/api/chat",
        Some(json!({"messages": [{"role": "user", "content": "hi"}]})),
    );
    assert_eq!(code, 409, "{refused}");
    let counters = &launcher.status()["launcher"]["counters"];
    assert_eq!(counters["no_model"], 2, "{counters}");
    assert!(counters["api"].as_u64().unwrap() >= 2, "{counters}");

    let output = rig.run(&["status"]);
    assert_eq!(output.status.code(), Some(0));
    let said = stdout(&output);
    assert!(said.contains("launcher  running at"), "{said}");
    assert!(said.contains("model     none loaded"), "{said}");
    let output = rig.run(&["--json", "status"]);
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["launcher"]["pid"], launcher.pid);
    assert_eq!(document["launcher"]["running"], true);
    assert!(document["launcher"]["control_token"].is_null());
    assert!(document["model"].is_null());

    // One launcher per state directory.
    let output = rig.run(&["launcher", "start", "--detach"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("already running"));

    let output = rig.run(&["launcher", "stop"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains("no model was loaded"));
    assert!(!alive(launcher.pid));
    assert!(!rig.state.join("launcher.json").exists());
}

#[test]
fn models_load_and_unload_under_one_launcher() {
    let rig = Rig::new("launcher-cycle");
    let launcher = rig.start_launcher();
    assert_eq!(launcher.get("/v1/models").0, 503);

    let output = rig.load(&["--parallel", "2"]);
    assert_eq!(output.status.code(), Some(0));
    let said = stdout(&output);
    assert!(
        said.contains(&format!("serving test/model at {}", launcher.endpoint)),
        "{said}"
    );
    assert!(said.contains("loom-launch unload"), "{said}");
    let (code, models) = launcher.get("/v1/models");
    assert_eq!(code, 200);
    assert_eq!(models["data"][0]["id"], "test-model");
    let current = rig.current();
    assert_eq!(current["endpoint"], launcher.endpoint);
    assert_eq!(current["launcher_pid"], launcher.pid);
    assert_eq!(current["max_sessions"], 2);
    let first_model = current["pid"].as_i64().unwrap();
    assert!(alive(first_model));
    assert_eq!(rig.launcher_record()["pid"], launcher.pid);
    let status = launcher.status();
    assert_eq!(status["launch"]["state"], "serving");
    assert_eq!(status["launch"]["kind"], "owned");
    assert_eq!(status["launcher"]["model"], "serving");
    assert_eq!(launcher.get("/healthz").1["model"], true);

    // A chat through the launcher lands in the run's request log.
    let output = rig.run(&["--json", "chat", "--prompt", "hello", "--max-tokens", "8"]);
    assert_eq!(output.status.code(), Some(0));
    let run_directory = PathBuf::from(current["run_directory"].as_str().unwrap());
    let requests = std::fs::read_to_string(run_directory.join("requests.jsonl")).unwrap();
    assert!(requests.contains("chat_completions"), "{requests}");

    // A second load is refused while one serves.
    let output = rig.load(&[]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("already being served"));

    // `unload`: back to 503, same launcher.
    let output = rig.run(&["unload"]);
    assert_eq!(output.status.code(), Some(0));
    let said = stdout(&output);
    assert!(said.contains("unloaded test/model"), "{said}");
    assert!(said.contains("the launcher keeps running"), "{said}");
    assert_eq!(launcher.get("/v1/models").0, 503);
    assert!(!rig.state.join("current.json").exists());
    assert!(!alive(first_model), "the unloaded server is still running");
    wait_until("the native port to free", 10, || port_free(rig.port));
    assert!(alive(launcher.pid));
    assert_eq!(rig.launcher_record()["pid"], launcher.pid);
    let meta = read(&run_directory.join("meta.json"));
    assert_eq!(meta["state"], "stopped");
    assert_eq!(meta["source"], "start");
    let unload = launcher
        .jobs()
        .into_iter()
        .find(|j| j["kind"] == "unload")
        .expect("the unload is an Activity row");
    assert_eq!(unload["state"], "done");
    assert_eq!(unload["result"]["repo"], "test/model");
    let output = rig.run(&["unload"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stdout(&output).contains("no model is loaded"));

    // Load again: the same endpoint answers at once.
    let output = rig.load(&[]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(launcher.get("/v1/models").0, 200);
    assert_eq!(rig.current()["endpoint"], launcher.endpoint);
    let second_model = rig.current()["pid"].as_i64().unwrap();
    assert_ne!(second_model, first_model);

    // `stop` unloads only, and says so.
    let output = rig.run(&["stop"]);
    assert_eq!(output.status.code(), Some(0));
    let said = stdout(&output);
    assert!(said.contains("unloaded test/model"), "{said}");
    assert!(
        said.contains("the launcher is still running")
            && said.contains("`loom-launch launcher stop`"),
        "{said}"
    );
    assert!(!alive(second_model));
    assert!(alive(launcher.pid));
    assert_eq!(launcher.get("/v1/models").0, 503);
    assert_eq!(launcher.status()["launch"]["state"], "none");
    let output = rig.run(&["launcher", "status"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains("model     none loaded"));

    // The page loads one too; `launcher stop` ends the model and itself.
    let (code, accepted) = launcher.call(
        "POST",
        "/api/serve",
        Some(json!({"path": rig.card.to_str().unwrap(), "repo": "test/model", "device": "task"})),
    );
    assert_eq!(
        code, 400,
        "the page may not load an unlisted directory: {accepted}"
    );
    let output = rig.load(&[]);
    assert_eq!(output.status.code(), Some(0));
    let third_model = rig.current()["pid"].as_i64().unwrap();
    let output = rig.run(&["launcher", "stop"]);
    assert_eq!(output.status.code(), Some(0));
    let said = stdout(&output);
    assert!(said.contains("unloaded test/model"), "{said}");
    assert!(!alive(launcher.pid));
    assert!(!alive(third_model), "the model outlived its launcher");
    assert!(!rig.state.join("launcher.json").exists());
    assert!(!rig.state.join("current.json").exists());
    wait_until("the native port to free", 10, || port_free(rig.port));
    let output = rig.run(&["launcher", "stop"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stdout(&output).contains("no launcher is running"));
}

#[test]
fn a_model_that_dies_leaves_the_launcher_up_and_says_so() {
    let rig = Rig::new("launcher-crash");
    let launcher = rig.start_launcher();
    assert_eq!(rig.load(&[]).status.code(), Some(0));
    let model = rig.current()["pid"].as_i64().unwrap();
    // As a page that has been polling would: from here on a status answers
    // at once, so it can see the exit before the launcher has handled it.
    assert_eq!(launcher.status()["launch"]["state"], "serving");
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe {
        libc::kill(model as i32, libc::SIGKILL);
    }
    wait_until("the launcher to notice", 20, || {
        launcher.status()["launch"]["state"] == "none"
    });
    assert!(alive(launcher.pid));
    assert_eq!(launcher.get("/v1/models").0, 503);
    assert_eq!(launcher.get("/healthz").1["model"], false);
    assert!(!rig.state.join("current.json").exists());
    let exit = launcher
        .jobs()
        .into_iter()
        .find(|j| j["kind"] == "exit")
        .expect("the exit is an Activity row");
    assert_eq!(exit["state"], "failed");
    assert_eq!(exit["target"], "test/model");
    assert!(
        exit["error"]
            .as_str()
            .unwrap()
            .contains("exited on its own"),
        "{exit}"
    );
    assert!(
        exit["steps"][0]["detail"]
            .as_str()
            .unwrap()
            .contains("signal 9"),
        "{exit}"
    );
    let status = launcher.status();
    assert_eq!(status["launcher"]["last_exit"]["repo"], "test/model");
    assert_eq!(status["launcher"]["last_exit"]["exit_code"], 137);
    assert_eq!(rig.launcher_record()["last_exit"]["exit_code"], 137);
    let output = rig.run(&["status"]);
    assert!(stdout(&output).contains("last exit test/model exited on its own"));
    // It loads again under the same launcher.
    assert_eq!(rig.load(&[]).status.code(), Some(0));
    assert_eq!(launcher.get("/v1/models").0, 200);
    assert_eq!(rig.launcher_record()["pid"], launcher.pid);
    assert_eq!(rig.run(&["stop", "--all"]).status.code(), Some(0));
    assert!(!alive(launcher.pid));
}

#[test]
fn an_unload_cancels_a_model_still_preparing() {
    let rig = Rig::new("launcher-cancel");
    let launcher = rig.start_launcher();
    let mut load = rig
        .command(&[
            "start",
            rig.card.to_str().unwrap(),
            "--device",
            "task",
            "--wait",
            "120",
        ])
        .env("LOOM_FAKE_SERVE_PREPARE_MS", "60000")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_until("the load to start preparing", 30, || {
        launcher.status()["launcher"]["model"] == "starting"
            && launcher.jobs().iter().any(|j| {
                j["kind"] == "serve"
                    && j["state"] == "running"
                    && j["line_count"].as_u64().unwrap_or(0) >= 3
            })
    });
    let output = rig.run(&["unload"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains("cancelled the load of test/model"));
    let code = wait_child(&mut load, 30);
    assert_eq!(code, Some(1), "start reports the cancelled load");
    let serve = launcher
        .jobs()
        .into_iter()
        .find(|j| j["kind"] == "serve")
        .unwrap();
    assert_eq!(serve["state"], "failed");
    assert!(
        serve["error"].as_str().unwrap().contains("cancelled"),
        "{serve}"
    );
    assert!(
        launcher.jobs().iter().all(|j| j["kind"] != "exit"),
        "a cancel is not a crash"
    );
    assert!(!rig.state.join("current.json").exists());
    wait_until("the cancelled server to release its port", 15, || {
        port_free(rig.port)
    });
    assert!(alive(launcher.pid));
    assert_eq!(launcher.status()["launch"]["state"], "none");
    // Nothing of it is left: the launcher has no children.
    assert!(
        children(launcher.pid).is_empty(),
        "{:?}",
        children(launcher.pid)
    );
}

fn wait_child(child: &mut Child, seconds: u64) -> Option<i32> {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return status.code();
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("the child did not exit");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn children(pid: i64) -> Vec<i64> {
    loom_launch::launcher::children_of(pid)
        .into_iter()
        .filter(|child| alive(*child))
        .collect()
}

#[test]
fn interrupting_a_foreground_launcher_unloads_its_model_and_exits() {
    let rig = Rig::new("launcher-foreground");
    let proxy_port = free_port();
    let log = std::fs::File::create(rig.root.join("launcher.log")).unwrap();
    let mut child = rig
        .command(&[
            "launcher",
            "start",
            "--port",
            &rig.port.to_string(),
            "--proxy-port",
            &proxy_port.to_string(),
            "--loom-serve",
            rig.shim.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(log.try_clone().unwrap())
        .stderr(log)
        .spawn()
        .unwrap();
    let endpoint = format!("http://127.0.0.1:{proxy_port}");
    wait_until("the launcher", 30, || {
        ureq::get(&format!("{endpoint}/healthz")).call().is_ok()
    });
    let record = rig.launcher_record();
    assert_eq!(record["pid"], child.id() as i64);
    assert_eq!(record["detached"], false);
    assert_eq!(rig.load(&[]).status.code(), Some(0));
    let model = rig.current()["pid"].as_i64().unwrap();
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe {
        libc::kill(child.id() as i32, libc::SIGINT);
    }
    assert_eq!(wait_child(&mut child, 60), Some(0));
    assert!(!alive(model), "the model outlived its launcher");
    assert!(!rig.state.join("current.json").exists());
    assert!(!rig.state.join("launcher.json").exists());
    let printed = std::fs::read_to_string(rig.root.join("launcher.log")).unwrap();
    assert!(printed.contains("^C stops it"), "{printed}");
    assert!(
        printed.contains("stopped the launcher; unloaded test/model"),
        "{printed}"
    );
}

#[test]
fn a_detached_launcher_ignores_hangups_and_serve_is_still_one_step() {
    let rig = Rig::new("launcher-serve");
    // `serve --detach`: a launcher and its model in one step.
    let output = rig.run(&[
        "serve",
        rig.card.to_str().unwrap(),
        "--loom-serve",
        rig.shim.to_str().unwrap(),
        "--port",
        &rig.port.to_string(),
        "--device",
        "task",
        "--detach",
        "--wait",
        "60",
    ]);
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains("serving test/model at http://127.0.0.1:"));
    let record = rig.launcher_record();
    let pid = record["pid"].as_i64().unwrap();
    assert_eq!(record["detached"], true);
    assert_eq!(rig.current()["launcher_pid"], pid);
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe {
        libc::kill(pid as i32, libc::SIGHUP);
    }
    std::thread::sleep(Duration::from_millis(500));
    assert!(alive(pid), "a detached launcher ended on SIGHUP");
    // `serve --detach` with a launcher running loads under it.
    assert_eq!(rig.run(&["unload"]).status.code(), Some(0));
    let output = rig.run(&[
        "serve",
        rig.card.to_str().unwrap(),
        "--device",
        "task",
        "--detach",
        "--wait",
        "60",
    ]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(rig.current()["launcher_pid"], pid);
    // A foreground `serve` refuses to start a second launcher.
    let output = rig.run(&["serve", rig.card.to_str().unwrap(), "--device", "task"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("loom-launch start"));
    let output = rig.run(&["stop", "--all"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains("stopped the launcher"));
    assert!(!alive(pid));
    assert!(!rig.state.join("launcher.json").exists());
    assert!(!rig.state.join("current.json").exists());
}

#[test]
fn a_record_from_before_the_split_is_still_stopped() {
    let rig = Rig::new("launcher-legacy");
    // A server as a pre-split `serve` left it: current.json names the model
    // and the (gone) process that supervised it; no launcher.json.
    let mut model = rig
        .command(&[
            "fake-serve",
            "--device=task",
            &format!("--listen=127.0.0.1:{}", rig.port),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let endpoint = format!("http://127.0.0.1:{}", rig.port);
    wait_until("the stand-in model", 20, || {
        ureq::get(&format!("{endpoint}/healthz")).call().is_ok()
    });
    let mut gone = Command::new("true").spawn().unwrap();
    let gone_pid = gone.id();
    let _ = gone.wait();
    std::fs::create_dir_all(&rig.state).unwrap();
    std::fs::write(
        rig.state.join("current.json"),
        json!({"run_id": "legacy-1", "pid": model.id(), "launcher_pid": gone_pid,
               "endpoint": endpoint, "server_endpoint": endpoint, "repo": "test/model"})
        .to_string(),
    )
    .unwrap();
    let output = rig.run(&["status"]);
    assert!(stdout(&output).contains("model     test/model"));
    // `launcher stop` cleans it up although no launcher service runs.
    let output = rig.run(&["launcher", "stop"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(
        stdout(&output).contains("started without a launcher service"),
        "{}",
        stdout(&output)
    );
    wait_child(&mut model, 20);
    assert!(!rig.state.join("current.json").exists());
}

#[test]
fn a_launcher_killed_outright_leaves_nothing_behind() {
    let rig = Rig::new("launcher-killed");
    let launcher = rig.start_launcher();
    let mut load = rig
        .command(&[
            "start",
            rig.card.to_str().unwrap(),
            "--device",
            "task",
            "--wait",
            "120",
        ])
        .env("LOOM_FAKE_SERVE_PREPARE_MS", "60000")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // The server is recorded as soon as it is spawned, before it serves.
    wait_until("the preparing server in launcher.json", 30, || {
        rig.launcher_record()["model"]["pid"].as_i64().is_some()
    });
    let model = rig.launcher_record()["model"]["pid"].as_i64().unwrap();
    assert!(alive(model));
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe {
        libc::kill(launcher.pid as i32, libc::SIGKILL);
    }
    wait_until("the launcher to die", 10, || !alive(launcher.pid));
    assert!(alive(model), "the server should have been orphaned");
    let _ = wait_child(&mut load, 30);
    let output = rig.run(&["launcher", "stop"]);
    assert!(
        stdout(&output).contains("no longer running"),
        "{}",
        stdout(&output)
    );
    wait_until("the orphaned server to stop", 15, || !alive(model));
    assert!(!rig.state.join("launcher.json").exists());
    assert!(!rig.state.join("current.json").exists());
}

#[test]
fn an_unload_right_after_a_load_from_the_page_cancels_it() {
    let rig = Rig::new("launcher-race");
    // Installed under a name the page may load.
    let output = rig.run(&[
        "install",
        "--from-path",
        rig.card.to_str().unwrap(),
        "local/card",
    ]);
    assert_eq!(output.status.code(), Some(0));
    let output = rig
        .command(&[
            "launcher",
            "start",
            "--detach",
            "--port",
            &rig.port.to_string(),
            "--loom-serve",
            rig.shim.to_str().unwrap(),
        ])
        .env("LOOM_FAKE_SERVE_PREPARE_MS", "60000")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let record = rig.launcher_record();
    let launcher = Launcher {
        pid: record["pid"].as_i64().unwrap(),
        endpoint: record["endpoint"].as_str().unwrap().to_string(),
        state: rig.state.clone(),
    };
    // However the two requests interleave with the load's planning, the
    // unload is accepted and the load ends cancelled.
    let (code, load) = launcher.call(
        "POST",
        "/api/serve",
        Some(json!({"repo": "local/card", "device": "task"})),
    );
    assert_eq!(code, 202, "{load}");
    let (code, unload) = launcher.call("POST", "/api/unload", Some(json!({})));
    assert_eq!(code, 202, "{unload}");
    wait_until("both jobs to end", 60, || {
        launcher
            .jobs()
            .iter()
            .all(|j| j["state"] == "done" || j["state"] == "failed")
    });
    let jobs = launcher.jobs();
    let serve = jobs.iter().find(|j| j["kind"] == "serve").unwrap();
    assert_eq!(serve["state"], "failed");
    assert!(
        serve["error"].as_str().unwrap().contains("cancelled"),
        "{serve}"
    );
    let unload = jobs.iter().find(|j| j["kind"] == "unload").unwrap();
    assert_eq!(unload["result"]["cancelled"], "local/card", "{unload}");
    wait_until("no server left", 15, || children(launcher.pid).is_empty());
    assert!(port_free(rig.port));
    // The cancel does not carry over to the next load.
    let (code, _) = launcher.call(
        "POST",
        "/api/serve",
        Some(json!({"repo": "local/card", "device": "task"})),
    );
    assert_eq!(code, 202);
    wait_until("the next load to start preparing", 30, || {
        launcher.status()["launcher"]["model"] == "starting" && !children(launcher.pid).is_empty()
    });
    assert_eq!(rig.run(&["launcher", "stop"]).status.code(), Some(0));
    assert!(!alive(launcher.pid));
}
