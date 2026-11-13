//! Prepared scenarios and local profiles, end to end against a recording stub.
//!
//! The fixtures are the documents of the Qwen3.8-27B GPU/NPU hybrid
//! (`tests/fixtures`): a package that carries its own `deployment.json` and
//! `devices.json`. `loom-serve` is a shim that records its argv and runs
//! `loom-launch fake-serve`, which refuses `--devices` next to the single-device
//! flags as the real server does. The hub is a dead port or a local recorder,
//! so no test reaches the network. Scratch data lives under Cargo's target
//! directory, never in /tmp.

use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};

const DEAD_HUB: &str = "http://127.0.0.1:9";

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

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// A copy of one fixture package at `destination`.
fn package(name: &str, destination: &Path) -> PathBuf {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(fixture(name)).unwrap().flatten() {
        std::fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
    }
    destination.to_path_buf()
}

/// An ordinary profile: a card and its generator config, no scenario documents.
fn plain_profile(destination: &Path) -> PathBuf {
    std::fs::create_dir_all(destination).unwrap();
    std::fs::write(
        destination.join("model.json"),
        r#"{"schema_version": 1, "model": {"id": "test/plain", "name": "Plain test model"},
  "services": {"chat": {"kind": "text", "version": 1}},
  "presets": {"chat": {"service": "chat", "execution": {"rows": 2, "context_capacity": 8192, "max_tokens": 512}}}}"#,
    )
    .unwrap();
    std::fs::write(
        destination.join("llm_config.json"),
        r#"{"arch": "test", "gguf": "plain-Q8_0.gguf", "context_length": 32768}"#,
    )
    .unwrap();
    destination.to_path_buf()
}

/// A `loom-serve` that writes its arguments, one per line, then runs fake-serve.
fn recording_shim(directory: &Path) -> (PathBuf, PathBuf) {
    let argv = directory.join("argv.txt");
    let path = directory.join("loom-serve");
    std::fs::write(
        &path,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{}'\nexec '{}' fake-serve \"$@\"\n",
            argv.display(),
            binary()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    (path, argv)
}

fn recorded_argv(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn launcher(state: &Path, hub: &str, arguments: &[&str]) -> Command {
    let mut command = Command::new(binary());
    command
        .arg("--state")
        .arg(state)
        .arg("--hub")
        .arg(hub)
        .args(arguments)
        .env_remove("LOOM_LAUNCH_API_KEY")
        .env_remove("LOOM_SERVE")
        .env_remove("LOOM_LAUNCH_PROFILES")
        .env("LOOM_FAKE_SERVE_TOKENS", "16")
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

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn json(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).expect("stdout is one JSON document")
}

fn read(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn get(url: &str) -> Value {
    ureq::get(url).call().unwrap().into_json().unwrap()
}

/// A hub that answers every listing with no models and records each request line.
struct RecordingHub {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl RecordingHub {
    fn start() -> RecordingHub {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                seen.lock().unwrap().push(line.trim().to_string());
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).unwrap_or(0) == 0 || header.trim().is_empty() {
                        break;
                    }
                }
                let mut stream = stream;
                let _ = stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n[]",
                );
            }
        });
        RecordingHub { url, requests }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

fn profiles_by_path(document: &Value) -> Vec<(String, Value)> {
    document["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["source"] == "local")
        .map(|p| (p["path"].as_str().unwrap().to_string(), p.clone()))
        .collect()
}

#[test]
fn a_prepared_scenario_runs_its_own_deployment_and_devices() {
    let root = workspace("scenario-serve");
    let state = root.join("state");
    let (serve, argv) = recording_shim(&root);
    let hybrid = package("hybrid-package", &root.join("Qwen3.8-27B-hybrid"));
    let port = free_port();
    let port_text = port.to_string();
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &[
            "serve",
            hybrid.to_str().unwrap(),
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port_text,
            "--workers",
            "4",
            "--wait",
            "60",
            "--detach",
        ],
    ));
    assert_eq!(output.status.code(), Some(0));
    assert!(stdout(&output).contains(
        "scenario   running the package's own deployment qwen3.8-27b across amdgpu + xdna"
    ));

    // The exact command: the package's own documents, no composed deployment, no --device.
    let expected = vec![
        format!("--devices={}/devices.json", hybrid.display()),
        format!("--model={}/deployment.json", hybrid.display()),
        format!("--listen=127.0.0.1:{port}"),
        "--workers=4".to_string(),
    ];
    assert_eq!(recorded_argv(&argv), expected);

    let current = read(&state.join("current.json"));
    assert_eq!(current["kind"], "scenario");
    assert_eq!(current["device"], "amdgpu + xdna");
    assert_eq!(current["deployment"], "qwen3.8-27b");
    assert_eq!(current["context_length"], 40960);
    assert_eq!(current["rows"], 1);
    assert_eq!(current["max_tokens"], 4096);
    let mut command = vec![serve.to_string_lossy().into_owned()];
    command.extend(expected.iter().cloned());
    assert_eq!(current["command"], serde_json::json!(command));
    assert_eq!(current["scenario"]["devices"], "amdgpu + xdna");
    assert_eq!(
        current["scenario"]["device_uris"]["xdna"],
        "xdna://0?columns=8"
    );
    assert_eq!(
        current["scenario"]["deployment_path"],
        format!("{}/deployment.json", hybrid.display())
    );
    let run_directory = PathBuf::from(current["run_directory"].as_str().unwrap());
    assert!(!run_directory.join("serve.json").exists());
    assert!(!run_directory.join("devices.json").exists());
    let meta = read(&run_directory.join("meta.json"));
    assert_eq!(meta["kind"], "scenario");
    assert_eq!(meta["device"], "amdgpu + xdna");
    assert_eq!(meta["deployment_name"], "qwen3.8-27b");
    assert_eq!(meta["command"], current["command"]);
    assert_eq!(meta["settings"]["context_length"], 40960);
    assert_eq!(meta["settings"]["rows"], 1);
    assert_eq!(meta["settings"]["max_tokens"], 4096);

    // The proxy and its dashboard front the scenario like any launch.
    let endpoint = current["endpoint"].as_str().unwrap().to_string();
    let status = get(&format!("{endpoint}/api/status"));
    let launch = &status["launch"];
    assert_eq!(launch["launch_kind"], "scenario");
    assert_eq!(launch["device"], "amdgpu + xdna");
    assert_eq!(launch["deployment"], "qwen3.8-27b");
    assert_eq!(launch["context_length"], 40960);
    assert_eq!(launch["rows"], 1);
    assert_eq!(launch["max_tokens"], 4096);
    assert_eq!(launch["command"], current["command"]);
    assert_eq!(status["scenario"]["kind"], "scenario");
    assert_eq!(status["scenario"]["deployment"], "qwen3.8-27b");
    assert_eq!(status["current"]["kind"], "scenario");
    let models = get(&format!("{endpoint}/v1/models"));
    assert_eq!(models["data"][0]["id"], "qwen3.8-27b");

    let output = run(&mut launcher(&state, DEAD_HUB, &["status"]));
    assert!(stdout(&output)
        .contains("scenario  the package's own deployment qwen3.8-27b across amdgpu + xdna"));
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &["--json", "chat", "--prompt", "hello", "--max-tokens", "8"],
    ));
    assert_eq!(output.status.code(), Some(0));
    assert!(stderr(&output).contains("% of 40960"));
    let output = run(&mut launcher(&state, DEAD_HUB, &["stop", "--all"]));
    assert_eq!(output.status.code(), Some(0));
    assert!(!state.join("current.json").exists());
}

#[test]
fn options_that_contradict_a_scenario_are_refused() {
    let root = workspace("scenario-refuse");
    let state = root.join("state");
    let (serve, argv) = recording_shim(&root);
    let hybrid = package("hybrid-publish", &root.join("hybrid"));
    let target = hybrid.to_str().unwrap();
    let serve_path = serve.to_str().unwrap();
    for (option, value) in [
        ("--context", "50000"),
        ("--max-tokens", "1024"),
        ("--parallel", "2"),
        ("--draft-depth", "3"),
        ("--device", "amdgpu://"),
    ] {
        for detach in [false, true] {
            let mut arguments = vec!["serve", target, "--loom-serve", serve_path, option, value];
            if detach {
                arguments.push("--detach");
            }
            let output = run(&mut launcher(&state, DEAD_HUB, &arguments));
            assert_eq!(output.status.code(), Some(1), "{option}");
            let message = stderr(&output);
            assert!(message.contains("is a prepared scenario"), "{message}");
            assert!(message.contains("owns its capacities"), "{message}");
            assert!(message.contains(option), "{message}");
            assert!(
                message.contains(&format!("{}/deployment.json", hybrid.display())),
                "{message}"
            );
            assert!(
                message.contains(&format!("{}/devices.json", hybrid.display())),
                "{message}"
            );
            assert!(message.contains("context 40960"), "{message}");
        }
    }
    // Nothing was started or recorded.
    assert!(!argv.exists());
    assert!(!state.join("runs").exists());
    assert!(!state.join("logs").join("detached.log").exists());

    // Pool options are honoured through a run copy of devices.json, since
    // loom-serve refuses them next to --devices.
    let port = free_port().to_string();
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &[
            "serve",
            target,
            "--loom-serve",
            serve_path,
            "--port",
            &port,
            "--memory-bytes",
            "48000000000",
            "--pool-backing",
            "elastic",
            "--wait",
            "60",
            "--detach",
        ],
    ));
    assert_eq!(output.status.code(), Some(0));
    let current = read(&state.join("current.json"));
    let run_directory = PathBuf::from(current["run_directory"].as_str().unwrap());
    let recorded = recorded_argv(&argv);
    assert_eq!(
        recorded[0],
        format!("--devices={}/devices.json", run_directory.display())
    );
    assert_eq!(
        recorded[1],
        format!("--model={}/deployment.json", hybrid.display())
    );
    assert!(!recorded.iter().any(|a| a.starts_with("--memory_bytes")
        || a.starts_with("--pool_backing")
        || a.starts_with("--device=")));
    let devices = read(&run_directory.join("devices.json"));
    assert_eq!(devices["pools"]["model"]["memory_bytes"], 48000000000_i64);
    assert_eq!(devices["pools"]["model"]["pool_backing"], "elastic");
    assert_eq!(devices["pools"]["exchange"]["pool_backing"], "fixed");
    assert_eq!(
        read(&hybrid.join("devices.json"))["pools"]["model"]["memory_bytes"],
        52000000000_i64
    );
    assert_eq!(current["scenario"]["overrides"]["pool"], "model");
    let output = run(&mut launcher(&state, DEAD_HUB, &["stop", "--all"]));
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn local_profiles_are_listed_without_asking_the_hub() {
    let root = workspace("scenario-local");
    let state = root.join("state");
    let profiles = root.join("profiles");
    let hybrid = package("hybrid-publish", &profiles.join("Qwen3.8-27B-hybrid-loom"));
    let plain = plain_profile(&profiles.join("plain"));
    let profiles_text = profiles.to_str().unwrap();

    // A hub that answers: only listings reach it, never the local profiles' documents.
    let hub = RecordingHub::start();
    let output = run(&mut launcher(
        &state,
        &hub.url,
        &["--json", "search", "--profiles", profiles_text],
    ));
    assert_eq!(output.status.code(), Some(0));
    let found = profiles_by_path(&json(&output));
    let paths: Vec<&str> = found.iter().map(|(p, _)| p.as_str()).collect();
    assert_eq!(paths.len(), 2, "{paths:?}");
    assert!(paths.contains(&hybrid.to_str().unwrap()));
    assert!(paths.contains(&plain.to_str().unwrap()));
    let (_, hybrid_row) = found
        .iter()
        .find(|(p, _)| p == hybrid.to_str().unwrap())
        .unwrap();
    assert_eq!(hybrid_row["source"], "local");
    assert_eq!(hybrid_row["scenario"], true);
    assert_eq!(hybrid_row["scenario_devices"], "amdgpu + xdna");
    assert_eq!(hybrid_row["installed"], false);
    assert_eq!(hybrid_row["profile"]["context_length"], 40960);
    assert_eq!(hybrid_row["profile"]["native_context_length"], 262144);
    assert_eq!(hybrid_row["profile"]["quant"], "Q4_K_M");
    assert_eq!(hybrid_row["profile"]["memory_bytes"], 52000000000_i64);
    assert!(hybrid_row["compatibility"]["score"].is_number());
    let (_, plain_row) = found
        .iter()
        .find(|(p, _)| p == plain.to_str().unwrap())
        .unwrap();
    assert_eq!(plain_row["scenario"], false);
    assert_eq!(plain_row["profile"]["context_length"], 8192);
    assert_eq!(plain_row["profile"]["native_context_length"], 32768);
    let requests = hub.requests();
    assert!(!requests.is_empty());
    for request in &requests {
        assert!(request.starts_with("GET /api/models?"), "{request}");
        assert!(
            !request.contains("qwen3.8") && !request.contains("plain"),
            "{request}"
        );
    }

    // A dead hub hides nothing local, without --offline; the environment
    // variable names roots like the flag does.
    let output = run(launcher(&state, DEAD_HUB, &["--json", "search"])
        .env("LOOM_LAUNCH_PROFILES", profiles_text));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(profiles_by_path(&json(&output)).len(), 2);
    assert!(stderr(&output).contains("the hub did not answer"));
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &["search", "qwen3.8", "--profiles", profiles_text],
    ));
    assert_eq!(output.status.code(), Some(0));
    let table = stdout(&output);
    assert!(
        table.contains(&format!("local {}", hybrid.display())),
        "{table}"
    );
    assert!(table.contains("model native 262144"), "{table}");
    assert!(table.contains("scenario on amdgpu + xdna"), "{table}");
    assert!(!table.contains("test/plain"), "{table}");

    // The dashboard lists them the same way.
    let port = free_port();
    let log = std::fs::File::create(root.join("dashboard.log")).unwrap();
    let mut dashboard = launcher(
        &state,
        &hub.url,
        &[
            "dashboard",
            "--port",
            &port.to_string(),
            "--profiles",
            profiles_text,
        ],
    )
    .stdout(log.try_clone().unwrap())
    .stderr(log)
    .spawn()
    .unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut models = Value::Null;
    for _ in 0..100 {
        if let Ok(response) = ureq::get(&format!("{base}/api/models?q=")).call() {
            models = response.into_json().unwrap();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let _ = dashboard.kill();
    let _ = dashboard.wait();
    let rows: Vec<&Value> = models["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["source"] == "local")
        .collect();
    assert_eq!(rows.len(), 2, "{models}");
    let row = rows
        .iter()
        .find(|p| p["path"] == hybrid.to_str().unwrap())
        .unwrap();
    assert_eq!(row["scenario"], true);
    assert_eq!(row["context_length"], 40960);
    assert_eq!(row["native_context_length"], 262144);
    assert_eq!(row["quant"], "Q4_K_M");
    assert!(row["fit"].is_number());
    assert!(row["decode"].is_number());
    for request in hub.requests() {
        assert!(request.starts_with("GET /api/models?"), "{request}");
        assert!(!request.contains("qwen3.8"), "{request}");
    }
}

#[test]
fn install_from_path_registers_a_package_that_is_then_servable() {
    let root = workspace("scenario-install");
    let state = root.join("state");
    let (serve, argv) = recording_shim(&root);
    let hybrid = package(
        "hybrid-package",
        &root.join("bundles").join("Qwen3.8-27B-hybrid"),
    );
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &["install", "--from-path", hybrid.to_str().unwrap()],
    ));
    assert_eq!(output.status.code(), Some(0));
    let text = stdout(&output);
    assert!(text.contains("in place (nothing was copied"), "{text}");
    assert!(text.contains("serve it with: loom-launch serve local/Qwen3.8-27B-hybrid"));
    let installs = read(&state.join("installs.json"));
    let record = &installs["installs"]["local/Qwen3.8-27B-hybrid"];
    assert_eq!(record["path"], hybrid.to_str().unwrap());
    assert_eq!(record["source"], "local");
    assert_eq!(record["scenario"], true);
    assert!(record.get("scenario_revision").is_none());

    // Visible without --profiles or the hub, and explained without the hub.
    let output = run(&mut launcher(&state, DEAD_HUB, &["--json", "search"]));
    let found = profiles_by_path(&json(&output));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].1["installed"], true);
    assert_eq!(found[0].1["repo"], "local/Qwen3.8-27B-hybrid");
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &["show", "local/Qwen3.8-27B-hybrid"],
    ));
    assert_eq!(output.status.code(), Some(0));
    let shown = stdout(&output);
    assert!(shown.contains("scenario     deployment qwen3.8-27b on amdgpu + xdna"));
    assert!(shown.contains(&format!(
        "serve with: loom-launch serve {}",
        hybrid.display()
    )));
    assert!(!shown.contains("install with"));

    let port = free_port().to_string();
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &[
            "serve",
            "local/Qwen3.8-27B-hybrid",
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port,
            "--wait",
            "60",
            "--detach",
        ],
    ));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        recorded_argv(&argv)[..2],
        [
            format!("--devices={}/devices.json", hybrid.display()),
            format!("--model={}/deployment.json", hybrid.display()),
        ]
    );
    let current = read(&state.join("current.json"));
    assert_eq!(current["repo"], "local/Qwen3.8-27B-hybrid");
    assert_eq!(current["kind"], "scenario");
    assert_eq!(
        run(&mut launcher(&state, DEAD_HUB, &["stop", "--all"]))
            .status
            .code(),
        Some(0)
    );

    // --copy copies the package but links weights instead of copying them.
    let plain = plain_profile(&root.join("bundles").join("plain"));
    std::fs::write(plain.join("weights.gguf"), b"GGUF").unwrap();
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &[
            "--json",
            "install",
            "--from-path",
            plain.to_str().unwrap(),
            "--copy",
        ],
    ));
    assert_eq!(output.status.code(), Some(0));
    let document = json(&output);
    assert_eq!(document["mode"], "copy");
    assert_eq!(document["install"]["scenario"], false);
    assert!(
        read(&state.join("installs.json"))["installs"]["local/plain"]
            .get("scenario")
            .is_none()
    );
    let copied = state.join("models").join("local-plain");
    assert_eq!(document["install"]["path"], copied.to_str().unwrap());
    assert!(copied.join("model.json").is_file());
    let link = std::fs::symlink_metadata(copied.join("weights.gguf")).unwrap();
    assert!(link.file_type().is_symlink());
    assert_eq!(
        std::fs::read_link(copied.join("weights.gguf")).unwrap(),
        plain.join("weights.gguf")
    );
}

#[test]
fn show_reports_the_served_context_and_suggests_the_path() {
    let root = workspace("scenario-show");
    let state = root.join("state");
    let hybrid = package("hybrid-publish", &root.join("hybrid"));
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &["show", hybrid.to_str().unwrap()],
    ));
    assert_eq!(output.status.code(), Some(0));
    let text = stdout(&output);
    assert!(
        text.contains("  context      40960 tokens (model native 262144)\n"),
        "{text}"
    );
    assert!(text.contains(&format!(
        "serve with: loom-launch serve {}\n",
        hybrid.display()
    )));
    assert!(!text.contains("install with"), "{text}");
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &["--json", "show", hybrid.to_str().unwrap()],
    ));
    let document = json(&output);
    assert_eq!(document["context_length"], 40960);
    assert_eq!(document["native_context_length"], 262144);
    assert_eq!(document["scenario"]["devices"], "amdgpu + xdna");

    // As an ordinary profile (no scenario documents), --context may widen the
    // served context up to the native one and no further.
    std::fs::remove_file(hybrid.join("devices.json")).unwrap();
    std::fs::remove_file(hybrid.join("deployment.json")).unwrap();
    let output = run(&mut launcher(
        &state,
        DEAD_HUB,
        &[
            "serve",
            hybrid.to_str().unwrap(),
            "--context",
            "300000",
            "--loom-serve",
            "/bin/false",
        ],
    ));
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("context 300000 exceeds the model's 262144 tokens"));
}

const HYBRID_REPO: &str = "ilintar/Qwen3.8-27B-hybrid-loom";
const HYBRID_REVISION: &str = "3d5342e3317741e55995fa568ba907e528d811e9";

/// A hub serving one repository's files at one revision (anything else is
/// 404), recording each request line.
struct ScriptedHub {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl ScriptedHub {
    fn start(repo: &'static str, revision: &'static str, files: Vec<(&str, Vec<u8>)>) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let files: Vec<(String, Vec<u8>)> = files
            .into_iter()
            .map(|(name, body)| (format!("/{repo}/resolve/{revision}/{name}"), body))
            .collect();
        let entry = serde_json::json!({
            "id": repo,
            "sha": revision,
            "tags": ["loom", "loom-serve"],
            "siblings": [{"rfilename": "model.json"}],
        });
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).unwrap_or(0) == 0 || header.trim().is_empty() {
                        break;
                    }
                }
                let line = line.trim().to_string();
                seen.lock().unwrap().push(line.clone());
                let path = line.split(' ').nth(1).unwrap_or("").to_string();
                let found: Option<Vec<u8>> = if path.starts_with("/api/models?") {
                    Some(serde_json::to_vec(&serde_json::json!([entry])).unwrap())
                } else if path.starts_with(&format!("/api/models/{repo}")) {
                    Some(serde_json::to_vec(&entry).unwrap())
                } else {
                    let file = path.replace("/resolve/main/", &format!("/resolve/{revision}/"));
                    files
                        .iter()
                        .find(|(p, _)| *p == file)
                        .map(|(_, b)| b.clone())
                };
                let (status, body) = match found {
                    Some(body) => ("200 OK", body),
                    None => ("404 Not Found", b"Entry not found".to_vec()),
                };
                let mut stream = stream;
                let _ = stream.write_all(
                    format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: \
                         {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                );
                let _ = stream.write_all(&body);
            }
        });
        ScriptedHub { url, requests }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

/// A `loom-fetch` that installs what the card declares (the card, its config
/// and measurements; never deployment.json or devices.json) and reports the
/// pinned revision the way the real one does, per file transport.
fn fake_fetch(root: &Path, card_from: &Path) -> PathBuf {
    let script = root.join("loom-fetch");
    std::fs::write(
        &script,
        format!(
            r#"#!/bin/sh
out=""
for arg in "$@"; do
  case "$arg" in --output=*) out="${{arg#--output=}}";; esac
done
printf '%s\n' "$@" > '{argv}'
mkdir -p "$out"
for name in model.json llm_config.json info.json; do
  [ -f '{source}'/$name ] && cp '{source}'/$name "$out/$name"
done
echo "loom-fetch  Complete  3/3 files" >&2
cat <<'JSON'
{{"model": "qwen3.8-27b", "preset": "chat", "files": [{{"path": "model.json", "transport": {{"kind": "hf", "repository": "{repo}", "revision": "{revision}"}}}}]}}
JSON
"#,
            argv = root.join("fetch-argv.txt").display(),
            source = card_from.display(),
            repo = HYBRID_REPO,
            revision = HYBRID_REVISION,
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    script
}

fn fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture("hybrid-publish").join(name)).unwrap()
}

#[test]
fn an_installed_hub_scenario_carries_its_own_documents_and_serves_them() {
    let root = workspace("scenario-hub-install");
    let state = root.join("state");
    let (serve, argv) = recording_shim(&root);
    let fetch = fake_fetch(&root, &fixture("hybrid-publish"));
    let hub = ScriptedHub::start(
        HYBRID_REPO,
        HYBRID_REVISION,
        [
            "model.json",
            "info.json",
            "llm_config.json",
            "deployment.json",
            "devices.json",
        ]
        .into_iter()
        .map(|name| (name, fixture_bytes(name)))
        .collect(),
    );
    let output = run(launcher(&state, &hub.url, &[])
        .env("HF_HUB_CACHE", root.join("empty-hf-cache"))
        .args([
            "install",
            HYBRID_REPO,
            "--revision",
            HYBRID_REVISION,
            "--materialize",
            "share",
            "--loom-fetch",
            fetch.to_str().unwrap(),
        ]));
    assert_eq!(output.status.code(), Some(0));
    let installed = state.join("models").join("ilintar-Qwen3.8-27B-hybrid-loom");
    let text = stdout(&output);
    assert!(
        text.contains(&format!(
            "  scenario  deployment qwen3.8-27b on amdgpu + xdna (deployment.json + \
             devices.json from the hub at {HYBRID_REVISION})\n"
        )),
        "{text}"
    );
    assert!(!stderr(&output).contains("warning"), "{}", stderr(&output));
    // Both documents are the published ones, next to the installed card.
    for name in ["deployment.json", "devices.json"] {
        assert_eq!(
            std::fs::read(installed.join(name)).unwrap(),
            fixture_bytes(name)
        );
    }
    assert!(hub.requests().contains(&format!(
        "GET /{HYBRID_REPO}/resolve/{HYBRID_REVISION}/deployment.json HTTP/1.1"
    )));
    assert!(recorded_argv(&root.join("fetch-argv.txt"))
        .contains(&format!("huggingface:{HYBRID_REPO}@{HYBRID_REVISION}")));
    let record = &read(&state.join("installs.json"))["installs"][HYBRID_REPO];
    assert_eq!(record["scenario"], true);
    assert_eq!(record["scenario_revision"], HYBRID_REVISION);
    assert_eq!(record["revision"], HYBRID_REVISION);
    assert_eq!(record["path"], installed.to_str().unwrap());

    // `show` of the installed name explains the installed scenario.
    let output = run(&mut launcher(&state, &hub.url, &["show", HYBRID_REPO]));
    assert!(stdout(&output).contains("scenario     deployment qwen3.8-27b on amdgpu + xdna"));
    let output = run(&mut launcher(
        &state,
        &hub.url,
        &["--json", "show", HYBRID_REPO],
    ));
    assert_eq!(json(&output)["scenario"]["devices"], "amdgpu + xdna");
    let output = run(&mut launcher(
        &state,
        &hub.url,
        &["--json", "search", "hybrid"],
    ));
    let row = json(&output)["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["repo"] == HYBRID_REPO)
        .cloned()
        .unwrap();
    assert_eq!(row["installed"], true);
    assert_eq!(row["scenario"], true);
    assert_eq!(row["scenario_devices"], "amdgpu + xdna");

    // `serve` runs the scenario: the installed documents, no --device.
    let port = free_port();
    let port_text = port.to_string();
    let output = run(&mut launcher(
        &state,
        &hub.url,
        &[
            "serve",
            HYBRID_REPO,
            "--loom-serve",
            serve.to_str().unwrap(),
            "--port",
            &port_text,
            "--wait",
            "60",
            "--detach",
        ],
    ));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        recorded_argv(&argv),
        [
            format!("--devices={}/devices.json", installed.display()),
            format!("--model={}/deployment.json", installed.display()),
            format!("--listen=127.0.0.1:{port}"),
        ]
    );
    let current = read(&state.join("current.json"));
    assert_eq!(current["kind"], "scenario");
    assert_eq!(current["device"], "amdgpu + xdna");
    let endpoint = current["endpoint"].as_str().unwrap().to_string();
    let models = get(&format!("{endpoint}/api/models"));
    let listed = models["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["repo"] == HYBRID_REPO)
        .cloned()
        .unwrap();
    assert_eq!(listed["scenario"], true);
    assert_eq!(listed["scenario_devices"], "amdgpu + xdna");
    assert_eq!(listed["running"], true);
    let installs = get(&format!("{endpoint}/api/installs"));
    assert_eq!(installs["installs"][0]["scenario"], true);
    assert_eq!(
        run(&mut launcher(&state, &hub.url, &["stop", "--all"]))
            .status
            .code(),
        Some(0)
    );
}

#[test]
fn a_profile_without_scenario_documents_installs_as_before() {
    let root = workspace("scenario-hub-plain");
    let state = root.join("state");
    let plain = plain_profile(&root.join("published"));
    let fetch = fake_fetch(&root, &plain);
    let hub = ScriptedHub::start(
        HYBRID_REPO,
        HYBRID_REVISION,
        vec![(
            "model.json",
            std::fs::read(plain.join("model.json")).unwrap(),
        )],
    );
    let output = run(launcher(&state, &hub.url, &[])
        .env("HF_HUB_CACHE", root.join("empty-hf-cache"))
        .args([
            "install",
            HYBRID_REPO,
            "--loom-fetch",
            fetch.to_str().unwrap(),
        ]));
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        stderr(&output).matches("warning").count(),
        0,
        "{}",
        stderr(&output)
    );
    assert!(
        !stdout(&output).contains("  scenario  "),
        "{}",
        stdout(&output)
    );
    let installed = state.join("models").join("ilintar-Qwen3.8-27B-hybrid-loom");
    assert!(installed.join("model.json").is_file());
    assert!(!installed.join("deployment.json").exists());
    assert!(!installed.join("devices.json").exists());
    let record = read(&state.join("installs.json"))["installs"][HYBRID_REPO].clone();
    let mut keys: Vec<String> = record.as_object().unwrap().keys().cloned().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "installed_at",
            "path",
            "preset",
            "repo",
            "revision",
            "size_bytes"
        ]
    );
    // Asked once, at the pinned revision, then left alone.
    let asked: Vec<String> = hub
        .requests()
        .into_iter()
        .filter(|r| r.contains(".json HTTP") && r.contains("/devices.json"))
        .collect();
    assert_eq!(
        asked,
        [format!(
            "GET /{HYBRID_REPO}/resolve/{HYBRID_REVISION}/devices.json HTTP/1.1"
        )]
    );
    let output = run(&mut launcher(
        &state,
        &hub.url,
        &["--json", "show", HYBRID_REPO],
    ));
    assert_eq!(json(&output)["scenario"], Value::Null);
}

#[test]
fn a_hub_failure_after_the_fetch_keeps_the_install_and_says_so() {
    let root = workspace("scenario-hub-down");
    let state = root.join("state");
    let fetch = fake_fetch(&root, &fixture("hybrid-publish"));
    let output = run(launcher(&state, DEAD_HUB, &[])
        .env("HF_HUB_CACHE", root.join("empty-hf-cache"))
        .args([
            "install",
            HYBRID_REPO,
            "--loom-fetch",
            fetch.to_str().unwrap(),
        ]));
    assert_eq!(output.status.code(), Some(0));
    let warning = stderr(&output);
    assert!(
        warning.contains(&format!(
            "warning: could not check {HYBRID_REPO}@{HYBRID_REVISION} for deployment.json and \
             devices.json: cannot reach {DEAD_HUB}/{HYBRID_REPO}/resolve/{HYBRID_REVISION}/\
             deployment.json"
        )),
        "{warning}"
    );
    assert!(
        warning.contains("The install itself succeeded"),
        "{warning}"
    );
    let record = read(&state.join("installs.json"))["installs"][HYBRID_REPO].clone();
    assert!(record.get("scenario").is_none());

    // --offline never asks the hub; documents in the local hub cache are used.
    let cache = root.join("hf-cache");
    let snapshot = cache
        .join("models--ilintar--Qwen3.8-27B-hybrid-loom")
        .join("snapshots")
        .join(HYBRID_REVISION);
    std::fs::create_dir_all(&snapshot).unwrap();
    for name in ["deployment.json", "devices.json"] {
        std::fs::write(snapshot.join(name), fixture_bytes(name)).unwrap();
    }
    let hub = RecordingHub::start();
    let output = run(launcher(&state, &hub.url, &[])
        .env("HF_HUB_CACHE", &cache)
        .args([
            "install",
            HYBRID_REPO,
            "--offline",
            "--output",
            root.join("offline").to_str().unwrap(),
            "--loom-fetch",
            fetch.to_str().unwrap(),
        ]));
    assert_eq!(output.status.code(), Some(0));
    assert!(hub.requests().is_empty(), "{:?}", hub.requests());
    assert!(recorded_argv(&root.join("fetch-argv.txt")).contains(&"--offline".to_string()));
    assert!(stdout(&output).contains("from the local hub cache at"));
    let record = read(&state.join("installs.json"))["installs"][HYBRID_REPO].clone();
    assert_eq!(record["scenario"], true);
    assert!(root.join("offline").join("devices.json").is_file());
}
