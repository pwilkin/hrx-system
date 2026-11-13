//! The `loom-launch` command line.
//!
//! `search` finds published Loom profiles, `install` acquires one, `show`
//! explains one. `launcher start|status|stop` runs the launcher service: one
//! instrumented OpenAI endpoint and its dashboard that stay up with or without
//! a model. `start` loads a model under it (starting it first if needed),
//! `unload` and `stop` unload the model and leave the launcher serving, and
//! `serve` is the one-shot form (launcher and model together). `chat` talks to
//! a model and `bench` measures one. `status`, `logs`, `hardware` and `doctor`
//! observe and diagnose; `dashboard` runs a launcher in the foreground.
//!
//! Exit codes: 0 success, 1 not found / not running / refused, 2 usage.

use crate::bench::{format_metrics, format_published, Runner};
use crate::chat::{describe_turn, one_shot, run_repl, stdin_line, ChatSession};
use crate::client::Client;
use crate::dashboard::Service;
use crate::deployment::{context_settings, expand_user, ContextRequest};
use crate::errors::{Error, ErrorKind, Result};
use crate::hardware::{detect, evaluate, Compatibility, Hardware};
use crate::hub::{Hub, HubModel, SearchQuery, PROFILE_TAG};
use crate::install::{
    default_models_directory, directory_size, fetch_command, install_and_record, InstallOptions,
    InstallResult,
};
use crate::launcher::{
    public_record, running_launcher, stop_launcher, stop_legacy, Launcher, LauncherOptions,
    LauncherStop, Remote, DETACHED_ENV,
};
use crate::local::{self, is_local_install, LocalProfile};
use crate::profile::{Profile, ProfileDocuments, CARD_DOCUMENT};
use crate::scenario::Scenario;
use crate::server::{
    describe_state, plan_launch, plan_scenario, read_current, running_endpoint, serve_command,
    LaunchOptions, LaunchPlan, StopOutcome, Supervisor,
};
use crate::signals;
use crate::store::{
    state_directory, BenchmarkStore, InstallRegistry, InstalledModel, ProfileCache, RunStore,
};
use crate::values::{
    arr, dumps_pretty, get, grouped, human_bytes, human_rate, obj, py_str, str_value, strict_int,
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DEFAULT_PORT: u16 = 8080;

#[derive(Parser, Debug)]
#[command(
    name = "loom-launch",
    version,
    about = "Search, install, serve and benchmark published Loom profiles behind an \
             instrumented OpenAI-compatible endpoint."
)]
pub struct Cli {
    /// Launcher state directory (default $LOOM_LAUNCH_STATE or ~/.local/state/loom-launch)
    #[arg(long, global = true, value_name = "DIR")]
    pub state: Option<String>,
    /// Model hub endpoint (default $LOOM_HUB_ENDPOINT or https://huggingface.co)
    #[arg(long, global = true, value_name = "URL")]
    pub hub: Option<String>,
    /// Emit machine-readable JSON
    #[arg(long, global = true)]
    pub json: bool,
    /// Directory of local profiles: every directory under it holding a
    /// model.json is listed and servable without the hub (repeatable; also
    /// $LOOM_LAUNCH_PROFILES, a path list)
    #[arg(long = "profiles", global = true, value_name = "DIR")]
    pub profiles: Vec<String>,
    #[command(subcommand)]
    pub command: Option<Verb>,
}

#[derive(Subcommand, Debug)]
pub enum Verb {
    /// Find published Loom profiles and local ones
    Search(SearchArgs),
    /// Explain a profile
    Show(ShowArgs),
    /// Install a published profile, or register a local package directory
    Install(InstallArgs),
    /// Run the launcher service: one endpoint and dashboard, with or without a model
    #[command(subcommand)]
    Launcher(LauncherVerb),
    /// Load a model under the launcher (starting the launcher if none is running)
    Start(StartArgs),
    /// Unload the model; the launcher keeps serving
    Unload(UnloadArgs),
    /// Serve a profile in one step: a launcher and its model together
    Serve(ServeArgs),
    /// Talk to a served profile
    Chat(ChatArgs),
    /// Benchmark a profile
    Bench(BenchArgs),
    /// Report the launcher and the loaded model
    Status,
    /// Show a launch's server log
    Logs(LogsArgs),
    /// Unload the model (the launcher keeps running; --all stops it too)
    Stop(StopArgs),
    /// Describe this machine
    Hardware,
    /// Check the launcher's environment
    Doctor(DoctorArgs),
    /// Run a launcher in the foreground on the dashboard port (no model needed)
    Dashboard(DashboardArgs),
    /// A GPU-free stand-in for loom-serve (tests and rehearsals)
    #[command(hide = true)]
    FakeServe {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        arguments: Vec<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum LauncherVerb {
    /// Start the launcher service (it needs no model)
    Start(LauncherStartArgs),
    /// Report the launcher service
    Status,
    /// Stop the launcher service, unloading its model
    Stop(LauncherStopArgs),
}

/// The launcher's listener and the tools it runs models with.
#[derive(Args, Debug, Clone)]
pub struct ListenerArgs {
    /// Address the launcher's endpoint and dashboard listen on (0.0.0.0 for
    /// the local network); model servers always stay on 127.0.0.1
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Native listener port of the models loaded under the launcher
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// The launcher's client-facing port (0 = ephemeral)
    #[arg(long, default_value_t = 0)]
    pub proxy_port: u16,
    /// Require this bearer token from clients
    #[arg(long, env = "LOOM_LAUNCH_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Let the dashboard chat's web_fetch tool fetch public http(s) URLs for every model
    #[arg(long)]
    pub allow_web: bool,
    /// Path to the native server
    #[arg(long)]
    pub loom_serve: Option<String>,
    /// Path to loom-fetch, for installs started from the page
    #[arg(long)]
    pub loom_fetch: Option<String>,
}

#[derive(Args, Debug)]
pub struct LauncherStartArgs {
    #[command(flatten)]
    pub listener: ListenerArgs,
    /// With --detach: seconds to wait for the launcher to answer
    #[arg(long, default_value_t = 30.0)]
    pub wait: f64,
    /// Run in the background, detached from this shell; `launcher stop` ends it
    #[arg(long)]
    pub detach: bool,
}

#[derive(Args, Debug)]
pub struct LauncherStopArgs {
    /// Seconds the launcher gets to unload its model and exit before SIGKILL
    #[arg(long, default_value_t = 45.0)]
    pub wait: f64,
}

#[derive(Args, Debug)]
pub struct StartArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Context length in tokens
    #[arg(long)]
    pub context: Option<i64>,
    /// Default output token limit
    #[arg(long)]
    pub max_tokens: Option<i64>,
    /// Resident sessions (rows and max_sessions)
    #[arg(long)]
    pub parallel: Option<i64>,
    /// Speculative draft depth, 0 to disable
    #[arg(long)]
    pub draft_depth: Option<i64>,
    /// Named checkpoints
    #[arg(long)]
    pub checkpoint_capacity: Option<i64>,
    /// Retained idle histories
    #[arg(long)]
    pub session_cache: Option<i64>,
    /// Retained input prefixes
    #[arg(long)]
    pub prefix_cache: Option<i64>,
    /// Packed prefill shape
    #[arg(long)]
    pub prefill_capacity: Option<i64>,
    /// HAL device URI (default: detected)
    #[arg(long)]
    pub device: Option<String>,
    /// Server worker threads
    #[arg(long)]
    pub workers: Option<i64>,
    /// fixed or elastic pool backing
    #[arg(long)]
    pub pool_backing: Option<String>,
    /// Pool memory budget (default: the card's memory_bytes)
    #[arg(long)]
    pub memory_bytes: Option<i64>,
    /// Let the dashboard chat's web_fetch tool fetch public URLs while this model serves
    #[arg(long)]
    pub allow_web: bool,
    /// Seconds to wait for model preparation
    #[arg(long, default_value_t = 1800.0)]
    pub wait: f64,
    /// Refuse to acquire a profile that is not installed
    #[arg(long)]
    pub no_install: bool,
    /// Native listener port for this model (default: the launcher's --port)
    #[arg(long)]
    pub port: Option<u16>,
    /// Address the launcher listens on, when this command has to start it
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// The launcher's client-facing port, when this command has to start it
    #[arg(long, default_value_t = 0)]
    pub proxy_port: u16,
    /// API key the launcher requires, when this command has to start it
    #[arg(long, env = "LOOM_LAUNCH_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Path to the native server, when this command has to start the launcher
    #[arg(long)]
    pub loom_serve: Option<String>,
}

#[derive(Args, Debug)]
pub struct UnloadArgs {
    /// Seconds to wait for the model to drain before SIGKILL
    #[arg(long, default_value_t = 30.0)]
    pub wait: f64,
}

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Name fragment, or an owner prefix ("loom-fun/")
    pub query: Option<String>,
    /// Restrict to one hub owner
    #[arg(long)]
    pub author: Option<String>,
    /// Require an additional hub tag (repeatable)
    #[arg(long = "tag")]
    pub tags: Vec<String>,
    /// Maximum profiles to show
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
    /// List every match, not only installable profiles
    #[arg(long)]
    pub no_filter: bool,
    /// Skip reading each profile's documents (quantization, context, benchmarks)
    #[arg(long)]
    pub no_details: bool,
    /// Use cached profile documents only (no network)
    #[arg(long)]
    pub offline: bool,
    /// Seconds to wait for each hub request
    #[arg(long, default_value_t = 60.0)]
    pub wait: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub enum OnOff {
    On,
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub enum Think {
    On,
    Off,
    Auto,
}

impl Think {
    fn as_option(self) -> Option<bool> {
        match self {
            Think::On => Some(true),
            Think::Off => Some(false),
            Think::Auto => None,
        }
    }
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    /// owner/name, a local package directory, or a path to its card
    pub target: String,
    /// Render this user prompt through the package's chat policy
    #[arg(long)]
    pub prompt: Option<String>,
    /// System prompt for --prompt
    #[arg(long)]
    pub system: Option<String>,
    /// Thinking control for --prompt
    #[arg(long, value_enum)]
    pub thinking: Option<OnOff>,
    /// Print the profile document
    #[arg(long)]
    pub raw: bool,
    /// Pin a branch, tag or commit
    #[arg(long)]
    pub revision: Option<String>,
    /// Use cached profile documents only
    #[arg(long)]
    pub offline: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub enum Materialize {
    Auto,
    Copy,
    Share,
}

#[derive(Args, Debug)]
pub struct InstallArgs {
    /// owner/name, or a hub URL; with --from-path, the name to register under
    /// (default local/<directory name>)
    #[arg(required_unless_present = "from_path")]
    pub target: Option<String>,
    /// Make a package directory on this machine visible without the hub:
    /// registered in place (default) or copied with --copy
    #[arg(long, value_name = "DIR")]
    pub from_path: Option<String>,
    /// With --from-path: copy the package into the state directory's models/
    /// (GGUF files are linked, not copied) instead of registering it in place
    #[arg(long, requires = "from_path")]
    pub copy: bool,
    /// Installation directory
    #[arg(long)]
    pub output: Option<String>,
    /// Named card preset to install
    #[arg(long)]
    pub preset: Option<String>,
    /// How installs treat programs and weights
    #[arg(long, value_enum, default_value = "copy")]
    pub materialize: Materialize,
    /// Pin a branch, tag or commit
    #[arg(long)]
    pub revision: Option<String>,
    /// Path to loom-fetch
    #[arg(long)]
    pub loom_fetch: Option<String>,
    /// Forbid network requests: loom-fetch runs --offline, and a profile's
    /// deployment.json/devices.json come only from next to the card or the
    /// local hub cache
    #[arg(long)]
    pub offline: bool,
}

#[derive(Args, Debug, Clone)]
pub struct ProfileArgs {
    /// owner/name, a local package directory (or its card), a local profile's
    /// name, or nothing for the last install
    pub target: Option<String>,
    /// Pin a branch, tag or commit
    #[arg(long)]
    pub revision: Option<String>,
    /// Card preset to select
    #[arg(long)]
    pub preset: Option<String>,
    /// Use cached profile documents only
    #[arg(long)]
    pub offline: bool,
    /// Path to loom-fetch (when the profile must be installed first)
    #[arg(long)]
    pub loom_fetch: Option<String>,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    /// Context length in tokens
    #[arg(long)]
    pub context: Option<i64>,
    /// Default output token limit
    #[arg(long)]
    pub max_tokens: Option<i64>,
    /// Resident sessions (rows and max_sessions)
    #[arg(long)]
    pub parallel: Option<i64>,
    /// Speculative draft depth, 0 to disable
    #[arg(long)]
    pub draft_depth: Option<i64>,
    /// Named checkpoints
    #[arg(long)]
    pub checkpoint_capacity: Option<i64>,
    /// Retained idle histories
    #[arg(long)]
    pub session_cache: Option<i64>,
    /// Retained input prefixes
    #[arg(long)]
    pub prefix_cache: Option<i64>,
    /// Packed prefill shape
    #[arg(long)]
    pub prefill_capacity: Option<i64>,
    /// HAL device URI (default: detected)
    #[arg(long)]
    pub device: Option<String>,
    /// Path to the native server
    #[arg(long)]
    pub loom_serve: Option<String>,
    /// Address the proxy and dashboard listen on (0.0.0.0 for the local
    /// network); the native listener always stays on 127.0.0.1
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Native listener port
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// Server worker threads
    #[arg(long)]
    pub workers: Option<i64>,
    /// fixed or elastic pool backing
    #[arg(long)]
    pub pool_backing: Option<String>,
    /// Pool memory budget (default: the card's memory_bytes)
    #[arg(long)]
    pub memory_bytes: Option<i64>,
    /// Serve the native endpoint without the instrumentation proxy
    #[arg(long)]
    pub no_proxy: bool,
    /// Instrumentation proxy port (0 = ephemeral)
    #[arg(long, default_value_t = 0)]
    pub proxy_port: u16,
    /// Require this bearer token from clients
    #[arg(long, env = "LOOM_LAUNCH_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Let the dashboard chat's web_fetch tool fetch public http(s) URLs
    #[arg(long)]
    pub allow_web: bool,
    /// Seconds to wait for model preparation
    #[arg(long, default_value_t = 1800.0)]
    pub wait: f64,
    /// Run the server in the background and report its endpoint
    #[arg(long)]
    pub detach: bool,
    /// Refuse to acquire a profile that is not installed
    #[arg(long)]
    pub no_install: bool,
}

#[derive(Args, Debug, Clone)]
pub struct EndpointArgs {
    /// Talk to an existing endpoint instead of launching one
    #[arg(long)]
    pub base_url: Option<String>,
    /// Bearer token for the endpoint
    #[arg(long, env = "LOOM_LAUNCH_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Per-request timeout in seconds
    #[arg(long, default_value_t = 900.0)]
    pub request_timeout: f64,
    /// Leave a launch started for this command running afterwards (detached)
    #[arg(long)]
    pub keep_server: bool,
    /// Context length in tokens (for a launch this command starts)
    #[arg(long)]
    pub context: Option<i64>,
    /// Resident sessions (for a launch this command starts)
    #[arg(long)]
    pub parallel: Option<i64>,
    /// HAL device URI
    #[arg(long)]
    pub device: Option<String>,
    /// Path to the native server
    #[arg(long)]
    pub loom_serve: Option<String>,
    /// Native listener port
    #[arg(long, default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// Server worker threads
    #[arg(long)]
    pub workers: Option<i64>,
    /// Serve the native endpoint without the instrumentation proxy
    #[arg(long)]
    pub no_proxy: bool,
    /// Seconds to wait for preparation
    #[arg(long, default_value_t = 1800.0)]
    pub wait: f64,
}

#[derive(Args, Debug)]
pub struct ChatArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    #[command(flatten)]
    pub endpoint: EndpointArgs,
    /// Send one prompt and exit
    #[arg(long)]
    pub prompt: Option<String>,
    /// System prompt
    #[arg(long)]
    pub system: Option<String>,
    /// Output token limit per turn
    #[arg(long)]
    pub max_tokens: Option<i64>,
    /// Sampling temperature
    #[arg(long)]
    pub temperature: Option<f64>,
    /// Request or suppress the package's reasoning block
    #[arg(long, value_enum, default_value = "auto")]
    pub think: Think,
    /// Named retained session (X-Loom-Session)
    #[arg(long)]
    pub session: Option<String>,
    /// JSON object passed to the package's chat policy (chat_template_kwargs)
    #[arg(long)]
    pub template_kwargs: Option<String>,
    /// Write the transcript here on exit
    #[arg(long)]
    pub save: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
pub enum Suite {
    Quick,
    Standard,
    Full,
}

impl Suite {
    fn name(self) -> &'static str {
        match self {
            Suite::Quick => "quick",
            Suite::Standard => "standard",
            Suite::Full => "full",
        }
    }
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[command(flatten)]
    pub profile: ProfileArgs,
    #[command(flatten)]
    pub endpoint: EndpointArgs,
    /// Metric set to measure
    #[arg(long, value_enum, default_value = "standard")]
    pub suite: Suite,
    /// Samples per metric (median reported)
    #[arg(long, default_value_t = 1)]
    pub repeats: usize,
    /// Thinking control during measurement
    #[arg(long, value_enum, default_value = "off")]
    pub think: Think,
    /// Named retained session (X-Loom-Session) for every request
    #[arg(long)]
    pub session: Option<String>,
    /// List stored runs
    #[arg(long)]
    pub list: bool,
    /// Print one stored run by id
    #[arg(long, value_name = "ID")]
    pub show: Option<String>,
    /// Do not keep the result
    #[arg(long)]
    pub no_store: bool,
}

#[derive(Args, Debug)]
pub struct LogsArgs {
    /// Lines to show
    #[arg(long, default_value_t = 40)]
    pub lines: usize,
    /// Run id (default: the running launch, else the latest run)
    #[arg(long)]
    pub run: Option<String>,
    /// Follow the log
    #[arg(long)]
    pub follow: bool,
    /// Show events.jsonl instead of serve.log
    #[arg(long)]
    pub events: bool,
}

#[derive(Args, Debug)]
pub struct StopArgs {
    /// Seconds to wait after SIGTERM before SIGKILL
    #[arg(long, default_value_t = 30.0)]
    pub wait: f64,
    /// Stop the launcher too (same as `launcher stop`)
    #[arg(long)]
    pub all: bool,
}

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Path to the native server
    #[arg(long)]
    pub loom_serve: Option<String>,
    /// Path to loom-fetch
    #[arg(long)]
    pub loom_fetch: Option<String>,
}

#[derive(Args, Debug)]
pub struct DashboardArgs {
    /// Port the launcher's endpoint and dashboard listen on
    #[arg(long, default_value_t = 8090)]
    pub port: u16,
    /// Address to listen on (0.0.0.0 for the local network); model servers
    /// loaded from the page stay on 127.0.0.1
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Let the dashboard chat's web_fetch tool fetch public http(s) URLs
    #[arg(long)]
    pub allow_web: bool,
    /// Path to the native server, for launches started from the page
    #[arg(long)]
    pub loom_serve: Option<String>,
    /// Path to loom-fetch, for installs started from the page
    #[arg(long)]
    pub loom_fetch: Option<String>,
}

/// Global context every verb shares.
pub struct Ctx {
    pub state: PathBuf,
    pub hub_endpoint: Option<String>,
    pub json: bool,
    /// Directories local profiles are listed from (`--profiles`, `$LOOM_LAUNCH_PROFILES`).
    pub profile_roots: Vec<PathBuf>,
}

impl Ctx {
    pub(crate) fn hub(&self, timeout: Duration) -> Hub {
        Hub::new(self.hub_endpoint.as_deref(), timeout)
    }

    /// Human progress goes to stderr when stdout carries JSON.
    fn say(&self, line: &str) {
        if self.json {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
    }
}

/// Runs the command line; returns the process exit code.
pub fn main() -> i32 {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return error.exit_code();
        }
    };
    run(cli)
}

pub fn run(cli: Cli) -> i32 {
    let Some(command) = cli.command else {
        use clap::CommandFactory;
        let _ = Cli::command().print_help();
        println!();
        return 2;
    };
    let ctx = Ctx {
        state: state_directory(cli.state.as_deref()),
        hub_endpoint: cli.hub.clone(),
        json: cli.json,
        profile_roots: local::profile_roots(&cli.profiles),
    };
    let outcome = match command {
        Verb::Search(args) => cmd_search(&ctx, &args),
        Verb::Show(args) => cmd_show(&ctx, &args),
        Verb::Install(args) => cmd_install(&ctx, &args),
        Verb::Launcher(LauncherVerb::Start(args)) => cmd_launcher_start(&ctx, &args),
        Verb::Launcher(LauncherVerb::Status) => cmd_launcher_status(&ctx),
        Verb::Launcher(LauncherVerb::Stop(args)) => cmd_launcher_stop(&ctx, args.wait),
        Verb::Start(args) => cmd_start(&ctx, &args),
        Verb::Unload(args) => cmd_unload(&ctx, args.wait),
        Verb::Serve(args) => cmd_serve(&ctx, &args),
        Verb::Chat(args) => cmd_chat(&ctx, &args),
        Verb::Bench(args) => cmd_bench(&ctx, &args),
        Verb::Status => cmd_status(&ctx),
        Verb::Logs(args) => cmd_logs(&ctx, &args),
        Verb::Stop(args) => cmd_stop(&ctx, &args),
        Verb::Hardware => cmd_hardware(&ctx),
        Verb::Doctor(args) => cmd_doctor(&ctx, &args),
        Verb::Dashboard(args) => cmd_dashboard(&ctx, &args),
        Verb::FakeServe { arguments } => Ok(crate::testing::fake_serve_main(&arguments)),
    };
    match outcome {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

fn print_json(value: &Value) {
    println!("{}", dumps_pretty(value));
}

// -- profile resolution -----------------------------------------------------

/// A profile plus the local card a launch can serve.
pub(crate) struct Resolved {
    pub(crate) profile: Profile,
    pub(crate) installed: InstalledModel,
}

impl Resolved {
    pub(crate) fn card_path(&self) -> PathBuf {
        PathBuf::from(&self.installed.path).join(CARD_DOCUMENT)
    }
}

/// A local card path or installation directory named on the command line.
fn local_target(target: &str) -> Option<PathBuf> {
    let candidate = expand_user(Path::new(target));
    if candidate.is_file() {
        return Some(candidate);
    }
    if candidate.is_dir() && candidate.join(CARD_DOCUMENT).is_file() {
        return Some(candidate);
    }
    None
}

/// Reads a profile from the hub, falling back to cached documents.
fn profile_from_repo(
    ctx: &Ctx,
    hub: &Hub,
    repo: &str,
    revision: Option<&str>,
    offline: bool,
    model: Option<HubModel>,
) -> Result<Profile> {
    let cache = ProfileCache::new(&ctx.state);
    if !offline {
        match Profile::documents_from_hub(hub, repo, revision, model) {
            Ok(documents) => {
                let _ = cache.put(repo, &documents.to_cache(repo));
                return Ok(Profile::from_documents(repo, &documents, None));
            }
            Err(error) if !error.is_hub() => return Err(error),
            Err(error) => {
                let Some(cached) = cache.get_any(repo) else {
                    return Err(Error::launcher(format!("cannot read {repo}: {error}")));
                };
                eprintln!("warning: {error}; using cached documents for {repo}");
                return Ok(from_cache(repo, &cached, revision));
            }
        }
    }
    match cache.get_any(repo) {
        Some(cached) => Ok(from_cache(repo, &cached, revision)),
        None => Err(Error::launcher(format!(
            "{repo} is not cached; run without --offline to read it from the hub"
        ))),
    }
}

pub(crate) fn from_cache(
    repo: &str,
    cached: &Map<String, Value>,
    revision: Option<&str>,
) -> Profile {
    let mut documents = ProfileDocuments::from_cache(cached);
    if let Some(revision) = revision {
        documents.revision = Some(revision.to_string());
    }
    Profile::from_documents(repo, &documents, None)
}

/// Adds the hub's sidecar documents (measurements, chat policy, README) to an
/// installed card. The installed `model.json` stays authoritative, since it is
/// the card the server will load.
fn enrich(
    ctx: &Ctx,
    profile: Profile,
    repo: &str,
    revision: Option<&str>,
    offline: bool,
) -> Profile {
    let cache = ProfileCache::new(&ctx.state);
    let mut documents = cache.get(repo).map(|d| ProfileDocuments::from_cache(&d));
    if documents.is_none() && !offline {
        let hub = ctx.hub(Duration::from_secs(30));
        match Profile::documents_from_hub(&hub, repo, revision, None) {
            Ok(fetched) => {
                let _ = cache.put(repo, &fetched.to_cache(repo));
                documents = Some(fetched);
            }
            Err(_) => documents = None,
        }
    }
    if documents.is_none() {
        documents = cache
            .get_any(repo)
            .map(|d| ProfileDocuments::from_cache(&d));
    }
    let Some(remote) = documents else {
        return profile;
    };
    let Some(path) = profile.path.clone() else {
        return Profile::from_documents(repo, &remote, None);
    };
    let Ok((_, local)) = Profile::local_documents(Path::new(&path)) else {
        return profile;
    };
    let pick = |local: &Map<String, Value>, remote: &Map<String, Value>| {
        if local.is_empty() {
            remote.clone()
        } else {
            local.clone()
        }
    };
    let merged = ProfileDocuments {
        model_card: pick(&local.model_card, &remote.model_card),
        info: pick(&local.info, &remote.info),
        llm_config: pick(&local.llm_config, &remote.llm_config),
        readme: local.readme.or(remote.readme),
        tags: remote.tags,
        revision: revision.map(str::to_string).or(remote.revision),
    };
    Profile::from_documents(repo, &merged, Some(path))
}

pub(crate) fn local_profile(
    ctx: &Ctx,
    installed: &InstalledModel,
    offline: bool,
) -> Result<Profile> {
    let profile = Profile::from_local(Path::new(&installed.path), Some(&installed.repo))?;
    // A package registered from a path has no hub documents to add, and
    // reading it never asks the hub.
    if is_local_install(installed) {
        return Ok(profile);
    }
    Ok(enrich(
        ctx,
        profile,
        &installed.repo,
        installed.revision.as_deref(),
        offline,
    ))
}

/// A profile for a local card path, enriched when the registry knows its repo.
fn profile_for_local(ctx: &Ctx, local: &Path, offline: bool) -> Result<(Profile, InstalledModel)> {
    let directory = if local.is_dir() {
        local.to_path_buf()
    } else {
        local.parent().unwrap_or(Path::new(".")).to_path_buf()
    };
    let registry = InstallRegistry::new(&ctx.state);
    match registry.find_by_path(&directory) {
        Some(installed) => Ok((local_profile(ctx, &installed, offline)?, installed)),
        None => {
            let profile = Profile::from_local(local, None)?;
            let installed = InstalledModel {
                repo: profile.repo.clone(),
                path: crate::deployment::absolute(&directory)
                    .to_string_lossy()
                    .into_owned(),
                source: Some("local".into()),
                ..Default::default()
            };
            Ok((profile, installed))
        }
    }
}

pub(crate) fn resolve(ctx: &Ctx, args: &ProfileArgs, install: bool) -> Result<Resolved> {
    let registry = InstallRegistry::new(&ctx.state);
    let Some(target) = args.target.as_deref() else {
        let installed = registry.latest().ok_or_else(|| {
            Error::launcher(
                "no target given and nothing installed; run `loom-launch search`, \
                 then `loom-launch install OWNER/NAME`",
            )
        })?;
        return Ok(Resolved {
            profile: local_profile(ctx, &installed, args.offline)?,
            installed,
        });
    };
    if let Some(local) = local_target(target) {
        let (profile, installed) = profile_for_local(ctx, &local, args.offline)?;
        return Ok(Resolved { profile, installed });
    }
    if let Some(installed) = registry.get(target) {
        if installed.card_path().is_file() {
            return Ok(Resolved {
                profile: local_profile(ctx, &installed, args.offline)?,
                installed,
            });
        }
    }
    if let Some((found, profile)) = local::find(&ctx.state, &ctx.profile_roots, target)? {
        return Ok(Resolved {
            installed: local_installed(&found, &profile),
            profile,
        });
    }
    if !target.contains('/') {
        return Err(Error::launcher(format!(
            "{target} is neither a local profile, an installed card nor an owner/name profile"
        )));
    }
    if !install {
        return Err(Error::launcher(format!(
            "{target} is not installed; run `loom-launch install {target}`"
        )));
    }
    let (result, installed) = install_target(
        ctx,
        target,
        &InstallOptions {
            preset: args.preset.clone(),
            revision: args.revision.clone(),
            explicit_fetch: args.loom_fetch.clone(),
            offline: args.offline,
            ..Default::default()
        },
    )?;
    if let Some(warning) = &result.documents.warning {
        eprintln!("warning: {warning}");
    }
    Ok(Resolved {
        profile: local_profile(ctx, &installed, args.offline)?,
        installed,
    })
}

/// The install record a local profile is served from.
fn local_installed(found: &LocalProfile, profile: &Profile) -> InstalledModel {
    found.registered.clone().unwrap_or_else(|| InstalledModel {
        repo: profile.repo.clone(),
        path: found.directory.to_string_lossy().into_owned(),
        source: Some("local".into()),
        ..Default::default()
    })
}

// -- launch planning ---------------------------------------------------------

/// How the caller spells launcher options in a refusal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Spelling {
    /// `--max-tokens`
    Flags,
    /// `max_tokens` (dashboard request fields)
    Fields,
}

/// What one launch will run: a deployment composed from the profile and the
/// request, or a prepared scenario exactly as its package declares it.
pub(crate) enum Prepared {
    Composed(crate::deployment::ContextSettings),
    Scenario(Box<Scenario>),
}

impl Prepared {
    pub(crate) fn settings(&self) -> &crate::deployment::ContextSettings {
        match self {
            Prepared::Composed(settings) => settings,
            Prepared::Scenario(scenario) => &scenario.settings,
        }
    }

    pub(crate) fn scenario(&self) -> Option<&Scenario> {
        match self {
            Prepared::Composed(_) => None,
            Prepared::Scenario(scenario) => Some(scenario),
        }
    }

    pub(crate) fn plan(
        &self,
        resolved: &Resolved,
        run: crate::store::RunDirectory,
        options: &LaunchOptions,
    ) -> Result<LaunchPlan> {
        match self {
            Prepared::Composed(settings) => plan_launch(
                &resolved.profile,
                &resolved.card_path(),
                settings,
                run,
                options,
                None,
            ),
            Prepared::Scenario(scenario) => {
                plan_scenario(&resolved.profile, scenario, run, options)
            }
        }
    }
}

/// The options of `request` (and `device`) a scenario would contradict.
fn contradicting(
    request: &ContextRequest,
    device: Option<&str>,
    spelling: Spelling,
) -> Vec<String> {
    let set = [
        ("context", request.context_length.is_some()),
        ("max_tokens", request.max_tokens.is_some()),
        ("parallel", request.parallel.is_some()),
        ("draft_depth", request.draft_depth.is_some()),
        ("checkpoint_capacity", request.checkpoint_capacity.is_some()),
        ("session_cache", request.session_cache_capacity.is_some()),
        ("prefix_cache", request.prefix_cache_capacity.is_some()),
        ("prefill_capacity", request.prefill_capacity.is_some()),
        ("cache_target_bytes", request.cache_target_bytes.is_some()),
        ("continuation_epochs", request.continuation_epochs.is_some()),
        ("pending_requests", request.pending_requests.is_some()),
        ("device", device.is_some_and(|d| !d.is_empty())),
    ];
    set.iter()
        .filter(|(_, present)| *present)
        .map(|(name, _)| match spelling {
            Spelling::Flags => format!("--{}", name.replace('_', "-")),
            Spelling::Fields => name.to_string(),
        })
        .collect()
}

/// Resolves what a launch of `resolved` will run, refusing options that
/// contradict a scenario before anything is started or written.
pub(crate) fn prepare(
    resolved: &Resolved,
    request: &ContextRequest,
    device: Option<&str>,
    spelling: Spelling,
) -> Result<Prepared> {
    if let Some(scenario) = Scenario::detect(Path::new(&resolved.installed.path))? {
        scenario.refuse(&contradicting(request, device, spelling))?;
        return Ok(Prepared::Scenario(Box::new(scenario)));
    }
    Ok(Prepared::Composed(context_settings(
        &resolved.profile,
        request,
    )?))
}

fn install_target(
    ctx: &Ctx,
    repo: &str,
    options: &InstallOptions,
) -> Result<(InstallResult, InstalledModel)> {
    let mut options = options.clone();
    if options.output.is_none() {
        options.output = Some(default_models_directory(&ctx.state).join(repo.replace('/', "-")));
    }
    let hub = ctx.hub(Duration::from_secs(5));
    if !hub.is_default_endpoint() {
        options.hub_endpoint = Some(hub.endpoint.clone());
    }
    let quiet = ctx.json;
    install_and_record(repo, Some(&ctx.state), &options, &move |line: &str| {
        if !quiet {
            eprintln!("  fetch: {line}");
        }
    })
}

// -- search ------------------------------------------------------------------

pub(crate) struct SearchRow {
    pub(crate) model: HubModel,
    pub(crate) profile: Option<Profile>,
    pub(crate) compat: Option<Compatibility>,
    pub(crate) error: Option<String>,
    /// Set for a profile on this machine (`source: "local"`).
    pub(crate) local: Option<LocalProfile>,
}

impl SearchRow {
    pub(crate) fn source(&self) -> &'static str {
        if self.local.is_some() {
            "local"
        } else {
            "hub"
        }
    }

    pub(crate) fn path(&self) -> Option<String> {
        self.local
            .as_ref()
            .map(|l| l.directory.to_string_lossy().into_owned())
    }

    /// Whether the launcher can serve it without installing: a local profile
    /// under the models directory or registered, or a hub profile with an
    /// install record.
    pub(crate) fn installed(&self, installs: &[InstalledModel]) -> bool {
        match &self.local {
            Some(local) => local.installed,
            None => installs.iter().any(|m| m.repo == self.model.repo),
        }
    }

    /// The installed scenario this row serves: a local one, or the install
    /// of a hub profile whose directory carries both documents.
    fn installed_scenario(&self, installs: &[InstalledModel]) -> Option<Scenario> {
        match &self.local {
            Some(local) => local.scenario(),
            None => installs
                .iter()
                .find(|m| m.repo == self.model.repo)
                .and_then(|m| Scenario::detect(Path::new(&m.path)).ok().flatten()),
        }
    }

    /// Whether it carries its own `deployment.json` and `devices.json`: an
    /// installed profile is judged by its directory, a hub one by its files.
    pub(crate) fn scenario(&self, installs: &[InstalledModel]) -> bool {
        match &self.local {
            Some(local) => local.scenario,
            None => {
                self.installed_scenario(installs).is_some()
                    || self.model.has_file(&[
                        crate::scenario::DEPLOYMENT_DOCUMENT,
                        crate::scenario::DEVICES_DOCUMENT,
                    ])
            }
        }
    }

    /// The devices an installed scenario spans (`amdgpu + xdna`).
    pub(crate) fn scenario_devices(&self, installs: &[InstalledModel]) -> Option<String> {
        self.installed_scenario(installs).map(|s| s.device_text())
    }
}

/// Local profiles matching a search, read from their own files only.
pub(crate) fn local_rows(
    ctx: &Ctx,
    hardware: &Hardware,
    query: Option<&str>,
    author: Option<&str>,
) -> Vec<SearchRow> {
    let query = query.unwrap_or("");
    let author = author
        .map(str::to_string)
        .or_else(|| query.strip_suffix('/').map(str::to_string));
    local::discover(&ctx.state, &ctx.profile_roots)
        .into_iter()
        .filter_map(|found| {
            let (profile, error) = match found.read() {
                Ok(profile) => (Some(profile), None),
                Err(error) => (None, Some(error.message)),
            };
            if let Some(profile) = &profile {
                if !local::matches_query(profile, &found.directory, query, author.as_deref()) {
                    return None;
                }
            } else if !query.is_empty() || author.is_some() {
                return None;
            }
            let repo = profile.as_ref().map(|p| p.repo.clone()).unwrap_or_else(|| {
                found
                    .directory
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            Some(SearchRow {
                model: HubModel {
                    repo,
                    ..Default::default()
                },
                compat: profile.as_ref().map(|p| evaluate(hardware, p)),
                profile,
                error,
                local: Some(found),
            })
        })
        .collect()
}

/// One search row as `search --json` prints it.
fn search_document(row: &SearchRow, installs: &[InstalledModel]) -> Value {
    let mut document = row.model.describe();
    document["source"] = json!(row.source());
    document["path"] = json!(row.path());
    document["installed"] = json!(row.installed(installs));
    document["scenario"] = json!(row.scenario(installs));
    document["scenario_devices"] = json!(row.scenario_devices(installs));
    document["profile"] = row
        .profile
        .as_ref()
        .map(Profile::describe)
        .unwrap_or(Value::Null);
    document["compatibility"] = row
        .compat
        .as_ref()
        .map(Compatibility::describe)
        .unwrap_or(Value::Null);
    document["error"] = json!(row.error);
    document
}

pub(crate) fn sort_rows(rows: &mut [SearchRow]) {
    rows.sort_by(|a, b| {
        let score = |row: &SearchRow| row.compat.as_ref().map(|c| c.score).unwrap_or(0.0);
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.model.repo.cmp(&b.model.repo))
            .then_with(|| a.path().cmp(&b.path()))
    });
}

fn cmd_search(ctx: &Ctx, args: &SearchArgs) -> Result<i32> {
    let hub = ctx.hub(Duration::from_secs_f64(args.wait.max(1.0)));
    let hardware = detect();
    let mut offline = args.offline;
    let mut warning = None;
    let found: Vec<(HubModel, Option<ProfileDocuments>)> = if offline {
        offline_profiles(ctx, args)
    } else {
        let listed = if args.no_filter {
            hub.search(&SearchQuery {
                query: args.query.clone(),
                author: args.author.clone(),
                tags: args.tags.clone(),
                limit: args.limit,
                with_files: true,
            })
        } else {
            let mut tags = vec![PROFILE_TAG.to_string()];
            tags.extend(args.tags.iter().cloned());
            hub.profiles(
                args.query.as_deref(),
                args.author.as_deref(),
                args.limit,
                &tags,
            )
        };
        match listed {
            Ok(models) => models.into_iter().map(|m| (m, None)).collect(),
            // A missing hub must not hide what is on this machine.
            Err(error) => {
                offline = true;
                warning = Some(format!(
                    "the hub did not answer ({error}); showing local and cached profiles"
                ));
                offline_profiles(ctx, args)
            }
        }
    };
    let mut rows = details(ctx, &hub, &hardware, found, !args.no_details, offline);
    rows.extend(local_rows(
        ctx,
        &hardware,
        args.query.as_deref(),
        args.author.as_deref(),
    ));
    sort_rows(&mut rows);
    let installs = InstallRegistry::new(&ctx.state).list();
    if let Some(warning) = &warning {
        eprintln!("warning: {warning}");
    }
    if ctx.json {
        let profiles: Vec<Value> = rows
            .iter()
            .map(|row| search_document(row, &installs))
            .collect();
        print_json(&json!({
            "query": args.query,
            "count": rows.len(),
            "offline": offline,
            "warning": warning,
            "hardware": hardware.describe(),
            "profiles": profiles,
        }));
        return Ok(0);
    }
    if rows.is_empty() {
        println!("no matching profiles");
        return Ok(1);
    }
    let header = format!(
        "{:<52} {:<9} {:>8} {:>9} {:>9} {:>5}",
        "profile", "quant", "ctx", "memory", "decode", "fit"
    );
    println!("{header}");
    println!("{}", "-".repeat(header.len()));
    for row in &rows {
        let profile = row.profile.as_ref();
        let decode = profile
            .and_then(|p| p.performance.as_ref())
            .and_then(|p| p.measurement("decode"))
            .and_then(|m| m.value)
            .filter(|v| *v != 0.0);
        println!(
            "{:<52} {:<9} {:>8} {:>9} {:>9} {:>5}",
            row.model.repo,
            profile
                .and_then(|p| p.quant.clone())
                .unwrap_or_else(|| "-".into()),
            profile
                .and_then(|p| p.context_length)
                .filter(|c| *c != 0)
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".into()),
            profile
                .map(|p| human_bytes(p.memory_bytes))
                .unwrap_or_else(|| "-".into()),
            decode.map(|d| grouped(d, 1)).unwrap_or_else(|| "-".into()),
            row.compat
                .as_ref()
                .map(|c| format!("{:.2}", c.score))
                .unwrap_or_else(|| "-".into()),
        );
        if let Some(path) = row.path() {
            let mut notes = vec![format!("local {path}")];
            if let Some(native) = profile
                .and_then(|p| p.native_context_length)
                .filter(|n| Some(*n) != profile.and_then(|p| p.context_length))
            {
                notes.push(format!("model native {native}"));
            }
            if let Some(devices) = row.scenario_devices(&installs) {
                notes.push(format!("scenario on {devices}"));
            }
            println!("  {}", notes.join("; "));
        }
        if let Some(error) = &row.error {
            println!("  ! {error}");
        }
    }
    println!();
    println!(
        "{} profile(s); `loom-launch install OWNER/NAME` acquires one, \
         `loom-launch show OWNER/NAME` explains it; `loom-launch serve DIR` serves a local one",
        rows.len()
    );
    Ok(0)
}

/// Reads each listed profile's documents, eight at a time.
pub(crate) fn details(
    ctx: &Ctx,
    hub: &Hub,
    hardware: &Hardware,
    found: Vec<(HubModel, Option<ProfileDocuments>)>,
    with_details: bool,
    offline: bool,
) -> Vec<SearchRow> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let slots: Vec<Mutex<Option<SearchRow>>> = found.iter().map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8.min(found.len().max(1)) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::SeqCst);
                let Some((model, documents)) = found.get(index) else {
                    break;
                };
                let mut row = SearchRow {
                    model: model.clone(),
                    profile: None,
                    compat: None,
                    error: None,
                    local: None,
                };
                if with_details {
                    let profile = match documents {
                        Some(documents) => {
                            Ok(Profile::from_documents(&model.repo, documents, None))
                        }
                        None => profile_from_repo(
                            ctx,
                            hub,
                            &model.repo,
                            model.revision.as_deref(),
                            offline,
                            Some(model.clone()),
                        ),
                    };
                    match profile {
                        Ok(profile) => {
                            row.compat = Some(evaluate(hardware, &profile));
                            row.profile = Some(profile);
                        }
                        Err(error) => row.error = Some(error.message),
                    }
                }
                *slots[index].lock().unwrap() = Some(row);
            });
        }
    });
    slots
        .into_iter()
        .filter_map(|slot| slot.into_inner().unwrap())
        .collect()
}

/// Cached profiles matching a search, without the network.
pub(crate) fn offline_profiles(
    ctx: &Ctx,
    args: &SearchArgs,
) -> Vec<(HubModel, Option<ProfileDocuments>)> {
    let registry = InstallRegistry::new(&ctx.state);
    let installs = registry.list();
    let query = args.query.clone().unwrap_or_default().to_lowercase();
    let author = args
        .author
        .clone()
        .or_else(|| query.strip_suffix('/').map(str::to_string));
    let mut found = Vec::new();
    for (stem, document) in ProfileCache::new(&ctx.state).entries() {
        let documents = ProfileDocuments::from_cache(&document);
        if documents.model_card.is_empty() {
            continue;
        }
        let repo = str_value(get(&document, "repo"))
            .map(str::to_string)
            .or_else(|| {
                installs
                    .iter()
                    .find(|m| m.repo.replace('/', "-") == stem)
                    .map(|m| m.repo.clone())
            })
            .unwrap_or_else(|| guess_repo(&stem, documents.readme.as_deref().unwrap_or("")));
        if let Some(author) = &author {
            if !repo
                .to_lowercase()
                .starts_with(&format!("{}/", author.to_lowercase()))
            {
                continue;
            }
        } else if !query.is_empty() && !repo.to_lowercase().contains(&query) {
            continue;
        }
        let model = HubModel {
            repo,
            revision: documents.revision.clone(),
            tags: documents.tags.clone(),
            ..Default::default()
        };
        found.push((model, Some(documents)));
        if found.len() >= args.limit {
            break;
        }
    }
    found
}

/// Recovers `owner/name` from a cache file stem (`owner-name`), preferring a
/// split the README mentions.
fn guess_repo(stem: &str, readme: &str) -> String {
    for (index, character) in stem.char_indices() {
        if character == '-' {
            let candidate = format!("{}/{}", &stem[..index], &stem[index + 1..]);
            if readme.contains(&candidate) {
                return candidate;
            }
        }
    }
    match stem.split_once('-') {
        Some((owner, name)) => format!("{owner}/{name}"),
        None => stem.to_string(),
    }
}

// -- show --------------------------------------------------------------------

/// What `show` explains: a hub profile, or a package on this machine.
fn shown_profile(ctx: &Ctx, args: &ShowArgs) -> Result<Profile> {
    if let Some(local) = local_target(&args.target) {
        return Ok(profile_for_local(ctx, &local, args.offline)?.0);
    }
    if let Some(installed) = InstallRegistry::new(&ctx.state)
        .get(&args.target)
        .filter(|m| m.card_path().is_file())
    {
        if is_local_install(&installed) {
            return local_profile(ctx, &installed, true);
        }
        // An installed hub profile is explained from what was installed (its
        // scenario documents included), unless another revision was asked for.
        let pinned = args.revision.as_deref();
        if pinned.is_none() || pinned == installed.revision.as_deref() {
            return local_profile(ctx, &installed, args.offline);
        }
    }
    if let Some((_, profile)) = local::find(&ctx.state, &ctx.profile_roots, &args.target)? {
        return Ok(profile);
    }
    let hub = ctx.hub(Duration::from_secs(60));
    profile_from_repo(
        ctx,
        &hub,
        &args.target,
        args.revision.as_deref(),
        args.offline,
        None,
    )
}

fn cmd_show(ctx: &Ctx, args: &ShowArgs) -> Result<i32> {
    let profile = shown_profile(ctx, args)?;
    let hardware = detect();
    let compat = evaluate(&hardware, &profile);
    if args.raw {
        print_json(&profile.describe());
        return Ok(0);
    }
    if ctx.json {
        let mut document = profile.describe();
        document["compatibility"] = compat.describe();
        document["scenario"] = json!(profile_scenario(&profile).map(|s| s.describe()));
        if let Some(prompt) = &args.prompt {
            document["rendered_prompt"] = json!(render_prompt(&profile, prompt, args).ok());
        }
        print_json(&document);
        return Ok(0);
    }
    print_profile(&profile, &compat);
    if let Some(prompt) = &args.prompt {
        match render_prompt(&profile, prompt, args) {
            Ok(rendered) => {
                println!("rendered prompt");
                println!("{}", "-".repeat(62));
                println!("{rendered}");
            }
            Err(error) => {
                eprintln!("error: {error}");
                return Ok(1);
            }
        }
    }
    Ok(0)
}

fn render_prompt(profile: &Profile, prompt: &str, args: &ShowArgs) -> Result<String> {
    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    messages.push(json!({"role": "user", "content": prompt}));
    let thinking = args.thinking.map(|t| t == OnOff::On);
    profile.render(&messages, thinking, None)
}

fn print_profile(profile: &Profile, compat: &Compatibility) {
    let revision = profile
        .revision
        .as_ref()
        .map(|r| format!(" @ {r}"))
        .unwrap_or_default();
    println!("{}{revision}", profile.repo);
    println!("  model        {}  ({})", profile.model_id, profile.name);
    println!(
        "  quant        {}",
        profile.quant.as_deref().unwrap_or("unknown")
    );
    println!(
        "  architecture {}",
        profile.architecture.as_deref().unwrap_or("unknown")
    );
    println!("  context      {}", profile.context_text());
    println!("  memory       {}", human_bytes(profile.memory_bytes));
    if let Some(device) = &profile.device_hint {
        println!("  measured on  {device}");
    }
    if let Some(scenario) = profile_scenario(profile) {
        println!(
            "  scenario     deployment {} on {} ({} + {})",
            scenario.name(),
            scenario.device_text(),
            crate::scenario::DEPLOYMENT_DOCUMENT,
            crate::scenario::DEVICES_DOCUMENT
        );
    }
    if let Some(execution) = &profile.execution {
        let dash = |v: Option<i64>| {
            v.filter(|x| *x != 0)
                .map(|x| x.to_string())
                .unwrap_or_else(|| "-".into())
        };
        println!(
            "  preset       {}  rows {}  prefill {}  max_tokens {}  draft {}",
            execution.preset.as_deref().unwrap_or("-"),
            dash(execution.rows),
            dash(execution.prefill_capacity),
            dash(execution.max_tokens),
            execution
                .draft_depth
                .map(|d| d.to_string())
                .unwrap_or_else(|| "-".into())
        );
    }
    if let Some(chat) = &profile.chat {
        let terminal = chat.terminal();
        println!(
            "  chat         thinking {}{}  tools {}  terminal {}",
            chat.thinking_mode(),
            if chat.default_thinking() {
                " (default on)"
            } else {
                ""
            },
            chat.tool_call_format().unwrap_or_else(|| "-".into()),
            if terminal.is_empty() {
                "-".to_string()
            } else {
                terminal.escape_debug().to_string()
            }
        );
    }
    if !profile.features.is_empty() {
        let mut enabled: Vec<&str> = profile
            .features
            .iter()
            .filter(|f| f.enabled())
            .map(|f| f.name.as_str())
            .collect();
        enabled.sort();
        println!(
            "  features     {}",
            if enabled.is_empty() {
                "none recorded".to_string()
            } else {
                enabled.join(", ")
            }
        );
    }
    if !profile.gates.is_empty() {
        println!(
            "  gates        {} passed, {} not passing",
            profile.passed_gates().len(),
            profile.failed_gates().len()
        );
    }
    println!(
        "  fit          {:.2}  {}",
        compat.score,
        compat.notes.join("; ")
    );
    if let Some(performance) = &profile.performance {
        println!();
        println!("published measurements");
        println!("{}", "-".repeat(62));
        for line in format_published(profile) {
            println!("{line}");
        }
        if let Some(name) = &performance.baseline_name {
            println!(
                "{}",
                format!(
                    "baseline     {name} {}",
                    performance.baseline_version.as_deref().unwrap_or("")
                )
                .trim_end()
            );
        }
        if !performance.extra_baselines.is_empty() {
            let names: Vec<&str> = performance
                .extra_baselines
                .keys()
                .map(String::as_str)
                .collect();
            println!("also compared with {}", names.join(", "));
        }
    }
    println!();
    println!("{}", next_step(profile));
}

/// The scenario a local profile's directory carries.
fn profile_scenario(profile: &Profile) -> Option<Scenario> {
    let path = profile.path.as_deref()?;
    Scenario::detect(Path::new(path)).ok().flatten()
}

/// What to run next: a local package is served by its path, a hub profile is
/// installed by its repository.
pub(crate) fn next_step(profile: &Profile) -> String {
    match &profile.path {
        Some(path) => format!(
            "serve with: loom-launch serve {}",
            crate::deployment::absolute(Path::new(path)).display()
        ),
        None => format!("install with: loom-launch install {}", profile.repo),
    }
}

// -- install -----------------------------------------------------------------

fn cmd_install(ctx: &Ctx, args: &InstallArgs) -> Result<i32> {
    if let Some(directory) = &args.from_path {
        return install_from_path(ctx, directory, args.target.as_deref(), args.copy);
    }
    let Some(target) = args.target.as_deref() else {
        return Err(Error::launcher(
            "install needs OWNER/NAME, or --from-path DIR",
        ));
    };
    let options = InstallOptions {
        output: args.output.as_ref().map(|o| expand_user(Path::new(o))),
        preset: args.preset.clone(),
        revision: args.revision.clone(),
        materialize: Some(
            match args.materialize {
                Materialize::Auto => "auto",
                Materialize::Copy => "copy",
                Materialize::Share => "share",
            }
            .to_string(),
        ),
        explicit_fetch: args.loom_fetch.clone(),
        hub_endpoint: None,
        offline: args.offline,
    };
    let (result, model) = install_target(ctx, target, &options)?;
    if let Some(warning) = &result.documents.warning {
        eprintln!("warning: {warning}");
    }
    for warning in result.profile_documents.warnings() {
        eprintln!("warning: {warning}");
    }
    if ctx.json {
        print_json(&json!({"result": result.describe(), "install": model.describe()}));
        return Ok(0);
    }
    println!("installed {} into {}", model.repo, model.path);
    println!("  card      {}", model.card_path().display());
    println!("  size      {}", human_bytes(model.size_bytes));
    if let Some(preset) = &result.preset {
        println!("  preset    {preset}");
    }
    if let Some(revision) = &result.revision {
        println!("  revision  {revision}");
    }
    if let Some(summary) = result.documents.summary() {
        println!("  scenario  {summary}");
    }
    if let Some(summary) = result.profile_documents.summary() {
        println!("  policy    {summary}");
    }
    println!("  elapsed   {:.1}s", result.seconds);
    println!();
    println!("serve it with: loom-launch serve {}", model.repo);
    Ok(0)
}

/// `local/<directory name>`, with characters a hub name cannot hold replaced.
fn local_name(directory: &Path) -> String {
    let name: String = directory
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    let name = name.trim_start_matches(['.', '-']);
    format!("local/{}", if name.is_empty() { "package" } else { name })
}

/// Copies a package directory, keeping symbolic links as links (a relative
/// one that leaves the package is made absolute) and linking GGUF files to
/// their source instead of copying them: weights stay where they are.
fn copy_package(source: &Path, destination: &Path, linked: &mut Vec<PathBuf>) -> Result<()> {
    let io = |path: &Path, error: std::io::Error| {
        Error::launcher(format!("cannot copy {}: {error}", path.display()))
    };
    std::fs::create_dir_all(destination).map_err(|e| io(destination, e))?;
    let mut entries: Vec<_> = std::fs::read_dir(source)
        .map_err(|e| io(source, e))?
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let from = entry.path();
        let to = destination.join(entry.file_name());
        let metadata = std::fs::symlink_metadata(&from).map_err(|e| io(&from, e))?;
        if metadata.file_type().is_symlink() {
            let target = std::fs::read_link(&from).map_err(|e| io(&from, e))?;
            let resolved = if target.is_absolute() {
                target.clone()
            } else {
                crate::deployment::absolute(&source.join(&target))
            };
            let kept = if target.is_relative() && resolved.starts_with(source) {
                target
            } else {
                resolved
            };
            std::os::unix::fs::symlink(&kept, &to).map_err(|e| io(&to, e))?;
        } else if metadata.is_dir() {
            copy_package(&from, &to, linked)?;
        } else if from
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            std::os::unix::fs::symlink(&from, &to).map_err(|e| io(&to, e))?;
            linked.push(from);
        } else {
            std::fs::copy(&from, &to).map_err(|e| io(&from, e))?;
        }
    }
    Ok(())
}

/// One package directory registered for serving.
pub(crate) struct Registered {
    pub(crate) model: InstalledModel,
    pub(crate) source: PathBuf,
    pub(crate) copied: bool,
    pub(crate) linked: Vec<PathBuf>,
    pub(crate) scenario: Option<Scenario>,
    pub(crate) profile: Profile,
    pub(crate) profile_documents: crate::install::ProfileDocumentsStatus,
    /// The hub repository the card came from, as its lock records it.
    pub(crate) origin: Option<String>,
}

impl Registered {
    pub(crate) fn describe(&self) -> Value {
        json!({
            "install": self.model.describe(),
            "mode": if self.copied { "copy" } else { "register" },
            "source": self.source.to_string_lossy(),
            "linked": self.linked.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
            "scenario": self.scenario.as_ref().map(Scenario::describe),
            "profile_documents": self.profile_documents.describe(),
        })
    }

    /// What to do when the package has no chat policy, or `None`.
    pub(crate) fn policy_hint(&self) -> Option<String> {
        if self.profile_documents.has(crate::profile::CONFIG_DOCUMENT) {
            return None;
        }
        let remedy = match &self.origin {
            Some(origin) => format!(
                "`loom-launch install {origin}` fetches it from the hub, or copy it next to the card"
            ),
            None => "copy the package's llm_config.json next to its model.json".to_string(),
        };
        Some(format!(
            "without llm_config.json there is no chat policy (thinking toggle, tool-call \
             format); {remedy}"
        ))
    }
}

/// Registers a package directory (in place, or copied into the state's
/// `models/`) and gives it the profile documents local copies can supply.
pub(crate) fn register_path(
    ctx: &Ctx,
    directory: &str,
    name: Option<&str>,
    copy: bool,
) -> Result<Registered> {
    let source = crate::deployment::absolute(&expand_user(Path::new(directory)));
    let source = if source.is_file() {
        source.parent().unwrap_or(Path::new("/")).to_path_buf()
    } else {
        source
    };
    let profile = Profile::from_local(&source, None)?;
    let scenario = Scenario::detect(&source)?;
    let repo = match name {
        Some(name) => {
            let valid = name.split_once('/').is_some_and(|(owner, rest)| {
                !owner.is_empty() && !rest.is_empty() && !rest.contains('/')
            });
            if !valid {
                return Err(Error::launcher(format!(
                    "{name} is not an owner/name to register {} under",
                    source.display()
                )));
            }
            name.to_string()
        }
        None => local_name(&source),
    };
    let mut linked = Vec::new();
    let path = if copy {
        let destination = default_models_directory(&ctx.state).join(repo.replace('/', "-"));
        if destination.exists() {
            return Err(Error::launcher(format!(
                "{} already exists; remove it, or register {} in place without --copy",
                destination.display(),
                source.display()
            )));
        }
        copy_package(&source, &destination, &mut linked)?;
        destination
    } else {
        source.clone()
    };
    // The documents the card does not declare (chat policy, measurements)
    // come from local copies only: a package registered from a path never asks
    // the hub. The lock names the hub revision its card came from, if any.
    let (origin, revision) = crate::install::document_origin(&source)
        .map(|(origin, revision)| (Some(origin), revision))
        .unwrap_or((None, None));
    let profile_documents = crate::install::acquire_profile_documents(
        &path,
        origin.as_deref().unwrap_or(&repo),
        revision.as_deref(),
        true,
        None,
        Some(&ctx.state),
    );
    let profile = Profile::from_local(&path, Some(&repo)).unwrap_or(profile);
    let model = InstalledModel {
        repo: repo.clone(),
        path: path.to_string_lossy().into_owned(),
        preset: profile.preset().map(str::to_string),
        revision: None,
        installed_at: Some(crate::values::iso_utc(crate::values::now_seconds())),
        size_bytes: Some(directory_size(&path) as i64),
        source: Some("local".into()),
        scenario: scenario.is_some().then_some(true),
        scenario_revision: None,
    };
    InstallRegistry::new(&ctx.state).record(&model)?;
    Ok(Registered {
        model,
        source,
        copied: copy,
        linked,
        scenario,
        profile,
        profile_documents,
        origin,
    })
}

/// `install --from-path`: makes a package directory servable without the hub.
fn install_from_path(ctx: &Ctx, directory: &str, name: Option<&str>, copy: bool) -> Result<i32> {
    let registered = register_path(ctx, directory, name, copy)?;
    if ctx.json {
        print_json(&registered.describe());
        return Ok(0);
    }
    for warning in registered.profile_documents.warnings() {
        eprintln!("warning: {warning}");
    }
    let (source, path, repo) = (
        &registered.source,
        Path::new(&registered.model.path),
        &registered.model.repo,
    );
    if copy {
        println!(
            "copied {} into {} as {repo}",
            source.display(),
            path.display()
        );
        if !registered.linked.is_empty() {
            println!(
                "  linked    {} GGUF file(s) to their source instead of copying them",
                registered.linked.len()
            );
        }
    } else {
        println!(
            "registered {} as {repo} in place (nothing was copied; it is served from where it is)",
            source.display()
        );
    }
    println!("  card      {}", registered.model.card_path().display());
    println!("  context   {}", registered.profile.context_text());
    if let Some(scenario) = &registered.scenario {
        println!(
            "  scenario  deployment {} on {}",
            scenario.name(),
            scenario.device_text()
        );
    }
    if let Some(summary) = registered.profile_documents.summary() {
        println!("  policy    {summary}");
    }
    if let Some(hint) = registered.policy_hint() {
        println!("  note      {hint}");
    }
    println!();
    println!("serve it with: loom-launch serve {repo}");
    Ok(0)
}

// -- serve -------------------------------------------------------------------

fn cmd_serve(ctx: &Ctx, args: &ServeArgs) -> Result<i32> {
    let api_key = args.api_key.clone().filter(|k| !k.is_empty());
    if args.no_proxy && !crate::network::is_loopback_host(&args.host) {
        return Err(Error::launcher(format!(
            "--host {} would expose the native listener without the proxy; the native listener \
             stays on 127.0.0.1, so drop --no-proxy to serve on {} through the proxy",
            args.host, args.host
        )));
    }
    let request = ContextRequest {
        context_length: args.context,
        max_tokens: args.max_tokens,
        parallel: args.parallel,
        draft_depth: args.draft_depth,
        checkpoint_capacity: args.checkpoint_capacity,
        session_cache_capacity: args.session_cache,
        prefix_cache_capacity: args.prefix_cache,
        prefill_capacity: args.prefill_capacity,
        ..Default::default()
    };
    if args.detach {
        // A refusal is reported here, not in the background launch's log.
        if let Some(scenario) = local_scenario(ctx, args.profile.target.as_deref())? {
            scenario.refuse(&contradicting(
                &request,
                args.device.as_deref(),
                Spelling::Flags,
            ))?;
        }
        // A launcher already runs: the model is loaded under it.
        if !args.no_proxy && Remote::connect(&ctx.state).is_some() {
            return cmd_start(ctx, &start_arguments(args));
        }
        return detach(ctx, args.wait, api_key.as_deref());
    }
    if args.no_proxy {
        return serve_without_launcher(ctx, args, &request);
    }
    if let Some(record) = running_launcher(&ctx.state) {
        return Err(Error::launcher(format!(
            "a launcher is already running at {} (pid {}); load the profile under it with \
             `loom-launch start {}`, or end it with `loom-launch launcher stop`",
            py_str(get(&record, "endpoint")),
            py_str(get(&record, "pid")),
            args.profile.target.as_deref().unwrap_or("PROFILE"),
        )));
    }
    let resolved = resolve(ctx, &args.profile, !args.no_install)?;
    let prepared = prepare(&resolved, &request, args.device.as_deref(), Spelling::Flags)?;
    let detached = detached_here();
    let launcher = Launcher::start(LauncherOptions {
        service: service(
            ctx,
            &args.host,
            args.proxy_port,
            args.port,
            args.allow_web,
            args.loom_serve.clone(),
            args.profile.loom_fetch.clone(),
            args.wait,
            !ctx.json,
        ),
        api_key: api_key.clone(),
        detached,
        log: detached.then(|| ctx.state.join("logs").join("detached.log")),
    })?;
    let quiet = ctx.json;
    let on_planned: crate::dashboard::OnPlanned = Box::new(move |plan: &LaunchPlan| {
        if quiet {
            return;
        }
        let settings = &plan.settings;
        println!("launching {}", plan.profile.repo);
        println!("  run       {}", plan.run.root.display());
        match &plan.scenario {
            Some(scenario) => {
                for line in scenario_summary(plan, scenario) {
                    println!("{line}");
                }
            }
            None => println!(
                "  deployment {}  device {}",
                plan.deployment_name(),
                plan.device_uri
            ),
        }
        println!(
            "  context   {} tokens, max {} output{}",
            settings.context_length.unwrap_or(0),
            settings.max_tokens.unwrap_or(0),
            settings
                .rows
                .map(|r| format!(", {r} rows"))
                .unwrap_or_default()
        );
    });
    let retry = load_retry(ctx, &resolved, &request, args.device.as_deref());
    let load = crate::dashboard::Load {
        resolved,
        prepared,
        device: args.device.clone(),
        port: Some(args.port),
        allow_web: args.allow_web,
        workers: args.workers,
        pool_backing: args.pool_backing.clone(),
        memory_bytes: args.memory_bytes,
        wait: Some(args.wait),
        source: "serve",
        retry,
        on_planned: Some(on_planned),
    };
    let context = launcher.context().clone();
    let accepted = match context.load(load) {
        Ok(accepted) => accepted,
        Err(message) => {
            launcher.shutdown(Duration::from_secs(10));
            return Err(Error::launcher(message));
        }
    };
    let id = str_value(&accepted["job"]).unwrap_or("").to_string();
    let job = loop {
        match context.jobs().state_of(&id) {
            Some(crate::dashboard::JobState::Done | crate::dashboard::JobState::Failed) | None => {
                break context.jobs().get(&id, 40).unwrap_or(Value::Null);
            }
            _ => std::thread::sleep(Duration::from_millis(100)),
        }
    };
    let supervisor = match (str_value(&job["state"]), context.owned_supervisor()) {
        (Some("done"), Some(supervisor)) => supervisor,
        _ => {
            launcher.shutdown(Duration::from_secs(10));
            let message = str_value(&job["error"])
                .unwrap_or("the model stopped before it served")
                .to_string();
            return Err(Error::launch(message));
        }
    };
    if !ctx.json {
        let endpoint = launcher.endpoint().to_string();
        println!("  endpoint  {endpoint}");
        let (bind_host, bind_port) = launcher.listener();
        // The key goes to the terminal only; redirected output (a log file,
        // `--detach`'s detached.log) gets the URLs without it.
        let terminal = std::io::IsTerminal::is_terminal(&std::io::stdout());
        let shown_key = api_key.as_deref().filter(|_| terminal);
        crate::dashboard::print_network_access(&bind_host, bind_port, shown_key, true);
        print_dashboard_line(&endpoint, shown_key, api_key.is_some(), "  dashboard ");
        println!(
            "  server    {} (instrumented by the launcher)",
            supervisor.server_endpoint()
        );
        if args.allow_web {
            println!("  web       the chat's web_fetch tool may fetch public URLs (--allow-web)");
        }
        let prepared = supervisor
            .run()
            .metadata
            .get("prepare_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        println!(
            "  prepared in {prepared:.1}s; ^C stops it, `loom-launch unload` unloads the model \
             and keeps this launcher serving"
        );
    }
    launcher.wait_for_stop();
    let metrics = context
        .metrics()
        .map(|m| m.snapshot())
        .unwrap_or(Value::Null);
    let record = public_record(launcher.record());
    launcher.shutdown(Duration::from_secs(30));
    let code = supervisor.exit_status().unwrap_or(0);
    if ctx.json {
        print_json(
            &json!({"run": supervisor.run().describe(), "exit_code": code, "launcher": record}),
        );
    } else {
        let p50 = &metrics["ttft_seconds"]["p50"];
        println!(
            "stopped: {} request(s), {} completion tokens, ttft p50 {}",
            metrics["requests"],
            metrics["completion_tokens"],
            if p50.is_null() {
                "-".to_string()
            } else {
                format!("{p50}s")
            }
        );
        println!("log: {}", supervisor.run().serve_log().display());
    }
    Ok(0)
}

/// `serve --no-proxy`: the native listener is the endpoint; no launcher,
/// no dashboard, and the command ends when the server does.
fn serve_without_launcher(ctx: &Ctx, args: &ServeArgs, request: &ContextRequest) -> Result<i32> {
    let resolved = resolve(ctx, &args.profile, !args.no_install)?;
    let prepared = prepare(&resolved, request, args.device.as_deref(), Spelling::Flags)?;
    let settings = prepared.settings().clone();
    let options = LaunchOptions {
        device: args.device.clone(),
        host: "127.0.0.1".into(),
        port: args.port,
        workers: args.workers,
        pool_backing: args.pool_backing.clone(),
        memory_bytes: args.memory_bytes,
        proxy: false,
        startup_timeout: args.wait,
        loom_serve: args.loom_serve.clone(),
        hub_endpoint: ctx.hub_endpoint.clone(),
        profile_roots: ctx.profile_roots.clone(),
        launcher_command: crate::server::redacted_command(&std::env::args().collect::<Vec<_>>()),
        ..Default::default()
    };
    let mut metadata = Map::new();
    metadata.insert("repo".into(), json!(resolved.profile.repo));
    metadata.insert("source".into(), json!("serve"));
    let run = RunStore::new(&ctx.state).create(None, metadata)?;
    let plan = prepared.plan(&resolved, run.clone(), &options)?;
    let on_output: Option<crate::server::OutputCallback> = if ctx.json {
        None
    } else {
        Some(Box::new(|line: &str| {
            let mut stdout = std::io::stdout().lock();
            let _ = writeln!(stdout, "  | {line}");
            let _ = stdout.flush();
        }))
    };
    let supervisor = Supervisor::new(plan, options, &ctx.state, on_output);
    signals::install();
    if !ctx.json {
        println!("launching {} (no proxy)", resolved.profile.repo);
        println!("  run       {}", run.root.display());
        println!(
            "  context   {} tokens, max {} output",
            settings.context_length.unwrap_or(0),
            settings.max_tokens.unwrap_or(0)
        );
    }
    supervisor.start()?;
    if !ctx.json {
        println!(
            "  endpoint  {} (native, not instrumented)",
            supervisor.endpoint()
        );
        println!("  ^C stops it");
    }
    let code = supervisor.wait();
    if ctx.json {
        print_json(&json!({"run": supervisor.run().describe(), "exit_code": code}));
    } else {
        println!("log: {}", supervisor.run().serve_log().display());
    }
    Ok(code)
}

/// `serve --detach` with a launcher already running: the same as `start`.
fn start_arguments(args: &ServeArgs) -> StartArgs {
    StartArgs {
        profile: args.profile.clone(),
        context: args.context,
        max_tokens: args.max_tokens,
        parallel: args.parallel,
        draft_depth: args.draft_depth,
        checkpoint_capacity: args.checkpoint_capacity,
        session_cache: args.session_cache,
        prefix_cache: args.prefix_cache,
        prefill_capacity: args.prefill_capacity,
        device: args.device.clone(),
        workers: args.workers,
        pool_backing: args.pool_backing.clone(),
        memory_bytes: args.memory_bytes,
        allow_web: args.allow_web,
        wait: args.wait,
        no_install: args.no_install,
        // The default port means "the launcher's own".
        port: (args.port != DEFAULT_PORT).then_some(args.port),
        host: args.host.clone(),
        proxy_port: args.proxy_port,
        api_key: args.api_key.clone(),
        loom_serve: args.loom_serve.clone(),
    }
}

/// The request a retry from the page repeats: by name when the registry
/// knows the profile, else by its directory.
fn load_retry(
    ctx: &Ctx,
    resolved: &Resolved,
    request: &ContextRequest,
    device: Option<&str>,
) -> Value {
    let mut body = json!({
        "repo": resolved.profile.repo,
        "context": request.context_length,
        "max_tokens": request.max_tokens,
        "parallel": request.parallel,
        "draft_depth": request.draft_depth,
        "device": device,
    });
    if InstallRegistry::new(&ctx.state)
        .get(&resolved.profile.repo)
        .is_none()
    {
        body["path"] = json!(resolved.installed.path);
    }
    json!({"route": "/api/serve", "body": body})
}

fn print_dashboard_line(endpoint: &str, shown_key: Option<&str>, keyed: bool, prefix: &str) {
    match (shown_key, keyed) {
        (Some(key), _) => println!(
            "{prefix}{endpoint}/?access_token={}",
            crate::proxy::percent_encode(key)
        ),
        (None, true) => println!("{prefix}{endpoint}/?access_token=<your --api-key>"),
        (None, false) => println!("{prefix}{endpoint}/"),
    }
}

// -- the launcher service ------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn service(
    ctx: &Ctx,
    host: &str,
    port: u16,
    server_port: u16,
    allow_web: bool,
    loom_serve: Option<String>,
    loom_fetch: Option<String>,
    startup_timeout: f64,
    echo: bool,
) -> Service {
    Service {
        state_root: ctx.state.clone(),
        hub_endpoint: ctx.hub_endpoint.clone(),
        host: host.to_string(),
        port,
        server_port,
        loom_serve,
        loom_fetch,
        allow_web,
        profile_roots: ctx.profile_roots.clone(),
        command: crate::server::redacted_command(&std::env::args().collect::<Vec<_>>()),
        startup_timeout,
        echo,
    }
}

/// Whether this process is a launcher started with `--detach`.
fn detached_here() -> bool {
    std::env::var(DETACHED_ENV).is_ok_and(|v| v == "1")
}

fn launcher_log(ctx: &Ctx) -> PathBuf {
    ctx.state.join("logs").join("launcher.log")
}

/// The global options a background `loom-launch` needs to share this state.
fn global_arguments(ctx: &Ctx) -> Vec<String> {
    let mut arguments = vec![
        "--state".to_string(),
        ctx.state.to_string_lossy().into_owned(),
    ];
    if let Some(hub) = &ctx.hub_endpoint {
        arguments.extend(["--hub".to_string(), hub.clone()]);
    }
    for root in &ctx.profile_roots {
        arguments.extend([
            "--profiles".to_string(),
            root.to_string_lossy().into_owned(),
        ]);
    }
    arguments
}

/// `launcher start` for a background launcher with these listener options
/// (the API key travels in the environment, never in the argument list).
#[allow(clippy::too_many_arguments)]
fn launcher_arguments(
    ctx: &Ctx,
    host: &str,
    port: u16,
    proxy_port: u16,
    allow_web: bool,
    loom_serve: Option<&str>,
    loom_fetch: Option<&str>,
) -> Vec<String> {
    let mut arguments = global_arguments(ctx);
    arguments.extend([
        "launcher".to_string(),
        "start".to_string(),
        format!("--host={host}"),
        format!("--port={port}"),
        format!("--proxy-port={proxy_port}"),
    ]);
    if allow_web {
        arguments.push("--allow-web".into());
    }
    if let Some(path) = loom_serve {
        arguments.push(format!("--loom-serve={path}"));
    }
    if let Some(path) = loom_fetch {
        arguments.push(format!("--loom-fetch={path}"));
    }
    arguments
}

/// Runs this executable with `arguments` in its own session (it ignores this
/// shell from then on), output appended to `log_path`.
fn spawn_background(
    arguments: &[String],
    log_path: &Path,
    api_key: Option<&str>,
) -> Result<std::process::Child> {
    use std::os::unix::process::CommandExt;
    let executable = std::env::current_exe()
        .map_err(|error| Error::launch(format!("cannot find this executable: {error}")))?;
    if let Some(directory) = log_path.parent() {
        std::fs::create_dir_all(directory).map_err(|error| {
            Error::launch(format!("cannot create {}: {error}", directory.display()))
        })?;
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|error| Error::launch(format!("cannot open {}: {error}", log_path.display())))?;
    let log_err = log
        .try_clone()
        .map_err(|error| Error::launch(format!("cannot open {}: {error}", log_path.display())))?;
    let mut command = std::process::Command::new(executable);
    command
        .args(arguments)
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(log_err)
        .env(DETACHED_ENV, "1");
    if let Some(key) = api_key {
        command.env("LOOM_LAUNCH_API_KEY", key);
    }
    // SAFETY: setsid(2) is async-signal-safe and touches no Rust state.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    command
        .spawn()
        .map_err(|error| Error::launch(format!("cannot start the background launcher: {error}")))
}

fn log_tail(path: &Path, count: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(count)..].join("\n")
}

/// Starts a launcher in the background and waits until it answers; returns
/// its record.
fn start_background_launcher(
    ctx: &Ctx,
    arguments: &[String],
    api_key: Option<&str>,
    wait: f64,
) -> Result<Map<String, Value>> {
    let log_path = launcher_log(ctx);
    let mut child = spawn_background(arguments, &log_path, api_key)?;
    let pid = child.id() as i64;
    let deadline = Instant::now() + Duration::from_secs_f64(wait.max(1.0));
    loop {
        if let Ok(Some(_)) = child.try_wait() {
            return Err(Error::launch(format!(
                "the launcher exited; see {}\n{}",
                log_path.display(),
                log_tail(&log_path, 6)
            )));
        }
        if let Some(record) = crate::launcher::read_launcher(&ctx.state) {
            if strict_int(get(&record, "pid")) == Some(pid) && Remote::connect(&ctx.state).is_some()
            {
                return Ok(record);
            }
        }
        if Instant::now() >= deadline {
            // The child runs in its own session, so nothing else would stop it.
            // SAFETY: killpg(2) has no memory-safety preconditions.
            unsafe {
                libc::killpg(pid as i32, libc::SIGTERM);
            }
            return Err(Error::launch(format!(
                "the launcher did not answer within {wait:.0}s; see {}",
                log_path.display()
            )));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Where a launcher answers and how to use it.
fn print_launcher(record: &Map<String, Value>, shown_key: Option<&str>, foreground: bool) {
    let endpoint = py_str(get(record, "endpoint"));
    let keyed = truthy_value(get(record, "auth"));
    println!("launcher  {endpoint} (pid {})", py_str(get(record, "pid")));
    let bind_host = str_value(get(record, "bind_host")).unwrap_or("127.0.0.1");
    let port = strict_int(get(record, "port")).unwrap_or(0) as u16;
    crate::dashboard::print_network_access(bind_host, port, shown_key, true);
    print_dashboard_line(&endpoint, shown_key, keyed, "  dashboard ");
    match strict_int(get(record, "server_port")) {
        Some(0) | None => println!("  models    served on a free 127.0.0.1 port per load"),
        Some(port) => println!("  models    served on 127.0.0.1:{port} when loaded"),
    }
    if truthy_value(get(record, "allow_web")) {
        println!("  web       the chat's web_fetch tool may fetch public URLs (--allow-web)");
    }
    println!("  state     {}", py_str(get(record, "state_directory")));
    if let Some(log) = str_value(get(record, "log")) {
        println!("  log       {log}");
    }
    println!("  no model is loaded; load one with: loom-launch start PROFILE");
    if foreground {
        println!("  ^C stops it (and unloads its model)");
    } else {
        println!("  stop it with: loom-launch launcher stop");
    }
}

fn truthy_value(value: &Value) -> bool {
    crate::values::truthy(value)
}

fn cmd_launcher_start(ctx: &Ctx, args: &LauncherStartArgs) -> Result<i32> {
    let listener = &args.listener;
    let api_key = listener.api_key.clone().filter(|k| !k.is_empty());
    if let Some(record) = running_launcher(&ctx.state) {
        return Err(Error::launcher(format!(
            "a launcher is already running at {} (pid {}); `loom-launch launcher stop` ends it",
            py_str(get(&record, "endpoint")),
            py_str(get(&record, "pid")),
        )));
    }
    let terminal = std::io::IsTerminal::is_terminal(&std::io::stdout());
    let shown_key = api_key.as_deref().filter(|_| terminal);
    if args.detach {
        let arguments = launcher_arguments(
            ctx,
            &listener.host,
            listener.port,
            listener.proxy_port,
            listener.allow_web,
            listener.loom_serve.as_deref(),
            listener.loom_fetch.as_deref(),
        );
        let record = start_background_launcher(ctx, &arguments, api_key.as_deref(), args.wait)?;
        if ctx.json {
            print_json(&public_record(&record));
        } else {
            print_launcher(&record, shown_key, false);
        }
        return Ok(0);
    }
    let detached = detached_here();
    let launcher = Launcher::start(LauncherOptions {
        service: service(
            ctx,
            &listener.host,
            listener.proxy_port,
            listener.port,
            listener.allow_web,
            listener.loom_serve.clone(),
            listener.loom_fetch.clone(),
            1800.0,
            !ctx.json,
        ),
        api_key: api_key.clone(),
        detached,
        log: detached.then(|| launcher_log(ctx)),
    })?;
    if ctx.json {
        print_json(&public_record(launcher.record()));
    } else {
        print_launcher(launcher.record(), shown_key, !detached);
    }
    launcher.wait_for_stop();
    let unloaded = launcher.shutdown(Duration::from_secs(30));
    if !ctx.json {
        match unloaded {
            Some(repo) => println!("stopped the launcher; unloaded {repo}"),
            None => println!("stopped the launcher"),
        }
    }
    Ok(0)
}

/// `dashboard`: a launcher in the foreground on the dashboard's port.
fn cmd_dashboard(ctx: &Ctx, args: &DashboardArgs) -> Result<i32> {
    let launcher = Launcher::start(LauncherOptions {
        service: service(
            ctx,
            &args.host,
            args.port,
            0,
            args.allow_web,
            args.loom_serve.clone(),
            args.loom_fetch.clone(),
            1800.0,
            false,
        ),
        api_key: None,
        detached: false,
        log: None,
    })?;
    let (host, port) = launcher.listener();
    let endpoint = launcher.endpoint().to_string();
    println!("loom-launch dashboard at {endpoint}/");
    crate::dashboard::print_network_access(&host, port, None, false);
    println!(
        "  launcher  {endpoint} carries the model loaded from the page (pid {})",
        std::process::id()
    );
    println!("  state     {}", ctx.state.display());
    if args.allow_web {
        println!("  web       the chat's web_fetch tool may fetch public URLs (--allow-web)");
    }
    if let Some((endpoint, record)) = running_endpoint(&ctx.state) {
        println!(
            "  showing   the running launch of {} at {endpoint}",
            py_str(get(&record, "repo"))
        );
    }
    println!("  ^C stops it (and any model loaded from the page)");
    launcher.wait_for_stop();
    let context = launcher.context().clone();
    if context.owned_supervisor().is_some() || context.preparing() {
        println!("stopping the model loaded from the page");
    }
    launcher.shutdown(Duration::from_secs(30));
    Ok(0)
}

/// The launcher, as `status --json` and `launcher status --json` show it:
/// its record (without the control token) and its live summary.
fn launcher_document(remote: Option<&Remote>) -> Value {
    let Some(remote) = remote else {
        return Value::Null;
    };
    let mut document = public_record(&remote.record);
    document["running"] = json!(true);
    if let Ok(status) = remote.loom_status() {
        document["live"] = get(&status, "launcher").clone();
    }
    document
}

/// The launcher's lines of `status` and `launcher status`.
fn print_launcher_status(remote: &Remote) -> Value {
    let record = &remote.record;
    let live = remote
        .loom_status()
        .map(|status| get(&status, "launcher").clone())
        .unwrap_or(Value::Null);
    let uptime = live["uptime_seconds"]
        .as_f64()
        .map(|s| format!(", up {}", crate::values::human_duration(s)))
        .unwrap_or_default();
    println!(
        "launcher  running at {}/ (pid {}{uptime})",
        remote.endpoint(),
        py_str(get(record, "pid"))
    );
    println!("  started   {}", py_str(get(record, "started_at")));
    println!(
        "  listening {}:{}{}",
        py_str(get(record, "bind_host")),
        py_str(get(record, "port")),
        if truthy_value(get(record, "auth")) {
            " (API key required)"
        } else {
            ""
        }
    );
    for url in arr(get(record, "urls")).iter().filter_map(Value::as_str) {
        println!("  network   {url}");
    }
    let counters = obj(&live["counters"]);
    if !counters.is_empty() {
        let count = |key: &str| strict_int(get(counters, key)).unwrap_or(0);
        println!(
            "  answered  page {}, api {}, loom {}, health {}, /v1 without a model {}",
            count("page"),
            count("api"),
            count("loom"),
            count("health"),
            count("no_model")
        );
    }
    if truthy_value(get(record, "detached")) {
        if let Some(log) = str_value(get(record, "log")) {
            println!("  log       {log}");
        }
    }
    let exit = obj(&live["last_exit"]);
    if !exit.is_empty() {
        println!(
            "  last exit {} exited on its own ({}) at {}",
            py_str(get(exit, "repo")),
            py_str(get(exit, "detail")),
            py_str(get(exit, "at"))
        );
    }
    live
}

fn cmd_launcher_status(ctx: &Ctx) -> Result<i32> {
    let remote = Remote::connect(&ctx.state);
    if ctx.json {
        print_json(&json!({
            "launcher": launcher_document(remote.as_ref()),
            "model": read_current(&ctx.state),
        }));
        return Ok(if remote.is_some() { 0 } else { 1 });
    }
    let Some(remote) = remote else {
        println!("no launcher is running");
        println!("state directory: {}", ctx.state.display());
        return Ok(1);
    };
    let live = print_launcher_status(&remote);
    let repo = py_str(&live["repo"]);
    match str_value(&live["model"]) {
        Some("serving") => {
            println!("  model     serving {repo} (`loom-launch status` for its counters)")
        }
        Some("starting") => println!("  model     loading {repo}"),
        Some("stopping") => println!("  model     unloading {repo}"),
        _ => println!("  model     none loaded (`loom-launch start PROFILE` loads one)"),
    }
    Ok(0)
}

/// Unloads the launcher's model and follows the job; `None` when no model
/// was loaded.
fn unload_with(remote: &Remote, wait: f64) -> Result<Option<Map<String, Value>>> {
    let Some(accepted) = remote.unload_within(wait)? else {
        return Ok(None);
    };
    let id = str_value(get(&accepted, "job")).unwrap_or("").to_string();
    let job = remote.follow(
        &id,
        Duration::from_secs_f64(wait.max(0.0) + 60.0),
        &mut |_| {},
        &|| false,
    )?;
    Ok(Some(job))
}

/// `unloaded REPO (pid N, exit status C)` from an unload job.
fn unloaded_line(job: &Map<String, Value>) -> String {
    let result = obj(get(job, "result"));
    if let Some(repo) = str_value(get(result, "cancelled")) {
        return format!("cancelled the load of {repo} before it served");
    }
    if str_value(get(job, "state")) == Some("failed") {
        return format!(
            "the unload of {} failed: {}",
            py_str(get(job, "target")),
            py_str(get(job, "error"))
        );
    }
    let mut facts = Vec::new();
    if let Some(pid) = strict_int(get(result, "pid")) {
        facts.push(format!("pid {pid}"));
    }
    if let Some(code) = strict_int(get(result, "exit_code")) {
        facts.push(format!("exit status {code}"));
    }
    format!(
        "unloaded {}{}{}",
        py_str(get(result, "repo")),
        if facts.is_empty() {
            String::new()
        } else {
            format!(" ({})", facts.join(", "))
        },
        if truthy_value(get(result, "forced")) {
            " with SIGKILL"
        } else {
            ""
        }
    )
}

fn unload_document(job: Option<&Map<String, Value>>, remote: &Remote) -> Value {
    let result = job
        .map(|j| obj(get(j, "result")).clone())
        .unwrap_or_default();
    json!({
        "unloaded": job.and(get(&result, "repo").as_str()),
        "cancelled": get(&result, "cancelled"),
        "pid": get(&result, "pid"),
        "exit_code": get(&result, "exit_code"),
        "forced": get(&result, "forced"),
        "job": job.map(|j| j.get("id").cloned()),
        "launcher": {"running": true, "endpoint": remote.endpoint(), "pid": remote.pid()},
    })
}

fn cmd_unload(ctx: &Ctx, wait: f64) -> Result<i32> {
    let Some(remote) = Remote::connect(&ctx.state) else {
        // No launcher service: a launch of before the split is stopped with
        // the process that supervised it.
        return match stop_legacy(&ctx.state, Duration::from_secs_f64(wait.max(0.0))) {
            Some(StopOutcome::Stopped {
                pid,
                repo,
                run_id,
                forced,
            }) => {
                if ctx.json {
                    print_json(&json!({"stopped": pid, "repo": repo, "run_id": run_id,
                                       "forced": forced, "launcher": null}));
                } else {
                    println!(
                        "stopped launch {pid} ({repo}){}; it ran without a launcher service, so \
                         nothing keeps serving",
                        if forced { " with SIGKILL" } else { "" }
                    );
                }
                Ok(0)
            }
            _ => {
                println!("no model is loaded and no launcher is running");
                Ok(1)
            }
        };
    };
    let job = unload_with(&remote, wait)?;
    if ctx.json {
        print_json(&unload_document(job.as_ref(), &remote));
        return Ok(if job.is_some() { 0 } else { 1 });
    }
    match &job {
        Some(job) => println!("{}", unloaded_line(job)),
        None => println!("no model is loaded"),
    }
    println!(
        "the launcher keeps running at {}/ (pid {}); `loom-launch launcher stop` ends it",
        remote.endpoint(),
        remote.pid().map(|p| p.to_string()).unwrap_or_default()
    );
    Ok(if job.is_some() { 0 } else { 1 })
}

fn cmd_launcher_stop(ctx: &Ctx, wait: f64) -> Result<i32> {
    // A launch of before the split (its supervisor is not this launcher)
    // is stopped first; the launcher then unloads its own model and exits.
    let legacy = stop_legacy(&ctx.state, Duration::from_secs(30));
    let outcome = stop_launcher(&ctx.state, Duration::from_secs_f64(wait.max(1.0)));
    let legacy_stopped = match &legacy {
        Some(StopOutcome::Stopped {
            pid, repo, forced, ..
        }) => Some(json!({"pid": pid, "repo": repo, "forced": forced})),
        _ => None,
    };
    if ctx.json {
        let launcher = match &outcome {
            LauncherStop::Stopped {
                pid,
                endpoint,
                model,
                forced,
            } => json!({"stopped": pid, "endpoint": endpoint, "unloaded": model, "forced": forced}),
            LauncherStop::NotRunning { stale } => json!({"stopped": null, "stale": stale}),
        };
        print_json(&json!({"launcher": launcher, "legacy": legacy_stopped}));
    } else {
        if let Some(StopOutcome::Stopped {
            pid, repo, forced, ..
        }) = &legacy
        {
            println!(
                "stopped launch {pid} ({repo}){}, started without a launcher service",
                if *forced { " with SIGKILL" } else { "" }
            );
        }
        match &outcome {
            LauncherStop::Stopped {
                pid,
                endpoint,
                model,
                forced,
            } => {
                let unloaded = match model {
                    Some(repo) => format!("unloaded {repo}"),
                    None => "no model was loaded".to_string(),
                };
                println!("stopped the launcher (pid {pid}) at {endpoint}; {unloaded}");
                if *forced {
                    println!(
                        "  it did not exit within {wait:.0}s, so it was killed and the model \
                         server it supervised was stopped"
                    );
                }
            }
            LauncherStop::NotRunning { stale: true } => {
                println!("removed the record of a launcher that is no longer running")
            }
            LauncherStop::NotRunning { stale: false } if legacy_stopped.is_none() => {
                println!("no launcher is running")
            }
            LauncherStop::NotRunning { .. } => {}
        }
    }
    let stopped = matches!(outcome, LauncherStop::Stopped { .. }) || legacy_stopped.is_some();
    Ok(if stopped { 0 } else { 1 })
}

/// `start`: loads a model under the running launcher, starting one first
/// (in the background) when none runs.
fn cmd_start(ctx: &Ctx, args: &StartArgs) -> Result<i32> {
    let request = ContextRequest {
        context_length: args.context,
        max_tokens: args.max_tokens,
        parallel: args.parallel,
        draft_depth: args.draft_depth,
        checkpoint_capacity: args.checkpoint_capacity,
        session_cache_capacity: args.session_cache,
        prefix_cache_capacity: args.prefix_cache,
        prefill_capacity: args.prefill_capacity,
        ..Default::default()
    };
    // Everything that can be refused is refused before a launcher starts.
    let resolved = resolve(ctx, &args.profile, !args.no_install)?;
    prepare(&resolved, &request, args.device.as_deref(), Spelling::Flags)?;
    let (remote, started) = match Remote::connect(&ctx.state) {
        Some(remote) => (remote, false),
        None => {
            if let Some((endpoint, record)) = running_endpoint(&ctx.state) {
                return Err(Error::launcher(format!(
                    "a launch of {} is already running at {endpoint} (pid {}), started without a \
                     launcher service; stop it with `loom-launch stop`, then retry",
                    py_str(get(&record, "repo")),
                    py_str(get(&record, "pid")),
                )));
            }
            let api_key = args.api_key.clone().filter(|k| !k.is_empty());
            let arguments = launcher_arguments(
                ctx,
                &args.host,
                args.port.unwrap_or(DEFAULT_PORT),
                args.proxy_port,
                false,
                args.loom_serve.as_deref(),
                args.profile.loom_fetch.as_deref(),
            );
            let record = start_background_launcher(ctx, &arguments, api_key.as_deref(), 30.0)?;
            ctx.say(&format!(
                "started the launcher at {} (pid {}); log {}",
                py_str(get(&record, "endpoint")),
                py_str(get(&record, "pid")),
                launcher_log(ctx).display()
            ));
            let remote = Remote::connect(&ctx.state)
                .ok_or_else(|| Error::launch("the launcher started but does not answer"))?;
            (remote, true)
        }
    };
    let body = json!({
        "path": resolved.installed.path,
        "repo": resolved.profile.repo,
        "context": request.context_length,
        "max_tokens": request.max_tokens,
        "parallel": request.parallel,
        "draft_depth": request.draft_depth,
        "checkpoint_capacity": request.checkpoint_capacity,
        "session_cache": request.session_cache_capacity,
        "prefix_cache": request.prefix_cache_capacity,
        "prefill_capacity": request.prefill_capacity,
        "device": args.device,
        "workers": args.workers,
        "pool_backing": args.pool_backing,
        "memory_bytes": args.memory_bytes,
        "allow_web": args.allow_web,
        "wait": args.wait,
        "port": args.port,
    });
    let accepted = remote.load(&body)?;
    let id = str_value(get(&accepted, "job")).unwrap_or("").to_string();
    let endpoint = remote.endpoint();
    if !ctx.json {
        let settings = obj(get(&accepted, "settings"));
        println!(
            "loading   {} under the launcher at {endpoint} (native port {})",
            resolved.profile.repo,
            py_str(get(&accepted, "port"))
        );
        println!(
            "  context   {} tokens, max {} output{}",
            py_str(get(settings, "context_length")),
            py_str(get(settings, "max_tokens")),
            strict_int(get(settings, "rows"))
                .map(|r| format!(", {r} rows"))
                .unwrap_or_default()
        );
    }
    signals::install();
    let quiet = ctx.json;
    let followed = remote.follow(
        &id,
        Duration::from_secs_f64(args.wait.max(1.0) + 60.0),
        &mut |line: &str| {
            if !quiet {
                println!("  | {line}");
            }
        },
        &signals::stop_requested,
    );
    let job = match followed {
        Ok(job) => job,
        Err(_) if signals::stop_requested() => {
            let _ = remote.unload();
            eprintln!(
                "interrupted: the load of {} was cancelled; the launcher keeps running at {endpoint}",
                resolved.profile.repo
            );
            return Ok(130);
        }
        Err(error) => return Err(error),
    };
    if str_value(get(&job, "state")) != Some("done") {
        let mut message = py_str(get(&job, "error"));
        if let Some(hint) = str_value(get(&job, "hint")) {
            message.push_str(&format!("\n{hint}"));
        }
        message.push_str(&format!("\nthe launcher keeps running at {endpoint}"));
        return Err(Error::launch(message));
    }
    let record = read_current(&ctx.state).unwrap_or_default();
    if ctx.json {
        print_json(&json!({
            "launcher": public_record(&remote.record),
            "started_launcher": started,
            "model": record,
            "job": job,
        }));
        return Ok(0);
    }
    let result = obj(get(&job, "result"));
    println!("serving {} at {endpoint}", py_str(get(result, "repo")));
    if str_value(get(&record, "kind")) == Some("scenario") {
        println!(
            "  scenario   running the package's own deployment {} across {}",
            py_str(get(&record, "deployment")),
            py_str(get(&record, "device"))
        );
    } else if !record.is_empty() {
        println!("  deployment {}", py_str(get(&record, "deployment")));
    }
    println!(
        "  server     {} (native listener)",
        py_str(get(result, "server_endpoint"))
    );
    let keyed = truthy_value(get(&remote.record, "auth"));
    print_dashboard_line(&endpoint, None, keyed, "  dashboard  ");
    if let Some(run) = str_value(get(&record, "run_directory")) {
        println!("  run        {run}");
    }
    if let Some(seconds) = get(result, "prepare_seconds").as_f64() {
        println!("  prepared in {seconds:.1}s");
    }
    println!(
        "  unload with: loom-launch unload (the launcher keeps running; \
         `loom-launch launcher stop` ends it)"
    );
    Ok(0)
}

/// The lines `serve` prints about a scenario: whose deployment runs, on which devices.
fn scenario_summary(plan: &LaunchPlan, scenario: &crate::server::ScenarioPlan) -> Vec<String> {
    let uris: Vec<String> = scenario
        .scenario
        .device_uris()
        .iter()
        .map(|(name, uri)| format!("{name} {}", py_str(uri)))
        .collect();
    let mut lines = vec![
        format!(
            "  scenario  running the package's own deployment {} across {}",
            plan.deployment_name(),
            plan.device_uri
        ),
        format!(
            "  model     {}",
            scenario.scenario.deployment_path.display()
        ),
        format!(
            "  devices   {} ({})",
            scenario.devices_path.display(),
            uris.join(", ")
        ),
    ];
    if !scenario.overrides.is_empty() {
        let changed: Vec<String> = scenario
            .overrides
            .iter()
            .filter(|(key, _)| key.as_str() != "pool")
            .map(|(key, value)| format!("{key} {}", py_str(value)))
            .collect();
        lines.push(format!(
            "            pool {} changed for this run: {} (the package's {} is untouched)",
            py_str(&scenario.overrides["pool"]),
            changed.join(", "),
            scenario.scenario.devices_path.display()
        ));
    }
    lines
}

/// The scenario a target names without resolving it through the hub: a local
/// directory, a registered install or a local profile.
fn local_scenario(ctx: &Ctx, target: Option<&str>) -> Result<Option<Scenario>> {
    let registry = InstallRegistry::new(&ctx.state);
    let directory = match target {
        None => registry.latest().map(|m| PathBuf::from(m.path)),
        Some(target) => match local_target(target) {
            Some(local) => Some(local),
            None => match registry.get(target) {
                Some(installed) => Some(PathBuf::from(installed.path)),
                None => local::find(&ctx.state, &ctx.profile_roots, target)?
                    .map(|(found, _)| found.directory),
            },
        },
    };
    match directory {
        Some(directory) => Scenario::detect(&directory),
        None => Ok(None),
    }
}

/// Runs `serve` in the background (its own session: a launcher and its
/// model that ignore this shell), then reports its endpoint.
fn detach(ctx: &Ctx, wait: f64, api_key: Option<&str>) -> Result<i32> {
    // The key is shown on a terminal only, never in redirected output.
    let shown_key = api_key.filter(|_| std::io::IsTerminal::is_terminal(&std::io::stdout()));
    let arguments: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--detach")
        .collect();
    let log_path = ctx.state.join("logs").join("detached.log");
    // The key reaches the child through the environment it already inherits.
    let mut child = spawn_background(&arguments, &log_path, None)?;
    let child_pid = child.id() as i64;
    let deadline = Instant::now() + Duration::from_secs_f64(wait.max(1.0) + 5.0);
    while Instant::now() < deadline {
        if let Ok(Some(_)) = child.try_wait() {
            return Err(Error::launch(format!(
                "the launch exited; see {}\n{}",
                log_path.display(),
                log_tail(&log_path, 6)
            )));
        }
        if let Some(record) = read_current(&ctx.state) {
            let ours = strict_int(get(&record, "launcher_pid")) == Some(child_pid);
            if ours && str_value(get(&record, "endpoint")).is_some() {
                if ctx.json {
                    let mut document = Value::Object(record.clone());
                    document["log"] = json!(log_path.to_string_lossy());
                    document["launcher"] = crate::launcher::read_launcher(&ctx.state)
                        .map(|r| public_record(&r))
                        .unwrap_or(Value::Null);
                    print_json(&document);
                } else {
                    println!(
                        "serving {} at {}",
                        py_str(get(&record, "repo")),
                        py_str(get(&record, "endpoint"))
                    );
                    for url in arr(get(&record, "urls")).iter().filter_map(Value::as_str) {
                        match shown_key {
                            Some(key) => println!(
                                "  network    {url}?access_token={}",
                                crate::proxy::percent_encode(key)
                            ),
                            None => println!("  network    {url}"),
                        }
                    }
                    if str_value(get(&record, "kind")) == Some("scenario") {
                        println!(
                            "  scenario   running the package's own deployment {} across {}",
                            py_str(get(&record, "deployment")),
                            py_str(get(&record, "device"))
                        );
                    } else {
                        println!("  deployment {}", py_str(get(&record, "deployment")));
                    }
                    println!(
                        "  server     {} (native listener)",
                        py_str(get(&record, "server_endpoint"))
                    );
                    print_dashboard_line(
                        &py_str(get(&record, "endpoint")),
                        shown_key,
                        api_key.is_some(),
                        "  dashboard  ",
                    );
                    println!(
                        "  launcher   pid {child_pid}; it keeps serving when the model is unloaded",
                    );
                    println!("  log        {}", log_path.display());
                    println!(
                        "  unload with: loom-launch unload; stop everything with: loom-launch stop --all"
                    );
                }
                return Ok(0);
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    // The child runs in its own session, so nothing else would stop it.
    // SAFETY: killpg(2) has no memory-safety preconditions.
    unsafe {
        libc::killpg(child_pid as i32, libc::SIGTERM);
    }
    Err(Error::launch(format!(
        "the launch did not become ready; see {}",
        log_path.display()
    )))
}

// -- endpoint resolution -----------------------------------------------------

/// A client-facing endpoint and the profile that describes it.
struct Endpoint {
    url: String,
    profile: Profile,
    supervisor: Option<Supervisor>,
    /// A launcher this command loaded the model under (unloaded afterwards).
    loaded_under: Option<Remote>,
    model: Option<String>,
    context_length: Option<i64>,
    max_sessions: Option<i64>,
    /// The bearer token to use when the caller gave none.
    token: Option<String>,
}

impl Endpoint {
    /// Stops a launch this command started, or unloads the model it loaded
    /// under a launcher (no-op otherwise).
    fn finish(&self) {
        if let Some(supervisor) = &self.supervisor {
            supervisor.stop(Duration::from_secs(30));
        }
        if let Some(remote) = &self.loaded_under {
            if let Ok(Some(accepted)) = remote.unload() {
                let id = str_value(get(&accepted, "job")).unwrap_or("").to_string();
                let _ = remote.follow(&id, Duration::from_secs(90), &mut |_| {}, &|| false);
            }
        }
    }
}

fn endpoint_for(ctx: &Ctx, profile: &ProfileArgs, endpoint: &EndpointArgs) -> Result<Endpoint> {
    let token = endpoint.api_key.as_deref().filter(|k| !k.is_empty());
    if let Some(base_url) = &endpoint.base_url {
        let client = Client::new(base_url, token, Duration::from_secs(10))?;
        let profile = profile_for_endpoint(ctx, profile, &client, None);
        let model = served_model(&client, &profile, None);
        return Ok(Endpoint {
            url: base_url.trim_end_matches('/').to_string(),
            profile,
            supervisor: None,
            loaded_under: None,
            model,
            context_length: None,
            max_sessions: None,
            token: None,
        });
    }
    if let Some(found) = running(ctx, profile, endpoint, token)? {
        return Ok(found);
    }
    // A launcher with no model: the model is loaded under it for this command.
    if let Some(remote) = Remote::connect(&ctx.state) {
        return load_for_command(ctx, profile, endpoint, remote, token);
    }
    if endpoint.keep_server {
        start_detached(ctx, profile, endpoint)?;
        return running(ctx, profile, endpoint, token)?
            .ok_or_else(|| Error::launch("the background launch is not answering"));
    }
    let resolved = resolve(ctx, profile, true)?;
    let prepared = prepare(
        &resolved,
        &ContextRequest {
            context_length: endpoint.context,
            parallel: endpoint.parallel,
            ..Default::default()
        },
        endpoint.device.as_deref(),
        Spelling::Flags,
    )?;
    let settings = prepared.settings().clone();
    let options = LaunchOptions {
        device: endpoint.device.clone(),
        port: endpoint.port,
        workers: endpoint.workers,
        proxy: !endpoint.no_proxy,
        auth_token: token.map(str::to_string),
        startup_timeout: endpoint.wait,
        loom_serve: endpoint.loom_serve.clone(),
        hub_endpoint: ctx.hub_endpoint.clone(),
        profile_roots: ctx.profile_roots.clone(),
        ..Default::default()
    };
    let mut metadata = Map::new();
    metadata.insert("repo".into(), json!(resolved.profile.repo));
    metadata.insert("source".into(), json!("ephemeral"));
    let run = RunStore::new(&ctx.state).create(None, metadata)?;
    let plan = prepared.plan(&resolved, run, &options)?;
    let supervisor = Supervisor::new(plan, options, &ctx.state, None);
    ctx.say(&format!(
        "launching {} for this command",
        resolved.profile.repo
    ));
    // A signal while this command owns the launch stops the server first.
    let guard = supervisor.clone();
    signals::on_stop(move || {
        guard.stop(Duration::from_secs(15));
        eprintln!("\ninterrupted");
        std::process::exit(130);
    });
    supervisor.start()?;
    let client = Client::new(&supervisor.endpoint(), token, Duration::from_secs(10))?;
    let model = served_model(&client, &resolved.profile, None);
    Ok(Endpoint {
        url: supervisor.endpoint(),
        profile: resolved.profile,
        supervisor: Some(supervisor),
        loaded_under: None,
        model,
        context_length: settings.context_length,
        max_sessions: settings.max_sessions.or(settings.rows),
        token: None,
    })
}

/// Loads the profile under the running launcher for one `chat` or `bench`;
/// with `--keep-server` it stays loaded afterwards.
fn load_for_command(
    ctx: &Ctx,
    profile: &ProfileArgs,
    endpoint: &EndpointArgs,
    remote: Remote,
    token: Option<&str>,
) -> Result<Endpoint> {
    let resolved = resolve(ctx, profile, true)?;
    let request = ContextRequest {
        context_length: endpoint.context,
        parallel: endpoint.parallel,
        ..Default::default()
    };
    let prepared = prepare(
        &resolved,
        &request,
        endpoint.device.as_deref(),
        Spelling::Flags,
    )?;
    let settings = prepared.settings().clone();
    ctx.say(&format!(
        "loading {} under the launcher at {} for this command",
        resolved.profile.repo,
        remote.endpoint()
    ));
    let accepted = remote.load(&json!({
        "path": resolved.installed.path,
        "repo": resolved.profile.repo,
        "context": endpoint.context,
        "parallel": endpoint.parallel,
        "device": endpoint.device,
        "workers": endpoint.workers,
        "wait": endpoint.wait,
    }))?;
    let id = str_value(get(&accepted, "job")).unwrap_or("").to_string();
    let job = remote.follow(
        &id,
        Duration::from_secs_f64(endpoint.wait.max(1.0) + 60.0),
        &mut |_| {},
        &signals::stop_requested,
    )?;
    if str_value(get(&job, "state")) != Some("done") {
        return Err(Error::launch(py_str(get(&job, "error"))));
    }
    let url = remote.endpoint();
    let fallback = str_value(get(&remote.record, "control_token")).map(str::to_string);
    let client = Client::new(&url, token.or(fallback.as_deref()), Duration::from_secs(10))?;
    let model = served_model(
        &client,
        &resolved.profile,
        read_current(&ctx.state).as_ref(),
    );
    Ok(Endpoint {
        url,
        profile: resolved.profile,
        supervisor: None,
        loaded_under: (!endpoint.keep_server).then_some(remote),
        model,
        context_length: settings.context_length,
        max_sessions: settings.max_sessions.or(settings.rows),
        token: fallback,
    })
}

fn running(
    ctx: &Ctx,
    profile: &ProfileArgs,
    _endpoint: &EndpointArgs,
    token: Option<&str>,
) -> Result<Option<Endpoint>> {
    let Some((url, record)) = running_endpoint(&ctx.state) else {
        return Ok(None);
    };
    eprintln!(
        "using the running launch of {} at {url}",
        py_str(get(&record, "repo"))
    );
    // A model served by this state's launcher accepts its control token when
    // the caller has no key of its own.
    let fallback = running_launcher(&ctx.state)
        .filter(|launcher| {
            strict_int(get(launcher, "pid")).is_some()
                && strict_int(get(launcher, "pid")) == strict_int(get(&record, "launcher_pid"))
        })
        .and_then(|launcher| str_value(get(&launcher, "control_token")).map(str::to_string));
    let client = Client::new(&url, token.or(fallback.as_deref()), Duration::from_secs(10))?;
    let described = profile_for_endpoint(ctx, profile, &client, Some(&record));
    let model = served_model(&client, &described, Some(&record));
    Ok(Some(Endpoint {
        url,
        profile: described,
        supervisor: None,
        loaded_under: None,
        model,
        context_length: strict_int(get(&record, "context_length")),
        max_sessions: strict_int(get(&record, "max_sessions")),
        token: fallback,
    }))
}

/// `--keep-server`: start the launch as `serve --detach` so it outlives this command.
fn start_detached(ctx: &Ctx, profile: &ProfileArgs, endpoint: &EndpointArgs) -> Result<()> {
    let executable = std::env::current_exe()
        .map_err(|error| Error::launch(format!("cannot find this executable: {error}")))?;
    let mut arguments = vec![
        "--state".to_string(),
        ctx.state.to_string_lossy().into_owned(),
    ];
    if let Some(hub) = &ctx.hub_endpoint {
        arguments.extend(["--hub".to_string(), hub.clone()]);
    }
    for root in &ctx.profile_roots {
        arguments.extend([
            "--profiles".to_string(),
            root.to_string_lossy().into_owned(),
        ]);
    }
    arguments.push("serve".into());
    if let Some(target) = &profile.target {
        arguments.push(target.clone());
    }
    let mut option = |name: &str, value: Option<String>| {
        if let Some(value) = value {
            arguments.push(format!("--{name}={value}"));
        }
    };
    option("revision", profile.revision.clone());
    option("preset", profile.preset.clone());
    option("loom-fetch", profile.loom_fetch.clone());
    option("context", endpoint.context.map(|v| v.to_string()));
    option("parallel", endpoint.parallel.map(|v| v.to_string()));
    option("device", endpoint.device.clone());
    option("loom-serve", endpoint.loom_serve.clone());
    option("port", Some(endpoint.port.to_string()));
    option("workers", endpoint.workers.map(|v| v.to_string()));
    option("wait", Some(endpoint.wait.to_string()));
    if endpoint.no_proxy {
        arguments.push("--no-proxy".into());
    }
    if profile.offline {
        arguments.push("--offline".into());
    }
    arguments.push("--detach".into());
    let mut command = std::process::Command::new(executable);
    command.args(&arguments).stdout(std::process::Stdio::null());
    if let Some(key) = endpoint.api_key.as_deref().filter(|k| !k.is_empty()) {
        // The key travels in the environment, never in a visible argument list.
        command.env("LOOM_LAUNCH_API_KEY", key);
    }
    let status = command
        .status()
        .map_err(|error| Error::launch(format!("cannot start the background launch: {error}")))?;
    if !status.success() {
        return Err(Error::launch("the background launch did not become ready"));
    }
    Ok(())
}

/// The name an endpoint answers to, which need not be the card's model id.
pub(crate) fn served_model(
    client: &Client,
    profile: &Profile,
    record: Option<&Map<String, Value>>,
) -> Option<String> {
    let names = client.model_names(Some(Duration::from_secs(10)));
    if names.is_empty() {
        return None;
    }
    let deployment = record.and_then(|r| str_value(get(r, "deployment")).map(str::to_string));
    let candidates = [
        Some(profile.model_id.clone()),
        Some(profile.slug()),
        Some(profile.name.clone()),
        deployment,
    ];
    for candidate in candidates.into_iter().flatten() {
        if names.contains(&candidate) {
            return Some(candidate);
        }
    }
    names.into_iter().next()
}

/// The profile a conversation describes; an unknown endpoint gets a placeholder
/// named after the model it serves.
pub(crate) fn profile_for_endpoint(
    ctx: &Ctx,
    args: &ProfileArgs,
    client: &Client,
    record: Option<&Map<String, Value>>,
) -> Profile {
    // A launch records the package directory it serves; that describes it
    // without the hub, whatever its name.
    let recorded_path = record
        .and_then(|r| str_value(get(r, "profile_path")))
        .filter(|p| Path::new(p).join(CARD_DOCUMENT).is_file())
        .map(str::to_string);
    let target = args.target.clone().or_else(|| {
        recorded_path
            .clone()
            .or_else(|| record.and_then(|r| str_value(get(r, "repo")).map(str::to_string)))
    });
    if let Some(target) = target {
        let found = if let Some(local) = local_target(&target) {
            profile_for_local(ctx, &local, args.offline).map(|(p, _)| p)
        } else if let Some(installed) = InstallRegistry::new(&ctx.state).get(&target) {
            local_profile(ctx, &installed, args.offline)
        } else if let Some((_, profile)) =
            local::find(&ctx.state, &ctx.profile_roots, &target).unwrap_or(None)
        {
            Ok(profile)
        } else if target.contains('/') {
            let hub = ctx.hub(Duration::from_secs(30));
            profile_from_repo(
                ctx,
                &hub,
                &target,
                args.revision.as_deref(),
                args.offline,
                None,
            )
        } else {
            Err(Error::launcher(format!("{target} is not a known profile")))
        };
        match found {
            Ok(profile) => return profile,
            Err(error) => eprintln!("warning: {error}; describing the endpoint by its served name"),
        }
    }
    let names = client.model_names(Some(Duration::from_secs(5)));
    Profile::named(names.first().map(String::as_str).unwrap_or("unknown"))
}

// -- chat --------------------------------------------------------------------

fn template_kwargs(text: Option<&str>) -> Result<Map<String, Value>> {
    let Some(text) = text.filter(|t| !t.is_empty()) else {
        return Ok(Map::new());
    };
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(map)) => Ok(map),
        Ok(_) => Err(Error::launcher("--template-kwargs must be a JSON object")),
        Err(error) => Err(Error::launcher(format!(
            "--template-kwargs is not JSON: {error}"
        ))),
    }
}

fn cmd_chat(ctx: &Ctx, args: &ChatArgs) -> Result<i32> {
    let kwargs = template_kwargs(args.template_kwargs.as_deref())?;
    let endpoint = endpoint_for(ctx, &args.profile, &args.endpoint)?;
    let outcome = chat_with(ctx, args, &endpoint, kwargs);
    endpoint.finish();
    outcome
}

fn chat_with(
    ctx: &Ctx,
    args: &ChatArgs,
    endpoint: &Endpoint,
    kwargs: Map<String, Value>,
) -> Result<i32> {
    let client = Client::new(
        &endpoint.url,
        args.endpoint
            .api_key
            .as_deref()
            .filter(|k| !k.is_empty())
            .or(endpoint.token.as_deref()),
        Duration::from_secs_f64(args.endpoint.request_timeout.max(1.0)),
    )?;
    let mut session = ChatSession::new(&client, endpoint.profile.clone());
    if let Some(model) = &endpoint.model {
        session.model = Some(model.clone());
    }
    if let Some(thinking) = args.think.as_option() {
        session.thinking = Some(thinking);
    }
    session.max_tokens = args.max_tokens;
    session.temperature = args.temperature;
    session.session = args.session.clone();
    session.context_length = endpoint.context_length;
    session.template_kwargs = kwargs;
    session.set_system(args.system.as_deref());
    let save = |session: &ChatSession<'_>| {
        if let Some(path) = &args.save {
            match std::fs::write(path, dumps_pretty(&session.transcript())) {
                Ok(()) => eprintln!("transcript written to {path}"),
                Err(error) => eprintln!("error: cannot write {path}: {error}"),
            }
        }
    };
    if let Some(prompt) = &args.prompt {
        let mut stdout = |text: &str| {
            if !ctx.json {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
            }
        };
        let outcome = one_shot(&mut session, prompt, &mut stdout);
        let code = match &outcome {
            Ok(result) => {
                if ctx.json {
                    print_json(&describe_turn(result));
                }
                0
            }
            Err(failure) => {
                eprintln!("error: {}", failure.0);
                1
            }
        };
        save(&session);
        return Ok(code);
    }
    println!(
        "chatting with {} at {}",
        endpoint.profile.repo, endpoint.url
    );
    println!("type /help for commands, /exit or Ctrl-D to leave");
    let mut write = |text: &str| {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    };
    let code = run_repl(&mut session, &mut |prompt| stdin_line(prompt), &mut write);
    save(&session);
    Ok(code)
}

// -- bench -------------------------------------------------------------------

fn cmd_bench(ctx: &Ctx, args: &BenchArgs) -> Result<i32> {
    let store = BenchmarkStore::new(&ctx.state);
    if args.list {
        return Ok(list_runs(ctx, &store));
    }
    if let Some(id) = &args.show {
        return match store.load(id) {
            Some(record) => {
                print_record(ctx, &record);
                Ok(0)
            }
            None => {
                eprintln!("error: no stored run {id}");
                Ok(1)
            }
        };
    }
    let endpoint = endpoint_for(ctx, &args.profile, &args.endpoint)?;
    let outcome = bench_with(ctx, args, &store, &endpoint);
    endpoint.finish();
    outcome
}

fn bench_with(
    ctx: &Ctx,
    args: &BenchArgs,
    store: &BenchmarkStore,
    endpoint: &Endpoint,
) -> Result<i32> {
    let client = Client::new(
        &endpoint.url,
        args.endpoint
            .api_key
            .as_deref()
            .filter(|k| !k.is_empty())
            .or(endpoint.token.as_deref()),
        Duration::from_secs_f64(args.endpoint.request_timeout.max(1.0)),
    )?;
    let progress = |message: &str| {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "  {message}");
        let _ = out.flush();
    };
    let mut runner = Runner::new(&client, &endpoint.profile);
    if let Some(model) = &endpoint.model {
        runner.model = model.clone();
    }
    runner.repeats = args.repeats.max(1);
    runner.thinking = args.think.as_option();
    runner.session = args.session.clone();
    runner.max_sessions = args.endpoint.parallel.or(endpoint.max_sessions).or(endpoint
        .profile
        .execution
        .as_ref()
        .and_then(|e| e.rows));
    if !ctx.json {
        runner.on_progress = Some(&progress);
    }
    ctx.say(&format!(
        "benchmarking {} ({} suite) at {}",
        endpoint.profile.repo,
        args.suite.name(),
        endpoint.url
    ));
    let mut record = runner.run(args.suite.name())?;
    record.insert("endpoint".into(), json!(endpoint.url));
    if !args.no_store {
        let path = store.save(&mut record)?;
        record.insert("path".into(), json!(path.to_string_lossy()));
    }
    let ok = record.get("ok").and_then(Value::as_bool).unwrap_or(false);
    if ctx.json {
        print_json(&Value::Object(record));
    } else {
        for line in format_metrics(&record) {
            println!("{line}");
        }
        print_notes(&record);
        if let Some(path) = str_value(get(&record, "path")) {
            println!("\nstored: {path}");
        }
    }
    Ok(if ok { 0 } else { 1 })
}

fn print_notes(record: &Map<String, Value>) {
    let notes = arr(get(record, "notes"));
    if !notes.is_empty() {
        println!();
        for note in notes {
            println!("note: {}", py_str(note));
        }
    }
}

fn list_runs(ctx: &Ctx, store: &BenchmarkStore) -> i32 {
    let records = store.list(None);
    if ctx.json {
        print_json(&json!({"runs": records}));
        return 0;
    }
    if records.is_empty() {
        println!("no stored benchmark runs");
        return 0;
    }
    println!(
        "{:<26} {:<44} {:<10} {:<5} started",
        "id", "profile", "suite", "ok"
    );
    for record in &records {
        println!(
            "{:<26} {:<44} {:<10} {:<5} {}",
            py_str(get(record, "id")),
            py_str(get(record, "profile")),
            py_str(get(record, "suite")),
            py_str(get(record, "ok")),
            py_str(get(record, "started_at"))
        );
    }
    0
}

fn print_record(ctx: &Ctx, record: &Map<String, Value>) {
    if ctx.json {
        print_json(&Value::Object(record.clone()));
        return;
    }
    println!(
        "{}  suite {}  {}",
        py_str(get(record, "profile")),
        py_str(get(record, "suite")),
        py_str(get(record, "started_at"))
    );
    for line in format_metrics(record) {
        println!("{line}");
    }
    print_notes(record);
}

// -- observation -------------------------------------------------------------

fn cmd_status(ctx: &Ctx) -> Result<i32> {
    let record = read_current(&ctx.state);
    let remote = Remote::connect(&ctx.state);
    if ctx.json {
        print_json(&json!({
            "state": describe_state(&ctx.state),
            "launcher": launcher_document(remote.as_ref()),
            "model": record,
            "current": record,
        }));
        return Ok(0);
    }
    match &remote {
        Some(remote) => {
            print_launcher_status(remote);
        }
        None => println!("launcher  none running (`loom-launch launcher start` starts one)"),
    }
    let Some(record) = record else {
        if remote.is_some() {
            println!("model     none loaded (`loom-launch start PROFILE` loads one)");
            return Ok(0);
        }
        println!("model     none loaded");
        println!("no launch is running");
        println!("state directory: {}", ctx.state.display());
        return Ok(1);
    };
    println!(
        "model     {} @ {}",
        py_str(get(&record, "repo")),
        py_str(get(&record, "endpoint"))
    );
    if str_value(get(&record, "kind")) == Some("scenario") {
        println!(
            "  scenario  the package's own deployment {} across {}",
            py_str(get(&record, "deployment")),
            py_str(get(&record, "device"))
        );
    } else {
        println!("  deployment {}", py_str(get(&record, "deployment")));
    }
    println!("  server    {}", py_str(get(&record, "server_endpoint")));
    println!("  pid       {}", py_str(get(&record, "pid")));
    println!("  started   {}", py_str(get(&record, "started_at")));
    if let Some(pid) = strict_int(get(&record, "pid")) {
        if signals::exited(pid) {
            println!(
                "  probe     the server (pid {pid}) is gone; `loom-launch stop` clears the record"
            );
            return Ok(1);
        }
    }
    let ours = remote.as_ref().is_some_and(|remote| {
        remote.pid().is_some() && remote.pid() == strict_int(get(&record, "launcher_pid"))
    });
    if !ours && remote.is_some() {
        println!(
            "  launcher  started without a launcher service (pid {})",
            py_str(get(&record, "launcher_pid"))
        );
    }
    let token = std::env::var("LOOM_LAUNCH_API_KEY")
        .ok()
        .filter(|t| !t.is_empty())
        .or_else(|| {
            remote
                .as_ref()
                .filter(|_| ours)
                .and_then(|r| str_value(get(&r.record, "control_token")).map(str::to_string))
        });
    let endpoint = str_value(get(&record, "endpoint")).unwrap_or("");
    let client = Client::new(endpoint, token.as_deref(), Duration::from_secs(5))?;
    let (status, document) = match client.get_json("/loom/status", Some(Duration::from_secs(5))) {
        Ok(answer) => answer,
        Err(error) => {
            println!("  probe     failed: {error}");
            return Ok(1);
        }
    };
    if status != 200 {
        println!("  probe     HTTP {status}");
        return Ok(1);
    }
    let metrics = obj(get(obj(get(&document, "proxy")), "metrics"));
    let count = |key: &str| strict_int(get(metrics, key)).unwrap_or(0);
    println!(
        "  traffic   {} request(s), {} completion tokens, in flight {}",
        count("requests"),
        count("completion_tokens"),
        count("in_flight")
    );
    let ttft = obj(get(metrics, "ttft_seconds"));
    let decode = obj(get(metrics, "decode_tokens_per_second"));
    let seconds = |value: &Value| match value.as_f64() {
        Some(_) => format!("{value}s"),
        None => "-".to_string(),
    };
    println!(
        "  latency   ttft p50 {} p95 {}, decode mean {}",
        seconds(get(ttft, "p50")),
        seconds(get(ttft, "p95")),
        human_rate(get(decode, "mean").as_f64())
    );
    Ok(0)
}

fn cmd_logs(ctx: &Ctx, args: &LogsArgs) -> Result<i32> {
    let runs = RunStore::new(&ctx.state);
    let run = match &args.run {
        Some(id) => Some(runs.open(id)),
        None => read_current(&ctx.state)
            .and_then(|r| str_value(get(&r, "run_id")).map(|id| runs.open(id)))
            .filter(|r| r.root.is_dir())
            .or_else(|| runs.latest()),
    };
    let Some(run) = run else {
        eprintln!("error: no runs recorded");
        return Ok(1);
    };
    let path = if args.events {
        run.event_log()
    } else {
        run.serve_log()
    };
    if !path.is_file() {
        eprintln!("error: {} does not exist", path.display());
        return Ok(1);
    }
    let bytes = std::fs::read(&path).unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    for line in &lines[lines.len().saturating_sub(args.lines)..] {
        println!("{line}");
    }
    if args.follow {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(&path)
            .map_err(|error| Error::launcher(format!("cannot open {}: {error}", path.display())))?;
        let mut position = file.seek(SeekFrom::End(0)).unwrap_or(bytes.len() as u64);
        loop {
            let length = std::fs::metadata(&path)
                .map(|m| m.len())
                .unwrap_or(position);
            if length < position {
                position = 0;
            }
            if length > position {
                let _ = file.seek(SeekFrom::Start(position));
                let mut chunk = Vec::new();
                let _ = (&mut file).take(length - position).read_to_end(&mut chunk);
                position += chunk.len() as u64;
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(&chunk);
                let _ = out.flush();
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
    Ok(0)
}

/// `stop`: unloads the model and says the launcher is still running (it
/// keeps its endpoint and dashboard); `--all` stops the launcher too.
fn cmd_stop(ctx: &Ctx, args: &StopArgs) -> Result<i32> {
    if args.all {
        return cmd_launcher_stop(ctx, args.wait.max(0.0) + 15.0);
    }
    if let Some(remote) = Remote::connect(&ctx.state) {
        let job = unload_with(&remote, args.wait)?;
        if ctx.json {
            let mut document = unload_document(job.as_ref(), &remote);
            if let Some(job) = &job {
                let result = obj(get(job, "result"));
                document["stopped"] = get(result, "pid").clone();
                document["repo"] = get(result, "repo").clone();
                document["run_id"] = get(result, "run").clone();
            }
            print_json(&document);
        } else {
            match &job {
                Some(job) => println!("{}", unloaded_line(job)),
                None => println!("no model is loaded"),
            }
            println!(
                "the launcher is still running at {}/ (pid {}); `loom-launch launcher stop` ends \
                 it (or `loom-launch stop --all`)",
                remote.endpoint(),
                remote.pid().map(|p| p.to_string()).unwrap_or_default()
            );
        }
        return Ok(if job.is_some() { 0 } else { 1 });
    }
    // No launcher answers: a record it left behind is cleaned up (its model
    // stopped), and a launch of before the split is stopped as it always was.
    if crate::launcher::read_launcher(&ctx.state).is_some() {
        if let LauncherStop::Stopped { pid, model, .. } = stop_launcher(
            &ctx.state,
            Duration::from_secs_f64(args.wait.max(1.0) + 15.0),
        ) {
            println!(
                "stopped an unresponsive launcher (pid {pid}){}",
                model.map(|m| format!(" and {m}")).unwrap_or_default()
            );
            return Ok(0);
        }
    }
    let wait = Duration::from_secs_f64(args.wait.max(0.0));
    match crate::server::stop_current(&ctx.state, wait) {
        StopOutcome::NotRunning => {
            println!("no launch is running");
            Ok(1)
        }
        StopOutcome::NoPid => {
            eprintln!("error: the current record has no pid");
            Ok(1)
        }
        StopOutcome::Stopped {
            pid,
            repo,
            run_id,
            forced,
        } => {
            if ctx.json {
                print_json(
                    &json!({"stopped": pid, "repo": repo, "run_id": run_id, "forced": forced}),
                );
            } else {
                println!(
                    "stopped launch {pid} ({repo}){}",
                    if forced { " with SIGKILL" } else { "" }
                );
            }
            Ok(0)
        }
    }
}

fn cmd_hardware(ctx: &Ctx) -> Result<i32> {
    let hardware = detect();
    let document = hardware.describe();
    if ctx.json {
        print_json(&document);
        return Ok(0);
    }
    println!("system   {} {}", hardware.system, hardware.machine);
    println!("cpu      {}", hardware.cpu.as_deref().unwrap_or("unknown"));
    println!("memory   {}", py_str(&document["memory"]));
    match hardware.gpu() {
        Some(gpu) => println!(
            "gpu      {} ({})",
            gpu.name,
            gpu.gfx_arch.as_deref().unwrap_or("arch unknown")
        ),
        None => println!("gpu      none detected"),
    }
    println!(
        "npu      {}",
        if hardware.npu {
            "present"
        } else {
            "not detected"
        }
    );
    println!("device   {}", hardware.default_device_uri());
    Ok(0)
}

fn cmd_doctor(ctx: &Ctx, args: &DoctorArgs) -> Result<i32> {
    let mut checks: Vec<(String, Option<bool>, String)> = Vec::new();
    match serve_command(args.loom_serve.as_deref()) {
        Ok(path) => checks.push(("loom-serve".into(), Some(true), path)),
        Err(error) => checks.push(("loom-serve".into(), Some(false), error.message)),
    }
    match fetch_command(args.loom_fetch.as_deref()) {
        Ok(command) => checks.push(("loom-fetch".into(), Some(true), command.join(" "))),
        Err(error) => checks.push(("loom-fetch".into(), Some(false), error.message)),
    }
    let hub = ctx.hub(Duration::from_secs(20));
    match hub.profiles(Some("loom"), None, 1, &[PROFILE_TAG.to_string()]) {
        Ok(found) => checks.push((
            "hub".into(),
            Some(true),
            format!("{} ({} profile(s) sampled)", hub.endpoint, found.len()),
        )),
        Err(error) if error.kind == ErrorKind::HubAuth => {
            checks.push(("hub".into(), None, format!("{}: {error}", hub.endpoint)))
        }
        Err(error) => checks.push((
            "hub".into(),
            None,
            format!("{} unreachable: {error}", hub.endpoint),
        )),
    }
    let probe = ctx.state.join("doctor.tmp");
    let state_check = std::fs::create_dir_all(&ctx.state)
        .and_then(|_| std::fs::write(&probe, "ok"))
        .and_then(|_| std::fs::remove_file(&probe));
    match state_check {
        Ok(()) => checks.push(("state".into(), Some(true), ctx.state.display().to_string())),
        Err(error) => checks.push((
            "state".into(),
            Some(false),
            format!("{}: {error}", ctx.state.display()),
        )),
    }
    let hardware = detect();
    checks.push((
        "device".into(),
        Some(!hardware.devices.is_empty()),
        format!(
            "{}{}",
            hardware.default_device_uri(),
            if hardware.devices.is_empty() {
                " (no AMD GPU detected)"
            } else {
                ""
            }
        ),
    ));
    let ok = checks.iter().all(|(_, status, _)| *status != Some(false));
    if ctx.json {
        let documents: Vec<Value> = checks
            .iter()
            .map(|(name, status, detail)| json!({"check": name, "ok": status, "detail": detail}))
            .collect();
        print_json(&json!({"checks": documents}));
    } else {
        for (name, status, detail) in &checks {
            let marker = match status {
                Some(true) => "ok  ",
                Some(false) => "FAIL",
                None => "warn",
            };
            println!("{marker} {name:<12} {detail}");
        }
    }
    Ok(if ok { 0 } else { 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Cli {
        let mut argv = vec!["loom-launch"];
        argv.extend_from_slice(arguments);
        Cli::try_parse_from(argv).unwrap()
    }

    #[test]
    fn every_verb_parses() {
        for verb in ["show", "install", "serve", "chat", "bench"] {
            let cli = parse(&[verb, "loom-fun/a-loom"]);
            assert!(cli.command.is_some(), "{verb}");
        }
        for verb in ["status", "stop", "hardware", "doctor", "logs", "dashboard"] {
            assert!(parse(&[verb]).command.is_some(), "{verb}");
        }
        assert!(matches!(
            parse(&["search", "loom"]).command,
            Some(Verb::Search(_))
        ));
    }

    #[test]
    fn serving_defaults_are_loopback_and_instrumented() {
        let Some(Verb::Serve(args)) = parse(&["serve", "loom-fun/a-loom"]).command else {
            panic!("serve did not parse");
        };
        assert_eq!(args.host, "127.0.0.1");
        assert_eq!(args.port, 8080);
        assert!(!args.no_proxy);
        assert!(!args.detach);
        assert_eq!(args.wait, 1800.0);
        assert_eq!(args.loom_serve, None);
        assert_eq!(args.proxy_port, 0);
    }

    #[test]
    fn context_and_parallel_are_available_on_every_launching_verb() {
        let Some(Verb::Serve(serve)) = parse(&[
            "serve",
            "x/y",
            "--context",
            "8192",
            "--parallel",
            "2",
            "--draft-depth",
            "0",
        ])
        .command
        else {
            panic!()
        };
        assert_eq!(
            (serve.context, serve.parallel, serve.draft_depth),
            (Some(8192), Some(2), Some(0))
        );
        let Some(Verb::Chat(chat)) = parse(&["chat", "x/y", "--context", "8192"]).command else {
            panic!()
        };
        assert_eq!(chat.endpoint.context, Some(8192));
        let Some(Verb::Bench(bench)) = parse(&["bench", "x/y", "--context", "8192"]).command else {
            panic!()
        };
        assert_eq!(bench.endpoint.context, Some(8192));
    }

    #[test]
    fn bench_defaults_measure_a_comparable_suite() {
        let Some(Verb::Bench(args)) = parse(&["bench", "loom-fun/a-loom"]).command else {
            panic!()
        };
        assert_eq!(args.suite, Suite::Standard);
        assert_eq!(args.repeats, 1);
        assert_eq!(args.think, Think::Off);
        assert!(!args.no_store);
    }

    #[test]
    fn chat_defaults_follow_the_package_and_allow_an_endpoint() {
        let Some(Verb::Chat(args)) = parse(&[
            "chat",
            "--base-url",
            "http://127.0.0.1:8080",
            "--prompt",
            "hi",
        ])
        .command
        else {
            panic!()
        };
        assert_eq!(args.think, Think::Auto);
        assert_eq!(
            args.endpoint.base_url.as_deref(),
            Some("http://127.0.0.1:8080")
        );
        assert_eq!(args.prompt.as_deref(), Some("hi"));
    }

    #[test]
    fn global_flags_work_on_either_side_of_the_verb() {
        let cli = parse(&["status", "--json", "--state", "x"]);
        assert!(cli.json);
        assert_eq!(cli.state.as_deref(), Some("x"));
        let Some(Verb::Search(args)) = parse(&["--json", "search", "loom"]).command else {
            panic!()
        };
        assert!(!args.no_details);
        assert_eq!(args.limit, 20);
        assert!(args.tags.is_empty());
    }

    #[test]
    fn usage_errors_exit_with_two() {
        let error = Cli::try_parse_from(["loom-launch", "bench", "--suite", "huge"]).unwrap_err();
        assert_eq!(error.exit_code(), 2);
        let none = Cli::try_parse_from(["loom-launch"]).unwrap();
        assert_eq!(run(none), 2);
    }

    #[test]
    fn status_reports_nothing_running_in_a_fresh_state() {
        let root = crate::store::tests::scratch("cli-status");
        let cli = parse(&["--state", root.to_str().unwrap(), "status"]);
        assert_eq!(run(cli), 1);
    }

    #[test]
    fn template_kwargs_must_be_an_object() {
        assert!(
            template_kwargs(Some("{\"preserve_thinking\": true}")).unwrap()["preserve_thinking"]
                == true
        );
        assert!(template_kwargs(Some("[1]")).is_err());
        assert!(template_kwargs(Some("{")).is_err());
        assert!(template_kwargs(None).unwrap().is_empty());
    }

    #[test]
    fn cache_stems_map_back_to_repositories() {
        assert_eq!(
            guess_repo(
                "loom-fun-unsloth-Qwen3-4B-loom",
                "see huggingface.co/loom-fun/unsloth-Qwen3-4B-loom"
            ),
            "loom-fun/unsloth-Qwen3-4B-loom"
        );
        assert_eq!(guess_repo("a-b", ""), "a/b");
    }
}
