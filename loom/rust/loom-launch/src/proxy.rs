//! The launcher's fronting proxy: one instrumented OpenAI endpoint.
//!
//! The native server owns inference. This proxy sits in front of it and records
//! what a client experienced: time to first token, decode rate, token usage,
//! status and bytes, per request, as JSONL and as Prometheus counters. It also
//! serves the launcher's own routes under `/loom/` (and the dashboard routes), so
//! a client sees one address for both inference and instrumentation.
//!
//! Each request runs on its own thread. Server-sent events are relayed line by
//! line through the raw connection writer with explicit chunked framing and a
//! flush per line, so nothing is buffered between the server and the client.

use crate::dashboard;
use crate::errors::{Error, Result};
use crate::metrics::{route_name, Metrics, RequestRecord, LAUNCHER_PREFIX};
use crate::store::JsonlWriter;
use crate::values::{dumps_pretty, get, now_seconds, obj, random_hex_id, str_value, strict_int};
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Headers that describe one hop only and are never forwarded.
const HOP_BY_HOP: [&str; 12] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
];

/// Where the native server listens.
#[derive(Clone, Debug)]
pub struct Upstream {
    pub host: String,
    pub port: u16,
}

impl Upstream {
    pub fn describe(&self) -> Value {
        json!({"host": self.host, "port": self.port})
    }

    pub(crate) fn base_url(&self) -> String {
        if self.host.contains(':') {
            format!("http://[{}]:{}", self.host, self.port)
        } else {
            format!("http://{}:{}", self.host, self.port)
        }
    }
}

/// Where `/v1/*` goes while a model is loaded: its native listener and the
/// run's request log.
#[derive(Clone)]
pub struct Target {
    pub upstream: Upstream,
    pub request_log: Option<Arc<JsonlWriter>>,
    /// Who pointed the route here (a run id), so only its owner clears it.
    pub owner: String,
}

/// The proxy's upstream, which a launcher points at the model it loads and
/// clears when the model goes away. With no target, `/v1/*` answers 503.
#[derive(Default)]
pub struct Route {
    target: RwLock<Option<Arc<Target>>>,
}

impl Route {
    /// A route that always leads to `upstream` (a launch with its own proxy).
    pub fn fixed(upstream: Upstream, request_log: Option<Arc<JsonlWriter>>) -> Arc<Route> {
        let route = Route::default();
        route.point(Target {
            upstream,
            request_log,
            owner: String::new(),
        });
        Arc::new(route)
    }

    pub fn current(&self) -> Option<Arc<Target>> {
        self.target.read().unwrap().clone()
    }

    pub fn upstream(&self) -> Option<Upstream> {
        self.current().map(|target| target.upstream.clone())
    }

    pub fn point(&self, target: Target) {
        *self.target.write().unwrap() = Some(Arc::new(target));
    }

    /// Clears the route when `owner` is the one that set it; true if it did.
    pub fn clear_if(&self, owner: &str) -> bool {
        let mut slot = self.target.write().unwrap();
        if slot.as_ref().is_some_and(|t| t.owner == owner) {
            *slot = None;
            return true;
        }
        false
    }
}

/// Extra launch facts merged into `/loom/status`.
pub type StatusProvider = Arc<dyn Fn() -> Map<String, Value> + Send + Sync>;

/// The document a launcher answers `/healthz` with itself.
pub type HealthProvider = Arc<dyn Fn() -> Value + Send + Sync>;

/// What `/v1/*` answers while no model is loaded.
pub const NO_MODEL: &str = "no model is loaded";

/// The OpenAI-shaped body of a `/v1/*` request that reached a launcher with no model.
pub fn no_model_body() -> Value {
    error_body(NO_MODEL, "loom_launch_error", "UNAVAILABLE")
}

/// Requests a launcher answered itself, by route group (`page`, `api`,
/// `loom`, `health`, `no_model`).
#[derive(Debug, Default)]
pub struct LauncherCounters {
    counts: std::sync::Mutex<std::collections::BTreeMap<&'static str, u64>>,
}

impl LauncherCounters {
    pub fn count(&self, group: &'static str) {
        *self.counts.lock().unwrap().entry(group).or_insert(0) += 1;
    }

    pub fn snapshot(&self) -> Value {
        let counts = self.counts.lock().unwrap();
        let mut document = Map::new();
        for group in ["page", "api", "loom", "health", "no_model"] {
            document.insert(group.into(), json!(counts.get(group).copied().unwrap_or(0)));
        }
        Value::Object(document)
    }
}

/// How to start one proxy.
#[derive(Clone)]
pub struct ProxyOptions {
    /// Where `/v1/*` goes: a fixed native listener, or a launcher's route.
    pub route: Arc<Route>,
    pub bind_host: String,
    /// Zero picks an ephemeral port.
    pub bind_port: u16,
    pub auth_token: Option<String>,
    pub request_timeout: Duration,
    pub request_log: Option<Arc<JsonlWriter>>,
    pub status_provider: Option<StatusProvider>,
    /// When set, the dashboard routes (`/`, `/api/*`) are served with this state root.
    pub dashboard_state: Option<PathBuf>,
    /// Model hub the dashboard searches and installs from (default: the launcher's).
    pub hub_endpoint: Option<String>,
    /// Let the dashboard's `POST /api/fetch` fetch public URLs (`--allow-web`).
    pub allow_web: bool,
    /// Directories the dashboard lists local profiles from.
    pub profile_roots: Vec<PathBuf>,
    /// A launcher service: the dashboard loads and unloads models behind
    /// `route`, and `/healthz` is answered here, model or not.
    pub service: Option<dashboard::Service>,
    /// A secret the launcher's own command line presents (from
    /// `launcher.json`) in place of the API key; it also lets `start` load a
    /// package directory the page could not name.
    pub control_token: Option<String>,
}

impl ProxyOptions {
    pub fn new(upstream: Upstream) -> ProxyOptions {
        ProxyOptions::routed(Route::fixed(upstream, None))
    }

    /// A proxy whose upstream follows `route`.
    pub fn routed(route: Arc<Route>) -> ProxyOptions {
        ProxyOptions {
            route,
            bind_host: "127.0.0.1".into(),
            bind_port: 0,
            auth_token: None,
            request_timeout: Duration::from_secs(900),
            request_log: None,
            status_provider: None,
            dashboard_state: None,
            hub_endpoint: None,
            allow_web: false,
            profile_roots: Vec::new(),
            service: None,
            control_token: None,
        }
    }
}

/// Carries requests to one upstream and records what each client experienced.
///
/// The proxy forwards every `/v1/*` request through one; the dashboard's chat
/// route uses the same forwarder, so its turns are counted like any other.
pub(crate) struct Forwarder {
    pub(crate) route: Arc<Route>,
    pub(crate) metrics: Arc<Metrics>,
    agent: ureq::Agent,
}

impl Forwarder {
    pub(crate) fn new(
        upstream: Upstream,
        metrics: Arc<Metrics>,
        request_log: Option<Arc<JsonlWriter>>,
        request_timeout: Duration,
    ) -> Forwarder {
        Forwarder::routed(
            Route::fixed(upstream, request_log),
            metrics,
            request_timeout,
        )
    }

    /// A forwarder whose upstream follows `route` (a launcher's).
    pub(crate) fn routed(
        route: Arc<Route>,
        metrics: Arc<Metrics>,
        request_timeout: Duration,
    ) -> Forwarder {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(request_timeout)
            .timeout_write(request_timeout)
            .redirects(0)
            .max_idle_connections(0)
            .build();
        Forwarder {
            route,
            metrics,
            agent,
        }
    }

    /// The native listener requests go to right now, if any.
    pub(crate) fn upstream(&self) -> Option<Upstream> {
        self.route.upstream()
    }
}

struct Shared {
    forwarder: Arc<Forwarder>,
    status_provider: Option<StatusProvider>,
    health: Option<HealthProvider>,
    auth_token: Option<String>,
    control_token: Option<String>,
    address: String,
    dashboard: Option<Arc<dashboard::Context>>,
    counters: Arc<LauncherCounters>,
}

/// An instrumented front end for one native server.
pub struct Proxy {
    shared: Arc<Shared>,
    server: Arc<tiny_http::Server>,
    thread: Option<JoinHandle<()>>,
    pub host: String,
    pub port: u16,
}

impl Proxy {
    pub fn start(options: ProxyOptions, metrics: Arc<Metrics>) -> Result<Proxy> {
        let server = tiny_http::Server::http((options.bind_host.as_str(), options.bind_port))
            .map_err(|error| {
                Error::launch(format!(
                    "cannot bind the proxy to {}:{}: {error}",
                    options.bind_host, options.bind_port
                ))
            })?;
        let local = server
            .server_addr()
            .to_ip()
            .ok_or_else(|| Error::launch("the proxy did not bind an IP address"))?;
        let host = local.ip().to_string();
        let port = local.port();
        // Bound to every interface, the proxy is still reached locally over
        // loopback; that is the address clients on this machine record.
        let address = crate::network::http_url(&crate::network::local_connect_host(&host), port);
        let auth_token = options.auth_token.filter(|t| !t.is_empty());
        let control_token = options.control_token.filter(|t| !t.is_empty());
        // A fixed route carries the request log a caller gave the proxy.
        if let (Some(log), Some(target)) = (options.request_log, options.route.current()) {
            if target.request_log.is_none() {
                let mut target = (*target).clone();
                target.request_log = Some(log);
                options.route.point(target);
            }
        }
        let forwarder = Arc::new(Forwarder::routed(
            options.route,
            metrics,
            options.request_timeout,
        ));
        let counters = Arc::new(LauncherCounters::default());
        let launcher = options.service.is_some();
        let dashboard = match options.service {
            Some(service) => Some(dashboard::Context::launcher(
                service,
                dashboard::Front {
                    forwarder: forwarder.clone(),
                    endpoint: address.clone(),
                    auth: auth_token.is_some(),
                    counters: counters.clone(),
                },
            )),
            None => options.dashboard_state.as_ref().map(|state| {
                dashboard::Context::hosted(dashboard::Hosted {
                    state_root: state.clone(),
                    hub_endpoint: options.hub_endpoint.clone(),
                    forwarder: forwarder.clone(),
                    status_provider: options.status_provider.clone(),
                    endpoint: address.clone(),
                    auth_token: auth_token.clone(),
                    allow_web: options.allow_web,
                    profile_roots: options.profile_roots.clone(),
                })
            }),
        };
        if let Some(context) = &dashboard {
            context.set_listener(&host, port);
        }
        // A launcher reports itself: its status and health do not depend on
        // a model being loaded.
        let (status_provider, health) = match (&dashboard, launcher) {
            (Some(context), true) => (
                Some(context.status_provider()),
                Some(context.health_provider()),
            ),
            _ => (options.status_provider, None),
        };
        let shared = Arc::new(Shared {
            forwarder,
            status_provider,
            health,
            auth_token,
            control_token,
            address,
            dashboard,
            counters,
        });
        let server = Arc::new(server);
        let accept_server = server.clone();
        let accept_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("loom-launch-proxy".into())
            .spawn(move || {
                for request in accept_server.incoming_requests() {
                    let shared = accept_shared.clone();
                    let _ = std::thread::Builder::new()
                        .name("loom-launch-request".into())
                        .spawn(move || handle(&shared, request));
                }
            })
            .map_err(|error| Error::launch(format!("cannot start the proxy thread: {error}")))?;
        Ok(Proxy {
            shared,
            server,
            thread: Some(thread),
            host,
            port,
        })
    }

    pub fn address(&self) -> &str {
        &self.shared.address
    }

    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.shared.forwarder.metrics
    }

    /// The dashboard these routes serve (a launcher's, or a launch's own).
    pub fn dashboard(&self) -> Option<&Arc<dashboard::Context>> {
        self.shared.dashboard.as_ref()
    }

    pub fn status(&self) -> Map<String, Value> {
        status_document(&self.shared)
    }

    pub fn stop(&mut self) {
        if let Some(dashboard) = &self.shared.dashboard {
            dashboard.close();
        }
        self.server.unblock();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Proxy {
    fn drop(&mut self) {
        self.stop();
    }
}

fn status_document(shared: &Shared) -> Map<String, Value> {
    let mut status = Map::new();
    status.insert(
        "proxy".into(),
        json!({
            "address": shared.address,
            "upstream": shared.forwarder.upstream().map(|u| u.describe()),
            "metrics": shared.forwarder.metrics.snapshot(),
        }),
    );
    if let Some(provider) = &shared.status_provider {
        for (key, value) in provider() {
            status.insert(key, value);
        }
    }
    status
}

/// Whether the native listener answers right now.
///
/// The gauge behind `loom_launch_upstream_up` is otherwise only written when a
/// request reaches the server, so a launch that has served nothing yet would
/// report itself down. The metrics route probes instead.
fn upstream_up(upstream: Option<&Upstream>) -> bool {
    let Some(upstream) = upstream else {
        return false;
    };
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_millis(800))
        .build();
    match agent
        .get(&format!("{}/healthz", upstream.base_url()))
        .call()
    {
        Ok(_) | Err(ureq::Error::Status(_, _)) => true,
        Err(ureq::Error::Transport(_)) => false,
    }
}

/// The bearer token a request carries, if any.
fn bearer(headers: &[(String, String)]) -> Option<&str> {
    let value = header(headers, "authorization")?;
    let (scheme, rest) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| rest.trim())
        .filter(|t| !t.is_empty())
}

/// Whether the request presents the launcher's own control token.
fn controlled(shared: &Shared, headers: &[(String, String)]) -> bool {
    match (&shared.control_token, bearer(headers)) {
        (Some(token), Some(offered)) => offered == token,
        _ => false,
    }
}

fn authorized(shared: &Shared, headers: &[(String, String)]) -> bool {
    let Some(token) = &shared.auth_token else {
        return true;
    };
    bearer(headers) == Some(token.as_str()) || controlled(shared, headers)
}

/// The dashboard's event stream cannot set headers (EventSource), so the
/// dashboard routes also accept the token as `access_token` in the query.
fn dashboard_authorized(shared: &Shared, headers: &[(String, String)], path: &str) -> bool {
    if authorized(shared, headers) {
        return true;
    }
    let query = path.split_once('?').map(|(_, q)| q).unwrap_or("");
    match (&shared.auth_token, query_value(query, "access_token")) {
        (Some(token), Some(offered)) => !offered.is_empty() && &offered == token,
        _ => false,
    }
}

/// Request headers as lowercase name/value pairs.
pub(crate) fn request_headers(request: &tiny_http::Request) -> Vec<(String, String)> {
    request
        .headers()
        .iter()
        .map(|h| {
            (
                h.field.as_str().as_str().to_ascii_lowercase(),
                h.value.as_str().to_string(),
            )
        })
        .collect()
}

pub(crate) fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

pub(crate) fn json_response(
    status: u16,
    document: &Value,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    raw_response(
        status,
        "application/json",
        dumps_pretty(document).into_bytes(),
    )
}

pub(crate) fn raw_response(
    status: u16,
    content_type: &str,
    payload: Vec<u8>,
) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let mut response = tiny_http::Response::from_data(payload).with_status_code(status);
    if let Ok(header) = tiny_http::Header::from_bytes("Content-Type", content_type) {
        response.add_header(header);
    }
    response
}

fn error_body(message: &str, kind: &str, code: &str) -> Value {
    json!({"error": {"message": message, "type": kind, "code": code}})
}

fn handle(shared: &Arc<Shared>, mut request: tiny_http::Request) {
    let method = request.method().as_str().to_uppercase();
    let path = request.url().to_string();
    let target = path.split('?').next().unwrap_or("").to_string();
    let headers = request_headers(&request);
    let client = request.remote_addr().map(|a| a.to_string());
    let launcher = shared.health.is_some();
    if path.starts_with(LAUNCHER_PREFIX) {
        if launcher {
            shared.counters.count("loom");
        }
        serve_launcher(shared, request, &path, &headers);
        return;
    }
    if let Some(context) = &shared.dashboard {
        if dashboard::claims(&target) {
            if launcher {
                shared.counters.count(if target.starts_with("/api/") {
                    "api"
                } else {
                    "page"
                });
            }
            if !dashboard::is_public(&target) && !dashboard_authorized(shared, &headers, &path) {
                let _ = request.respond(json_response(
                    401,
                    &error_body(
                        "the launcher requires a bearer token",
                        "loom_launch_auth_error",
                        "PERMISSION_DENIED",
                    ),
                ));
                return;
            }
            let trusted = controlled(shared, &headers);
            dashboard::handle_trusted(context, request, &method, &path, trusted);
            return;
        }
    }
    // A launcher is healthy with or without a model; it says which.
    if let (Some(health), "/healthz") = (&shared.health, target.as_str()) {
        shared.counters.count("health");
        let _ = request.respond(json_response(200, &health()));
        return;
    }
    // Health probes stay anonymous so tooling can find a protected launch.
    if target != "/healthz" && !authorized(shared, &headers) {
        let _ = request.respond(json_response(
            401,
            &error_body(
                "the launcher requires a bearer token",
                "loom_launch_auth_error",
                "PERMISSION_DENIED",
            ),
        ));
        return;
    }
    let mut body = Vec::new();
    let _ = request.as_reader().read_to_end(&mut body);
    if launcher && shared.forwarder.route.current().is_none() {
        shared.counters.count("no_model");
    }
    shared
        .forwarder
        .forward(request, &method, &path, &headers, body, client);
}

fn serve_launcher(
    shared: &Shared,
    request: tiny_http::Request,
    path: &str,
    headers: &[(String, String)],
) {
    if !authorized(shared, headers) {
        let _ = request.respond(json_response(
            401,
            &error_body(
                "the launcher requires a bearer token",
                "loom_launch_auth_error",
                "PERMISSION_DENIED",
            ),
        ));
        return;
    }
    let (target, query) = path.split_once('?').unwrap_or((path, ""));
    let response = match target.strip_prefix(LAUNCHER_PREFIX).unwrap_or("") {
        "metrics" => {
            shared
                .forwarder
                .metrics
                .set_upstream_up(upstream_up(shared.forwarder.upstream().as_ref()));
            raw_response(
                200,
                "text/plain; version=0.0.4",
                shared.forwarder.metrics.prometheus().into_bytes(),
            )
        }
        "status" => json_response(200, &Value::Object(status_document(shared))),
        "requests" => {
            let limit = query_value(query, "limit")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(50);
            json_response(
                200,
                &json!({"object": "list", "data": shared.forwarder.metrics.recent(limit)}),
            )
        }
        _ => json_response(
            404,
            &error_body(
                &format!("{path} is not a launcher route"),
                "loom_launch_error",
                "NOT_FOUND",
            ),
        ),
    };
    let _ = request.respond(response);
}

/// One query-string value, undecoded beyond `+` and `%XX`.
pub(crate) fn query_value(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        (key == name).then(|| percent_decode(value))
    })
}

/// Percent-encodes everything but RFC 3986 unreserved characters.
pub(crate) fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

pub(crate) fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => out.push(b' '),
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(value) => {
                        out.push(value);
                        index += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            other => out.push(other),
        }
        index += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[derive(Default)]
struct Observation {
    status: u16,
    error: Option<String>,
    ttft: Option<f64>,
    chunks: usize,
    usage: Map<String, Value>,
    response_bytes: usize,
    model: Option<String>,
    streamed: bool,
}

impl Forwarder {
    /// Forwards one request, relays the answer (streams unbuffered) and
    /// records it in the metrics and the request log.
    pub(crate) fn forward(
        &self,
        request: tiny_http::Request,
        method: &str,
        path: &str,
        headers: &[(String, String)],
        body: Vec<u8>,
        client: Option<String>,
    ) {
        let started_at = now_seconds();
        let started = Instant::now();
        let record_id = header(headers, "x-request-id")
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .unwrap_or_else(random_hex_id);
        let mut model = None;
        let mut stream_requested = false;
        if !body.is_empty() {
            if let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(&body) {
                model = str_value(get(&payload, "model")).map(str::to_string);
                stream_requested = crate::values::truthy(get(&payload, "stream"));
            }
        }
        let Some(target) = self.route.current() else {
            // No model behind the launcher: answered here, never forwarded.
            let _ = request.respond(json_response(503, &no_model_body()));
            self.metrics.set_upstream_up(false);
            self.metrics.observe(&RequestRecord {
                request_id: record_id,
                method: method.to_string(),
                path: path.to_string(),
                route: route_name(path).to_string(),
                started_at,
                status: 503,
                stream: stream_requested,
                model,
                duration_seconds: Some(started.elapsed().as_secs_f64()),
                request_bytes: body.len(),
                client,
                error: Some(NO_MODEL.into()),
                ..Default::default()
            });
            return;
        };
        let upstream = &target.upstream;
        self.metrics.begin();
        let mut outgoing = self
            .agent
            .request(method, &format!("{}{}", upstream.base_url(), path));
        for (name, value) in headers {
            if !HOP_BY_HOP.contains(&name.as_str()) {
                outgoing = outgoing.set(name, value);
            }
        }
        outgoing = outgoing.set("Accept-Encoding", "identity");
        let sent = if body.is_empty() {
            outgoing.call()
        } else {
            outgoing.send_bytes(&body)
        };
        let mut observation = Observation {
            model,
            ..Default::default()
        };
        match sent {
            Ok(response) | Err(ureq::Error::Status(_, response)) => {
                self.metrics.set_upstream_up(true);
                relay(request, response, started, &mut observation);
            }
            Err(ureq::Error::Transport(error)) => {
                self.metrics.set_upstream_up(false);
                let message = format!(
                    "upstream {}:{} failed: {error}",
                    upstream.host, upstream.port
                );
                observation.error = Some(format!("{:?}: {error}", error.kind()));
                observation.status = 502;
                let mut body = error_body(&message, "loom_launch_proxy_error", "UNAVAILABLE");
                body["error"]["request_id"] = json!(record_id);
                let _ = request.respond(json_response(502, &body));
            }
        }
        self.metrics.end();
        let (prompt_tokens, completion_tokens, cached_tokens) = usage_totals(&observation.usage);
        let record = RequestRecord {
            request_id: record_id,
            method: method.to_string(),
            path: path.to_string(),
            route: route_name(path).to_string(),
            started_at,
            status: observation.status,
            stream: stream_requested || observation.streamed,
            model: observation.model,
            ttft_seconds: observation.ttft,
            duration_seconds: Some(started.elapsed().as_secs_f64()),
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            chunks: observation.chunks,
            request_bytes: body.len(),
            response_bytes: observation.response_bytes,
            client,
            error: observation.error,
        };
        self.metrics.observe(&record);
        if let Some(log) = &target.request_log {
            log.write(&record.describe());
        }
    }
}

/// Copies one upstream response to the client, observing usage as it passes.
fn relay(
    request: tiny_http::Request,
    response: ureq::Response,
    started: Instant,
    observation: &mut Observation,
) {
    observation.status = response.status();
    let content_type = response.header("Content-Type").unwrap_or("").to_string();
    let cache_control = response.header("Cache-Control").map(str::to_string);
    if content_type.starts_with("text/event-stream") {
        observation.streamed = true;
        let mut head = format!("HTTP/1.1 {} {}\r\n", response.status(), reason(&response));
        head.push_str(&format!("Content-Type: {content_type}\r\n"));
        if let Some(cache) = &cache_control {
            head.push_str(&format!("Cache-Control: {cache}\r\n"));
        }
        head.push_str(
            "X-Loom-Launch: 1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        );
        let mut writer = request.into_writer();
        let mut client_open = writer
            .write_all(head.as_bytes())
            .and_then(|_| writer.flush())
            .is_ok();
        let mut reader = BufReader::new(response.into_reader());
        let mut line = Vec::new();
        loop {
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(error) => {
                    observation.error = Some(format!("upstream stream failed: {error}"));
                    break;
                }
            }
            observation.response_bytes += line.len();
            if client_open {
                let mut frame = format!("{:x}\r\n", line.len()).into_bytes();
                frame.extend_from_slice(&line);
                frame.extend_from_slice(b"\r\n");
                client_open = writer
                    .write_all(&frame)
                    .and_then(|_| writer.flush())
                    .is_ok();
                if !client_open {
                    // Dropping the upstream reader closes the connection, which
                    // lets the server cancel a generation nobody will read.
                    observation.error = Some("client disconnected".into());
                }
            }
            if let Some(payload) = event_payload(&line) {
                observation.chunks += 1;
                if observation.ttft.is_none() && has_content(&payload) {
                    observation.ttft = Some(started.elapsed().as_secs_f64());
                }
                if let Some(Value::Object(usage)) = payload.get("usage") {
                    observation.usage = usage.clone();
                }
            }
            if !client_open {
                break;
            }
        }
        if client_open {
            let _ = writer.write_all(b"0\r\n\r\n").and_then(|_| writer.flush());
        }
        return;
    }
    let mut data = Vec::new();
    if let Err(error) = response.into_reader().read_to_end(&mut data) {
        observation.error = Some(format!("upstream body failed: {error}"));
    }
    observation.response_bytes = data.len();
    if let Ok(Value::Object(document)) = serde_json::from_slice::<Value>(&data) {
        if let Some(Value::Object(usage)) = document.get("usage") {
            observation.usage = usage.clone();
        }
        if observation.model.is_none() {
            observation.model = str_value(get(&document, "model")).map(str::to_string);
        }
    }
    let mut reply = tiny_http::Response::from_data(data).with_status_code(observation.status);
    for (name, value) in [
        ("Content-Type", Some(content_type.as_str())),
        ("Cache-Control", cache_control.as_deref()),
        ("X-Loom-Launch", Some("1")),
    ] {
        if let Some(value) = value.filter(|v| !v.is_empty()) {
            if let Ok(header) = tiny_http::Header::from_bytes(name, value) {
                reply.add_header(header);
            }
        }
    }
    let _ = request.respond(reply);
}

fn reason(response: &ureq::Response) -> String {
    let text = response.status_text().trim();
    if text.is_empty() {
        "OK".to_string()
    } else {
        text.to_string()
    }
}

/// One SSE line's JSON payload, ignoring framing and `[DONE]`.
fn event_payload(line: &[u8]) -> Option<Map<String, Value>> {
    let payload = line.strip_prefix(b"data:")?.trim_ascii();
    if payload.is_empty() || payload == b"[DONE]" {
        return None;
    }
    match serde_json::from_slice::<Value>(payload) {
        Ok(Value::Object(document)) => Some(document),
        _ => None,
    }
}

fn has_content(payload: &Map<String, Value>) -> bool {
    let Some(Value::Array(choices)) = payload.get("choices") else {
        return false;
    };
    let Some(Value::Object(choice)) = choices.first() else {
        return false;
    };
    let delta = match choice.get("delta") {
        Some(value) if crate::values::truthy(value) => value,
        _ => choice.get("message").unwrap_or(&Value::Null),
    };
    let delta = obj(delta);
    ["reasoning_content", "content", "tool_calls"]
        .iter()
        .any(|key| delta.get(*key).is_some_and(crate::values::truthy))
}

fn usage_totals(usage: &Map<String, Value>) -> (Option<i64>, Option<i64>, Option<i64>) {
    (
        strict_int(get(usage, "prompt_tokens")),
        strict_int(get(usage, "completion_tokens")),
        strict_int(get(
            obj(get(usage, "prompt_tokens_details")),
            "cached_tokens",
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::Client;
    use crate::store::tests::scratch;
    use crate::testing::{FakeModelConfig, FakeOpenAIServer};

    /// A fake model behind an instrumented proxy, for one test.
    struct Instrumented {
        _model: FakeOpenAIServer,
        proxy: Proxy,
        log_path: PathBuf,
        base_url: String,
    }

    impl Instrumented {
        fn new(name: &str, token: Option<&str>, config: FakeModelConfig) -> Instrumented {
            let model = FakeOpenAIServer::start(config).unwrap();
            let port: u16 = model.address.rsplit(':').next().unwrap().parse().unwrap();
            let log_path = scratch(name).join("requests.jsonl");
            let mut options = ProxyOptions::new(Upstream {
                host: "127.0.0.1".into(),
                port,
            });
            options.request_log = Some(Arc::new(JsonlWriter::open(&log_path).unwrap()));
            options.auth_token = token.map(str::to_string);
            options.status_provider = Some(Arc::new(|| {
                let mut status = Map::new();
                status.insert("launch".into(), json!({"repo": "loom-fun/a-loom"}));
                status
            }));
            let proxy = Proxy::start(options, Arc::new(Metrics::default())).unwrap();
            let base_url = proxy.address().to_string();
            Instrumented {
                _model: model,
                proxy,
                log_path,
                base_url,
            }
        }

        fn client(&self, token: Option<&str>) -> Client {
            Client::new(&self.base_url, token, Duration::from_secs(30)).unwrap()
        }

        /// The request log, once it holds at least `count` records (or after 3 s):
        /// the proxy appends a record after the client already has its answer.
        fn records_after(&self, count: usize) -> Vec<Value> {
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            loop {
                let records = self.records();
                if records.len() >= count || std::time::Instant::now() > deadline {
                    return records;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        fn records(&self) -> Vec<Value> {
            std::fs::read_to_string(&self.log_path)
                .unwrap_or_default()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }
    }

    fn user(text: &str) -> Vec<Value> {
        vec![json!({"role": "user", "content": text})]
    }

    #[test]
    fn a_streamed_turn_passes_through_and_is_recorded() {
        let instrumented = Instrumented::new("proxy-stream", None, FakeModelConfig::default());
        let client = instrumented.client(None);
        let mut deltas = 0;
        let result = client.chat(
            &user("hello there friend"),
            Some("test/model"),
            true,
            &[],
            &Map::new(),
            &mut |_| deltas += 1,
        );
        assert_eq!(result.status, 200);
        assert!(result.text.starts_with(" token0"), "{}", result.text);
        assert_eq!(result.usage.as_ref().unwrap().completion_tokens, Some(8));
        assert!(result.ttft_seconds.is_some());
        assert!(deltas >= 9);
        let records = instrumented.records_after(1);
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record["route"], "chat_completions");
        assert_eq!(record["status"], 200);
        assert_eq!(record["stream"], true);
        assert_eq!(record["model"], "test/model");
        assert_eq!(record["prompt_tokens"], 3);
        assert_eq!(record["completion_tokens"], 8);
        assert_eq!(record["cached_tokens"], 0);
        assert!(record["chunks"].as_u64().unwrap() >= 9);
        assert!(record["response_bytes"].as_u64().unwrap() > 0);
        assert!(record["request_bytes"].as_u64().unwrap() > 0);
        assert!(record["ttft_seconds"].is_number());
        assert!(record["error"].is_null());
        let mut keys: Vec<_> = record.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        let mut expected = vec![
            "id",
            "time",
            "method",
            "path",
            "route",
            "model",
            "status",
            "stream",
            "duration_seconds",
            "ttft_seconds",
            "prompt_tokens",
            "completion_tokens",
            "cached_tokens",
            "decode_tokens_per_second",
            "chunks",
            "request_bytes",
            "response_bytes",
            "client",
            "error",
        ];
        expected.sort();
        assert_eq!(keys, expected);
        let text = instrumented.proxy.metrics().prometheus();
        assert!(text.contains("loom_launch_requests_total{route=\"chat_completions\",status=\"200\",stream=\"stream\"} 1"));
        assert!(text.contains("loom_launch_prompt_tokens_total 3"));
        assert!(text.contains("loom_launch_completion_tokens_total 8"));
        assert!(text.contains("loom_launch_cached_tokens_total 0"));
        assert!(text.contains("loom_launch_ttft_seconds{quantile=\"0.5\"}"));
        assert!(text.contains("loom_launch_decode_tokens_per_second{quantile=\"mean\"}"));
    }

    #[test]
    fn a_json_turn_is_recorded_with_usage_and_latency() {
        let instrumented = Instrumented::new("proxy-json", None, FakeModelConfig::default());
        let client = instrumented.client(None);
        let result = client.chat(
            &user("two words"),
            Some("test/model"),
            false,
            &[],
            &Map::new(),
            &mut |_| {},
        );
        assert!(result.text.starts_with(" token0"));
        let record = &instrumented.records_after(1)[0];
        assert_eq!(record["stream"], false);
        assert_eq!(record["completion_tokens"], 8);
        assert_eq!(record["prompt_tokens"], 2);
        assert!(record["duration_seconds"].is_number());
        let snapshot = instrumented.proxy.metrics().snapshot();
        assert_eq!(snapshot["requests"], 1);
        assert_eq!(snapshot["prompt_tokens"], 2);
        assert_eq!(snapshot["completion_tokens"], 8);
        assert_eq!(snapshot["by_route"]["chat_completions|200|json"], 1);
    }

    #[test]
    fn server_sent_events_are_relayed_without_buffering() {
        let config = FakeModelConfig {
            completion_tokens: 6,
            token_delay: Duration::from_millis(120),
            ..Default::default()
        };
        let instrumented = Instrumented::new("proxy-unbuffered", None, config);
        let client = instrumented.client(None);
        let result = client.chat(&user("hi"), None, true, &[], &Map::new(), &mut |_| {});
        let ttft = result.ttft_seconds.unwrap();
        let duration = result.duration_seconds.unwrap();
        assert!(duration >= 0.6, "{duration}");
        assert!(
            ttft < 0.45,
            "first token arrived after {ttft}s of {duration}s: buffered"
        );
        let record = &instrumented.records_after(1)[0];
        assert!(record["decode_tokens_per_second"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn discovery_routes_are_forwarded_untouched() {
        let instrumented = Instrumented::new("proxy-discovery", None, FakeModelConfig::default());
        let client = instrumented.client(None);
        assert_eq!(client.model_names(None), ["test/model"]);
        let (status, document) = client.get_json("/v1/status", None).unwrap();
        assert_eq!(status, 200);
        assert_eq!(document["models"]["test/model"]["state"], "ready");
        let record = &instrumented.records_after(1)[0];
        assert_eq!(record["route"], "models");
        assert_eq!(record["method"], "GET");
    }

    #[test]
    fn launcher_routes_are_served_locally() {
        let instrumented = Instrumented::new("proxy-launcher", None, FakeModelConfig::default());
        let client = instrumented.client(None);
        let (status, document) = client.get_json("/loom/status", None).unwrap();
        assert_eq!(status, 200);
        assert_eq!(document["launch"]["repo"], "loom-fun/a-loom");
        assert!(document["proxy"]["upstream"]["port"].as_u64().unwrap() > 0);
        let (status, body) = client.request("GET", "/loom/metrics", None, None).unwrap();
        assert_eq!(status, 200);
        assert!(String::from_utf8_lossy(&body).contains("loom_launch_requests_total"));
        let (status, document) = client.get_json("/loom/requests?limit=5", None).unwrap();
        assert_eq!(status, 200);
        assert_eq!(document["object"], "list");
        assert_eq!(
            client.request("GET", "/loom/absent", None, None).unwrap().0,
            404
        );
        // Launcher routes never reach the model, so nothing is logged as inference.
        assert!(instrumented.records().is_empty());
    }

    #[test]
    fn a_configured_token_is_required_and_accepted() {
        let instrumented =
            Instrumented::new("proxy-auth", Some("s3cret"), FakeModelConfig::default());
        let anonymous = instrumented.client(None);
        let (status, document) = anonymous.get_json("/v1/models", None).unwrap();
        assert_eq!(status, 401);
        assert_eq!(document["error"]["code"], "PERMISSION_DENIED");
        assert_eq!(anonymous.get_json("/loom/status", None).unwrap().0, 401);
        assert!(
            anonymous.healthy(Duration::from_secs(5)),
            "health probes stay anonymous"
        );
        let wrong = instrumented.client(Some("nope"));
        assert_eq!(wrong.get_json("/v1/models", None).unwrap().0, 401);
        let authorized = instrumented.client(Some("s3cret"));
        assert_eq!(authorized.model_names(None), ["test/model"]);
    }

    #[test]
    fn a_failed_upstream_becomes_a_recorded_bad_gateway() {
        let closed = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = closed.local_addr().unwrap().port();
        drop(closed);
        let log_path = scratch("proxy-502").join("requests.jsonl");
        let mut options = ProxyOptions::new(Upstream {
            host: "127.0.0.1".into(),
            port,
        });
        options.request_log = Some(Arc::new(JsonlWriter::open(&log_path).unwrap()));
        let proxy = Proxy::start(options, Arc::new(Metrics::default())).unwrap();
        let client = Client::new(proxy.address(), None, Duration::from_secs(5)).unwrap();
        let (status, document) = client.get_json("/v1/models", None).unwrap();
        assert_eq!(status, 502);
        assert_eq!(document["error"]["type"], "loom_launch_proxy_error");
        assert_eq!(document["error"]["code"], "UNAVAILABLE");
        assert!(document["error"]["request_id"].is_string());
        let result = client.chat(&user("hi"), None, true, &[], &Map::new(), &mut |_| {});
        assert_eq!(result.status, 502);
        assert!(result.error.is_some());
        let snapshot = proxy.metrics().snapshot();
        assert_eq!(snapshot["errors"], 2);
        assert_eq!(snapshot["upstream_up"], false);
        let record: Value = serde_json::from_str(
            std::fs::read_to_string(&log_path)
                .unwrap()
                .lines()
                .next()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(record["status"], 502);
        assert!(record["error"].is_string());
    }

    #[test]
    fn model_errors_are_forwarded_with_their_status() {
        let config = FakeModelConfig {
            fail_status: 429,
            ..Default::default()
        };
        let instrumented = Instrumented::new("proxy-429", None, config);
        let client = instrumented.client(None);
        let (status, document) = client
            .post_json(
                "/v1/chat/completions",
                &json!({"messages": user("hi")}),
                None,
            )
            .unwrap();
        assert_eq!(status, 429);
        assert_eq!(document["error"]["code"], "FAILED_PRECONDITION");
        assert_eq!(instrumented.records_after(1)[0]["status"], 429);
    }

    #[test]
    fn a_quiet_launch_reports_its_listener_as_up_in_prometheus() {
        let instrumented = Instrumented::new("proxy-metrics-up", None, FakeModelConfig::default());
        let (status, body) = instrumented
            .client(None)
            .request("GET", "/loom/metrics", None, None)
            .unwrap();
        assert_eq!(status, 200);
        let text = String::from_utf8(body).unwrap();
        assert!(
            text.contains("loom_launch_upstream_up 1"),
            "a launch that has served nothing must not report itself down: {text}"
        );
    }

    #[test]
    fn a_dead_listener_reports_itself_down_in_prometheus() {
        let proxy = Proxy::start(
            ProxyOptions::new(Upstream {
                host: "127.0.0.1".into(),
                port: 1,
            }),
            Arc::new(Metrics::default()),
        )
        .unwrap();
        let client = Client::new(proxy.address(), None, Duration::from_secs(5)).unwrap();
        let (status, body) = client.request("GET", "/loom/metrics", None, None).unwrap();
        assert_eq!(status, 200);
        let text = String::from_utf8(body).unwrap();
        assert!(text.contains("loom_launch_upstream_up 0"), "{text}");
    }

    #[test]
    fn upstream_health_follows_the_requests_that_reached_it() {
        let instrumented = Instrumented::new("proxy-health", None, FakeModelConfig::default());
        assert_eq!(
            instrumented.proxy.metrics().snapshot()["upstream_up"],
            false
        );
        assert!(!instrumented.client(None).model_names(None).is_empty());
        assert_eq!(instrumented.proxy.metrics().snapshot()["upstream_up"], true);
    }

    #[test]
    fn dashboard_seam_routes_answer() {
        let instrumented = Instrumented::new("proxy-dashboard", None, FakeModelConfig::default());
        drop(instrumented);
        let model = FakeOpenAIServer::start(FakeModelConfig::default()).unwrap();
        let port: u16 = model.address.rsplit(':').next().unwrap().parse().unwrap();
        let state = scratch("proxy-dashboard-state");
        let mut options = ProxyOptions::new(Upstream {
            host: "127.0.0.1".into(),
            port,
        });
        options.dashboard_state = Some(state);
        let proxy = Proxy::start(options, Arc::new(Metrics::default())).unwrap();
        let client = Client::new(proxy.address(), None, Duration::from_secs(5)).unwrap();
        assert_eq!(client.request("GET", "/", None, None).unwrap().0, 200);
        let (status, document) = client.get_json("/api/status", None).unwrap();
        assert_eq!(status, 200);
        assert_eq!(document["stored_runs"], 0);
        assert_eq!(
            client.get_json("/api/installs", None).unwrap().1["installs"],
            json!([])
        );
        // A launch's own dashboard never starts a second launch.
        assert_eq!(
            client
                .request("POST", "/api/serve", Some(&json!({})), None)
                .unwrap()
                .0,
            409
        );
        assert_eq!(
            client.get_json("/api/benchmarks/absent", None).unwrap().0,
            404
        );
        assert_eq!(client.model_names(None), ["test/model"]);
    }
}
