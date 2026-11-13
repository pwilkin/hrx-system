//! Request instrumentation recorded by the launcher's fronting proxy.
//!
//! One record describes one HTTP request the proxy carried. Records feed a
//! bounded in-memory window for percentiles and a JSONL log, and the counters are
//! exposed as Prometheus text with the Python launcher's metric names.

use crate::values::{iso_utc_millis, now_seconds, opt_f64, round_opt};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

/// Local endpoints the proxy answers itself.
pub const LAUNCHER_PREFIX: &str = "/loom/";
/// How many recent records feed percentile estimates.
pub const DEFAULT_WINDOW: usize = 1024;

/// Groups request paths into stable route labels for metrics.
pub fn route_name(path: &str) -> &'static str {
    let path = path.split('?').next().unwrap_or(path);
    if path.starts_with(LAUNCHER_PREFIX) {
        return "launcher";
    }
    match path {
        "/v1/chat/completions" => "chat_completions",
        "/v1/completions" => "completions",
        "/v1/embeddings" => "embeddings",
        "/v1/models" => "models",
        "/v1/status" => "status",
        "/v1/requests" => "requests",
        "/v1/sessions" => "sessions",
        "/v1/observe" => "observe",
        "/metrics" => "metrics",
        "/logz" => "logz",
        "/tokenize" => "tokenize",
        "/detokenize" => "detokenize",
        "/healthz" | "/readyz" => "health",
        _ if path.starts_with("/v1/models/") => "model_control",
        _ if path.starts_with("/v1/checkpoints/") => "checkpoints",
        _ if path.starts_with("/dashboard") => "dashboard",
        _ => "other",
    }
}

/// One measured request.
#[derive(Clone, Debug, Default)]
pub struct RequestRecord {
    pub request_id: String,
    pub method: String,
    pub path: String,
    pub route: String,
    pub started_at: f64,
    pub status: u16,
    pub stream: bool,
    pub model: Option<String>,
    pub ttft_seconds: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub chunks: usize,
    pub request_bytes: usize,
    pub response_bytes: usize,
    pub client: Option<String>,
    pub error: Option<String>,
}

impl RequestRecord {
    /// Rate over the decode interval after the first token arrived.
    pub fn decode_tokens_per_second(&self) -> Option<f64> {
        let completion = self.completion_tokens?;
        if completion < 2 {
            return None;
        }
        let interval = self.duration_seconds? - self.ttft_seconds?;
        if interval <= 0.0 {
            return None;
        }
        Some((completion - 1) as f64 / interval)
    }

    /// The `requests.jsonl` record, keys in the Python launcher's order.
    pub fn describe(&self) -> Value {
        json!({
            "id": self.request_id,
            "time": iso_utc_millis(self.started_at),
            "method": self.method,
            "path": self.path,
            "route": self.route,
            "model": self.model,
            "status": self.status,
            "stream": self.stream,
            "duration_seconds": opt_f64(round_opt(self.duration_seconds, 4)),
            "ttft_seconds": opt_f64(round_opt(self.ttft_seconds, 4)),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "decode_tokens_per_second": opt_f64(round_opt(self.decode_tokens_per_second(), 4)),
            "chunks": self.chunks,
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "client": self.client,
            "error": self.error,
        })
    }
}

/// Bounded recent observations for percentile estimates.
#[derive(Debug)]
struct Window {
    values: VecDeque<f64>,
    capacity: usize,
}

impl Window {
    fn new(capacity: usize) -> Window {
        Window {
            values: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn add(&mut self, value: Option<f64>) {
        if let Some(value) = value.filter(|v| !v.is_nan()) {
            if self.values.len() == self.capacity {
                self.values.pop_front();
            }
            self.values.push_back(value);
        }
    }

    /// Nearest-rank quantile with Python's round-half-even index.
    fn quantile(&self, fraction: f64) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        let mut ordered: Vec<f64> = self.values.iter().copied().collect();
        ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let last = ordered.len() - 1;
        let index = (fraction * last as f64).round_ties_even().max(0.0) as usize;
        Some(ordered[index.min(last)])
    }

    fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
    }
}

#[derive(Debug)]
struct State {
    counts: BTreeMap<(String, String, String), u64>,
    requests: u64,
    prompt_tokens: i64,
    completion_tokens: i64,
    cached_tokens: i64,
    request_bytes: u64,
    response_bytes: u64,
    errors: u64,
    in_flight: u64,
    upstream_up: bool,
    ttft: Window,
    duration: Window,
    decode: Window,
    recent: VecDeque<RequestRecord>,
    window: usize,
}

/// Cumulative counters, a recent-request window and Prometheus rendering.
#[derive(Debug)]
pub struct Metrics {
    started_at: f64,
    state: Mutex<State>,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics::new(DEFAULT_WINDOW)
    }
}

impl Metrics {
    pub fn new(window: usize) -> Metrics {
        Metrics {
            started_at: now_seconds(),
            state: Mutex::new(State {
                counts: BTreeMap::new(),
                requests: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
                cached_tokens: 0,
                request_bytes: 0,
                response_bytes: 0,
                errors: 0,
                in_flight: 0,
                upstream_up: false,
                ttft: Window::new(window),
                duration: Window::new(window),
                decode: Window::new(window),
                recent: VecDeque::new(),
                window: window.max(1),
            }),
        }
    }

    pub fn started_at(&self) -> f64 {
        self.started_at
    }

    pub fn begin(&self) {
        self.state.lock().unwrap().in_flight += 1;
    }

    pub fn end(&self) {
        let mut state = self.state.lock().unwrap();
        state.in_flight = state.in_flight.saturating_sub(1);
    }

    pub fn set_upstream_up(&self, up: bool) {
        self.state.lock().unwrap().upstream_up = up;
    }

    pub fn observe(&self, record: &RequestRecord) {
        let key = (
            record.route.clone(),
            record.status.to_string(),
            if record.stream { "stream" } else { "json" }.to_string(),
        );
        let mut state = self.state.lock().unwrap();
        state.requests += 1;
        *state.counts.entry(key).or_insert(0) += 1;
        state.prompt_tokens += record.prompt_tokens.unwrap_or(0);
        state.completion_tokens += record.completion_tokens.unwrap_or(0);
        state.cached_tokens += record.cached_tokens.unwrap_or(0);
        state.request_bytes += record.request_bytes as u64;
        state.response_bytes += record.response_bytes as u64;
        if record.error.is_some() || record.status >= 500 {
            state.errors += 1;
        }
        state.ttft.add(record.ttft_seconds);
        state.duration.add(record.duration_seconds);
        state.decode.add(record.decode_tokens_per_second());
        if state.recent.len() == state.window {
            state.recent.pop_front();
        }
        state.recent.push_back(record.clone());
    }

    /// The most recent records, newest first.
    pub fn recent(&self, limit: usize) -> Vec<Value> {
        let state = self.state.lock().unwrap();
        state
            .recent
            .iter()
            .rev()
            .take(limit)
            .map(RequestRecord::describe)
            .collect()
    }

    pub fn snapshot(&self) -> Value {
        let state = self.state.lock().unwrap();
        let by_route: Map<String, Value> = state
            .counts
            .iter()
            .map(|((route, status, stream), count)| {
                (format!("{route}|{status}|{stream}"), json!(count))
            })
            .collect();
        let r = |value: Option<f64>| opt_f64(round_opt(value, 4));
        json!({
            "uptime_seconds": crate::values::round_to(now_seconds() - self.started_at, 3),
            "requests": state.requests,
            "errors": state.errors,
            "in_flight": state.in_flight,
            "upstream_up": state.upstream_up,
            "by_route": by_route,
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "cached_tokens": state.cached_tokens,
            "request_bytes": state.request_bytes,
            "response_bytes": state.response_bytes,
            "ttft_seconds": {"p50": r(state.ttft.quantile(0.5)), "p95": r(state.ttft.quantile(0.95))},
            "duration_seconds": {"p50": r(state.duration.quantile(0.5)), "p95": r(state.duration.quantile(0.95))},
            "decode_tokens_per_second": {"mean": r(state.decode.mean()), "p50": r(state.decode.quantile(0.5))},
        })
    }

    /// Renders the current counters in the Prometheus text format.
    pub fn prometheus(&self) -> String {
        let state = self.state.lock().unwrap();
        let mut lines = vec![
            "# HELP loom_launch_requests_total Requests carried by the launcher".to_string(),
            "# TYPE loom_launch_requests_total counter".to_string(),
        ];
        for ((route, status, stream), count) in &state.counts {
            lines.push(format!(
                "loom_launch_requests_total{{route=\"{route}\",status=\"{status}\",stream=\"{stream}\"}} {count}"
            ));
        }
        let uptime = now_seconds() - self.started_at;
        let gauges: [(&str, String); 9] = [
            ("loom_launch_errors_total", state.errors.to_string()),
            ("loom_launch_in_flight", state.in_flight.to_string()),
            (
                "loom_launch_upstream_up",
                u8::from(state.upstream_up).to_string(),
            ),
            (
                "loom_launch_prompt_tokens_total",
                state.prompt_tokens.to_string(),
            ),
            (
                "loom_launch_completion_tokens_total",
                state.completion_tokens.to_string(),
            ),
            (
                "loom_launch_cached_tokens_total",
                state.cached_tokens.to_string(),
            ),
            (
                "loom_launch_request_bytes_total",
                state.request_bytes.to_string(),
            ),
            (
                "loom_launch_response_bytes_total",
                state.response_bytes.to_string(),
            ),
            ("loom_launch_uptime_seconds", format!("{uptime:?}")),
        ];
        for (name, value) in gauges {
            lines.push(format!("# TYPE {name} gauge"));
            lines.push(format!("{name} {value}"));
        }
        lines.push("# TYPE loom_launch_ttft_seconds gauge".to_string());
        for (quantile, value) in [
            ("0.5", state.ttft.quantile(0.5)),
            ("0.95", state.ttft.quantile(0.95)),
        ] {
            if let Some(value) = value {
                lines.push(format!(
                    "loom_launch_ttft_seconds{{quantile=\"{quantile}\"}} {value:.6}"
                ));
            }
        }
        lines.push("# TYPE loom_launch_decode_tokens_per_second gauge".to_string());
        if let Some(mean) = state.decode.mean() {
            lines.push(format!(
                "loom_launch_decode_tokens_per_second{{quantile=\"mean\"}} {mean:.6}"
            ));
        }
        if let Some(median) = state.decode.quantile(0.5) {
            lines.push(format!(
                "loom_launch_decode_tokens_per_second{{quantile=\"0.5\"}} {median:.6}"
            ));
        }
        lines.join("\n") + "\n"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> RequestRecord {
        RequestRecord {
            request_id: "r1".into(),
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            route: "chat_completions".into(),
            started_at: 0.0,
            status: 200,
            stream: true,
            model: Some("test/model".into()),
            ttft_seconds: Some(0.2),
            duration_seconds: Some(1.2),
            prompt_tokens: Some(40),
            completion_tokens: Some(11),
            cached_tokens: Some(0),
            chunks: 12,
            request_bytes: 100,
            response_bytes: 900,
            client: Some("127.0.0.1:5000".into()),
            error: None,
        }
    }

    #[test]
    fn routes_group_known_paths_and_keep_unknown_ones() {
        assert_eq!(route_name("/v1/chat/completions"), "chat_completions");
        assert_eq!(route_name("/v1/models?full=true"), "models");
        assert_eq!(route_name("/v1/models/variants"), "model_control");
        assert_eq!(route_name("/v1/checkpoints/scout"), "checkpoints");
        assert_eq!(route_name("/loom/metrics"), "launcher");
        assert_eq!(route_name("/healthz"), "health");
        assert_eq!(route_name("/something/else"), "other");
    }

    #[test]
    fn decode_rate_uses_the_interval_after_the_first_token() {
        assert_eq!(record().decode_tokens_per_second(), Some(10.0));
        let short = RequestRecord {
            completion_tokens: Some(1),
            ..record()
        };
        assert_eq!(short.decode_tokens_per_second(), None);
        let untimed = RequestRecord {
            ttft_seconds: None,
            ..record()
        };
        assert_eq!(untimed.decode_tokens_per_second(), None);
        let instant = RequestRecord {
            ttft_seconds: Some(1.2),
            ..record()
        };
        assert_eq!(instant.decode_tokens_per_second(), None);
    }

    #[test]
    fn records_describe_themselves_with_the_python_key_set() {
        let document = record().describe();
        assert_eq!(document["id"], "r1");
        assert_eq!(document["time"], "1970-01-01T00:00:00.000Z");
        assert_eq!(document["decode_tokens_per_second"], 10.0);
        assert_eq!(document["stream"], true);
        let keys: Vec<_> = document.as_object().unwrap().keys().cloned().collect();
        assert_eq!(
            keys,
            [
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
                "error"
            ]
        );
    }

    #[test]
    fn counters_accumulate_over_observations() {
        let metrics = Metrics::default();
        metrics.observe(&record());
        metrics.observe(&RequestRecord {
            request_id: "r2".into(),
            status: 500,
            route: "other".into(),
            ..record()
        });
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["requests"], 2);
        assert_eq!(snapshot["errors"], 1);
        assert_eq!(snapshot["prompt_tokens"], 80);
        assert_eq!(snapshot["completion_tokens"], 22);
        assert_eq!(snapshot["by_route"]["chat_completions|200|stream"], 1);
        assert_eq!(snapshot["ttft_seconds"]["p50"], 0.2);
    }

    #[test]
    fn in_flight_and_upstream_health_are_gauges() {
        let metrics = Metrics::default();
        metrics.begin();
        metrics.begin();
        metrics.set_upstream_up(true);
        assert_eq!(metrics.snapshot()["in_flight"], 2);
        metrics.end();
        assert_eq!(metrics.snapshot()["in_flight"], 1);
        assert_eq!(metrics.snapshot()["upstream_up"], true);
    }

    #[test]
    fn prometheus_text_reports_counters_and_quantiles() {
        let metrics = Metrics::default();
        metrics.observe(&record());
        metrics.set_upstream_up(true);
        let text = metrics.prometheus();
        assert!(text.contains(
            "loom_launch_requests_total{route=\"chat_completions\",status=\"200\",stream=\"stream\"} 1"
        ));
        assert!(text.contains("loom_launch_completion_tokens_total 11"));
        assert!(text.contains("loom_launch_prompt_tokens_total 40"));
        assert!(text.contains("loom_launch_cached_tokens_total 0"));
        assert!(text.contains("loom_launch_ttft_seconds{quantile=\"0.5\"} 0.200000"));
        assert!(text.contains("loom_launch_decode_tokens_per_second{quantile=\"mean\"} 10.000000"));
        assert!(text.contains("loom_launch_upstream_up 1"));
        assert!(text.contains("loom_launch_in_flight 0"));
        assert!(text.contains("loom_launch_errors_total 0"));
        assert!(text.contains("loom_launch_uptime_seconds "));
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn recent_records_are_newest_first_and_bounded() {
        let metrics = Metrics::new(2);
        for index in 0..4 {
            metrics.observe(&RequestRecord {
                request_id: format!("r{index}"),
                ..record()
            });
        }
        let ids: Vec<_> = metrics
            .recent(50)
            .iter()
            .map(|r| r["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, ["r3", "r2"]);
    }

    #[test]
    fn quantiles_use_python_rounding() {
        let mut window = Window::new(10);
        window.add(Some(1.0));
        window.add(Some(2.0));
        // round(0.5) == 0 in Python.
        assert_eq!(window.quantile(0.5), Some(1.0));
    }

    #[test]
    fn an_empty_metrics_registry_renders_a_valid_document() {
        let text = Metrics::default().prometheus();
        assert!(text.contains("loom_launch_requests_total"));
        assert!(!text.contains("quantile"));
    }
}
