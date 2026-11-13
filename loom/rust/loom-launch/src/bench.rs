//! Local benchmarks with the metric vocabulary a Loom card publishes.
//!
//! The rows mirror the publisher's measurement record: decode at a short prompt
//! and at 2k and 32k context, prefill over 1k through 32k documents, and four
//! concurrent streams. Each row keeps the samples it came from, so a stored run
//! can be read the way a card is read: against the published row and against
//! the hardware ceiling.

use crate::client::{ChatResult, Client};
use crate::errors::{Error, Result};
use crate::profile::{metric_label, Profile, METRIC_ORDER};
use crate::store::new_run_id;
use crate::values::{
    get, grouped, iso_utc, now_seconds, number_value, obj, opt_f64, round_opt, round_to, str_value,
};
use crate::workload::{decode_prompt, prefill_prompt, WORKLOAD_IDENTITY};
use serde_json::{json, Map, Value};
use std::time::Instant;

/// Metric sets a caller can request, cheapest first.
pub const SUITES: [(&str, &[&str]); 3] = [
    ("quick", &["decode", "prefill_1k", "prefill_2k"]),
    (
        "standard",
        &[
            "decode",
            "decode_2k",
            "prefill_1k",
            "prefill_2k",
            "prefill_8k",
            "streams4",
        ],
    ),
    (
        "full",
        &[
            "decode",
            "decode_2k",
            "decode_32k",
            "prefill_1k",
            "prefill_2k",
            "prefill_8k",
            "prefill_16k",
            "prefill_32k",
            "streams4",
        ],
    ),
];

pub fn suite_metrics(suite: &str) -> Option<&'static [&'static str]> {
    SUITES
        .iter()
        .find(|(name, _)| *name == suite)
        .map(|(_, m)| *m)
}

/// Document size (tokens) and output budget each metric measures.
fn metric_shape(metric: &str) -> Option<(usize, i64)> {
    Some(match metric {
        "decode" => (0, 300),
        "decode_2k" => (2000, 256),
        "decode_32k" => (32000, 128),
        "prefill_1k" => (1000, 4),
        "prefill_2k" => (2000, 4),
        "prefill_8k" => (8000, 4),
        "prefill_16k" => (16000, 4),
        "prefill_32k" => (32000, 4),
        _ => return None,
    })
}

/// One measured row, with its samples and the protocol that produced it.
#[derive(Clone, Debug, Default)]
pub struct MetricResult {
    pub metric: String,
    pub value: Option<f64>,
    pub unit: String,
    pub samples: Vec<f64>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub ttft_seconds: Option<f64>,
    pub duration_seconds: Option<f64>,
    pub note: Option<String>,
}

impl MetricResult {
    fn new(metric: &str) -> MetricResult {
        MetricResult {
            metric: metric.to_string(),
            unit: "tok/s".into(),
            ..Default::default()
        }
    }

    pub fn describe(&self) -> Value {
        json!({
            "metric": self.metric,
            "label": metric_label(&self.metric),
            "value": opt_f64(round_opt(self.value, 2)),
            "unit": self.unit,
            "samples": self.samples.iter().map(|s| round_to(*s, 2)).collect::<Vec<_>>(),
            "runs": self.samples.len(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ttft_seconds": opt_f64(round_opt(self.ttft_seconds, 2)),
            "duration_seconds": opt_f64(round_opt(self.duration_seconds, 2)),
            "note": self.note,
        })
    }
}

/// One unit of a suite's work, for a caller that shows determinate progress.
#[derive(Clone, Debug, PartialEq)]
pub enum BenchEvent<'a> {
    /// A request (or the four streams) is about to run: `done` of `total`
    /// units finished so far; `repeat` counts from 1.
    Begin {
        done: usize,
        total: usize,
        metric: &'a str,
        repeat: usize,
        repeats: usize,
    },
    /// A metric finished, with its median value or the reason it has none.
    Measured {
        metric: &'a str,
        value: Option<f64>,
        unit: &'a str,
        note: Option<&'a str>,
    },
}

/// Measures one suite against a live endpoint.
pub struct Runner<'a> {
    pub client: &'a Client,
    pub profile: &'a Profile,
    pub model: String,
    pub repeats: usize,
    pub thinking: Option<bool>,
    pub warmup: bool,
    pub session: Option<String>,
    /// Sessions the deployment admits, for the streams4 comparability note.
    pub max_sessions: Option<i64>,
    pub on_progress: Option<&'a (dyn Fn(&str) + Sync)>,
    /// Structured progress: one `Begin` per request, one `Measured` per metric.
    pub on_event: Option<&'a (dyn Fn(&BenchEvent) + Sync)>,
}

impl<'a> Runner<'a> {
    pub fn new(client: &'a Client, profile: &'a Profile) -> Runner<'a> {
        Runner {
            client,
            profile,
            model: profile.model_id.clone(),
            repeats: 1,
            thinking: Some(false),
            warmup: true,
            session: None,
            max_sessions: None,
            on_progress: None,
            on_event: None,
        }
    }

    fn emit(&self, event: BenchEvent) {
        if let Some(observer) = self.on_event {
            observer(&event);
        }
    }

    /// Requests one metric makes: its repeats, or one for the stream group and
    /// rows the harness cannot measure.
    fn units(&self, metric: &str) -> usize {
        if metric_shape(metric).is_some() {
            self.repeats.max(1)
        } else {
            1
        }
    }

    fn notify(&self, message: &str) {
        if let Some(progress) = self.on_progress {
            progress(message);
        }
    }

    fn chat(&self, content: &str, max_tokens: i64) -> ChatResult {
        let mut fields = Map::new();
        fields.insert("max_tokens".into(), json!(max_tokens));
        fields.insert("temperature".into(), json!(0));
        if let Some(thinking) = self.thinking {
            fields.insert("enable_thinking".into(), json!(thinking));
        }
        let headers: Vec<(String, String)> = self
            .session
            .iter()
            .map(|s| ("X-Loom-Session".to_string(), s.clone()))
            .collect();
        self.client.chat(
            &[json!({"role": "user", "content": content})],
            Some(&self.model),
            true,
            &headers,
            &fields,
            &mut |_| {},
        )
    }

    pub fn run(&self, suite: &str) -> Result<Map<String, Value>> {
        let Some(metrics) = suite_metrics(suite) else {
            let names: Vec<&str> = SUITES.iter().map(|(n, _)| *n).collect();
            return Err(Error::benchmark(format!(
                "unknown suite '{suite}'; choose one of {}",
                names.join(", ")
            )));
        };
        let started_at = now_seconds();
        let started = Instant::now();
        let total = usize::from(self.warmup) + metrics.iter().map(|m| self.units(m)).sum::<usize>();
        let mut done = 0;
        if self.warmup {
            self.notify("warm-up");
            self.emit(BenchEvent::Begin {
                done,
                total,
                metric: "warm-up",
                repeat: 1,
                repeats: 1,
            });
            self.chat(&decode_prompt(0), 32);
            done += 1;
        }
        let mut results: Vec<MetricResult> = Vec::new();
        for metric in metrics {
            let result = self.measure_from(metric, done, total);
            done += self.units(metric);
            self.emit(BenchEvent::Measured {
                metric,
                value: result.value,
                unit: &result.unit,
                note: result.note.as_deref(),
            });
            results.push(result);
        }
        let elapsed = started.elapsed().as_secs_f64();
        let mut measured = Map::new();
        for result in &results {
            measured.insert(result.metric.clone(), result.describe());
        }
        let comparison = compare(self.profile, &results);
        let mut notes: Vec<String> = results
            .iter()
            .filter_map(|r| r.note.as_ref().map(|n| format!("{}: {n}", r.metric)))
            .collect();
        if !comparison.is_empty() {
            notes.push(format!(
                "local rows use generated documents ({WORKLOAD_IDENTITY}); published rows were \
                 measured on the publisher's own documents and machine"
            ));
        }
        let mut record = Map::new();
        record.insert("id".into(), json!(new_run_id()));
        record.insert("kind".into(), json!("loom-launch-benchmark"));
        record.insert("suite".into(), json!(suite));
        record.insert("profile".into(), json!(self.profile.repo));
        record.insert("revision".into(), json!(self.profile.revision));
        record.insert("model".into(), json!(self.model));
        record.insert("started_at".into(), json!(iso_utc(started_at)));
        record.insert("elapsed_seconds".into(), json!(round_to(elapsed, 3)));
        record.insert(
            "harness".into(),
            json!({
                "workload": WORKLOAD_IDENTITY,
                "repeats": self.repeats,
                "thinking": self.thinking,
                "warmup": self.warmup,
                "parallel": self.max_sessions,
            }),
        );
        record.insert(
            "settings".into(),
            self.profile
                .execution
                .as_ref()
                .map(|e| e.describe())
                .unwrap_or(Value::Null),
        );
        record.insert("device".into(), json!(self.profile.device_hint));
        record.insert("metrics".into(), Value::Object(measured));
        record.insert("comparison".into(), Value::Array(comparison));
        record.insert("notes".into(), json!(notes));
        record.insert(
            "ok".into(),
            json!(results.iter().any(|r| r.value.is_some())),
        );
        Ok(record)
    }

    pub fn measure(&self, metric: &str) -> MetricResult {
        self.measure_from(metric, 0, self.units(metric))
    }

    /// [`Runner::measure`], announcing each request as unit `done + n` of `total`.
    fn measure_from(&self, metric: &str, done: usize, total: usize) -> MetricResult {
        let repeats = self.units(metric);
        let begin = |repeat: usize| {
            self.emit(BenchEvent::Begin {
                done: done + repeat,
                total,
                metric,
                repeat: repeat + 1,
                repeats,
            })
        };
        if metric_shape(metric).is_none() {
            begin(0);
        }
        if metric == "streams4" {
            return self.measure_streams(4, 128);
        }
        if metric == "spec_acceptance" {
            let mut result = MetricResult::new(metric);
            result.unit = "fraction".into();
            result.note = Some("the server does not report drafted-token acceptance here".into());
            return result;
        }
        let mut result = MetricResult::new(metric);
        let Some((document_tokens, max_tokens)) = metric_shape(metric) else {
            result.note = Some("unknown metric".into());
            return result;
        };
        let prefill = metric.starts_with("prefill");
        for repeat in 0..self.repeats.max(1) {
            begin(repeat);
            let rate;
            if prefill {
                let completion = self.chat(&prefill_prompt(document_tokens), max_tokens);
                let prompt_tokens = completion.usage.as_ref().and_then(|u| u.prompt_tokens);
                let ttft = completion.ttft_seconds;
                match (
                    prompt_tokens.filter(|p| *p != 0),
                    ttft.filter(|t| *t != 0.0),
                ) {
                    (Some(prompt), Some(ttft)) => {
                        rate = Some(prompt as f64 / ttft);
                        result.prompt_tokens = Some(prompt);
                        result.ttft_seconds = Some(ttft);
                    }
                    _ => {
                        result.note = Some(
                            completion
                                .error
                                .clone()
                                .unwrap_or_else(|| "no prompt token count reported".into()),
                        );
                        continue;
                    }
                }
            } else {
                let completion = self.chat(&decode_prompt(document_tokens), max_tokens);
                rate = completion.decode_tokens_per_second();
                let completed = completion.usage.as_ref().and_then(|u| u.completion_tokens);
                result.prompt_tokens = completion.usage.as_ref().and_then(|u| u.prompt_tokens);
                result.completion_tokens = completed;
                result.ttft_seconds = completion.ttft_seconds;
                result.duration_seconds = completion.duration_seconds;
                if let Some(error) = &completion.error {
                    result.note = Some(error.clone());
                    continue;
                }
                if let Some(completed) = completed {
                    if completed < max_tokens / 2 {
                        result.note = Some(format!(
                            "stopped after {completed} of {max_tokens} tokens; not a decode measurement"
                        ));
                        continue;
                    }
                }
            }
            match rate.filter(|r| *r > 0.0) {
                None => {
                    if result.note.is_none() {
                        result.note = Some("no usable measurement".into());
                    }
                }
                Some(rate) => {
                    result.samples.push(rate);
                    self.notify(&format!(
                        "{metric}: {} tok/s (repeat {}/{})",
                        grouped(rate, 1),
                        repeat + 1,
                        self.repeats.max(1)
                    ));
                }
            }
        }
        result.value = median(&result.samples);
        result
    }

    pub fn measure_streams(&self, streams: usize, max_tokens: i64) -> MetricResult {
        let mut result = MetricResult::new("streams4");
        if let Some(sessions) = self.max_sessions.filter(|s| (*s as usize) < streams) {
            // Measured, but it answers a narrower question than the published
            // row: four clients against fewer resident sessions queue.
            result.note = Some(format!(
                "deployment admits {sessions} session(s); the published row assumes four concurrent streams"
            ));
        }
        let window_start = Instant::now();
        let completions: Vec<ChatResult> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..streams)
                .map(|_| scope.spawn(|| self.chat(&decode_prompt(0), max_tokens)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or_default())
                .collect()
        });
        let window = window_start.elapsed().as_secs_f64();
        let mut tokens = Vec::new();
        for completion in &completions {
            if let Some(error) = &completion.error {
                result.note = Some(error.clone());
                continue;
            }
            if let Some(completed) = completion
                .usage
                .as_ref()
                .and_then(|u| u.completion_tokens)
                .filter(|c| *c != 0)
            {
                tokens.push(completed);
            }
        }
        if tokens.is_empty() || window <= 0.0 {
            if result.note.is_none() {
                result.note = Some("no usable measurements".into());
            }
            return result;
        }
        // Aggregate decode: every stream's decoded tokens over the shared window.
        let aggregate = tokens.iter().map(|c| (c - 1).max(1)).sum::<i64>() as f64 / window;
        result.samples.push(aggregate);
        result.value = Some(aggregate);
        result.completion_tokens = Some(tokens.iter().sum());
        result.duration_seconds = Some(window);
        self.notify(&format!(
            "streams4: {} tok/s aggregate over {}",
            grouped(aggregate, 1),
            tokens.len()
        ));
        result
    }
}

/// `statistics.median`: the middle value, or the mean of the two middle ones.
pub fn median(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = ordered.len() / 2;
    Some(if ordered.len() % 2 == 1 {
        ordered[middle]
    } else {
        (ordered[middle - 1] + ordered[middle]) / 2.0
    })
}

/// Compares measured rows with the card's published rows and ceiling.
///
/// Rows carry the Python launcher's keys (`metric, label, measured, published,
/// ratio, ceiling, roofline_fraction`) plus `value` (= measured) and `baseline`
/// (the card's baseline engine for that metric).
pub fn compare(profile: &Profile, results: &[MetricResult]) -> Vec<Value> {
    let mut rows = Vec::new();
    for metric in METRIC_ORDER {
        let Some(result) = results.iter().find(|r| r.metric == metric) else {
            continue;
        };
        let Some(value) = result.value else {
            continue;
        };
        let reference = profile
            .performance
            .as_ref()
            .and_then(|p| p.measurement(metric));
        let published = reference.and_then(|r| r.value);
        let baseline = reference.and_then(|r| r.baseline);
        let ceiling = reference.and_then(|r| r.ceiling);
        let mut row = Map::new();
        row.insert("metric".into(), json!(metric));
        row.insert("label".into(), json!(metric_label(metric)));
        row.insert("measured".into(), json!(round_to(value, 2)));
        row.insert("value".into(), json!(round_to(value, 2)));
        row.insert("published".into(), opt_f64(round_opt(published, 2)));
        row.insert("baseline".into(), opt_f64(round_opt(baseline, 2)));
        row.insert(
            "ratio".into(),
            opt_f64(
                published
                    .filter(|p| *p != 0.0)
                    .map(|p| round_to(value / p, 2)),
            ),
        );
        row.insert("ceiling".into(), opt_f64(round_opt(ceiling, 2)));
        row.insert(
            "roofline_fraction".into(),
            opt_f64(
                ceiling
                    .filter(|c| *c != 0.0)
                    .map(|c| round_to(value / c, 2)),
            ),
        );
        rows.push(Value::Object(row));
    }
    rows
}

fn optional_column(value: Option<f64>, render: impl Fn(f64) -> String) -> String {
    match value.filter(|v| *v != 0.0) {
        Some(v) => render(v),
        None => "-".to_string(),
    }
}

/// Renders one stored run as a terminal table (the Python `format_metrics`).
pub fn format_metrics(record: &Map<String, Value>) -> Vec<String> {
    let mut lines = vec![
        format!(
            "{:<22} {:>10} {:>10} {:>7} {:>9}",
            "metric", "value", "published", "ratio", "roofline"
        ),
        "-".repeat(62),
    ];
    let comparison: Vec<&Map<String, Value>> = crate::values::arr(get(record, "comparison"))
        .iter()
        .filter_map(Value::as_object)
        .collect();
    let Some(metrics) = record.get("metrics").and_then(Value::as_object) else {
        return lines;
    };
    for metric in METRIC_ORDER {
        let Some(entry) = metrics.get(metric).and_then(Value::as_object) else {
            continue;
        };
        let Some(value) = number_value(get(entry, "value")) else {
            continue;
        };
        let reference = comparison
            .iter()
            .rev()
            .find(|row| str_value(get(row, "metric")) == Some(metric))
            .copied()
            .unwrap_or_else(|| obj(&Value::Null));
        let published = number_value(get(reference, "published"));
        let ratio = number_value(get(reference, "ratio"));
        let fraction = number_value(get(reference, "roofline_fraction"));
        lines.push(format!(
            "{:<22} {:>10} {:>10} {:>7} {:>9}",
            metric,
            grouped(value, 1),
            optional_column(published, |p| grouped(p, 1)),
            optional_column(ratio, |r| format!("{r:.2}x")),
            optional_column(fraction, |f| format!("{:.0}%", f * 100.0)),
        ));
    }
    lines
}

/// Renders a card's stored measurements, with the baseline and ceiling.
pub fn format_published(profile: &Profile) -> Vec<String> {
    let mut lines = vec![
        format!(
            "{:<22} {:>10} {:>10} {:>7} {:>9}",
            "metric", "loom", "baseline", "ratio", "roofline"
        ),
        "-".repeat(62),
    ];
    let Some(performance) = &profile.performance else {
        lines.push("no measurements published".into());
        return lines;
    };
    for measurement in &performance.measurements {
        let Some(value) = measurement.value else {
            continue;
        };
        lines.push(format!(
            "{:<22} {:>10} {:>10} {:>7} {:>9}",
            measurement.metric,
            grouped(value, 1),
            optional_column(measurement.baseline, |b| grouped(b, 1)),
            optional_column(measurement.baseline_ratio(), |r| format!("{r:.2}x")),
            optional_column(measurement.roofline_fraction(), |f| format!(
                "{:.0}%",
                f * 100.0
            )),
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::ProfileDocuments;
    use crate::testing::{FakeModelConfig, FakeOpenAIServer};
    use std::time::Duration;

    fn profile() -> Profile {
        let card = json!({
            "schema_version": 1,
            "model": {"id": "test/model", "name": "Test model"},
            "services": {"chat": {"kind": "text", "version": 1}},
            "presets": {"chat": {"service": "chat", "execution": {
                "rows": 1, "context_capacity": 65536, "prefill_capacity": 1024,
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
        let config = json!({"arch": "test", "gguf": "test-Q4_K_M.gguf", "context_length": 65536});
        Profile::from_documents(
            "loom-fun/test-loom",
            &ProfileDocuments {
                model_card: obj(&card).clone(),
                info: obj(&info).clone(),
                llm_config: obj(&config).clone(),
                ..Default::default()
            },
            None,
        )
    }

    fn served(config: FakeModelConfig) -> (FakeOpenAIServer, Client) {
        let server = FakeOpenAIServer::start(config).unwrap();
        let client = Client::new(&server.url, None, Duration::from_secs(30)).unwrap();
        (server, client)
    }

    fn timed(tokens: usize) -> FakeModelConfig {
        FakeModelConfig {
            completion_tokens: tokens,
            first_token_delay: Duration::from_millis(1),
            ..Default::default()
        }
    }

    #[test]
    fn the_quick_suite_measures_every_row_it_names() {
        let (_server, client) = served(timed(160));
        let profile = profile();
        let mut runner = Runner::new(&client, &profile);
        runner.warmup = false;
        let record = runner.run("quick").unwrap();
        assert_eq!(record["ok"], true);
        assert_eq!(record["suite"], "quick");
        assert_eq!(record["harness"]["workload"], "loom-launch/text-v1");
        let measured = record["metrics"].as_object().unwrap();
        let mut names: Vec<_> = measured.keys().cloned().collect();
        names.sort();
        assert_eq!(names, ["decode", "prefill_1k", "prefill_2k"]);
        for entry in measured.values() {
            assert!(!entry["value"].is_null(), "{entry}");
            assert_eq!(entry["runs"], 1);
        }
        assert_eq!(measured["decode"]["completion_tokens"], 160);
        assert!(measured["prefill_1k"]["prompt_tokens"].as_i64().unwrap() > 500);
        assert_eq!(record["comparison"][0]["metric"], "decode");
        assert_eq!(record["comparison"][0]["published"], 10.0);
        assert_eq!(record["comparison"][0]["baseline"], 5.0);
        assert_eq!(
            record["comparison"][0]["measured"],
            record["comparison"][0]["value"]
        );
    }

    #[test]
    fn progress_counts_every_request_of_the_suite() {
        let (_server, client) = served(timed(160));
        let profile = profile();
        let mut runner = Runner::new(&client, &profile);
        runner.repeats = 2;
        let seen = std::sync::Mutex::new(Vec::new());
        let observe = |event: &BenchEvent| {
            seen.lock().unwrap().push(match event {
                BenchEvent::Begin {
                    done,
                    total,
                    metric,
                    repeat,
                    repeats,
                } => format!("{done}/{total} {metric} {repeat}/{repeats}"),
                BenchEvent::Measured { metric, value, .. } => {
                    format!("{metric} = {}", value.is_some())
                }
            })
        };
        runner.on_event = Some(&observe);
        runner.run("quick").unwrap();
        assert_eq!(
            *seen.lock().unwrap(),
            [
                "0/7 warm-up 1/1",
                "1/7 decode 1/2",
                "2/7 decode 2/2",
                "decode = true",
                "3/7 prefill_1k 1/2",
                "4/7 prefill_1k 2/2",
                "prefill_1k = true",
                "5/7 prefill_2k 1/2",
                "6/7 prefill_2k 2/2",
                "prefill_2k = true",
            ]
        );
    }

    #[test]
    fn short_completions_are_not_reported_as_decode_measurements() {
        let (_server, client) = served(FakeModelConfig {
            completion_tokens: 4,
            ..Default::default()
        });
        let profile = profile();
        let mut runner = Runner::new(&client, &profile);
        runner.warmup = false;
        let result = runner.measure("decode");
        assert_eq!(result.value, None);
        assert!(result.note.unwrap().contains("not a decode measurement"));
    }

    #[test]
    fn four_streams_report_an_aggregate_rate() {
        let (server, client) = served(timed(32));
        let profile = profile();
        let mut runner = Runner::new(&client, &profile);
        runner.max_sessions = Some(1);
        let result = runner.measure_streams(4, 128);
        assert_eq!(result.metric, "streams4");
        assert!(result.value.unwrap() > 0.0);
        assert_eq!(result.completion_tokens, Some(32 * 4));
        assert_eq!(server.completions().len(), 4);
        assert!(result
            .note
            .unwrap()
            .contains("deployment admits 1 session(s)"));
    }

    #[test]
    fn repeats_keep_samples_and_report_the_median() {
        let (_server, client) = served(timed(160));
        let profile = profile();
        let mut runner = Runner::new(&client, &profile);
        runner.repeats = 3;
        let result = runner.measure("prefill_1k");
        assert_eq!(result.samples.len(), 3);
        assert_eq!(result.value, median(&result.samples));
        assert_eq!(median(&[1.0, 3.0, 2.0, 10.0]), Some(2.5));
    }

    #[test]
    fn comparison_pairs_measured_rows_with_published_ones() {
        let profile = profile();
        let results = vec![
            MetricResult {
                metric: "decode".into(),
                value: Some(20.0),
                ..MetricResult::new("decode")
            },
            MetricResult {
                metric: "prefill_1k".into(),
                value: Some(100.0),
                ..MetricResult::new("prefill_1k")
            },
            MetricResult::new("prefill_2k"),
        ];
        let rows = compare(&profile, &results);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["published"], 10.0);
        assert_eq!(rows[0]["ratio"], 2.0);
        assert_eq!(rows[0]["ceiling"], 100.0);
        assert_eq!(rows[0]["roofline_fraction"], 0.2);
        assert_eq!(rows[1]["published"], 200.0);
        assert_eq!(rows[1]["ratio"], 0.5);
        assert!(rows[1]["ceiling"].is_null());
    }

    #[test]
    fn reports_render_the_python_table_layout() {
        let record = json!({
            "metrics": {
                "decode": {"value": 80.18},
                "prefill_1k": {"value": 2595.79},
                "streams4": {"value": 200.89},
            },
            "comparison": [
                {"metric": "decode", "published": 77.7, "ratio": 1.03, "roofline_fraction": 0.92},
                {"metric": "prefill_1k", "published": 2982.1, "ratio": 0.87, "roofline_fraction": 0.33},
            ],
        });
        let lines = format_metrics(obj(&record));
        assert_eq!(
            lines[0],
            "metric                      value  published   ratio  roofline"
        );
        assert_eq!(lines[1], "-".repeat(62));
        assert_eq!(
            lines[2],
            "decode                       80.2       77.7   1.03x       92%"
        );
        assert_eq!(
            lines[3],
            "prefill_1k                2,595.8    2,982.1   0.87x       33%"
        );
        assert_eq!(
            lines[4],
            "streams4                    200.9          -       -         -"
        );
        let published = format_published(&profile());
        assert_eq!(
            published[2],
            "decode                       10.0        5.0   2.00x       10%"
        );
        assert_eq!(
            published[3],
            "prefill_1k                  200.0          -       -         -"
        );
        let empty = Profile::from_documents(
            "x/y",
            &ProfileDocuments {
                model_card: obj(&json!({"model": {"id": "x/y"}})).clone(),
                ..Default::default()
            },
            None,
        );
        assert!(format_published(&empty)
            .iter()
            .any(|l| l.contains("no measurements published")));
    }

    #[test]
    fn an_unknown_suite_is_refused_before_measuring() {
        let (server, client) = served(FakeModelConfig::default());
        let profile = profile();
        let runner = Runner::new(&client, &profile);
        let error = runner.run("absent").unwrap_err();
        assert!(error.message.contains("unknown suite"));
        assert!(server.completions().is_empty());
    }
}
