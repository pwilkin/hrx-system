//! `/api/events`: one server-sent event per second for each open page.
//!
//! Every event carries the whole live picture (launch, counters, last request,
//! job), the counter samples the page has not seen yet (all of them on the
//! first event, so sparklines start with history) and the log lines appended
//! since the previous event, so the page needs no polling loop of its own.

use super::Context;
use crate::values::{dumps_compact, get, number_value, obj};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Samples kept for sparklines (one per second: ten minutes).
pub const HISTORY: usize = 600;
/// Most log bytes one event carries per file; a faster writer skips ahead.
const LOG_CHUNK: u64 = 256 * 1024;

/// One second of counters, as the sparklines plot them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    pub t: f64,
    /// The launch the counters belong to; a new launch restarts the history.
    pub key: String,
    pub requests: f64,
    pub errors: f64,
    pub prompt_tokens: f64,
    pub completion_tokens: f64,
    pub cached_tokens: f64,
    pub in_flight: f64,
    pub ttft_p50: Option<f64>,
    pub ttft_p95: Option<f64>,
    pub decode_mean: Option<f64>,
}

impl Sample {
    /// Reads a metrics snapshot (`Metrics::snapshot`, possibly absent).
    pub fn from_snapshot(t: f64, key: &str, snapshot: &Value) -> Sample {
        let snapshot = obj(snapshot);
        let number = |key: &str| number_value(get(snapshot, key)).unwrap_or(0.0);
        let nested = |outer: &str, inner: &str| number_value(get(obj(get(snapshot, outer)), inner));
        Sample {
            t,
            key: key.to_string(),
            requests: number("requests"),
            errors: number("errors"),
            prompt_tokens: number("prompt_tokens"),
            completion_tokens: number("completion_tokens"),
            cached_tokens: number("cached_tokens"),
            in_flight: number("in_flight"),
            ttft_p50: nested("ttft_seconds", "p50"),
            ttft_p95: nested("ttft_seconds", "p95"),
            decode_mean: nested("decode_tokens_per_second", "mean"),
        }
    }

    pub fn describe(&self) -> Value {
        json!({
            "t": crate::values::round_to(self.t, 3),
            "key": self.key,
            "requests": self.requests,
            "errors": self.errors,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "in_flight": self.in_flight,
            "ttft_p50": self.ttft_p50,
            "ttft_p95": self.ttft_p95,
            "decode_mean": self.decode_mean,
        })
    }
}

/// The recent samples of the launch currently shown.
#[derive(Debug, Default)]
pub struct History {
    samples: VecDeque<Sample>,
}

impl History {
    pub fn push(&mut self, sample: Sample) {
        if self
            .samples
            .back()
            .is_some_and(|last| last.key != sample.key)
        {
            self.samples.clear();
        }
        if self.samples.len() == HISTORY {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Samples taken after `t` (all of them when `t` is `None`).
    pub fn since(&self, t: Option<f64>) -> Vec<&Sample> {
        self.samples
            .iter()
            .filter(|s| t.is_none_or(|after| s.t > after))
            .collect()
    }
}

/// What one open page has already been sent.
#[derive(Debug, Default)]
pub struct Cursor {
    pub last_sample: Option<f64>,
    started: bool,
    run: Option<String>,
    serve: u64,
    events: u64,
}

impl Cursor {
    /// The log lines appended since the previous event. The first call only
    /// positions the cursor (the page reads the tail from `/api/logs`); a new
    /// run is announced with `reset` so the page reloads its tail.
    pub fn advance(&mut self, state_root: &Path) -> Value {
        let run = super::current_run(state_root);
        let id = run.as_ref().map(|r| r.run_id.clone());
        let lengths = |run: &Option<crate::store::RunDirectory>| match run {
            Some(run) => (file_length(&run.serve_log()), file_length(&run.event_log())),
            None => (0, 0),
        };
        if !self.started || id != self.run {
            let reset = self.started;
            self.started = true;
            self.run = id.clone();
            (self.serve, self.events) = lengths(&run);
            return json!({"run": id, "reset": reset, "serve": [], "events": []});
        }
        let (serve, events) = match &run {
            Some(run) => (
                appended(&run.serve_log(), &mut self.serve),
                appended(&run.event_log(), &mut self.events),
            ),
            None => (Vec::new(), Vec::new()),
        };
        json!({"run": id, "reset": false, "serve": serve, "events": events})
    }
}

fn file_length(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Complete lines written to `path` after `offset`, advancing it past them.
fn appended(path: &Path, offset: &mut u64) -> Vec<String> {
    let length = file_length(path);
    if length < *offset {
        // Truncated or replaced: start over.
        *offset = 0;
    }
    if length == *offset {
        return Vec::new();
    }
    let mut skipped = false;
    if length - *offset > LOG_CHUNK {
        *offset = length - LOG_CHUNK;
        skipped = true;
    }
    let Ok(mut file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    if file.seek(SeekFrom::Start(*offset)).is_err() {
        return Vec::new();
    }
    let mut chunk = Vec::new();
    if file.take(length - *offset).read_to_end(&mut chunk).is_err() {
        return Vec::new();
    }
    let Some(end) = chunk.iter().rposition(|b| *b == b'\n') else {
        return Vec::new();
    };
    *offset += end as u64 + 1;
    let text = String::from_utf8_lossy(&chunk[..end]);
    let mut lines: Vec<String> = text
        .split('\n')
        .map(|l| l.trim_end_matches('\r').to_string())
        .collect();
    if skipped && !lines.is_empty() {
        // The first line was entered mid-way.
        lines[0] = "…".to_string();
    }
    lines
}

fn write_chunk(writer: &mut dyn Write, data: &[u8]) -> std::io::Result<()> {
    if data.is_empty() {
        writer.write_all(b"0\r\n\r\n")?;
    } else {
        writer.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
        writer.write_all(data)?;
        writer.write_all(b"\r\n")?;
    }
    writer.flush()
}

/// Serves one event stream until the client leaves or the dashboard closes.
pub fn stream(context: Arc<Context>, request: tiny_http::Request) {
    let mut writer = request.into_writer();
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
                X-Accel-Buffering: no\r\nX-Content-Type-Options: nosniff\r\n\
                Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    if writer
        .write_all(head.as_bytes())
        .and_then(|_| writer.flush())
        .is_err()
    {
        return;
    }
    let mut cursor = Cursor::default();
    let mut sequence: u64 = 0;
    loop {
        let document = context.event(sequence, &mut cursor);
        let mut frame = String::new();
        if sequence == 0 {
            frame.push_str("retry: 2000\n");
        }
        frame.push_str(&format!(
            "id: {sequence}\nevent: snapshot\ndata: {}\n\n",
            dumps_compact(&document)
        ));
        if write_chunk(&mut *writer, frame.as_bytes()).is_err() {
            return;
        }
        sequence += 1;
        let next = Instant::now() + Duration::from_secs(1);
        while Instant::now() < next {
            if context.closed() {
                let _ = write_chunk(&mut *writer, b"");
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::tests::scratch;

    #[test]
    fn samples_read_snapshots_and_a_new_launch_restarts_history() {
        let snapshot = json!({
            "requests": 3, "errors": 1, "prompt_tokens": 40, "completion_tokens": 11,
            "cached_tokens": 2, "in_flight": 1,
            "ttft_seconds": {"p50": 0.2, "p95": null},
            "decode_tokens_per_second": {"mean": 10.0},
        });
        let sample = Sample::from_snapshot(1.0, "a", &snapshot);
        assert_eq!(sample.requests, 3.0);
        assert_eq!(sample.ttft_p50, Some(0.2));
        assert_eq!(sample.ttft_p95, None);
        assert_eq!(sample.decode_mean, Some(10.0));
        assert_eq!(Sample::from_snapshot(1.0, "", &Value::Null).requests, 0.0);
        let mut history = History::default();
        history.push(sample.clone());
        history.push(Sample {
            t: 2.0,
            ..sample.clone()
        });
        assert_eq!(history.since(Some(1.0)).len(), 1);
        assert_eq!(history.since(None).len(), 2);
        history.push(Sample {
            t: 3.0,
            key: "b".into(),
            ..sample
        });
        assert_eq!(history.since(None).len(), 1);
        assert_eq!(history.since(None)[0].key, "b");
    }

    #[test]
    fn appended_lines_are_complete_and_follow_truncation() {
        let root = scratch("events-append");
        let path = root.join("serve.log");
        std::fs::write(&path, "one\ntwo\npart").unwrap();
        let mut offset = 0;
        assert_eq!(appended(&path, &mut offset), ["one", "two"]);
        assert_eq!(appended(&path, &mut offset), Vec::<String>::new());
        std::fs::write(&path, "one\ntwo\npartial line\nthree\n").unwrap();
        assert_eq!(appended(&path, &mut offset), ["partial line", "three"]);
        std::fs::write(&path, "fresh\n").unwrap();
        assert_eq!(appended(&path, &mut offset), ["fresh"]);
    }
}
