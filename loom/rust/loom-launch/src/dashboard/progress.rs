//! Progress the dashboard derives from what its jobs observe.
//!
//! - An install reads `loom-fetch`'s console lines (`loom-fetch  <Stage>
//!   <details>`): the file count it reports (`3/15 files`), and per file the
//!   bytes its stage has handled (`[###.........]  25%  1.0 GiB / 4.0 GiB`),
//!   measured against the total the card's lock declares when one is known.
//! - A launch reads the server's output and the supervisor's events: the
//!   listener's `ready` event, i8-panel derivation (`i8-panel deriving`,
//!   `derived` or `reused`, and the loader's `validation_seconds`), the
//!   registration phases, and readiness. A stage has no measurable extent, so a
//!   launch reports its stages with their times and an indeterminate progress.

use super::jobs::{Progress, Step, StepState};
use crate::values::{arr, get, int_value, obj, str_value};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

// -- installs ------------------------------------------------------------------

/// One parsed `loom-fetch` console line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FetchLine {
    /// The console stage label: Metadata, Access, Output, Fetch, Download,
    /// Transfer, Read, Verify, Transform, Install, Copy, Complete or Warning.
    pub stage: String,
    pub percent: Option<u32>,
    pub bytes_done: Option<u64>,
    pub bytes_total: Option<u64>,
    pub path: Option<String>,
    pub files: Option<(u64, u64)>,
}

/// `0 bytes`, `121 bytes`, `20.8 KiB`, `1.5 GiB` (`loom.model.memory.format_bytes`).
pub fn parse_size(text: &str) -> Option<u64> {
    let (number, unit) = text.trim().split_once(' ')?;
    let value: f64 = number.parse().ok()?;
    let scale = match unit.trim() {
        "bytes" | "byte" | "B" => 1u64,
        "KiB" => 1 << 10,
        "MiB" => 1 << 20,
        "GiB" => 1 << 30,
        "TiB" => 1 << 40,
        "PiB" => 1 << 50,
        _ => return None,
    };
    (value >= 0.0).then(|| (value * scale as f64).round() as u64)
}

fn parse_files(chunk: &str) -> Option<(u64, u64)> {
    let counts = chunk.trim().strip_suffix(" files")?;
    let (done, total) = counts.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// Reads one line `loom-fetch` printed, or `None` for anything else.
pub fn parse_fetch_line(line: &str) -> Option<FetchLine> {
    let rest = line.trim_start().strip_prefix("loom-fetch ")?.trim_start();
    let (stage, details) = rest.split_once(' ').unwrap_or((rest, ""));
    if stage.is_empty() || !stage.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut parsed = FetchLine {
        stage: stage.to_string(),
        ..Default::default()
    };
    if matches!(stage, "Warning" | "Output" | "Metadata" | "Access") {
        return Some(parsed);
    }
    for chunk in details.split("  ").map(str::trim).filter(|c| !c.is_empty()) {
        if let Some(files) = parse_files(chunk) {
            parsed.files = Some(files);
        } else if chunk.starts_with('[') && chunk.ends_with(']') {
            // The bar itself; the percentage follows as its own chunk.
        } else if let Some(percent) = chunk.strip_suffix('%') {
            parsed.percent = percent.trim().parse().ok();
        } else if let Some((done, total)) = chunk.split_once(" / ") {
            if let (Some(done), Some(total)) = (parse_size(done), parse_size(total)) {
                parsed.bytes_done = Some(done);
                parsed.bytes_total = Some(total);
            }
        } else if let Some(done) = chunk.strip_suffix(" (total unknown)") {
            parsed.bytes_done = parse_size(done);
        } else if chunk.starts_with("ETA ")
            || chunk.ends_with("/s")
            || chunk.starts_with("received ")
            || chunk.ends_with("/s)")
        {
            // Rate and time estimates: shown in the output, not measured here.
        } else if parsed.path.is_none() {
            parsed.path = Some(chunk.to_string());
        }
    }
    Some(parsed)
}

/// The bytes a card's lock (`model.json.lock`) declares for the files an
/// install fetches: every listed file but those derived locally.
pub fn declared_bytes(lock: &Map<String, Value>) -> Option<u64> {
    let files = arr(get(lock, "files"));
    let sizes: Vec<u64> = files
        .iter()
        .map(obj)
        .filter(|file| get(file, "transform").is_null() && get(file, "derived").is_null())
        .filter_map(|file| int_value(get(file, "size")))
        .filter_map(|size| u64::try_from(size).ok())
        .collect();
    (!sizes.is_empty()).then(|| sizes.iter().sum())
}

/// What one install has shown so far.
#[derive(Clone, Debug, Default)]
pub struct FetchTracker {
    declared: Option<u64>,
    files: Option<(u64, u64)>,
    /// Bytes handled per file path, with the file's size when shown.
    per_path: BTreeMap<String, (u64, Option<u64>)>,
    last_path: Option<String>,
    stage: Option<String>,
    percent: Option<u32>,
    complete: bool,
}

/// The job step a `loom-fetch` stage belongs to.
fn fetch_step(stage: &str) -> Option<(&'static str, &'static str)> {
    Some(match stage {
        "Metadata" | "Access" | "Output" => ("card", "Read the card and check access"),
        "Fetch" | "Download" | "Transfer" | "Read" | "Verify" | "Transform" => {
            ("files", "Fetch and verify files")
        }
        "Install" | "Copy" => ("place", "Install files"),
        _ => return None,
    })
}

impl FetchTracker {
    /// `declared`: the card's total bytes, when its lock could be read first.
    pub fn new(declared: Option<u64>) -> FetchTracker {
        FetchTracker {
            declared,
            ..Default::default()
        }
    }

    /// Takes one output line; returns the step it moves the install to.
    pub fn observe(&mut self, line: &str) -> Option<(&'static str, &'static str)> {
        let parsed = parse_fetch_line(line)?;
        if let Some(files) = parsed.files {
            self.files = Some(files);
        }
        if parsed.stage == "Complete" {
            self.complete = true;
            for entry in self.per_path.values_mut() {
                if let Some(total) = entry.1 {
                    entry.0 = total;
                }
            }
            return None;
        }
        if parsed.stage == "Warning" {
            return None;
        }
        if let Some(path) = &parsed.path {
            // A new file in a byte-reporting stage means the previous one finished.
            if parsed.bytes_done.is_some() && self.last_path.as_deref() != Some(path) {
                if let Some(previous) = self.last_path.take() {
                    if let Some(entry) = self.per_path.get_mut(&previous) {
                        if let Some(total) = entry.1 {
                            entry.0 = entry.0.max(total);
                        }
                    }
                }
            }
            if let Some(done) = parsed.bytes_done {
                let entry = self.per_path.entry(path.clone()).or_insert((0, None));
                entry.0 = entry.0.max(done);
                if parsed.bytes_total.is_some() {
                    entry.1 = parsed.bytes_total;
                }
                self.last_path = Some(path.clone());
            }
        }
        self.stage = Some(match &parsed.path {
            Some(path) if parsed.stage != "Output" => format!("{} {path}", parsed.stage),
            _ => parsed.stage.clone(),
        });
        self.percent = parsed.percent;
        fetch_step(&parsed.stage)
    }

    fn bytes_done(&self) -> u64 {
        let seen: u64 = self.per_path.values().map(|(done, _)| *done).sum();
        match (self.complete, self.declared) {
            (true, Some(declared)) => declared,
            (_, Some(declared)) => seen.min(declared),
            _ => seen,
        }
    }

    fn bytes_total(&self) -> Option<u64> {
        self.declared.or_else(|| {
            let shown: Vec<u64> = self.per_path.values().filter_map(|(_, t)| *t).collect();
            (!shown.is_empty()).then(|| shown.iter().sum())
        })
    }

    /// Bytes against the declared total when the card declared one, else files.
    pub fn progress(&self) -> Progress {
        let label = self.stage.as_ref().map(|stage| match self.percent {
            Some(percent) if !self.complete => format!("{stage} ({percent}%)"),
            _ => stage.clone(),
        });
        let mut progress = match (self.declared, self.files) {
            (Some(declared), _) => {
                Progress::of(self.bytes_done() as f64, declared as f64, "bytes", label)
            }
            (None, Some((done, total))) if total > 0 => {
                Progress::of(done as f64, total as f64, "files", label)
            }
            _ => Progress::indeterminate("files", label),
        };
        if let Some((done, total)) = self.files {
            progress
                .detail
                .insert("files".into(), json!({"done": done, "total": total}));
        }
        if self.declared.is_some() || !self.per_path.is_empty() {
            progress.detail.insert(
                "bytes".into(),
                json!({"done": self.bytes_done(), "total": self.bytes_total()}),
            );
        }
        progress
    }
}

// -- launches ------------------------------------------------------------------

const START: (&str, &str) = ("start", "Start loom-serve");
const LISTEN: (&str, &str) = ("listen", "Open the listener");
const PANELS: (&str, &str) = ("panels", "Derive i8 panels");
const PREPARE: (&str, &str) = ("prepare", "Load weights and register");
const READY: (&str, &str) = ("ready", "Serve");

/// The stages of one launch, as its server and supervisor reveal them.
#[derive(Clone, Debug)]
pub struct LaunchTracker {
    steps: Vec<Step>,
}

impl LaunchTracker {
    pub fn new(now: f64) -> LaunchTracker {
        let mut tracker = LaunchTracker { steps: Vec::new() };
        tracker.begin(START, now);
        tracker
    }

    fn find(&mut self, name: &str) -> Option<&mut Step> {
        self.steps.iter_mut().find(|s| s.name == name)
    }

    fn begin(&mut self, (name, label): (&str, &str), now: f64) {
        if self.find(name).is_some() {
            return;
        }
        self.steps.push(Step {
            name: name.to_string(),
            label: label.to_string(),
            state: StepState::Running,
            started_at: Some(now),
            finished_at: None,
            detail: None,
        });
    }

    fn end(&mut self, stage: (&str, &str), now: f64, detail: Option<String>) {
        self.begin(stage, now);
        if let Some(step) = self.find(stage.0) {
            if step.state == StepState::Running {
                step.state = StepState::Done;
                step.finished_at = Some(now);
            }
            if detail.is_some() {
                step.detail = detail;
            }
        }
    }

    fn detail(&mut self, name: &str, detail: String) {
        if let Some(step) = self.find(name) {
            step.detail = Some(detail);
        }
    }

    fn listening(&mut self, now: f64, address: Option<&str>) {
        self.end(START, now, None);
        self.end(LISTEN, now, address.map(|a| format!("at {a}")));
        self.begin(PREPARE, now);
    }

    /// One line of server output; returns whether a stage changed.
    pub fn line(&mut self, line: &str, now: f64) -> bool {
        let before = self.steps.clone();
        let text = line.trim();
        if let Some(event) = crate::server::parse_event(text) {
            if str_value(get(&event, "event")) == Some("ready") {
                self.listening(now, str_value(get(&event, "address")));
            }
        } else if let Some(rest) = text.strip_prefix("i8-panel ") {
            self.end(START, now, None);
            if rest.starts_with("deriving ") {
                self.begin(PANELS, now);
                self.detail("panels", "deriving the panels from the weights".into());
            } else if rest.starts_with("derived ") {
                self.end(PANELS, now, Some("derived and cached".into()));
            } else if rest.starts_with("reused ") {
                self.end(PANELS, now, Some("reused the cached panels".into()));
            } else if let Some(seconds) = rest
                .split_whitespace()
                .find_map(|field| field.strip_prefix("validation_seconds="))
                .and_then(|s| s.parse::<f64>().ok())
            {
                let done = self.find("panels").map(|s| s.detail.clone());
                let prefix = done.flatten().unwrap_or_else(|| "checked".into());
                self.end(
                    PANELS,
                    now,
                    Some(format!("{prefix}; the loader took {seconds:.1} s")),
                );
            }
        } else if !text.is_empty() {
            // Any output means the process is up.
            self.end(START, now, None);
        }
        self.steps != before
    }

    /// One event the supervisor recorded; returns whether a stage changed.
    pub fn event(&mut self, event: &Map<String, Value>, now: f64) -> bool {
        let before = self.steps.clone();
        match str_value(get(event, "event")) {
            Some("listening") => self.listening(now, None),
            Some("registration") => {
                self.begin(PREPARE, now);
                if let Some(state) = str_value(get(event, "state")) {
                    self.detail("prepare", format!("registration: {state}"));
                }
            }
            _ => {}
        }
        self.steps != before
    }

    /// Readiness: every stage still running ends, and serving begins.
    pub fn ready(&mut self, now: f64, endpoint: &str) {
        for step in self.steps.iter_mut() {
            if step.state == StepState::Running {
                step.state = StepState::Done;
                step.finished_at = Some(now);
            }
        }
        self.end(READY, now, Some(format!("at {endpoint}")));
    }

    pub fn steps(&self) -> Vec<Step> {
        self.steps.clone()
    }

    /// Indeterminate: how many stages ended, and the one under way.
    pub fn progress(&self) -> Progress {
        let current = self
            .steps
            .iter()
            .rev()
            .find(|s| s.state == StepState::Running)
            .map(|s| s.label.clone());
        let mut progress = Progress::indeterminate("stages", current);
        progress.done = self
            .steps
            .iter()
            .filter(|s| s.state == StepState::Done)
            .count() as f64;
        progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_the_fetch_console_spelling() {
        assert_eq!(parse_size("0 bytes"), Some(0));
        assert_eq!(parse_size("121 bytes"), Some(121));
        assert_eq!(parse_size("20.8 KiB"), Some(21299));
        assert_eq!(parse_size("1.0 GiB"), Some(1 << 30));
        assert_eq!(parse_size("ten bytes"), None);
        assert_eq!(parse_size("3 parsecs"), None);
    }

    #[test]
    fn fetch_lines_yield_stage_files_bytes_and_path() {
        let verify = parse_fetch_line(
            "loom-fetch  Verify    [............]   0%  0 bytes / 20.8 KiB  ETA estimating  \
             program/prepare.loom  0/15 files",
        )
        .unwrap();
        assert_eq!(verify.stage, "Verify");
        assert_eq!(verify.percent, Some(0));
        assert_eq!(verify.bytes_done, Some(0));
        assert_eq!(verify.bytes_total, Some(21299));
        assert_eq!(verify.path.as_deref(), Some("program/prepare.loom"));
        assert_eq!(verify.files, Some((0, 15)));
        let download = parse_fetch_line(
            "loom-fetch  Download  [###.........]  25%  1.0 GiB / 4.0 GiB  50.0 MiB/s  ETA 1m 01s  \
             received 1.1 GiB (52.0 MiB/s)  parameter/model.gguf  3/15 files",
        )
        .unwrap();
        assert_eq!(download.percent, Some(25));
        assert_eq!(download.bytes_done, Some(1 << 30));
        assert_eq!(download.path.as_deref(), Some("parameter/model.gguf"));
        let complete = parse_fetch_line("loom-fetch  Complete  15/15 files").unwrap();
        assert_eq!(complete.files, Some((15, 15)));
        assert_eq!(complete.path, None);
        let output = parse_fetch_line("loom-fetch  Output    /models/a b").unwrap();
        assert_eq!(output.stage, "Output");
        assert_eq!(output.path, None);
        assert!(parse_fetch_line("installing a/b into /x").is_none());
        assert!(parse_fetch_line("{").is_none());
    }

    #[test]
    fn the_declared_total_skips_derived_files() {
        let lock = json!({"files": [
            {"origin": "@checkpoint/a.gguf", "size": 1000},
            {"origin": "@document/tokenizer.json", "size": 24},
            {"origin": "parameter/derived/x", "size": 99, "transform": {"kind": "i8-panel"}},
        ]});
        assert_eq!(declared_bytes(obj(&lock)), Some(1024));
        assert_eq!(declared_bytes(obj(&json!({"files": []}))), None);
    }

    #[test]
    fn an_install_reports_bytes_against_the_declared_total_then_completes() {
        let mut tracker = FetchTracker::new(Some(4 << 30));
        assert_eq!(
            tracker.observe("loom-fetch  Metadata  reading model metadata"),
            Some(("card", "Read the card and check access"))
        );
        tracker.observe("loom-fetch  Fetch     0/3 files");
        let step = tracker.observe(
            "loom-fetch  Download  [###.........]  25%  1.0 GiB / 2.0 GiB  ETA 1m  a.gguf  0/3 files",
        );
        assert_eq!(step, Some(("files", "Fetch and verify files")));
        let progress = tracker.progress();
        assert_eq!(progress.unit, "bytes");
        assert_eq!(progress.done, (1u64 << 30) as f64);
        assert_eq!(progress.total, Some((4u64 << 30) as f64));
        assert_eq!(progress.label.as_deref(), Some("Download a.gguf (25%)"));
        assert_eq!(progress.detail["files"], json!({"done": 0, "total": 3}));
        // The next file means the previous one finished.
        tracker.observe(
            "loom-fetch  Download  [............]   0%  0 bytes / 2.0 GiB  b.gguf  1/3 files",
        );
        assert_eq!(tracker.progress().done, (2u64 << 30) as f64);
        tracker.observe("loom-fetch  Complete  3/3 files");
        let done = tracker.progress();
        assert_eq!(done.done, (4u64 << 30) as f64);
        assert_eq!(done.detail["files"], json!({"done": 3, "total": 3}));
    }

    #[test]
    fn without_a_declared_total_an_install_counts_files() {
        let mut tracker = FetchTracker::new(None);
        assert!(tracker.progress().total.is_none());
        tracker.observe("loom-fetch  Install   4/15 files");
        let progress = tracker.progress();
        assert_eq!(progress.unit, "files");
        assert_eq!((progress.done, progress.total), (4.0, Some(15.0)));
        assert!(tracker.observe("fetching 1/2 files").is_none());
    }

    #[test]
    fn a_launch_moves_through_its_stages_from_output_and_events() {
        let mut tracker = LaunchTracker::new(0.0);
        assert_eq!(
            tracker.progress().label.as_deref(),
            Some("Start loom-serve")
        );
        assert!(tracker.line(
            "{\"event\":\"ready\",\"address\":\"127.0.0.1:8082\",\"models\":1}",
            1.0
        ));
        assert!(tracker.line("i8-panel deriving /m/parameter/derived-cache-raw/d0", 2.0));
        assert_eq!(
            tracker.progress().label.as_deref(),
            Some("Derive i8 panels")
        );
        tracker.line("i8-panel derived /m/parameter/derived-cache-raw/d0", 150.0);
        tracker.line(
            "i8-panel validation_seconds=178.578693 source_hashes=1 payload_hashes=0 verify=0",
            180.0,
        );
        tracker.event(
            obj(&json!({"event": "registration", "state": "preparing"})),
            181.0,
        );
        tracker.ready(185.0, "http://127.0.0.1:8099");
        let steps = tracker.steps();
        let names: Vec<&str> = steps.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["start", "listen", "prepare", "panels", "ready"]);
        assert!(steps.iter().all(|s| s.state == StepState::Done));
        assert_eq!(steps[1].detail.as_deref(), Some("at 127.0.0.1:8082"));
        assert_eq!(
            steps[3].detail.as_deref(),
            Some("derived and cached; the loader took 178.6 s")
        );
        assert_eq!(steps[3].started_at, Some(2.0));
        assert_eq!(steps[2].detail.as_deref(), Some("registration: preparing"));
        let progress = tracker.progress();
        assert!(progress.total.is_none());
        assert_eq!(progress.done, 5.0);
    }

    #[test]
    fn reused_panels_and_a_health_fallback_are_stages_too() {
        let mut tracker = LaunchTracker::new(0.0);
        tracker.event(obj(&json!({"event": "listening"})), 1.0);
        tracker.line("i8-panel reused /m/cache/d0", 2.0);
        let panels = tracker
            .steps()
            .into_iter()
            .find(|s| s.name == "panels")
            .unwrap();
        assert_eq!(panels.detail.as_deref(), Some("reused the cached panels"));
        assert_eq!(panels.state, StepState::Done);
        assert!(!tracker.line("i8-panel reused /m/cache/d0", 3.0));
    }
}
