//! Actions the dashboard runs in the background, each kept as one row the page
//! shows until it is dismissed: install (from the hub or a local path), launch,
//! stop, benchmark and the chat's web fetches.
//!
//! A job moves `queued` → `running` → `done` | `failed`. While it runs it
//! carries the current step, the steps seen so far with their times, a progress
//! measure (determinate only when the work has a known extent) and its output
//! lines; a finished job keeps its outcome, error, a hint on what to do next and
//! the request that started it, so the page can offer a retry after a reload.
//! Install, launch and benchmark share one slot (one at a time); stops and
//! fetches run beside them.

use crate::values::{iso_utc, now_seconds, round_to};
use serde_json::{json, Map, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Output lines a job keeps (older ones are counted, not kept).
const KEPT_LINES: usize = 400;
/// Jobs kept for the page, newest first; older finished ones are forgotten.
const KEPT_JOBS: usize = 30;
/// Kinds that share the single slot.
const EXCLUSIVE: [&str; 3] = ["install", "serve", "bench"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobState {
    Queued,
    Running,
    Done,
    Failed,
}

impl JobState {
    fn name(self) -> &'static str {
        match self {
            JobState::Queued => "queued",
            JobState::Running => "running",
            JobState::Done => "done",
            JobState::Failed => "failed",
        }
    }

    fn active(self) -> bool {
        matches!(self, JobState::Queued | JobState::Running)
    }
}

/// How far a job has come. `total: None` is indeterminate: the work has no
/// extent the launcher can measure, so the page shows activity, not a bar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Progress {
    pub done: f64,
    pub total: Option<f64>,
    pub unit: String,
    pub label: Option<String>,
    /// Secondary measures (files and bytes of an install).
    pub detail: Map<String, Value>,
}

impl Progress {
    pub fn indeterminate(unit: &str, label: Option<String>) -> Progress {
        Progress {
            unit: unit.to_string(),
            label,
            ..Default::default()
        }
    }

    pub fn of(done: f64, total: f64, unit: &str, label: Option<String>) -> Progress {
        Progress {
            done,
            total: Some(total),
            unit: unit.to_string(),
            label,
            detail: Map::new(),
        }
    }

    pub fn describe(&self) -> Value {
        let fraction = self
            .total
            .filter(|t| *t > 0.0)
            .map(|t| round_to((self.done / t).clamp(0.0, 1.0), 4));
        let mut document = json!({
            "done": self.done,
            "total": self.total,
            "unit": self.unit,
            "label": self.label,
            "fraction": fraction,
        });
        for (key, value) in &self.detail {
            document[key] = value.clone();
        }
        document
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

impl StepState {
    fn name(self) -> &'static str {
        match self {
            StepState::Pending => "pending",
            StepState::Running => "running",
            StepState::Done => "done",
            StepState::Failed => "failed",
            StepState::Skipped => "skipped",
        }
    }
}

/// One stage of a job, timed from when it was seen to start until it ended.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub name: String,
    pub label: String,
    pub state: StepState,
    pub started_at: Option<f64>,
    pub finished_at: Option<f64>,
    pub detail: Option<String>,
}

impl Step {
    fn describe(&self) -> Value {
        let elapsed = self
            .started_at
            .map(|start| round_to(self.finished_at.unwrap_or_else(now_seconds) - start, 1));
        json!({
            "name": self.name,
            "label": self.label,
            "state": self.state.name(),
            "started_at": self.started_at.map(iso_utc),
            "finished_at": self.finished_at.map(iso_utc),
            "elapsed_seconds": elapsed,
            "detail": self.detail,
        })
    }
}

#[derive(Debug)]
struct Job {
    id: String,
    kind: String,
    target: String,
    state: JobState,
    created_at: f64,
    started_at: Option<f64>,
    finished_at: Option<f64>,
    lines: VecDeque<String>,
    line_count: usize,
    progress: Option<Progress>,
    steps: Vec<Step>,
    result: Value,
    error: Option<String>,
    hint: Option<String>,
    /// The route and body that started it, for a retry.
    retry: Value,
    /// Extra facts the page shows (install source, suite, URL...).
    facts: Map<String, Value>,
    dismissed: bool,
}

impl Job {
    fn describe(&self, max_lines: usize) -> Value {
        let end = self.finished_at.unwrap_or_else(now_seconds);
        let lines: Vec<&String> = self
            .lines
            .iter()
            .skip(self.lines.len().saturating_sub(max_lines))
            .collect();
        let step = self
            .steps
            .iter()
            .rev()
            .find(|s| s.state == StepState::Running)
            .or_else(|| self.steps.last())
            .map(|s| s.label.clone());
        let mut document = json!({
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "state": self.state.name(),
            "created_at": iso_utc(self.created_at),
            "started_at": iso_utc(self.started_at.unwrap_or(self.created_at)),
            "finished_at": self.finished_at.map(iso_utc),
            "elapsed_seconds": round_to(end - self.started_at.unwrap_or(self.created_at), 1),
            "step": step,
            "steps": self.steps.iter().map(Step::describe).collect::<Vec<_>>(),
            "progress": self.progress.as_ref().map(Progress::describe),
            "line_count": self.line_count,
            "lines": lines,
            "last_line": self.lines.back(),
            "result": self.result,
            "error": self.error,
            "hint": self.hint,
            "retry": self.retry,
        });
        for (key, value) in &self.facts {
            document[key] = value.clone();
        }
        document
    }
}

/// The dashboard's jobs, newest first.
#[derive(Debug, Default)]
pub struct Jobs {
    jobs: Mutex<VecDeque<Job>>,
    counter: AtomicU64,
}

impl Jobs {
    /// Starts a job, `queued` until its worker calls [`JobHandle::begin`].
    /// Install, launch and benchmark refuse while one of them is active; the
    /// refusal names it.
    pub fn start(self: &Arc<Self>, kind: &str, target: &str) -> Result<JobHandle, String> {
        self.start_with(kind, target, Value::Null)
    }

    /// [`Jobs::start`] remembering the request a retry repeats (`{route, body}`).
    pub fn start_with(
        self: &Arc<Self>,
        kind: &str,
        target: &str,
        retry: Value,
    ) -> Result<JobHandle, String> {
        let mut jobs = self.jobs.lock().unwrap();
        if EXCLUSIVE.contains(&kind) {
            if let Some(job) = jobs
                .iter()
                .find(|j| j.state.active() && EXCLUSIVE.contains(&j.kind.as_str()))
            {
                let noun = noun(&job.kind);
                let article = if noun.starts_with(['a', 'e', 'i', 'o', 'u']) {
                    "an"
                } else {
                    "a"
                };
                return Err(format!(
                    "{article} {noun} of {} is still running (job {}); wait for it to finish",
                    job.target, job.id
                ));
            }
        }
        let id = format!(
            "{}-{}",
            kind,
            self.counter.fetch_add(1, Ordering::SeqCst) + 1
        );
        jobs.push_front(Job {
            id: id.clone(),
            kind: kind.to_string(),
            target: target.to_string(),
            state: JobState::Queued,
            created_at: now_seconds(),
            started_at: None,
            finished_at: None,
            lines: VecDeque::new(),
            line_count: 0,
            progress: None,
            steps: Vec::new(),
            result: Value::Null,
            error: None,
            hint: None,
            retry,
            facts: Map::new(),
            dismissed: false,
        });
        // Forget the oldest finished jobs; active ones always stay.
        while jobs.len() > KEPT_JOBS {
            match jobs.iter().rposition(|j| !j.state.active()) {
                Some(index) => {
                    jobs.remove(index);
                }
                None => break,
            }
        }
        Ok(JobHandle {
            observer: JobObserver {
                jobs: self.clone(),
                id: id.clone(),
            },
            id,
            finished: false,
        })
    }

    fn with_job(&self, id: &str, action: impl FnOnce(&mut Job)) {
        if let Some(job) = self.jobs.lock().unwrap().iter_mut().find(|j| j.id == id) {
            action(job);
        }
    }

    /// Adds an output line to job `id` while it is active.
    pub fn push_line(&self, id: &str, line: &str) {
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        self.with_job(id, |job| {
            if !job.state.active() {
                return;
            }
            job.line_count += 1;
            job.lines.push_back(line.to_string());
            while job.lines.len() > KEPT_LINES {
                job.lines.pop_front();
            }
        });
    }

    /// The active install, launch or benchmark: its kind and target.
    pub fn running(&self) -> Option<(String, String)> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|j| j.state.active() && EXCLUSIVE.contains(&j.kind.as_str()))
            .map(|j| (j.kind.clone(), j.target.clone()))
    }

    /// The active job of `kind`, if any: its id.
    pub fn active(&self, kind: &str) -> Option<String> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|j| j.state.active() && j.kind == kind)
            .map(|j| j.id.clone())
    }

    /// The latest install, launch or benchmark (running or finished), with its
    /// most recent output lines: the single job `/api/status` always reported.
    pub fn describe(&self, max_lines: usize) -> Value {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|j| EXCLUSIVE.contains(&j.kind.as_str()))
            .map(|job| job.describe(max_lines))
            .unwrap_or(Value::Null)
    }

    /// Every job not dismissed, newest first.
    pub fn list(&self, max_lines: usize) -> Value {
        Value::Array(
            self.jobs
                .lock()
                .unwrap()
                .iter()
                .filter(|j| !j.dismissed)
                .map(|j| j.describe(max_lines))
                .collect(),
        )
    }

    /// One job (dismissed or not) with up to `max_lines` of its output.
    pub fn get(&self, id: &str, max_lines: usize) -> Option<Value> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|j| j.id == id)
            .map(|j| j.describe(max_lines))
    }

    /// Hides a finished job from the list; an active one cannot be dismissed.
    pub fn dismiss(&self, id: &str) -> Result<(), String> {
        let mut jobs = self.jobs.lock().unwrap();
        let Some(job) = jobs.iter_mut().find(|j| j.id == id) else {
            return Err(format!("no job {id}"));
        };
        if job.state.active() {
            return Err(format!("{id} is still {}", job.state.name()));
        }
        job.dismissed = true;
        Ok(())
    }

    /// The state of the job with `id`, while it is kept.
    pub fn state_of(&self, id: &str) -> Option<JobState> {
        self.jobs
            .lock()
            .unwrap()
            .iter()
            .find(|j| j.id == id)
            .map(|j| j.state)
    }
}

/// `install` → "install", `serve` → "launch", `bench` → "benchmark".
fn noun(kind: &str) -> &str {
    match kind {
        "serve" => "launch",
        "bench" => "benchmark",
        other => other,
    }
}

/// Reports on one job from wherever its work happens (a supervisor's output
/// callback, a progress observer); only its [`JobHandle`] finishes it.
#[derive(Clone)]
pub struct JobObserver {
    jobs: Arc<Jobs>,
    pub id: String,
}

impl JobObserver {
    /// Marks the job running (its worker has started).
    pub fn begin(&self) {
        self.jobs.with_job(&self.id, |job| {
            if job.state == JobState::Queued {
                job.state = JobState::Running;
                job.started_at = Some(now_seconds());
            }
        });
    }

    pub fn line(&self, line: &str) {
        self.jobs.push_line(&self.id, line);
    }

    pub fn set_progress(&self, progress: Option<Progress>) {
        self.jobs.with_job(&self.id, |job| job.progress = progress);
    }

    pub fn set_fact(&self, key: &str, value: Value) {
        self.jobs.with_job(&self.id, |job| {
            job.facts.insert(key.to_string(), value);
        });
    }

    pub fn set_hint(&self, hint: &str) {
        self.jobs
            .with_job(&self.id, |job| job.hint = Some(hint.to_string()));
    }

    /// Replaces the step list (a tracker that derives all of it at once).
    pub fn set_steps(&self, steps: Vec<Step>) {
        self.jobs.with_job(&self.id, |job| job.steps = steps);
    }

    /// Starts step `name`, ending any other running step as done.
    pub fn step(&self, name: &str, label: &str) {
        let now = now_seconds();
        self.jobs.with_job(&self.id, |job| {
            for step in job.steps.iter_mut() {
                if step.state == StepState::Running && step.name != name {
                    step.state = StepState::Done;
                    step.finished_at = Some(now);
                }
            }
            match job.steps.iter_mut().find(|s| s.name == name) {
                Some(step) => {
                    step.state = StepState::Running;
                    step.label = label.to_string();
                }
                None => job.steps.push(Step {
                    name: name.to_string(),
                    label: label.to_string(),
                    state: StepState::Running,
                    started_at: Some(now),
                    finished_at: None,
                    detail: None,
                }),
            }
        });
    }

    /// Ends step `name` (adding it when it was never started) with `detail`.
    pub fn end_step(&self, name: &str, label: &str, state: StepState, detail: Option<String>) {
        let now = now_seconds();
        self.jobs.with_job(&self.id, |job| {
            match job.steps.iter_mut().find(|s| s.name == name) {
                Some(step) => {
                    step.state = state;
                    step.label = label.to_string();
                    step.finished_at = Some(now);
                    if detail.is_some() {
                        step.detail = detail;
                    }
                }
                None => job.steps.push(Step {
                    name: name.to_string(),
                    label: label.to_string(),
                    state,
                    started_at: Some(now),
                    finished_at: Some(now),
                    detail,
                }),
            }
        });
    }
}

/// The worker's side of one job; finishing it twice keeps the first outcome,
/// and dropping it unfinished (a panic, an early return) marks it failed.
pub struct JobHandle {
    observer: JobObserver,
    pub id: String,
    finished: bool,
}

impl std::ops::Deref for JobHandle {
    type Target = JobObserver;

    fn deref(&self) -> &JobObserver {
        &self.observer
    }
}

impl JobHandle {
    /// A reporter for callbacks that outlive a borrow of this handle.
    pub fn observer(&self) -> JobObserver {
        self.observer.clone()
    }

    pub fn finish(&mut self, outcome: Result<Value, String>) {
        if self.finished {
            return;
        }
        self.finished = true;
        let now = now_seconds();
        self.observer.jobs.with_job(&self.id, |job| {
            job.finished_at = Some(now);
            if job.started_at.is_none() {
                job.started_at = Some(job.created_at);
            }
            let failed = outcome.is_err();
            match outcome {
                Ok(result) => {
                    job.state = JobState::Done;
                    job.result = result;
                    // A finished job's measured extent is complete.
                    if let Some(progress) = job.progress.as_mut() {
                        if let Some(total) = progress.total {
                            progress.done = total;
                        }
                    }
                }
                Err(message) => {
                    job.state = JobState::Failed;
                    job.error = Some(message);
                }
            }
            for step in job.steps.iter_mut() {
                if step.state == StepState::Running {
                    step.state = if failed {
                        StepState::Failed
                    } else {
                        StepState::Done
                    };
                    step.finished_at = Some(now);
                }
            }
        });
    }
}

impl Drop for JobHandle {
    fn drop(&mut self) {
        if !self.finished {
            self.finish(Err("the job ended without reporting an outcome".into()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_exclusive_job_runs_at_a_time_and_keeps_its_outcome() {
        let jobs = Arc::new(Jobs::default());
        let mut first = jobs.start("install", "a/b").unwrap();
        assert_eq!(jobs.state_of(&first.id), Some(JobState::Queued));
        let refusal = jobs.start("bench", "a/b").err().unwrap();
        assert!(refusal.contains("install of a/b"), "{refusal}");
        first.begin();
        assert_eq!(jobs.describe(10)["state"], "running");
        first.line("fetching 1/2");
        first.line("  ");
        assert_eq!(jobs.describe(10)["lines"], json!(["fetching 1/2"]));
        assert_eq!(jobs.running(), Some(("install".into(), "a/b".into())));
        // Stops and fetches run beside the slot.
        let fetch = jobs.start("fetch", "https://example.com/").unwrap();
        assert_eq!(jobs.list(1).as_array().unwrap().len(), 2);
        drop(fetch);
        first.finish(Ok(json!({"path": "/x"})));
        first.finish(Err("ignored".into()));
        let described = jobs.describe(10);
        assert_eq!(described["state"], "done");
        assert_eq!(described["result"]["path"], "/x");
        assert!(jobs.running().is_none());
        let second = jobs.start("bench", "a/b").unwrap();
        assert_ne!(second.id, first.id);
        drop(second);
        assert_eq!(jobs.describe(10)["state"], "failed");
        assert_eq!(
            jobs.describe(10)["error"],
            "the job ended without reporting an outcome"
        );
    }

    #[test]
    fn progress_lines_are_bounded() {
        let jobs = Arc::new(Jobs::default());
        let handle = jobs.start("install", "a/b").unwrap();
        for index in 0..(KEPT_LINES + 10) {
            handle.line(&format!("line {index}"));
        }
        let described = jobs.describe(5);
        assert_eq!(described["line_count"], KEPT_LINES + 10);
        assert_eq!(described["lines"].as_array().unwrap().len(), 5);
        assert_eq!(described["last_line"], format!("line {}", KEPT_LINES + 9));
    }

    #[test]
    fn steps_progress_and_retries_are_described() {
        let jobs = Arc::new(Jobs::default());
        let mut handle = jobs
            .start_with(
                "serve",
                "a/b",
                json!({"route": "/api/serve", "body": {"repo": "a/b"}}),
            )
            .unwrap();
        handle.begin();
        handle.step("listen", "Open the listener");
        handle.set_progress(Some(Progress::indeterminate(
            "stages",
            Some("Open the listener".into()),
        )));
        let described = jobs.get(&handle.id, 5).unwrap();
        assert_eq!(described["step"], "Open the listener");
        assert_eq!(described["progress"]["total"], Value::Null);
        assert_eq!(described["progress"]["fraction"], Value::Null);
        handle.step("register", "Load weights and register");
        let steps = jobs.get(&handle.id, 5).unwrap()["steps"].clone();
        assert_eq!(steps[0]["state"], "done");
        assert_eq!(steps[1]["state"], "running");
        handle.end_step(
            "panels",
            "Derive i8 panels",
            StepState::Done,
            Some("reused".into()),
        );
        handle.set_progress(Some(Progress::of(1.0, 4.0, "files", None)));
        assert_eq!(
            jobs.get(&handle.id, 5).unwrap()["progress"]["fraction"],
            0.25
        );
        handle.set_hint("check the serve log");
        handle.finish(Err("the server exited".into()));
        let failed = jobs.get(&handle.id, 5).unwrap();
        assert_eq!(failed["state"], "failed");
        assert_eq!(failed["steps"][1]["state"], "failed");
        assert_eq!(failed["hint"], "check the serve log");
        assert_eq!(failed["retry"]["route"], "/api/serve");
        assert!(jobs.dismiss("absent").is_err());
        jobs.dismiss(&handle.id).unwrap();
        assert!(jobs.list(5).as_array().unwrap().is_empty());
        assert!(jobs.get(&handle.id, 5).is_some());
        let active = jobs.start("stop", "a/b").unwrap();
        assert!(jobs.dismiss(&active.id).unwrap_err().contains("queued"));
    }

    #[test]
    fn a_finished_determinate_job_reports_its_whole_extent() {
        let jobs = Arc::new(Jobs::default());
        let mut handle = jobs.start("bench", "a/b").unwrap();
        handle.begin();
        handle.set_progress(Some(Progress::of(2.0, 4.0, "requests", None)));
        handle.finish(Ok(Value::Null));
        assert_eq!(jobs.describe(1)["progress"]["done"], 4.0);
    }

    #[test]
    fn old_finished_jobs_are_forgotten_and_active_ones_kept() {
        let jobs = Arc::new(Jobs::default());
        let active = jobs.start("serve", "a/b").unwrap();
        for index in 0..(KEPT_JOBS + 5) {
            let mut fetch = jobs.start("fetch", &format!("u{index}")).unwrap();
            fetch.finish(Ok(Value::Null));
        }
        let listed = jobs.list(0);
        assert_eq!(listed.as_array().unwrap().len(), KEPT_JOBS);
        assert!(jobs.state_of(&active.id).is_some());
    }
}
