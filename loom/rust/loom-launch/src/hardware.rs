//! Local hardware facts and profile compatibility.
//!
//! Detection is best effort: every probe is optional, an absent tool answers
//! nothing, and a CPU-only machine still yields a usable description. The result
//! decides which device URI a launch defaults to and how a profile ranks for this
//! machine; it never overrides a profile's own requirements.

use crate::profile::{gfx_arch, Profile};
use crate::values::human_bytes;
use serde_json::{json, Value};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

const MEMINFO: &str = "/proc/meminfo";
const CPUINFO: &str = "/proc/cpuinfo";
const NPU_DEVICE: &str = "/dev/accel/accel0";

/// One execution device the launcher may select.
#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    pub name: String,
    pub driver: String,
    pub uri: String,
    pub gfx_arch: Option<String>,
}

impl Device {
    pub fn describe(&self) -> Value {
        json!({
            "name": self.name,
            "driver": self.driver,
            "uri": self.uri,
            "gfx_arch": self.gfx_arch,
        })
    }
}

/// What this machine offers a Loom deployment.
#[derive(Clone, Debug)]
pub struct Hardware {
    pub system: String,
    pub machine: String,
    pub cpu: Option<String>,
    pub total_memory_bytes: Option<i64>,
    pub devices: Vec<Device>,
    pub npu: bool,
}

impl Hardware {
    pub fn gpu(&self) -> Option<&Device> {
        self.devices.iter().find(|d| d.driver == "amdgpu")
    }

    pub fn gfx_arch(&self) -> Option<&str> {
        self.gpu().and_then(|d| d.gfx_arch.as_deref())
    }

    /// The URI a single-device launch selects when none was requested.
    pub fn default_device_uri(&self) -> String {
        self.gpu()
            .map(|d| d.uri.clone())
            .unwrap_or_else(|| "task".to_string())
    }

    pub fn describe(&self) -> Value {
        json!({
            "system": self.system,
            "machine": self.machine,
            "cpu": self.cpu,
            "total_memory_bytes": self.total_memory_bytes,
            "memory": human_bytes(self.total_memory_bytes),
            "gpu": self.gpu().map(Device::describe),
            "npu": self.npu,
            "devices": self.devices.iter().map(Device::describe).collect::<Vec<_>>(),
            "default_device_uri": self.default_device_uri(),
        })
    }
}

/// How well one profile fits this machine, with the evidence either way.
#[derive(Clone, Debug, Default)]
pub struct Compatibility {
    pub score: f64,
    pub fits_memory: Option<bool>,
    pub device_match: Option<bool>,
    pub notes: Vec<String>,
}

impl Compatibility {
    pub fn describe(&self) -> Value {
        json!({
            "score": crate::values::round_to(self.score, 3),
            "fits_memory": self.fits_memory,
            "device_match": self.device_match,
            "notes": self.notes,
        })
    }
}

/// Injectable probes, so detection can be tested without the machine.
pub struct Probes<'a> {
    pub run: &'a dyn Fn(&[&str]) -> Option<String>,
    pub read: &'a dyn Fn(&str) -> Option<String>,
    pub exists: &'a dyn Fn(&str) -> bool,
    /// `/sys/class/drm/*/device/uevent` contents, for the no-rocminfo fallback.
    pub drm_uevents: &'a dyn Fn() -> Vec<String>,
}

/// Describes this machine, using only the probes that answer.
pub fn detect() -> Hardware {
    let run = |argv: &[&str]| run_command(argv);
    let read = |path: &str| std::fs::read_to_string(path).ok();
    let exists = |path: &str| Path::new(path).exists();
    let uevents = || {
        let mut found = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                if let Ok(text) = std::fs::read_to_string(entry.path().join("device/uevent")) {
                    found.push(text);
                }
            }
        }
        found
    };
    detect_with(
        &Probes {
            run: &run,
            read: &read,
            exists: &exists,
            drm_uevents: &uevents,
        },
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

pub fn detect_with(probes: &Probes<'_>, system: &str, machine: &str) -> Hardware {
    let (gpu_name, gfx) = amd_gpu(probes);
    let mut devices = Vec::new();
    if gfx.is_some() || gpu_name.is_some() {
        devices.push(Device {
            name: gpu_name
                .clone()
                .or_else(|| gfx.clone())
                .unwrap_or_else(|| "AMD GPU".to_string()),
            driver: "amdgpu".to_string(),
            uri: "amdgpu://".to_string(),
            gfx_arch: gfx,
        });
    }
    Hardware {
        system: platform_system(system),
        machine: machine.to_string(),
        cpu: cpu_name(probes),
        total_memory_bytes: memory_total(probes),
        devices,
        npu: (probes.exists)(NPU_DEVICE),
    }
}

/// Python's `platform.system()` spelling.
fn platform_system(os: &str) -> String {
    match os {
        "linux" => "Linux".to_string(),
        "macos" => "Darwin".to_string(),
        "windows" => "Windows".to_string(),
        other => other.to_string(),
    }
}

/// Scores one profile for this machine; missing evidence is not failure.
pub fn evaluate(hardware: &Hardware, profile: &Profile) -> Compatibility {
    let mut notes = Vec::new();
    let mut score: f64 = 0.4;
    let mut fits_memory = None;
    if let Some(memory) = profile.memory_bytes.filter(|m| *m > 0) {
        match hardware.total_memory_bytes.filter(|t| *t > 0) {
            Some(total) => {
                let fits = memory <= (total as f64 * 0.9) as i64;
                fits_memory = Some(fits);
                if fits {
                    score += 0.25;
                    notes.push(format!(
                        "resident {} fits {}",
                        human_bytes(Some(memory)),
                        human_bytes(Some(total))
                    ));
                } else {
                    score -= 0.35;
                    notes.push(format!(
                        "resident {} does not fit {}",
                        human_bytes(Some(memory)),
                        human_bytes(Some(total))
                    ));
                }
            }
            None => notes.push(format!(
                "resident {}; total unknown",
                human_bytes(Some(memory))
            )),
        }
    }
    let mut device_match = None;
    if let Some(hint) = profile.gfx_arch.as_deref().filter(|h| !h.is_empty()) {
        let local = hardware.gfx_arch();
        let matched = Some(hint) == local;
        device_match = Some(matched);
        if matched {
            score += 0.2;
            notes.push(format!("measured on {hint}, matching this machine"));
        } else {
            score -= 0.15;
            notes.push(format!(
                "measured on {hint}, this machine reports {}",
                local.unwrap_or("no AMD GPU")
            ));
        }
    }
    if !profile.gates.is_empty() {
        let mut not_passing: Vec<&str> = profile
            .gates
            .iter()
            .filter(|g| g.status != "pass")
            .map(|g| g.name.as_str())
            .collect();
        not_passing.sort();
        if not_passing.is_empty() {
            score += 0.15;
            notes.push(format!("all {} recorded gates passed", profile.gates.len()));
        } else {
            score -= 0.1;
            notes.push(format!("gates not passing: {}", not_passing.join(", ")));
        }
    }
    Compatibility {
        score: score.clamp(0.0, 1.0),
        fits_memory,
        device_match,
        notes,
    }
}

fn memory_total(probes: &Probes<'_>) -> Option<i64> {
    let text = (probes.read)(MEMINFO)?;
    for line in text.lines() {
        if line.starts_with("MemTotal:") {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 2 && fields[1].chars().all(|c| c.is_ascii_digit()) {
                return fields[1].parse::<i64>().ok().map(|kib| kib * 1024);
            }
        }
    }
    None
}

fn cpu_name(probes: &Probes<'_>) -> Option<String> {
    let text = (probes.read)(CPUINFO)?;
    for line in text.lines() {
        if line.starts_with("model name") {
            let value = line.split_once(':').map(|(_, v)| v.trim()).unwrap_or("");
            return if value.is_empty() {
                None
            } else {
                Some(value.to_string())
            };
        }
    }
    None
}

/// The first GPU agent `rocminfo` reports: its marketing name and gfx target.
///
/// Agents are read as blocks so a CPU agent listed first does not lend the GPU
/// its marketing name; without `rocminfo`, an `amdgpu` DRM node still counts.
fn amd_gpu(probes: &Probes<'_>) -> (Option<String>, Option<String>) {
    let text = (probes.run)(&["rocminfo"]).unwrap_or_default();
    let mut blocks: Vec<Vec<&str>> = vec![Vec::new()];
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Agent ")
            && trimmed[6..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        {
            blocks.push(Vec::new());
        }
        blocks.last_mut().unwrap().push(trimmed);
    }
    for block in &blocks {
        let gfx = block.iter().find_map(|line| gfx_arch(line));
        if let Some(gfx) = gfx {
            let name = block.iter().find_map(|line| {
                line.strip_prefix("Marketing Name:")
                    .map(str::trim)
                    .filter(|n| !n.is_empty())
                    .map(str::to_string)
            });
            return (name, Some(gfx));
        }
    }
    if (probes.drm_uevents)()
        .iter()
        .any(|content| content.contains("amdgpu"))
    {
        return (Some("AMD GPU".to_string()), None);
    }
    (None, None)
}

fn run_command(argv: &[&str]) -> Option<String> {
    let (program, arguments) = argv.split_first()?;
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        use std::io::Read;
        let _ = stdout.read_to_string(&mut text);
        text
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let text = reader.join().ok()?;
                return if status.success() { Some(text) } else { None };
            }
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::tests::{card, config, documents, measurements, profile};
    use crate::profile::Profile;
    use std::collections::HashMap;

    const ROCMINFO: &str =
        "Agent 1\n  Name: gfx1151\n  Marketing Name: AMD Radeon 8060S Graphics\n";

    fn detected(rocminfo: Option<&str>, files: HashMap<&str, &str>) -> Hardware {
        let rocminfo = rocminfo.map(str::to_string);
        let run = move |_: &[&str]| rocminfo.clone();
        let read = |path: &str| files.get(path).map(|s| s.to_string());
        let exists = |_: &str| false;
        let uevents = Vec::new;
        detect_with(
            &Probes {
                run: &run,
                read: &read,
                exists: &exists,
                drm_uevents: &uevents,
            },
            "linux",
            "x86_64",
        )
    }

    fn machine() -> Hardware {
        detected(
            Some(ROCMINFO),
            HashMap::from([
                ("/proc/meminfo", "MemTotal: 131072000 kB\n"),
                ("/proc/cpuinfo", "model name\t: AMD Ryzen\n"),
            ]),
        )
    }

    #[test]
    fn detection_reads_memory_cpu_and_the_gpu_arch() {
        let hardware = machine();
        assert_eq!(hardware.total_memory_bytes, Some(131072000 * 1024));
        assert_eq!(hardware.cpu.as_deref(), Some("AMD Ryzen"));
        assert_eq!(hardware.gfx_arch(), Some("gfx1151"));
        assert_eq!(hardware.gpu().unwrap().name, "AMD Radeon 8060S Graphics");
        assert_eq!(hardware.default_device_uri(), "amdgpu://");
        assert!(!hardware.npu);
        assert_eq!(hardware.system, "Linux");
    }

    #[test]
    fn a_cpu_agent_listed_first_does_not_name_the_gpu() {
        let text =
            "Agent 1\n  Name: AMD RYZEN AI MAX+ 395\n  Marketing Name: AMD RYZEN AI MAX+ 395\n\
                    Agent 2\n  Name: gfx1151\n  Marketing Name: AMD Radeon 8060S Graphics\n";
        let hardware = detected(Some(text), HashMap::new());
        assert_eq!(hardware.gpu().unwrap().name, "AMD Radeon 8060S Graphics");
        let cpu_only = "Agent 1\n  Name: AMD EPYC\n  Marketing Name: AMD EPYC\n";
        assert_eq!(
            detected(Some(cpu_only), HashMap::new()).default_device_uri(),
            "task"
        );
    }

    #[test]
    fn a_cpu_only_machine_defaults_to_the_task_device() {
        let hardware = detected(
            None,
            HashMap::from([("/proc/meminfo", "MemTotal: 1000 kB\n")]),
        );
        assert!(hardware.devices.is_empty());
        assert_eq!(hardware.default_device_uri(), "task");
    }

    #[test]
    fn compatibility_rewards_fit_and_a_matching_device() {
        let hardware = machine();
        let good = evaluate(&hardware, &profile());
        assert_eq!(good.fits_memory, Some(true));
        assert_eq!(good.device_match, Some(true));
        assert!(good.score >= 0.7, "{}", good.score);
        assert!(good.notes.join(" ").contains("gates not passing: vision"));
        let mut info = measurements();
        info["card"] = json!({"memory_bytes": 200000000000_i64});
        let large = Profile::from_documents(
            "loom-fun/qwen3-4b-loom",
            &documents(card(None), info, config()),
            None,
        );
        let bad = evaluate(&hardware, &large);
        assert_eq!(bad.fits_memory, Some(false));
        assert!(bad.score < good.score);
        assert!(bad.notes.join(" ").contains("does not fit"));
    }
}
