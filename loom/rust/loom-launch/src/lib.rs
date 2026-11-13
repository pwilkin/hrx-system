//! Native launcher for published Loom model profiles.
//!
//! A profile is one published Loom package: a portable model card, its
//! parameter and program payloads, the chat policy its generator emitted, and
//! the measurements its publisher recorded. This crate discovers profiles on a
//! model hub, installs them through `loom-fetch`, composes a deployment document
//! for the local machine, and runs a launcher service that keeps one
//! instrumented OpenAI-compatible endpoint (and its dashboard) up while
//! `loom-serve` models are loaded and unloaded under it. It shares its state directory format
//! with the Python launcher (`loom/py/loom/launch`).

pub mod bench;
pub mod chat;
pub mod cli;
pub mod client;
pub mod dashboard;
pub mod deployment;
pub mod errors;
pub mod hardware;
pub mod hub;
pub mod install;
pub mod launcher;
pub mod local;
pub mod metrics;
pub mod network;
pub mod profile;
pub mod proxy;
pub mod scenario;
pub mod server;
pub mod signals;
pub mod store;
pub mod testing;
pub mod values;
pub mod workload;
