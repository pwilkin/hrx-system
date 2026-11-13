//! Launcher failures that carry an operator-actionable message.

use std::fmt;

/// What kind of operation failed; the CLI maps every kind to exit code 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A generic launcher refusal.
    Launcher,
    /// A published profile is missing or malformed.
    Profile,
    /// A profile could not be installed locally.
    Install,
    /// A server could not be started or reached.
    Launch,
    /// A benchmark run or its record could not be completed.
    Benchmark,
    /// The hub could not be contacted at all.
    HubUnreachable,
    /// The hub requires credentials for this repository.
    HubAuth,
    /// The hub reports the resource as absent (HTTP 404).
    HubNotFound,
    /// Any other hub failure (bad status, malformed document).
    Hub,
}

/// A launcher failure: a kind plus the message shown to the operator.
#[derive(Clone, Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
        }
    }

    pub fn launcher(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Launcher, message)
    }

    pub fn profile(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Profile, message)
    }

    pub fn install(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Install, message)
    }

    pub fn launch(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Launch, message)
    }

    pub fn benchmark(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Benchmark, message)
    }

    pub fn hub(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Hub, message)
    }

    /// True for hub failures (network, status, auth) as opposed to launcher refusals.
    pub fn is_hub(&self) -> bool {
        matches!(
            self.kind,
            ErrorKind::HubUnreachable
                | ErrorKind::HubAuth
                | ErrorKind::HubNotFound
                | ErrorKind::Hub
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
