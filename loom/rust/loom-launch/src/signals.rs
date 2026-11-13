//! SIGINT/SIGTERM handling and process signalling.
//!
//! The handler only records the signal and wakes a self-pipe; supervision code
//! polls [`stop_requested`] or registers a watcher with [`on_stop`] that runs in
//! ordinary thread context.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Once;

static STOP: AtomicBool = AtomicBool::new(false);
static SIGNAL: AtomicI32 = AtomicI32::new(0);
static PIPE_WRITE: AtomicI32 = AtomicI32::new(-1);
static INSTALL: Once = Once::new();

extern "C" fn on_signal(signal: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
    SIGNAL.store(signal, Ordering::SeqCst);
    let fd = PIPE_WRITE.load(Ordering::SeqCst);
    if fd >= 0 {
        let byte = 1u8;
        // SAFETY: write(2) is async-signal-safe; the buffer outlives the call.
        unsafe {
            libc::write(fd, &byte as *const u8 as *const libc::c_void, 1);
        }
    }
}

/// Routes SIGINT and SIGTERM to the stop flag (idempotent).
pub fn install() {
    INSTALL.call_once(|| {
        // SAFETY: a zeroed sigaction with an extern "C" handler is valid input.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_signal as *const () as usize;
            action.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(libc::SIGINT, &action, std::ptr::null_mut());
            libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut());
        }
    });
}

/// Ignores SIGHUP: a detached launcher outlives the terminal it came from.
pub fn ignore_hangup() {
    // SAFETY: setting a signal's disposition to SIG_IGN has no preconditions.
    unsafe {
        libc::signal(libc::SIGHUP, libc::SIG_IGN);
    }
}

/// True once SIGINT or SIGTERM arrived after [`install`].
pub fn stop_requested() -> bool {
    STOP.load(Ordering::SeqCst)
}

/// The signal that requested the stop, or zero.
pub fn received() -> i32 {
    SIGNAL.load(Ordering::SeqCst)
}

/// Installs the handler and runs `action` on a watcher thread when a signal
/// arrives. Only the first registration is honoured.
pub fn on_stop(action: impl FnOnce() + Send + 'static) {
    install();
    if PIPE_WRITE.load(Ordering::SeqCst) >= 0 {
        return;
    }
    let mut fds = [0 as libc::c_int; 2];
    // SAFETY: fds has room for the two descriptors pipe(2) returns.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return;
    }
    PIPE_WRITE.store(fds[1], Ordering::SeqCst);
    let read_end = fds[0];
    std::thread::spawn(move || {
        let mut byte = 0u8;
        loop {
            // SAFETY: reading one byte into a live local buffer.
            let count =
                unsafe { libc::read(read_end, &mut byte as *mut u8 as *mut libc::c_void, 1) };
            if count == 1
                || (count < 0
                    && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted)
            {
                break;
            }
        }
        action();
    });
    if stop_requested() {
        on_signal(received());
    }
}

/// Sends `signal` to `pid`; true when delivered.
pub fn send(pid: i32, signal: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: kill(2) has no memory-safety preconditions.
    unsafe { libc::kill(pid, signal) == 0 }
}

/// True while a process with this pid exists.
pub fn alive(pid: i64) -> bool {
    if pid <= 0 || pid > i32::MAX as i64 {
        return false;
    }
    // SAFETY: signal 0 only checks for existence and permission.
    let result = unsafe { libc::kill(pid as i32, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// True when `pid` is a zombie or gone (a reaped-later child still "exists").
pub fn exited(pid: i64) -> bool {
    if !alive(pid) {
        return true;
    }
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(text) => text
            .rsplit_once(')')
            .map(|(_, rest)| rest.trim_start().starts_with('Z'))
            .unwrap_or(false),
        Err(_) => false,
    }
}

pub const SIGINT: i32 = libc::SIGINT;
pub const SIGTERM: i32 = libc::SIGTERM;
pub const SIGKILL: i32 = libc::SIGKILL;
