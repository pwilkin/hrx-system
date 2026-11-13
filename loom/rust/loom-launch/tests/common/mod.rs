//! Support shared by the integration tests.

use std::io::Error;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::sync::Mutex;

/// The sockets holding the ports [`free_port`] handed out.
static HELD: Mutex<Vec<OwnedFd>> = Mutex::new(Vec::new());

/// A loopback port nothing else is given while this test process runs.
///
/// The port stays bound, with `SO_REUSEADDR` and without listening, until
/// the process exits. Linux never hands such a port to `bind(0)` or to an
/// outgoing connection, in this process or any other, while a server that
/// binds it with `SO_REUSEADDR` (as `std` and `tiny_http` do) still can.
pub fn free_port() -> u16 {
    // SAFETY: socket(2) has no memory-safety preconditions; the descriptor
    // it returns is owned by `socket` from here on.
    let socket = unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0);
        assert!(fd >= 0, "socket: {}", Error::last_os_error());
        OwnedFd::from_raw_fd(fd)
    };
    let fd = socket.as_raw_fd();
    let one: libc::c_int = 1;
    // SAFETY: `one` outlives the call and its size is passed with it.
    let set = unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_REUSEADDR,
            (&one as *const libc::c_int).cast(),
            size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    assert_eq!(set, 0, "SO_REUSEADDR: {}", Error::last_os_error());
    // SAFETY: an all-zero sockaddr_in is valid (port 0: any free port).
    let mut address: libc::sockaddr_in = unsafe { std::mem::zeroed() };
    address.sin_family = libc::AF_INET as libc::sa_family_t;
    address.sin_addr.s_addr = u32::from(std::net::Ipv4Addr::LOCALHOST).to_be();
    let mut length = size_of::<libc::sockaddr_in>() as libc::socklen_t;
    // SAFETY: `address` is a sockaddr_in of `length` bytes for both calls.
    unsafe {
        let pointer = (&mut address as *mut libc::sockaddr_in).cast::<libc::sockaddr>();
        assert_eq!(
            libc::bind(fd, pointer, length),
            0,
            "bind: {}",
            Error::last_os_error()
        );
        assert_eq!(
            libc::getsockname(fd, pointer, &mut length),
            0,
            "getsockname: {}",
            Error::last_os_error()
        );
    }
    HELD.lock().unwrap().push(socket);
    u16::from_be(address.sin_port)
}
