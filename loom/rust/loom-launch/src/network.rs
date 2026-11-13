//! Listen addresses: which hosts are loopback, and the URLs a listener on a
//! wider address can be reached at from the local network.

use std::net::{IpAddr, Ipv4Addr};

/// True for `127.0.0.0/8`, `::1` and `localhost`, the addresses only this
/// machine reaches.
pub fn is_loopback_host(host: &str) -> bool {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if bare.eq_ignore_ascii_case("localhost") {
        return true;
    }
    bare.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// True for `0.0.0.0` and `::`, which listen on every interface.
pub fn is_unspecified_host(host: &str) -> bool {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    bare.parse::<IpAddr>().is_ok_and(|ip| ip.is_unspecified())
}

/// `http://host:port` with IPv6 hosts bracketed.
pub fn http_url(host: &str, port: u16) -> String {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if bare.contains(':') {
        format!("http://[{bare}]:{port}")
    } else {
        format!("http://{bare}:{port}")
    }
}

/// The IPv4 addresses of this machine's interfaces that other machines can
/// reach: everything that is up except loopback and link-local.
pub fn interface_ipv4_addresses() -> Vec<Ipv4Addr> {
    let mut found = Vec::new();
    let mut list: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: getifaddrs(3) fills `list` with a linked list we only read and
    // release with freeifaddrs(3) below.
    if unsafe { libc::getifaddrs(&mut list) } != 0 {
        return found;
    }
    let mut cursor = list;
    while !cursor.is_null() {
        // SAFETY: `cursor` is a node of the list getifaddrs returned.
        let entry = unsafe { &*cursor };
        cursor = entry.ifa_next;
        let up = entry.ifa_flags & (libc::IFF_UP as libc::c_uint) != 0;
        if !up || entry.ifa_addr.is_null() {
            continue;
        }
        // SAFETY: ifa_addr is non-null and points at a sockaddr whose family
        // field says how to read the rest.
        let family = unsafe { (*entry.ifa_addr).sa_family } as libc::c_int;
        if family != libc::AF_INET {
            continue;
        }
        // SAFETY: an AF_INET address is a sockaddr_in.
        let inet = unsafe { &*(entry.ifa_addr as *const libc::sockaddr_in) };
        let address = Ipv4Addr::from(u32::from_be(inet.sin_addr.s_addr));
        if address.is_loopback() || address.is_link_local() || address.is_unspecified() {
            continue;
        }
        if !found.contains(&address) {
            found.push(address);
        }
    }
    // SAFETY: `list` came from getifaddrs and is released exactly once.
    unsafe { libc::freeifaddrs(list) };
    found
}

/// The URLs (`http://ip:port/`) a listener bound to `bind_host` answers on
/// from other machines: every interface address for `0.0.0.0` or `::`, the
/// address itself for a specific one, nothing for loopback.
pub fn reachable_urls(bind_host: &str, port: u16) -> Vec<String> {
    if is_loopback_host(bind_host) {
        return Vec::new();
    }
    if is_unspecified_host(bind_host) {
        return interface_ipv4_addresses()
            .into_iter()
            .map(|ip| format!("{}/", http_url(&ip.to_string(), port)))
            .collect();
    }
    vec![format!("{}/", http_url(bind_host, port))]
}

/// The address a local client connects to for a listener on `bind_host`:
/// loopback stands in for "every interface".
pub fn local_connect_host(bind_host: &str) -> String {
    let bare = bind_host.trim_start_matches('[').trim_end_matches(']');
    match bare.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) if ip.is_unspecified() => "127.0.0.1".into(),
        Ok(IpAddr::V6(ip)) if ip.is_unspecified() => "::1".into(),
        _ => bare.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hosts_are_classified() {
        for host in [
            "127.0.0.1",
            "127.3.4.5",
            "::1",
            "[::1]",
            "localhost",
            "LOCALHOST",
        ] {
            assert!(is_loopback_host(host), "{host}");
        }
        for host in ["0.0.0.0", "192.168.0.200", "::", "example.com"] {
            assert!(!is_loopback_host(host), "{host}");
        }
        assert!(is_unspecified_host("0.0.0.0") && is_unspecified_host("[::]"));
        assert!(!is_unspecified_host("192.168.0.200"));
        assert_eq!(local_connect_host("0.0.0.0"), "127.0.0.1");
        assert_eq!(local_connect_host("::"), "::1");
        assert_eq!(local_connect_host("192.168.0.200"), "192.168.0.200");
        assert_eq!(http_url("::1", 80), "http://[::1]:80");
        assert_eq!(http_url("10.0.0.2", 8099), "http://10.0.0.2:8099");
    }

    #[test]
    fn reachable_urls_follow_the_bind_address() {
        assert!(reachable_urls("127.0.0.1", 8099).is_empty());
        assert_eq!(
            reachable_urls("192.168.0.200", 8099),
            ["http://192.168.0.200:8099/"]
        );
        for url in reachable_urls("0.0.0.0", 8099) {
            assert!(
                url.starts_with("http://") && url.ends_with(":8099/"),
                "{url}"
            );
            assert!(!url.contains("127.0.0.1"), "{url}");
        }
        for address in interface_ipv4_addresses() {
            assert!(!address.is_loopback() && !address.is_link_local());
        }
    }
}
