//! `POST /api/fetch {url}`: the fetch proxy behind the chat's `web_fetch` tool.
//!
//! This route is deliberate: it lets the model, through the page, have the
//! launcher fetch one URL and read it back as text. The launcher then makes
//! requests on the model's behalf from inside the user's network, so the route
//! is off unless the launcher runs with `--allow-web`, and the address checks
//! here are what keep it from reaching the machine's own services (this
//! dashboard, `loom-serve`, anything listening on loopback or the LAN):
//!
//! - only `http` and `https` URLs without credentials;
//! - every address the host resolves to must be public: loopback, private,
//!   shared (CGNAT), link-local, unique-local, multicast, unspecified,
//!   documentation and other reserved ranges are refused, for IPv4, IPv6 and
//!   IPv4 embedded in IPv6;
//! - the connection goes to exactly the addresses that were checked (the HTTP
//!   client's resolver is pinned to them), so a second DNS answer cannot swap
//!   in a private address;
//! - redirects are not followed: the answer carries the `Location` so the
//!   model can ask for it, and that request is checked again;
//! - a 10 s budget, a 1 MiB response cap and an 8 KiB text cap.

use super::{error_reply, Reply};
use serde_json::{json, Value};
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs};
use std::time::Duration;

/// Whole-request budget: resolution aside, connect, headers and body.
pub(crate) const FETCH_TIMEOUT: Duration = Duration::from_secs(10);
/// Bytes read from the response body at most.
pub(crate) const BODY_CAP: usize = 1024 * 1024;
/// Bytes of text handed back to the model at most.
pub(crate) const TEXT_CAP: usize = 8 * 1024;
const URL_CAP: usize = 2048;
const USER_AGENT: &str = concat!(
    "Mozilla/5.0 (X11; Linux x86_64) loom-launch/",
    env!("CARGO_PKG_VERSION"),
    " (web_fetch for a local model)"
);
const ACCEPT: &str = "text/html,text/plain,application/json";

/// Resolves and checks host names for one fetch.
#[derive(Clone, Debug, Default)]
pub(crate) struct Fetcher {
    /// Test builds only: host names answered with a local stub's address and
    /// treated as public, so a test can stand in for the internet. Production
    /// builds have no such table, so the address check always applies.
    #[cfg(test)]
    pub(crate) stubs: Vec<(String, SocketAddr)>,
}

/// A refusal or failure, as the route answers it.
#[derive(Debug)]
pub(crate) struct Failure {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl Failure {
    fn new(status: u16, code: &'static str, message: impl Into<String>) -> Failure {
        Failure {
            status,
            code,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Failure {
        Failure::new(400, "INVALID_ARGUMENT", message)
    }

    fn refused(message: impl Into<String>) -> Failure {
        Failure::new(403, "PERMISSION_DENIED", message)
    }

    fn unavailable(message: impl Into<String>) -> Failure {
        Failure::new(502, "UNAVAILABLE", message)
    }

    pub(crate) fn reply(&self) -> Reply {
        error_reply(self.status, self.code, &self.message)
    }
}

/// `/api/fetch`: the document for one URL, or why it was refused.
pub(crate) fn fetch_document(
    fetcher: &Fetcher,
    url: Option<&str>,
) -> std::result::Result<Value, Failure> {
    let Some(url) = url.map(str::trim).filter(|u| !u.is_empty()) else {
        return Err(Failure::invalid(
            "url is required (an absolute http:// or https:// URL)",
        ));
    };
    fetcher.fetch(url)
}

/// Why an address may not be fetched, or `None` for a public one.
pub(crate) fn refused_kind(ip: IpAddr) -> Option<&'static str> {
    match ip {
        IpAddr::V4(ip) => refused_v4(ip),
        IpAddr::V6(ip) => refused_v6(ip),
    }
}

fn refused_v4(ip: Ipv4Addr) -> Option<&'static str> {
    let [a, b, c, _] = ip.octets();
    if ip.is_unspecified() || a == 0 {
        Some("unspecified")
    } else if ip.is_loopback() {
        Some("loopback")
    } else if ip.is_private() {
        Some("private")
    } else if a == 100 && (64..128).contains(&b) {
        Some("private (shared address space)")
    } else if ip.is_link_local() {
        Some("link-local")
    } else if ip.is_multicast() {
        Some("multicast")
    } else if ip.is_broadcast() || a >= 240 {
        Some("reserved")
    } else if (a, b, c) == (192, 0, 2) || (a, b, c) == (198, 51, 100) || (a, b, c) == (203, 0, 113)
    {
        Some("documentation")
    } else if (a, b, c) == (192, 0, 0) || (a == 198 && (b == 18 || b == 19)) {
        Some("reserved")
    } else {
        None
    }
}

fn refused_v6(ip: Ipv6Addr) -> Option<&'static str> {
    let segments = ip.segments();
    if ip.is_unspecified() {
        return Some("unspecified");
    }
    if ip.is_loopback() {
        return Some("loopback");
    }
    // IPv4-mapped (::ffff:a.b.c.d) addresses reach the embedded IPv4 address;
    // the deprecated IPv4-compatible form (::a.b.c.d) is refused outright.
    if let Some(v4) = ip.to_ipv4_mapped() {
        return refused_v4(v4);
    }
    if let Some(v4) = ip.to_ipv4() {
        return refused_v4(v4).or(Some("reserved (IPv4-compatible)"));
    }
    let embedded = |high: u16, low: u16| {
        Ipv4Addr::new((high >> 8) as u8, high as u8, (low >> 8) as u8, low as u8)
    };
    match segments[0] {
        // NAT64 (64:ff9b::/96) carries an IPv4 address in its last 32 bits.
        0x0064 if segments[1] == 0xff9b => refused_v4(embedded(segments[6], segments[7])),
        // 6to4 (2002::/16) carries one in bits 16..48.
        0x2002 => refused_v4(embedded(segments[1], segments[2])),
        0x2001 if segments[1] == 0x0db8 => Some("documentation"),
        // Teredo tunnels and the other IETF assignments in 2001::/23.
        0x2001 if segments[1] < 0x0200 => Some("reserved"),
        0x0100 if segments[1..4] == [0, 0, 0] => Some("reserved"),
        first if first & 0xfe00 == 0xfc00 => Some("unique-local"),
        first if first & 0xffc0 == 0xfe80 => Some("link-local"),
        first if first & 0xffc0 == 0xfec0 => Some("site-local"),
        first if first & 0xff00 == 0xff00 => Some("multicast"),
        _ => None,
    }
}

/// `host:port` as the HTTP client names it when it resolves a URL.
fn netloc(url: &url::Url) -> Option<String> {
    Some(format!(
        "{}:{}",
        url.host_str()?,
        url.port_or_known_default()?
    ))
}

impl Fetcher {
    /// The checked URL and the addresses it may connect to.
    fn admit(&self, text: &str) -> Result<(url::Url, Vec<SocketAddr>), Failure> {
        if text.len() > URL_CAP {
            return Err(Failure::invalid(format!(
                "the URL is longer than {URL_CAP} characters"
            )));
        }
        let mut url = url::Url::parse(text).map_err(|error| {
            Failure::invalid(format!(
                "{text} is not an absolute URL ({error}); give one starting with https:// or http://"
            ))
        })?;
        let scheme = url.scheme().to_string();
        if scheme != "http" && scheme != "https" {
            return Err(Failure::refused(format!(
                "only http and https URLs can be fetched, not {scheme}:"
            )));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(Failure::refused(
                "URLs carrying credentials are not fetched",
            ));
        }
        url.set_fragment(None);
        let port = url
            .port_or_known_default()
            .ok_or_else(|| Failure::invalid("the URL has no port"))?;
        let host = url
            .host()
            .ok_or_else(|| Failure::invalid("the URL has no host"))?
            .to_owned();
        let addresses: Vec<SocketAddr> = match &host {
            url::Host::Ipv4(ip) => vec![SocketAddr::new(IpAddr::V4(*ip), port)],
            url::Host::Ipv6(ip) => vec![SocketAddr::new(IpAddr::V6(*ip), port)],
            url::Host::Domain(name) => {
                let name = name.trim_end_matches('.').to_ascii_lowercase();
                if name == "localhost" || name.ends_with(".localhost") {
                    return Err(Failure::refused(format!(
                        "{name} is this machine; only public addresses can be fetched"
                    )));
                }
                #[cfg(test)]
                if let Some((_, stub)) = self.stubs.iter().find(|(stub, _)| *stub == name) {
                    return Ok((url, vec![*stub]));
                }
                let found: Vec<SocketAddr> = (name.as_str(), port)
                    .to_socket_addrs()
                    .map_err(|error| {
                        Failure::unavailable(format!("cannot resolve {name}: {error}"))
                    })?
                    .collect();
                if found.is_empty() {
                    return Err(Failure::unavailable(format!("{name} has no addresses")));
                }
                found
            }
        };
        for address in &addresses {
            if let Some(kind) = refused_kind(address.ip()) {
                let ip = address.ip();
                let what = match &host {
                    url::Host::Domain(name) => {
                        format!("{name} resolves to {ip}, {}", article(kind))
                    }
                    _ => format!("{ip} is {}", article(kind)),
                };
                return Err(Failure::refused(format!(
                    "{what} address; only public addresses can be fetched"
                )));
            }
        }
        Ok((url, addresses))
    }

    /// Fetches one URL and reduces it to text (see the module comment).
    pub(crate) fn fetch(&self, text: &str) -> Result<Value, Failure> {
        let (url, addresses) = self.admit(text)?;
        let pinned = netloc(&url).ok_or_else(|| Failure::invalid("the URL has no host"))?;
        let resolver = move |asked: &str| -> std::io::Result<Vec<SocketAddr>> {
            if asked.eq_ignore_ascii_case(&pinned) {
                Ok(addresses.clone())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!("{asked} was not checked"),
                ))
            }
        };
        let agent = ureq::AgentBuilder::new()
            .timeout(FETCH_TIMEOUT)
            .redirects(0)
            .try_proxy_from_env(false)
            .max_idle_connections(0)
            .user_agent(USER_AGENT)
            .resolver(resolver)
            .build();
        let response = match agent.request_url("GET", &url).set("Accept", ACCEPT).call() {
            Ok(response) | Err(ureq::Error::Status(_, response)) => response,
            Err(ureq::Error::Transport(error)) => {
                return Err(Failure::unavailable(format!(
                    "could not fetch {url}: {error}"
                )))
            }
        };
        let status = response.status();
        let content_type = response.header("Content-Type").unwrap_or("").to_string();
        let location = response
            .header("Location")
            .map(|l| url.join(l).map(|u| u.to_string()).unwrap_or(l.to_string()));
        let mut body = Vec::new();
        let read = response
            .into_reader()
            .take(BODY_CAP as u64 + 1)
            .read_to_end(&mut body);
        if let Err(error) = read {
            if body.is_empty() {
                return Err(Failure::unavailable(format!(
                    "could not read {url}: {error}"
                )));
            }
        }
        let body_truncated = body.len() > BODY_CAP;
        body.truncate(BODY_CAP);
        let (title, text) = readable(&content_type, &body);
        let (text, text_truncated) = cap(text, TEXT_CAP);
        let mut document = json!({
            "url": url.as_str(),
            "status": status,
            "content_type": content_type,
            "bytes": body.len(),
            "truncated": body_truncated || text_truncated,
            "text": text,
        });
        if let Some(title) = title.filter(|t| !t.is_empty()) {
            document["title"] = json!(title);
        }
        if let Some(location) = location.filter(|_| (300..400).contains(&status)) {
            document["location"] = json!(location);
            if document["text"].as_str().is_some_and(str::is_empty) {
                document["text"] = json!(format!(
                    "Redirected to {location}; fetch that URL to follow the redirect."
                ));
            }
        }
        Ok(document)
    }
}

fn article(kind: &str) -> String {
    let vowel = kind.starts_with(['a', 'e', 'i', 'o', 'u', 'I']);
    format!("{} {kind}", if vowel { "an" } else { "a" })
}

/// `text` cut to at most `limit` bytes on a character boundary.
fn cap(mut text: String, limit: usize) -> (String, bool) {
    if text.len() <= limit {
        return (text, false);
    }
    let mut end = limit;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    (text, true)
}

/// The body as text: HTML reduced to its readable content, other text kinds
/// decoded, binary kinds described instead of shown.
fn readable(content_type: &str, body: &[u8]) -> (Option<String>, String) {
    let lowered = content_type.to_ascii_lowercase();
    let mime = lowered.split(';').next().unwrap_or("").trim();
    let charset = lowered
        .split(';')
        .filter_map(|p| p.trim().strip_prefix("charset="))
        .next()
        .map(|c| c.trim_matches('"').to_string());
    let latin = charset.as_deref().is_some_and(|c| {
        matches!(
            c,
            "iso-8859-1" | "latin1" | "latin-1" | "windows-1252" | "cp1252" | "us-ascii"
        )
    });
    let text = if latin {
        body.iter().map(|&b| b as char).collect()
    } else {
        String::from_utf8_lossy(body).into_owned()
    };
    let sniffed = {
        let head = text
            .trim_start()
            .get(..15)
            .unwrap_or("")
            .to_ascii_lowercase();
        head.starts_with("<!doctype html") || head.starts_with("<html")
    };
    let textual = mime.is_empty()
        || mime.starts_with("text/")
        || mime.contains("json")
        || mime.contains("xml")
        || mime.contains("javascript");
    if mime.contains("html") || (sniffed && !mime.contains("json")) {
        let (title, text) = html_to_text(&text);
        return (title, text);
    }
    if !textual {
        return (
            None,
            format!(
                "(binary content of type {mime}, {} bytes, not shown)",
                body.len()
            ),
        );
    }
    (None, collapse_whitespace(&text, true))
}

/// Elements whose content is never readable text.
const SKIPPED: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "iframe", "object", "canvas", "head", "nav",
];
/// Elements that start a new line.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "br",
    "caption",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "html",
    "li",
    "main",
    "nav",
    "ol",
    "option",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tbody",
    "tfoot",
    "thead",
    "tr",
    "ul",
];

/// Index just past the `>` closing the tag that starts at `start`.
fn tag_end(bytes: &[u8], start: usize) -> usize {
    let mut quote = None;
    let mut index = start + 1;
    while index < bytes.len() {
        let byte = bytes[index];
        match quote {
            Some(q) if byte == q => quote = None,
            Some(_) => {}
            None if byte == b'"' || byte == b'\'' => quote = Some(byte),
            None if byte == b'>' => return index + 1,
            None => {}
        }
        index += 1;
    }
    bytes.len()
}

/// HTML reduced to readable text: scripts, styles, navigation and markup
/// removed, the usual entities decoded, block elements on their own lines,
/// whitespace collapsed. A page that marks its content with `<main>` (else
/// `<article>`) is reduced to that, so the 8 KiB the model sees are not spent
/// on menus. Returns the `<title>` separately.
pub(crate) fn html_to_text(html: &str) -> (Option<String>, String) {
    let (title, whole) = reduce_html(html);
    let lower = html.to_ascii_lowercase();
    for name in ["main", "article"] {
        if let Some(range) = element_span(&lower, name) {
            let (_, content) = reduce_html(&html[range]);
            if !content.is_empty() {
                return (title, content);
            }
        }
    }
    (title, whole)
}

/// From the first `<name ...>` to the end of the last `</name>`.
fn element_span(lower: &str, name: &str) -> Option<std::ops::Range<usize>> {
    let open = format!("<{name}");
    let start = lower.match_indices(&open).map(|(i, _)| i).find(|&i| {
        lower[i + open.len()..]
            .chars()
            .next()
            .is_some_and(|c| c == '>' || c.is_ascii_whitespace())
    })?;
    let close = format!("</{name}");
    let end = lower.rfind(&close).filter(|&e| e > start)?;
    let after = lower[end..]
        .find('>')
        .map(|p| end + p + 1)
        .unwrap_or(lower.len());
    Some(start..after)
}

fn reduce_html(html: &str) -> (Option<String>, String) {
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    let mut out = String::with_capacity(html.len() / 2);
    let mut title = None;
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'<' {
            let next = html[index..]
                .find('<')
                .map(|p| index + p)
                .unwrap_or(bytes.len());
            // Line breaks in HTML source are spaces; only elements break lines.
            out.extend(decode_entities(&html[index..next]).chars().map(|c| {
                if c.is_whitespace() {
                    ' '
                } else {
                    c
                }
            }));
            index = next;
            continue;
        }
        let rest = &lower[index + 1..];
        if let Some(comment) = rest.strip_prefix("!--") {
            index = comment
                .find("-->")
                .map(|p| index + 4 + p + 3)
                .unwrap_or(bytes.len());
            continue;
        }
        if rest.starts_with('!') || rest.starts_with('?') {
            index = tag_end(bytes, index);
            continue;
        }
        let closing = rest.starts_with('/');
        let name_start = index + 1 + usize::from(closing);
        let name_length = lower[name_start..]
            .bytes()
            .take_while(u8::is_ascii_alphanumeric)
            .count();
        if name_length == 0 {
            // A bare "<" in text, as in "a < b".
            out.push('<');
            index += 1;
            continue;
        }
        let name = &lower[name_start..name_start + name_length];
        let end = tag_end(bytes, index);
        if !closing && (SKIPPED.contains(&name) || name == "title") {
            let close = format!("</{name}");
            let (inner_end, after) = match lower[end..].find(&close) {
                Some(p) => (end + p, tag_end(bytes, end + p)),
                None => (bytes.len(), bytes.len()),
            };
            if name == "title" && title.is_none() {
                let text = collapse_whitespace(&decode_entities(&html[end..inner_end]), false);
                title = Some(text);
            } else if name == "head" {
                // A <title> inside <head> is still the title.
                if let Some(p) = lower[end..inner_end].find("<title") {
                    let open = end + p;
                    let start = tag_end(bytes, open);
                    let stop = lower[start..inner_end]
                        .find("</title")
                        .map(|q| start + q)
                        .unwrap_or(inner_end);
                    title = Some(collapse_whitespace(
                        &decode_entities(&html[start..stop]),
                        false,
                    ));
                }
            }
            index = after;
            continue;
        }
        if BLOCKS.contains(&name) {
            out.push('\n');
            if name == "li" && !closing {
                out.push_str("- ");
            }
        } else if name == "td" || name == "th" {
            out.push(' ');
        }
        index = end;
    }
    (title, collapse_whitespace(&out, false))
}

fn named_entity(name: &str) -> Option<char> {
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "ndash" => '–',
        "mdash" => '—',
        "hellip" => '…',
        "lsquo" => '‘',
        "rsquo" => '’',
        "ldquo" => '“',
        "rdquo" => '”',
        "laquo" => '«',
        "raquo" => '»',
        "middot" => '·',
        "bull" => '•',
        "copy" => '©',
        "reg" => '®',
        "trade" => '™',
        "deg" => '°',
        "times" => '×',
        "divide" => '÷',
        "plusmn" => '±',
        "euro" => '€',
        "pound" => '£',
        "yen" => '¥',
        "cent" => '¢',
        "sect" => '§',
        "para" => '¶',
        "shy" => '\u{ad}',
        "zwj" | "zwnj" => '\u{200b}',
        _ => return None,
    })
}

/// Decodes `&name;`, `&#NNN;` and `&#xHH;`; anything else stays as written.
pub(crate) fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find('&') {
        out.push_str(&rest[..position]);
        let tail = &rest[position..];
        let end = tail[1..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '#'))
            .map(|p| p + 1);
        let decoded = end
            .filter(|&e| tail[e..].starts_with(';') && e > 1)
            .and_then(|e| {
                let name = &tail[1..e];
                let value = if let Some(number) = name.strip_prefix('#') {
                    let code = match number.strip_prefix(['x', 'X']) {
                        Some(hex) => u32::from_str_radix(hex, 16).ok(),
                        None => number.parse::<u32>().ok(),
                    };
                    code.map(|c| {
                        char::from_u32(c)
                            .filter(|c| *c != '\0')
                            .unwrap_or('\u{fffd}')
                    })
                } else {
                    named_entity(name)
                };
                value.map(|v| (v, e + 1))
            });
        match decoded {
            Some((character, consumed)) => {
                out.push(character);
                rest = &tail[consumed..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Runs of spaces and tabs become one space and lines are trimmed; blank lines
/// are dropped, or with `keep_blank` at most one separates paragraphs.
pub(crate) fn collapse_whitespace(text: &str, keep_blank: bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = false;
    for line in text.lines() {
        let words: Vec<&str> = line
            .split(|c: char| c.is_whitespace() || c == '\u{200b}' || c == '\u{ad}')
            .filter(|w| !w.is_empty())
            .collect();
        if words.is_empty() {
            blank = keep_blank && !out.is_empty();
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
            if blank {
                out.push('\n');
            }
        }
        blank = false;
        out.push_str(&words.join(" "));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(text: &str) -> Option<&'static str> {
        refused_kind(text.parse().unwrap())
    }

    #[test]
    fn private_and_special_addresses_are_refused() {
        for (address, expected) in [
            ("127.0.0.1", "loopback"),
            ("127.9.9.9", "loopback"),
            ("0.0.0.0", "unspecified"),
            ("0.1.2.3", "unspecified"),
            ("10.1.2.3", "private"),
            ("172.16.0.1", "private"),
            ("172.31.255.255", "private"),
            ("192.168.0.200", "private"),
            ("100.64.0.1", "private (shared address space)"),
            ("169.254.169.254", "link-local"),
            ("224.0.0.1", "multicast"),
            ("255.255.255.255", "reserved"),
            ("240.0.0.1", "reserved"),
            ("192.0.2.10", "documentation"),
            ("::", "unspecified"),
            ("::1", "loopback"),
            ("fe80::1", "link-local"),
            ("fc00::1", "unique-local"),
            ("fd12:3456::1", "unique-local"),
            ("ff02::1", "multicast"),
            ("::ffff:127.0.0.1", "loopback"),
            ("::ffff:192.168.1.1", "private"),
            ("::7f00:1", "loopback"),
            ("64:ff9b::a00:1", "private"),
            ("2002:c0a8:0001::1", "private"),
            ("2001:db8::1", "documentation"),
            ("2001:0:4136:e378::1", "reserved"),
        ] {
            assert_eq!(kind(address), Some(expected), "{address}");
        }
        for public in [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",
            "172.32.0.1",
            "100.128.0.1",
            "2606:4700:4700::1111",
            "2a00:1450:4001::200e",
            "::ffff:8.8.8.8",
            "64:ff9b::808:808",
        ] {
            assert_eq!(kind(public), None, "{public}");
        }
    }

    #[test]
    fn urls_are_admitted_only_when_public() {
        let fetcher = Fetcher::default();
        let refused = |url: &str| {
            let failure = fetcher.admit(url).unwrap_err();
            (failure.status, failure.message)
        };
        let (status, message) = refused("file:///etc/passwd");
        assert_eq!(status, 403);
        assert!(message.contains("only http and https"), "{message}");
        assert_eq!(refused("ftp://example.com/").0, 403);
        assert_eq!(refused("not a url").0, 400);
        assert_eq!(refused("javascript:alert(1)").0, 403);
        let (status, message) = refused("http://localhost:8080/api/status");
        assert_eq!(status, 403);
        assert!(message.contains("this machine"), "{message}");
        assert_eq!(refused("http://LOCALHOST./").0, 403);
        assert_eq!(refused("http://app.localhost/").0, 403);
        let (status, message) = refused("http://127.0.0.1:8099/api/stop");
        assert_eq!(status, 403);
        assert!(message.contains("loopback"), "{message}");
        assert!(refused("http://192.168.0.200/").1.contains("private"));
        assert!(refused("http://[::1]:8080/").1.contains("loopback"));
        assert!(refused("http://[fd00::5]/").1.contains("unique-local"));
        // Alternative spellings of loopback normalize before the check.
        assert!(refused("http://2130706433/").1.contains("loopback"));
        assert!(refused("http://0x7f.1/").1.contains("loopback"));
        assert!(refused("http://127.1/").1.contains("loopback"));
        assert!(refused("http://user:pw@example.com/")
            .1
            .contains("credentials"));
        let long = format!("https://example.com/{}", "a".repeat(URL_CAP));
        assert_eq!(refused(&long).0, 400);
        let (url, addresses) = fetcher.admit("http://93.184.216.34/page#frag").unwrap();
        assert_eq!(url.as_str(), "http://93.184.216.34/page");
        assert_eq!(addresses, ["93.184.216.34:80".parse().unwrap()]);
    }

    #[test]
    fn html_is_reduced_to_readable_text() {
        let html = r#"<!DOCTYPE html><html><head><title>The  &amp; Title</title>
            <style>body { color: red }</style><script>var x = "<p>not text</p>";</script></head>
            <body><!-- hidden --><h1>Heading</h1><p>First&nbsp;paragraph with <b>bold</b>text
            and   spaces.</p><ul><li>one</li><li>two &lt;3 &#8364; &#x41;</li></ul>
            <table><tr><td>a</td><td>b</td></tr></table><p>a < b &unknown; &amp</p>
            <noscript>enable scripts</noscript><svg><text>drawn</text></svg></body></html>"#;
        let (title, text) = html_to_text(html);
        assert_eq!(title.as_deref(), Some("The & Title"));
        assert_eq!(
            text,
            "Heading\nFirst paragraph with boldtext and spaces.\n- one\n- two <3 € A\na b\na < b &unknown; &amp"
        );
        for absent in ["not text", "color", "hidden", "enable scripts", "drawn"] {
            assert!(!text.contains(absent), "{absent}: {text}");
        }
        // Menus are dropped, and marked content wins over the page around it.
        let page = "<html><title>T</title><body><nav><a>Home</a><a>About</a></nav>\
                    <div>Sidebar <mainly>x</mainly></div><main id=content><h1>Topic</h1>\
                    <p>The fact.</p></main><footer>Footer</footer></body></html>";
        let (title, text) = html_to_text(page);
        assert_eq!(
            (title.as_deref(), text.as_str()),
            (Some("T"), "Topic\nThe fact.")
        );
        let article = "<body><nav>Menu</nav><p>Intro</p><article><p>Body</p></article></body>";
        assert_eq!(html_to_text(article).1, "Body");
        assert_eq!(
            html_to_text("<body><nav>Menu</nav><p>Only</p></body>").1,
            "Only"
        );
        assert_eq!(
            collapse_whitespace("  a \t b \n\n\n\n c  ", true),
            "a b\n\nc"
        );
        assert_eq!(collapse_whitespace("a\n\n\nb", false), "a\nb");
        assert_eq!(decode_entities("&#0; &#xZZ; &;"), "\u{fffd} &#xZZ; &;");
    }

    #[test]
    fn bodies_are_decoded_by_type_and_capped() {
        let (_, text) = readable("application/json", b"{\"a\":  1}");
        assert_eq!(text, "{\"a\": 1}");
        let (_, text) = readable("text/plain; charset=ISO-8859-1", &[0x63, 0x61, 0x66, 0xe9]);
        assert_eq!(text, "café");
        let (title, text) = readable("", b"<html><title>t</title><p>x</p></html>");
        assert_eq!((title.as_deref(), text.as_str()), (Some("t"), "x"));
        let (_, text) = readable("image/png", &[0x89, 0x50]);
        assert!(text.contains("binary content of type image/png"), "{text}");
        let (text, cut) = cap("é".repeat(10), 5);
        assert_eq!((text.as_str(), cut), ("éé", true));
        assert_eq!(cap("short".into(), 8), ("short".to_string(), false));
    }
}
