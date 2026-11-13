//! Tolerant readers for documents whose publishers may add fields, plus the
//! number, byte, time and JSON formatting the launcher shares with the Python
//! reference implementation.
//!
//! Every reader answers "absent" instead of failing, so one unexpected field
//! cannot make an otherwise usable profile unreadable.

use serde_json::{Map, Value};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

static NULL: Value = Value::Null;

fn empty_map() -> &'static Map<String, Value> {
    static EMPTY: OnceLock<Map<String, Value>> = OnceLock::new();
    EMPTY.get_or_init(Map::new)
}

/// Returns `value` when it is an object, else an empty object.
pub fn obj(value: &Value) -> &Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => empty_map(),
    }
}

/// Reads `key` from an object, answering null when absent.
pub fn get<'a>(map: &'a Map<String, Value>, key: &str) -> &'a Value {
    map.get(key).unwrap_or(&NULL)
}

/// Shorthand for `obj(get(map, key))`.
pub fn child<'a>(map: &'a Map<String, Value>, key: &str) -> &'a Map<String, Value> {
    obj(get(map, key))
}

/// Returns `value` when it is an array, else an empty slice.
pub fn arr(value: &Value) -> &[Value] {
    match value {
        Value::Array(items) => items,
        _ => &[],
    }
}

/// Returns a nonempty string, treating null and empty text as absent.
pub fn str_value(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) if !text.is_empty() => Some(text),
        _ => None,
    }
}

/// Owned variant of [`str_value`].
pub fn string(value: &Value) -> Option<String> {
    str_value(value).map(str::to_string)
}

/// Returns a real boolean; numbers are not booleans in these documents.
pub fn bool_value(value: &Value) -> Option<bool> {
    value.as_bool()
}

/// Returns an integral count, accepting an integral floating point value.
pub fn int_value(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => {
            if let Some(v) = number.as_i64() {
                Some(v)
            } else if let Some(v) = number.as_u64() {
                i64::try_from(v).ok()
            } else {
                let v = number.as_f64()?;
                if v.is_finite() && v.fract() == 0.0 && v.abs() < 9.0e18 {
                    Some(v as i64)
                } else {
                    None
                }
            }
        }
        _ => None,
    }
}

/// Returns a strict JSON integer (Python `isinstance(value, int)`), no floats.
pub fn strict_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) if number.is_i64() || number.is_u64() => number
            .as_i64()
            .or_else(|| number.as_u64().and_then(|v| i64::try_from(v).ok())),
        _ => None,
    }
}

/// Returns a measurement as a float, accepting numeric strings.
pub fn number_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Python truthiness of a JSON value.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|v| v != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// JSON value for an optional string.
pub fn opt_str<S: AsRef<str>>(value: Option<S>) -> Value {
    match value {
        Some(text) => Value::String(text.as_ref().to_string()),
        None => Value::Null,
    }
}

/// JSON value for an optional integer.
pub fn opt_int<N: Into<i64>>(value: Option<N>) -> Value {
    match value {
        Some(number) => Value::from(number.into()),
        None => Value::Null,
    }
}

/// JSON value for an optional float (NaN and infinities become null).
pub fn opt_f64(value: Option<f64>) -> Value {
    match value {
        Some(v) if v.is_finite() => Value::from(v),
        _ => Value::Null,
    }
}

/// Rounds like Python's `round(value, digits)`: correctly rounded decimal.
pub fn round_to(value: f64, digits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.digits$}").parse::<f64>().unwrap_or(value)
}

/// `round_to` lifted over an optional value.
pub fn round_opt(value: Option<f64>, digits: usize) -> Option<f64> {
    value.map(|v| round_to(v, digits))
}

/// Formats with thousands separators, like Python's `f"{value:,.{decimals}f}"`.
pub fn grouped(value: f64, decimals: usize) -> String {
    let text = format!("{:.*}", decimals, value.abs());
    let (integer, fraction) = match text.find('.') {
        Some(index) => (&text[..index], &text[index..]),
        None => (text.as_str(), ""),
    };
    let mut out = String::new();
    let digits: Vec<char> = integer.chars().collect();
    for (index, digit) in digits.iter().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*digit);
    }
    let negative = value < 0.0 && text.chars().any(|c| c.is_ascii_digit() && c != '0');
    format!("{}{}{}", if negative { "-" } else { "" }, out, fraction)
}

/// Formats a byte count in binary units; an unknown size stays "unknown".
pub fn human_bytes(count: Option<i64>) -> String {
    human_bytes_precision(count, 1)
}

pub fn human_bytes_precision(count: Option<i64>, precision: usize) -> String {
    let Some(count) = count else {
        return "unknown".to_string();
    };
    let mut value = count as f64;
    for unit in ["B", "KiB", "MiB", "GiB"] {
        if value < 1024.0 {
            if unit == "B" {
                return format!("{} B", value as i64);
            }
            return format!("{value:.precision$} {unit}");
        }
        value /= 1024.0;
    }
    format!("{value:.precision$} TiB")
}

/// Formats a token rate; an unmeasured rate is a dash.
pub fn human_rate(value: Option<f64>) -> String {
    match value {
        None => "-".to_string(),
        Some(v) => format!("{} tok/s", grouped(v, 1)),
    }
}

/// Formats a duration at the precision a benchmark needs.
pub fn human_seconds(value: Option<f64>) -> String {
    match value {
        None => "-".to_string(),
        Some(v) if v < 1.0 => format!("{:.0} ms", v * 1000.0),
        Some(v) => format!("{v:.2} s"),
    }
}

/// An uptime as `45s`, `3m 05s`, `2h 04m` or `3d 02h`.
pub fn human_duration(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (days, hours, minutes, secs) = (
        total / 86400,
        (total % 86400) / 3600,
        (total % 3600) / 60,
        total % 60,
    );
    if days > 0 {
        format!("{days}d {hours:02}h")
    } else if hours > 0 {
        format!("{hours}h {minutes:02}m")
    } else if minutes > 0 {
        format!("{minutes}m {secs:02}s")
    } else {
        format!("{secs}s")
    }
}

/// Python `str()` of a JSON scalar, as the CLI tables print it.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        other => other.to_string(),
    }
}

// -- JSON output -----------------------------------------------------------

/// Recursively orders object keys, like Python's `sort_keys=True`.
pub fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for key in keys {
                out.insert(key.clone(), sorted(&map[key]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

/// `json.dumps(value, indent=1)`: one-space indentation, insertion order.
pub fn dumps_pretty(value: &Value) -> String {
    use serde::Serialize;
    let mut out = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    value
        .serialize(&mut serializer)
        .expect("JSON values always serialize");
    String::from_utf8(out).expect("serde_json writes UTF-8")
}

/// `json.dumps(value, indent=1, sort_keys=True)`, the launcher's file format.
pub fn dumps_sorted(value: &Value) -> String {
    dumps_pretty(&sorted(value))
}

/// `json.dumps(value, separators=(",", ":"))`, the JSONL record format.
pub fn dumps_compact(value: &Value) -> String {
    serde_json::to_string(value).expect("JSON values always serialize")
}

/// Parses bytes as a JSON object, answering an empty object otherwise.
pub fn parse_object(data: &[u8]) -> Map<String, Value> {
    match serde_json::from_slice::<Value>(data) {
        Ok(Value::Object(map)) => map,
        _ => Map::new(),
    }
}

// -- time ------------------------------------------------------------------

/// Seconds since the Unix epoch as a float.
pub fn now_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Civil UTC fields of a Unix timestamp: (year, month, day, hour, minute, second).
pub fn civil_utc(timestamp: f64) -> (i64, u32, u32, u32, u32, u32) {
    let secs = timestamp.floor() as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Howard Hinnant's days-from-civil inverse.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y };
    (
        year,
        m,
        d,
        (rem / 3600) as u32,
        ((rem % 3600) / 60) as u32,
        (rem % 60) as u32,
    )
}

/// `%Y-%m-%dT%H:%M:%SZ` in UTC.
pub fn iso_utc(timestamp: f64) -> String {
    let (y, mo, d, h, mi, s) = civil_utc(timestamp);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// `%Y-%m-%dT%H:%M:%S.mmmZ` in UTC (request records).
pub fn iso_utc_millis(timestamp: f64) -> String {
    let (y, mo, d, h, mi, s) = civil_utc(timestamp);
    let millis = ((timestamp.rem_euclid(1.0)) * 1000.0) as u32;
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{millis:03}Z")
}

/// `%Y%m%dT%H%M%S` in UTC, the run-id stamp.
pub fn compact_utc(timestamp: f64) -> String {
    let (y, mo, d, h, mi, s) = civil_utc(timestamp);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}")
}

/// A random 32-hex-digit identifier (the shape of a uuid4 hex string).
pub fn random_hex_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut parts = [0u64; 2];
    for (index, part) in parts.iter_mut().enumerate() {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(count);
        hasher.write_usize(index);
        hasher.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        hasher.write_u32(std::process::id());
        *part = hasher.finish();
    }
    format!("{:016x}{:016x}", parts[0], parts[1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tolerant_readers_answer_absent() {
        assert_eq!(str_value(&json!("")), None);
        assert_eq!(str_value(&json!("x")), Some("x"));
        assert_eq!(int_value(&json!(4.0)), Some(4));
        assert_eq!(int_value(&json!(4.5)), None);
        assert_eq!(int_value(&json!(true)), None);
        assert_eq!(number_value(&json!("2.5")), Some(2.5));
        assert_eq!(number_value(&json!(false)), None);
        assert!(obj(&json!([1])).is_empty());
    }

    #[test]
    fn human_formats_match_the_python_launcher() {
        assert_eq!(human_bytes(Some(1024)), "1.0 KiB");
        assert_eq!(human_bytes(Some(500)), "500 B");
        assert_eq!(human_bytes(None), "unknown");
        assert_eq!(human_bytes(Some(63_000_000_000)), "58.7 GiB");
        assert_eq!(grouped(1234567.891, 1), "1,234,567.9");
        assert_eq!(grouped(-1234.5, 1), "-1,234.5");
        assert_eq!(grouped(12.0, 1), "12.0");
        assert_eq!(human_rate(Some(2595.79)), "2,595.8 tok/s");
        assert_eq!(human_seconds(Some(0.25)), "250 ms");
        assert_eq!(round_to(80.18456, 2), 80.18);
    }

    #[test]
    fn python_json_layouts() {
        let value = json!({"b": 1, "a": {"d": [1, 2], "c": null}});
        assert_eq!(
            dumps_sorted(&value),
            "{\n \"a\": {\n  \"c\": null,\n  \"d\": [\n   1,\n   2\n  ]\n },\n \"b\": 1\n}"
        );
        assert_eq!(
            dumps_compact(&value),
            "{\"b\":1,\"a\":{\"d\":[1,2],\"c\":null}}"
        );
    }

    #[test]
    fn utc_formatting() {
        // 2026-11-12T03:43:47Z
        let stamp = 1_794_455_027.25;
        assert_eq!(iso_utc(stamp), "2026-11-12T03:43:47Z");
        assert_eq!(iso_utc_millis(stamp), "2026-11-12T03:43:47.250Z");
        assert_eq!(compact_utc(stamp), "20261112T034347");
        assert_eq!(iso_utc(0.0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc(951_782_400.0), "2000-02-29T00:00:00Z");
    }
}
