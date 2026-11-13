//! The embedded page: one HTML document with its style and script inline.
//!
//! The three sources live next to this file so they can be edited as what they
//! are; they are joined once, at first use. With scripts disabled the page
//! shows a server-rendered summary instead of its panels: the launch and what
//! it allows, the actions the dashboard ran, the installed profiles and the
//! stored benchmark runs.

use super::{current_run, headline, Context, Reply};
use crate::proxy::raw_response;
use crate::store::{BenchmarkStore, InstallRegistry};
use crate::values::{arr, get, human_bytes, number_value, str_value};
use serde_json::Value;
use std::path::Path;
use std::sync::OnceLock;

const HTML: &str = include_str!("page/index.html");
const STYLE: &str = include_str!("page/style.css");
const SCRIPT: &str = include_str!("page/app.js");
const SUMMARY_MARK: &str = "<!--STATIC-SUMMARY-->";

/// Nothing loads from anywhere but this listener; inline style and script are
/// the page itself. `worker-src blob:` admits the chat's JavaScript sandbox, a
/// worker built from a Blob per run; the worker inherits this policy, so code
/// in it cannot `eval`, import scripts or connect anywhere but this listener
/// (and the sandbox removes `fetch` and the other network APIs before the
/// snippet runs).
const SECURITY_POLICY: &str = "default-src 'none'; style-src 'unsafe-inline'; \
     script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; \
     worker-src blob:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

fn template() -> &'static str {
    static PAGE: OnceLock<String> = OnceLock::new();
    PAGE.get_or_init(|| {
        HTML.replacen("/*STYLE*/", STYLE, 1)
            .replacen("/*SCRIPT*/", SCRIPT, 1)
    })
}

pub(super) fn reply(context: &Context) -> Reply {
    let html = template().replacen(SUMMARY_MARK, &summary(context), 1);
    let mut reply = raw_response(200, "text/html; charset=utf-8", html.into_bytes());
    for (name, value) in [
        ("Cache-Control", "no-store"),
        ("Content-Security-Policy", SECURITY_POLICY),
        ("X-Content-Type-Options", "nosniff"),
        ("Referrer-Policy", "no-referrer"),
    ] {
        if let Ok(header) = tiny_http::Header::from_bytes(name, value) {
            reply.add_header(header);
        }
    }
    reply
}

pub(super) fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// An endpoint shown as `host:port` (the scheme adds nothing on a page).
fn bare(endpoint: &str) -> &str {
    endpoint
        .strip_prefix("http://")
        .unwrap_or(endpoint)
        .trim_end_matches('/')
}

/// The server-rendered summary shown when scripts are disabled.
fn summary(context: &Context) -> String {
    let live = context.live();
    let mut html = String::new();
    html.push_str(
        "<h2>Static summary</h2><p class=\"hint\">Scripts are disabled, so nothing \
                   here updates live. Reload for fresh numbers, or read the JSON routes linked \
                   at the bottom of the page.</p>",
    );
    html.push_str("<h3>Launch</h3>");
    match (&live.repo, &live.endpoint) {
        (Some(repo), Some(endpoint)) => html.push_str(&format!(
            "<p><strong>{}</strong> serving at <code>{}</code> on <code>{}</code>, context {}.</p>",
            escape(repo),
            escape(bare(endpoint)),
            escape(str_value(&live.fact("device")).unwrap_or("?")),
            escape(&crate::values::py_str(&live.fact("context_length"))),
        )),
        _ => html.push_str("<p>Nothing is running.</p>"),
    }
    let web = context.web_access(&live);
    html.push_str(&match (&web.reason, &web.command) {
        (None, _) => {
            "<p>Web access: on (the chat's <code>web_fetch</code> tool may fetch public pages).</p>"
                .to_string()
        }
        (Some(reason), Some(command)) => format!(
            "<p>Web access: {} — <code>{}</code></p>",
            escape(reason),
            escape(command)
        ),
        (Some(reason), None) => format!("<p>Web access: {}</p>", escape(reason)),
    });
    html.push_str("<h3>Activity</h3>");
    let jobs = context.jobs().list(0);
    if arr(&jobs).is_empty() {
        html.push_str("<p>No actions since this dashboard started.</p>");
    } else {
        html.push_str("<ul>");
        for job in arr(&jobs).iter().take(20) {
            let text = |key: &str| str_value(&job[key]).unwrap_or("").to_string();
            let elapsed = number_value(&job["elapsed_seconds"])
                .map(|s| format!(", {s:.1} s"))
                .unwrap_or_default();
            let active = matches!(text("state").as_str(), "queued" | "running");
            let detail = match (&job["error"], &job["step"]) {
                (Value::String(error), _) => format!(": {}", error.lines().next().unwrap_or("")),
                (_, Value::String(step)) if active => format!(" ({step})"),
                _ => String::new(),
            };
            let noun = match text("kind").as_str() {
                "serve" => "launch".to_string(),
                "bench" => "benchmark".to_string(),
                "fetch" => "web fetch".to_string(),
                other => other.to_string(),
            };
            html.push_str(&format!(
                "<li><strong>{}</strong> {} — {}{}{}</li>",
                escape(&noun),
                escape(&text("target")),
                escape(&text("state")),
                escape(&elapsed),
                escape(&detail),
            ));
        }
        html.push_str("</ul>");
    }
    let installs = InstallRegistry::new(&context.state_root).list();
    html.push_str("<h3>Installed profiles</h3>");
    if installs.is_empty() {
        html.push_str("<p>None yet.</p>");
    } else {
        html.push_str("<ul>");
        for installed in &installs {
            html.push_str(&format!(
                "<li><strong>{}</strong> — {}</li>",
                escape(&installed.repo),
                escape(&human_bytes(installed.size_bytes))
            ));
        }
        html.push_str("</ul>");
    }
    let runs = BenchmarkStore::new(&context.state_root).list(None);
    html.push_str("<h3>Stored benchmark runs</h3>");
    if runs.is_empty() {
        html.push_str("<p>None yet.</p>");
    } else {
        html.push_str(
            "<div class=\"table-wrap\"><table class=\"data\"><thead><tr><th scope=\"col\">Started</th>\
             <th scope=\"col\">Profile</th><th scope=\"col\">Suite</th><th scope=\"col\">Decode</th>\
             <th scope=\"col\">Record</th></tr></thead><tbody>",
        );
        for entry in runs.iter().rev().take(20) {
            let id = str_value(get(entry, "id")).unwrap_or("");
            let record = str_value(get(entry, "path"))
                .map(|path| crate::store::read_json(Path::new(path)))
                .unwrap_or_default();
            let decode = headline(&record)
                .iter()
                .find(|row| str_value(&row["metric"]) == Some("decode"))
                .and_then(|row| number_value(&row["value"]))
                .map(|v| format!("{} tok/s", crate::values::grouped(v, 1)))
                .unwrap_or_else(|| "—".into());
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td class=\"num\">{}</td>\
                 <td><a href=\"/api/benchmarks/{}\">JSON</a></td></tr>",
                escape(str_value(get(entry, "started_at")).unwrap_or("")),
                escape(str_value(get(entry, "profile")).unwrap_or("")),
                escape(str_value(get(entry, "suite")).unwrap_or("")),
                escape(&decode),
                escape(id),
            ));
        }
        html.push_str("</tbody></table></div>");
    }
    if let Some(run) = current_run(&context.state_root) {
        html.push_str(&format!(
            "<h3>Logs</h3><p>Run <code>{}</code>: <a href=\"/api/logs?lines=400\">serve.log and events</a>.</p>",
            escape(&run.run_id)
        ));
    }
    html
}
