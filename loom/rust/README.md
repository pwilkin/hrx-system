# Rust tools for Loom

| Crate | What it is |
| --- | --- |
| [`loom-launch`](loom-launch/README.md) | Native launcher for published Loom profiles: hub search, `loom-fetch` installs, `loom-serve` supervision behind an instrumented OpenAI-compatible proxy, chat and card-comparable benchmarks. Shares its state directory with the Python launcher (`loom/py/loom/launch`). |

Cargo is the supported build (`cargo build --release` inside a crate). Bazel
integration would need `rules_rust`, which this repository does not use, so
these crates are not wired into the Bazel build.

## Status

Experimental, and **not proposed for upstream review yet**. Two things would have
to be settled first:

1. This repository has no `rules_rust`, so the crate has no Bazel wiring and
   would land as a cargo-only island with its own dependency policy.
2. It duplicates the Python launcher's surface (`loom/py/loom/launch`). One of
   the two should survive review, and the choice depends on whether the launcher
   is expected to ship with the runtime or alongside it.

The state-directory format is shared with the Python launcher and still moving,
so the two can be switched between runs without losing installs or benchmarks.
