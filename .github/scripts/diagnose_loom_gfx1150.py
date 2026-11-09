#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Branch-only capture of the native Loom table lookup timeout."""

import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

PHASES = (
    "loom_run_hal_runtime_initialize",
    "loomc_compile_artifact",
    "loom_run_hal_prepared_candidate_prepare",
    "loom_run_hal_testbench_staging_upload",
    "loom_run_hal_dispatch_sequence_execute",
    "loom_run_hal_testbench_staging_readback",
    "loom_run_hal_testbench_staging_deinitialize",
    "loom_run_hal_runtime_deinitialize",
)


def run_under(arguments: list[str]) -> int:
    commands = [
        "set pagination off",
        "set confirm off",
        "set breakpoint pending on",
        "set startup-with-shell off",
        "set debuginfod enabled off",
        "set print elements 16",
        "set print repeats 4",
        "set print max-depth 3",
        "set max-value-size 4096",
        "handle SIGINT stop print nopass",
        "python",
        "captured_exit_code = 1",
        "def capture_exit(event):",
        "    global captured_exit_code",
        "    captured_exit_code = getattr(event, 'exit_code', 1)",
        "gdb.events.exited.connect(capture_exit)",
        "end",
    ]
    for phase in PHASES:
        commands.extend(
            [
                f"break {phase}",
                "commands",
                "silent",
                f'printf "LOOM_PHASE {phase}\\n"',
                "continue",
                "end",
            ]
        )
    commands.extend(
        [
            "run",
            'printf "LOOM_DIAGNOSTIC_THREAD_STACKS\\n"',
            "thread apply all backtrace full 20",
            "python",
            "if gdb.selected_inferior().pid:",
            "    gdb.execute('kill')",
            "    gdb.execute('quit 124')",
            "else:",
            "    gdb.execute('quit %d' % captured_exit_code)",
            "end",
        ]
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".gdb") as script:
        script.write("\n".join(commands) + "\n")
        script.flush()
        process = subprocess.Popen(
            ["gdb", "--quiet", "--batch", "-x", script.name, "--args", *arguments],
            start_new_session=True,
        )
        # Capture before Bazel's existing 60-second timeout retires the sandbox.
        capture_seconds = float(os.environ.get("LOOM_STACK_CAPTURE_SECONDS", "40"))
        try:
            return process.wait(timeout=capture_seconds)
        except subprocess.TimeoutExpired:
            print("LOOM_DIAGNOSTIC_CAPTURE_REQUESTED", flush=True)
            process.send_signal(signal.SIGINT)
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                print("LOOM_DIAGNOSTIC_DEBUGGER_DID_NOT_RETIRE", flush=True)
                return 124


def run_ci() -> int:
    if os.environ.get("LOOM_DIAGNOSTIC_CI") != "1":
        raise RuntimeError("The configuration step is restricted to the CI runner")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from build_tools.devtools import ci

    result_directory = Path(os.environ["LOOM_DIAGNOSTIC_RESULTS_DIRECTORY"])
    arguments = ci.parse_arguments(
        [
            "iree-bazel-amdgpu",
            "--amdgpu-target",
            "gfx1150",
            "--target",
            "//loom/target/amdgpu/test/corpus:"
            "memory_address_table_index_test_execute_amd_gpu_hardware_test",
            "--bazel-test-results-dir",
            str(result_directory),
        ]
    )
    steps = ci.steps_from_args(arguments)
    assert len(steps) == 2 and steps[0].name == "Configure Bazel"
    test = steps[1]
    steps[1] = replace(
        test,
        argv=test.argv[:4]
        + (
            "--nocache_test_results",
            "--test_output=all",
            "--strip=never",
            "--copt=-g",
            "--host_copt=-g",
            f"--run_under=python3 {Path(__file__).resolve()} --under /bin/bash",
        )
        + test.argv[4:],
    )
    ci.bazel_test_results.prepare_output_directory(result_directory)
    return ci.run_steps(steps, dry_run=False, keep_going=False, verbose=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--under":
        raise SystemExit(run_under(sys.argv[2:]))
    raise SystemExit(run_ci())
