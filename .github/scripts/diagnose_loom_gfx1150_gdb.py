# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read-only state capture, loaded by GDB on the diagnostic branch."""

import gdb

owners = {}


class CaptureOwner(gdb.Breakpoint):
    def __init__(self, function, name, expression):
        super().__init__(function, internal=True)
        self.name = name
        self.capture_expression = expression

    def stop(self):
        try:
            owners[self.name] = int(gdb.parse_and_eval(self.capture_expression))
            print(f"LOOM_OWNER {self.name}={owners[self.name]:#x}")
        except gdb.error as error:
            print(f"LOOM_OWNER_ERROR {self.name}: {error}")
        return False


CaptureOwner("loom_run_hal_testbench_staging_initialize", "staging", "out_staging")
CaptureOwner("iree_hal_amd_gpu_completion_main", "completion", "user_data")
CaptureOwner("iree_hal_amd_gpu_semaphore_wait", "wait_value", "value")
CaptureOwner("amdf_gpu_umd_event_wait", "native_event", "event")


def command(text):
    print(f"LOOM_STATE_COMMAND {text}")
    try:
        gdb.execute(text)
    except gdb.error as error:
        print(f"LOOM_STATE_ERROR {error}")


def capture_state():
    print(f"LOOM_STATE_OWNERS {owners}")
    if "native_event" in owners:
        command(f"print *(amdf_gpu_umd_event_t *){owners['native_event']}")
    if "completion" in owners:
        command(
            "set $loom_completion = (iree_hal_amd_gpu_completion_t *)"
            f"{owners['completion']}"
        )
        command("print *$loom_completion")
        count = int(gdb.parse_and_eval("$loom_completion->source_count"))
        for index in range(min(count, 32)):
            command(f"print $loom_completion->sources[{index}]")
        command("print $loom_completion->context->event_info")
    if "staging" not in owners:
        print("LOOM_STATE_ERROR no live staging owner was captured")
        return
    command(
        f"set $loom_staging = (loom_run_hal_testbench_staging_t *){owners['staging']}"
    )
    command("print *$loom_staging")
    command("print *$loom_staging->runtime")
    command(
        "set $loom_progress = (iree_hal_amd_gpu_semaphore_t *)"
        "$loom_staging->allocation.progress_semaphore"
    )
    command("print *$loom_progress")
    if int(gdb.parse_and_eval("$loom_progress->submitted.signal")):
        command("print *$loom_progress->submitted.signal")
        command("print *$loom_progress->submitted.signal->host_value")
    command("set $loom_transfer = $loom_staging->runtime->transfer_queue")
    command("print *$loom_transfer")
    vtable = str(gdb.parse_and_eval("$loom_transfer->resource.vtable"))
    if "iree_hal_amd_gpu_sdma_queue_vtable" not in vtable:
        print(f"LOOM_STATE_TRANSFER_VTABLE {vtable}")
        return
    command("set $loom_sdma = (iree_hal_amd_gpu_sdma_queue_t *)$loom_transfer")
    for field in (
        "options",
        "host_owners",
        "operation_count",
        "closed",
        "failure_code",
        "completed_epoch",
        "producer",
        "signal_growth",
        "signal_reuse",
    ):
        command(f"print $loom_sdma->{field}")
    command("set $loom_native = $loom_sdma->native")
    command("print *$loom_native")
    command("print *$loom_native->completion")
    if int(gdb.parse_and_eval("$loom_native->user.storage.mapping")):
        command("print $loom_native->user.storage.mapping->info")
        command("print *$loom_native->user.write_index")
        # The mapping contract makes the consumed index CPU-readable. The
        # doorbell is write-only MMIO and is deliberately never dereferenced.
        command("x/gx $loom_native->user.storage.mapping->info.read_index_address")
    command("x/256wx $loom_native->commands")
    slots = gdb.parse_and_eval("$loom_sdma->spans").type.range()[1] + 1
    for index in range(slots):
        operation = gdb.parse_and_eval(f"$loom_sdma->spans[{index}].operation")
        if int(operation):
            command(f"print $loom_sdma->spans[{index}]")
            command(f"print *$loom_sdma->spans[{index}].operation")


class CaptureState(gdb.Command):
    def __init__(self):
        super().__init__("loom-capture-state", gdb.COMMAND_USER)

    def invoke(self, argument, from_tty):
        try:
            capture_state()
        except gdb.error as error:
            print(f"LOOM_STATE_ERROR {error}")


CaptureState()
