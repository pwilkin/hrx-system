# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""State capture and a raw native-write probe on the diagnostic branch."""

import struct

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


class ProbeNativeWrite(gdb.Breakpoint):
    def __init__(self):
        super().__init__("iree_hal_amd_gpu_pm4_publish_accepted", internal=True)

    def stop(self):
        try:
            if int(gdb.parse_and_eval("position")) != 0:
                return False
            if not bool(
                gdb.parse_and_eval(
                    "submission->entry.kind == IREE_HAL_AMD_GPU_PM4_ENTRY_ALLOCATION"
                )
            ):
                return False
            address = int(gdb.parse_and_eval("queue->producer.ring.words"))
            signal = int(gdb.parse_and_eval("submission->signal->device_address"))
            expected = (
                0xC0064900,
                0x528,
                0x23010000,
                signal & 0xFFFFFFFF,
                signal >> 32,
                1,
                0,
                0,
            )
            inferior = gdb.selected_inferior()
            actual = struct.unpack("<8I", inferior.read_memory(address, 32))
            if actual != expected:
                print(f"LOOM_PROBE_ERROR unexpected initial packet {actual}")
                return True
            if int(gdb.parse_and_eval("queue->producer.ring.reserved_words")) != 32:
                print("LOOM_PROBE_ERROR unexpected initial publication extent")
                return True
            native = gdb.parse_and_eval(
                "((amdf_gpu_user_queue_t *)queue->storage.queue)->umd"
            ).dereference()
            control = native["plan"]["control"]
            offset = (
                int(control["error_payload_byte_offset"])
                + int(control["error_payload_byte_length"])
                + 63
            ) & ~63
            if (
                offset + 4 > int(control["storage"]["byte_length"])
                or not int(control["storage"]["native_flags"]) & 0x02000000
            ):
                print("LOOM_PROBE_ERROR no unused uncached control word")
                return True
            target = int(native["control"]["device_address"]) + offset
            owners["native_write_word"] = (
                int(native["control"]["host_pointer"]) + offset
            )
            initial = struct.unpack(
                "<I", inferior.read_memory(owners["native_write_word"], 4)
            )[0]
            if initial != 0:
                print(f"LOOM_PROBE_ERROR control padding is not zero: {initial:#x}")
                return True
            # Submit one confirmed write into unused uncached control storage.
            # No release, wait, shader, or event remains in this publication.
            # Host staging intentionally remains blocked; the marker and native
            # read index are the experiment's observations.
            replacement = [
                0xC0033700,
                0x00100200,
                target & 0xFFFFFFFF,
                target >> 32,
                0x6C6F6F6D,
                0xC0191000,
            ] + [0] * 26
            inferior.write_memory(address, struct.pack("<32I", *replacement))
            print(
                f"LOOM_PROBE_NATIVE_WRITE ring={address:#x} target={target:#x} "
                "expected=0x6c6f6f6d"
            )
            self.enabled = False
        except gdb.error as error:
            print(f"LOOM_PROBE_ERROR {error}")
            return True
        return False


ProbeNativeWrite()


def command(text):
    print(f"LOOM_STATE_COMMAND {text}")
    try:
        gdb.execute(text)
    except gdb.error as error:
        print(f"LOOM_STATE_ERROR {error}")


def capture_pm4():
    command("set $loom_pm4 = (iree_hal_amd_gpu_pm4_queue_t *)$loom_transfer")
    for field in (
        "options",
        "publisher",
        "growth",
        "closed",
        "native_failure",
        "pending",
        "producer.ring",
        "producer.count",
        "producer.frontier",
        "capacity",
        "completed",
        "retirement",
    ):
        command(f"print $loom_pm4->{field}")
    if int(gdb.parse_and_eval("$loom_pm4->storage.mapping")):
        command(
            "set $loom_umd_queue = "
            "((amdf_gpu_user_queue_t *)$loom_pm4->storage.queue)->umd"
        )
        command("print *$loom_umd_queue")
        command("print $loom_umd_queue->plan")
        command("print $loom_pm4->storage.mapping->info")
        command("print *$loom_pm4->producer.write_index")
        command("x/gx $loom_pm4->storage.mapping->info.read_index_address")
        command("x/256wx $loom_pm4->producer.ring.words")
    published = int(gdb.parse_and_eval("$loom_pm4->producer.count"))
    completed = int(gdb.parse_and_eval("$loom_pm4->completed.count"))
    mask = int(gdb.parse_and_eval("$loom_pm4->retirement.capacity_mask"))
    for position in range(completed, min(published, completed + 16)):
        slot = position & mask
        command(f"print $loom_pm4->retirement.records[{slot}]")
        command(f"print $loom_pm4->issued[{slot}].action")
        tagged = int(
            gdb.parse_and_eval(f"$loom_pm4->issued[{slot}].action.tagged_submission")
        )
        command(
            f"set $loom_submission = (iree_hal_amd_gpu_pm4_submission_t *){tagged & ~1}"
        )
        command("print *$loom_submission")
        count = int(gdb.parse_and_eval("$loom_submission->wait_count"))
        for index in range(min(count, 16)):
            command(f"print $loom_submission->waits[{index}]")
            if int(gdb.parse_and_eval(f"$loom_submission->waits[{index}].signal")):
                command(f"print *$loom_submission->waits[{index}].signal")
                command(f"print *$loom_submission->waits[{index}].signal->host_value")


def capture_state():
    print(f"LOOM_STATE_OWNERS {owners}")
    if "native_write_word" in owners:
        command(f"x/wx {owners['native_write_word']}")
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
    if "iree_hal_amd_gpu_pm4_queue_vtable" in vtable:
        capture_pm4()
        return
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
    published = int(gdb.parse_and_eval("$loom_native->published"))
    retired = int(gdb.parse_and_eval("$loom_native->retired"))
    for position in range(retired, min(published, retired + slots)):
        index = position % slots
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
