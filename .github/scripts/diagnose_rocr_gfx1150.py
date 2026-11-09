#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostic-only ROCr AQL barrier with no HRX or Loom code loaded."""

import ctypes as C
import os
import platform
import struct
import time
from pathlib import Path


class Handle(C.Structure):
    # Public HSA opaque agent/signal handle, passed by value.
    _fields_ = [("handle", C.c_uint64)]


class Queue(C.Structure):
    _fields_ = [
        # Public HSA queue producer mode.
        ("type", C.c_uint32),
        # Public queue feature mask.
        ("features", C.c_uint32),
        # Runtime-owned AQL packet storage.
        ("base_address", C.c_void_p),
        # Runtime-owned notification signal.
        ("doorbell_signal", Handle),
        # Number of packet slots.
        ("size", C.c_uint32),
        # Reserved ABI word.
        ("reserved", C.c_uint32),
        # Runtime-assigned queue identity.
        ("id", C.c_uint64),
    ]


def check(operation, status):
    print(f"ROCR_STATUS {operation}={status:#x}", flush=True)
    if status != 0:
        raise RuntimeError(f"{operation} failed with HSA status {status:#x}")


def report_host():
    print(f"ROCR_HOST {platform.uname()}", flush=True)
    for name in ("mes", "sched_policy", "cwsr", "hws_max_conc_proc"):
        path = Path("/sys/module/amdgpu/parameters") / name
        try:
            print(f"ROCR_HOST {path}={path.read_text().strip()}", flush=True)
        except OSError as error:
            print(f"ROCR_HOST_UNAVAILABLE {path}: {error}", flush=True)
    path = Path("/sys/class/kfd/kfd/proc")
    try:
        print(f"ROCR_HOST kfd_processes={sorted(p.name for p in path.iterdir())}")
    except OSError as error:
        print(f"ROCR_HOST_UNAVAILABLE {path}: {error}", flush=True)


def main():
    if os.environ.get("LOOM_DIAGNOSTIC_CI") != "1":
        raise RuntimeError("This hardware experiment is restricted to its CI workflow")
    report_host()
    path = os.environ["IREE_HAL_AMDGPU_LIBHSA_PATH"]
    print(f"ROCR_LIBRARY {path}", flush=True)
    library = C.CDLL(path)

    def bind(name, result, *arguments):
        function = getattr(library, name)
        function.restype = result
        function.argtypes = list(arguments)
        return function

    agent_callback_type = C.CFUNCTYPE(C.c_uint32, Handle, C.c_void_p)
    queue_callback_type = C.CFUNCTYPE(None, C.c_uint32, C.POINTER(Queue), C.c_void_p)
    initialize = bind("hsa_init", C.c_uint32)
    shutdown = bind("hsa_shut_down", C.c_uint32)
    iterate = bind("hsa_iterate_agents", C.c_uint32, agent_callback_type, C.c_void_p)
    agent_info = bind("hsa_agent_get_info", C.c_uint32, Handle, C.c_uint32, C.c_void_p)
    system_info = bind("hsa_system_get_info", C.c_uint32, C.c_uint32, C.c_void_p)
    create_queue = bind(
        "hsa_queue_create",
        C.c_uint32,
        Handle,
        C.c_uint32,
        C.c_uint32,
        queue_callback_type,
        C.c_void_p,
        C.c_uint32,
        C.c_uint32,
        C.POINTER(C.POINTER(Queue)),
    )
    destroy_queue = bind("hsa_queue_destroy", C.c_uint32, C.POINTER(Queue))
    reserve = bind(
        "hsa_queue_add_write_index_relaxed", C.c_uint64, C.POINTER(Queue), C.c_uint64
    )
    read_index = bind(
        "hsa_queue_load_read_index_scacquire", C.c_uint64, C.POINTER(Queue)
    )
    create_signal = bind(
        "hsa_signal_create",
        C.c_uint32,
        C.c_int64,
        C.c_uint32,
        C.c_void_p,
        C.POINTER(Handle),
    )
    destroy_signal = bind("hsa_signal_destroy", C.c_uint32, Handle)
    store_signal = bind("hsa_signal_store_screlease", None, Handle, C.c_int64)
    wait_signal = bind(
        "hsa_signal_wait_scacquire",
        C.c_int64,
        Handle,
        C.c_uint32,
        C.c_int64,
        C.c_uint64,
        C.c_uint32,
    )
    check("hsa_init", initialize())
    agents = []

    @agent_callback_type
    def visit(agent, unused):
        device = C.c_uint32()
        status = agent_info(agent, 17, C.byref(device))
        if status != 0:
            return status
        if device.value == 1:
            name = C.create_string_buffer(64)
            status = agent_info(agent, 0, name)
            if status != 0:
                return status
            print(f"ROCR_GPU name={name.value!r} handle={agent.handle:#x}", flush=True)
            if name.value == b"gfx1150":
                agents.append(Handle(agent.handle))
        return 0

    check("hsa_iterate_agents", iterate(visit, None))
    if len(agents) != 1:
        raise RuntimeError(f"Expected one visible gfx1150 agent, got {len(agents)}")
    minimum_size = C.c_uint32()
    check("queue_minimum_size", agent_info(agents[0], 13, C.byref(minimum_size)))
    frequency = C.c_uint64()
    check("timestamp_frequency", system_info(3, C.byref(frequency)))
    errors = []

    @queue_callback_type
    def queue_error(status, source, unused):
        errors.append(status)
        print(f"ROCR_QUEUE_ERROR status={status:#x}", flush=True)

    queue = C.POINTER(Queue)()
    check(
        "hsa_queue_create",
        create_queue(
            agents[0],
            max(64, minimum_size.value),
            1,
            queue_error,
            None,
            0xFFFFFFFF,
            0xFFFFFFFF,
            C.byref(queue),
        ),
    )
    completion = Handle()
    check("hsa_signal_create", create_signal(1, 0, None, C.byref(completion)))
    position = reserve(queue, 1)
    slot = queue.contents.base_address + (position & (queue.contents.size - 1)) * 64
    # Public hsa_barrier_and_packet_t: zero dependencies, completion at byte 56.
    # Publish the header last, then use ROCr's release doorbell operation.
    packet = bytearray(64)
    struct.pack_into("<H", packet, 0, 1)  # INVALID while recording.
    struct.pack_into("<Q", packet, 56, completion.handle)
    C.memmove(slot, bytes(packet), 64)
    C.c_uint16.from_address(slot).value = 3 | (2 << 9) | (2 << 11)
    print(
        f"ROCR_PUBLISH queue={queue.contents.id} position={position} completion=1",
        flush=True,
    )
    store_signal(queue.contents.doorbell_signal, position)
    # HSA permits spurious wakes. Observe one publication for ten seconds;
    # the workflow independently bounds the diagnostic process at 30 seconds.
    deadline = time.monotonic_ns() + 10_000_000_000
    value = 1
    while value != 0 and not errors:
        remaining = deadline - time.monotonic_ns()
        if remaining <= 0:
            break
        value = wait_signal(
            completion, 0, 0, remaining * frequency.value // 1_000_000_000, 0
        )
    consumed = read_index(queue)
    print(
        f"ROCR_OBSERVATION completion={value} read_index={consumed} expected_index={position + 1} errors={errors}",
        flush=True,
    )
    if value != 0 or errors:
        # Pending work retains all owners until process retirement; no buffer
        # is freed and no queue is reset or replayed after a failed observation.
        return 124
    check("hsa_queue_destroy", destroy_queue(queue))
    check("hsa_signal_destroy", destroy_signal(completion))
    check("hsa_shut_down", shutdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
