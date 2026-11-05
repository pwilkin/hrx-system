// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PARAMETER_MODULE_H_
#define LOOM_RUNTIME_PARAMETER_MODULE_H_

#include "iree/hal/vm/types.h"
#include "iree/vm/module.h"
#include "loom/runtime/parameter/sources.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates the source-facing "parameters" read capability:
//   gather(i32 source, buffer key, hal.semaphore wait, i64 wait_value,
//          hal.buffer descriptors, i64 descriptor_offset, i64 count,
//          hal.buffer target) -> (hal.semaphore, i64)
//
// Source ordinals identify the cold/request catalog, not resident bindings.
// Each 24-byte descriptor is three little-endian u64 byte values: parameter
// offset, target offset, and length. Target spans must not overlap. The wait
// timepoint covers descriptor production, destination availability, and
// previous consumers; NULL with value zero means no prerequisite. Returned
// readiness orders consumers and descriptor reuse. An empty gather retains and
// returns the exact input timepoint without allocating or submitting queue
// work. No execution frontier is captured or changed here.
//
// Retains the exact queue, its device, and the source catalog. Accepted reads
// independently retain their provider and buffers through actual retirement,
// even after rebinding or module removal. Failed or discarded readiness does
// not release those obligations. The hosting device group and registered
// reference types outlive every accepted operation and escaped reference.
iree_status_t loom_runtime_parameters_module_create(
    iree_vm_environment_t* environment, const iree_hal_vm_types_t* types,
    iree_hal_queue_t* queue, loom_runtime_parameter_sources_t* sources,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

// Publishes a retained immutable source catalog at a serialized invocation
// boundary. NULL clears access to all sources. Does not wait for or redirect
// already accepted reads; each captured its exact provider at submission.
void loom_runtime_parameters_module_bind_sources(
    iree_vm_module_t* module, loom_runtime_parameter_sources_t* sources);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PARAMETER_MODULE_H_
