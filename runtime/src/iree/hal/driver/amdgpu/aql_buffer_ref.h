// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMDGPU_AQL_BUFFER_REF_H_
#define IREE_HAL_DRIVER_AMDGPU_AQL_BUFFER_REF_H_

#include "iree/hal/driver/amdgpu/aql_command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates |buffer_ref| and resolves it to an AMDGPU device pointer.
iree_status_t iree_hal_amdgpu_aql_resolve_buffer_ref_device_pointer(
    iree_hal_buffer_ref_t buffer_ref, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access, uint8_t** out_device_pointer);

// Resolves a recorded static or dynamic command-buffer reference and validates
// its range, usage, and access before returning its AMDGPU device pointer.
iree_status_t iree_hal_amdgpu_aql_resolve_command_buffer_ref(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_command_buffer_binding_kind_t kind, uint32_t ordinal,
    uint64_t offset, uint64_t length, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access,
    iree_hal_buffer_ref_t* out_buffer_ref, uint8_t** out_device_pointer);

// Resolves a retained static dispatch binding source to an AMDGPU device
// pointer after staged transient-buffer materialization has completed.
iree_status_t iree_hal_amdgpu_aql_resolve_static_binding_source_pointer(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    uint64_t* out_binding_pointer);

// Resolves an indirect-parameter source to its validated buffer reference and
// AMDGPU device pointer. Legacy raw-pointer sources have an empty
// |out_buffer_ref| and cannot be inspected through a host binding.
iree_status_t iree_hal_amdgpu_aql_resolve_indirect_parameter_source(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    iree_hal_buffer_ref_t* out_buffer_ref,
    const uint32_t** out_device_workgroup_count);

// Tries to snapshot a validated indirect-parameter reference through its
// prepared AMDGPU host binding. Returns unavailable for device-only storage.
// AMDGPU host-visible allocations are coherent with their device binding.
iree_status_t iree_hal_amdgpu_aql_try_snapshot_host_visible_indirect_parameters(
    iree_hal_buffer_ref_t buffer_ref, bool* out_available,
    uint32_t out_workgroup_count[3]);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMDGPU_AQL_BUFFER_REF_H_
