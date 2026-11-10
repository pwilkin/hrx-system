// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/aql_buffer_ref.h"

#include "iree/hal/driver/amdgpu/buffer.h"

iree_status_t iree_hal_amdgpu_aql_resolve_buffer_ref_device_pointer(
    iree_hal_buffer_ref_t buffer_ref, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access, uint8_t** out_device_pointer) {
  *out_device_pointer = NULL;
  if (IREE_UNLIKELY(!buffer_ref.buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer dynamic binding resolved to a NULL buffer");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(buffer_ref.buffer), required_usage));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(buffer_ref.buffer), required_access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      buffer_ref.buffer, buffer_ref.offset, buffer_ref.length));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer_ref.buffer);
  uint8_t* device_pointer =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_pointer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer buffer must be backed by an AMDGPU allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer_ref.buffer), buffer_ref.offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer buffer device pointer offset overflows device "
        "size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer buffer device pointer offset exceeds host pointer "
        "size");
  }
  *out_device_pointer = device_pointer + (uintptr_t)device_offset;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_aql_resolve_command_buffer_ref(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_command_buffer_binding_kind_t kind, uint32_t ordinal,
    uint64_t offset, uint64_t length, iree_hal_buffer_usage_t required_usage,
    iree_hal_memory_access_t required_access,
    iree_hal_buffer_ref_t* out_buffer_ref, uint8_t** out_device_pointer) {
  memset(out_buffer_ref, 0, sizeof(*out_buffer_ref));
  *out_device_pointer = NULL;
  if (kind == IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC) {
    iree_hal_buffer_t* buffer =
        iree_hal_amdgpu_aql_command_buffer_static_buffer(command_buffer,
                                                         ordinal);
    if (IREE_UNLIKELY(!buffer)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AQL command-buffer static buffer ordinal %" PRIu32 " is invalid",
          ordinal);
    }
    *out_buffer_ref = iree_hal_make_buffer_ref(buffer, offset, length);
  } else if (kind == IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC) {
    iree_hal_buffer_ref_t dynamic_ref =
        iree_hal_make_indirect_buffer_ref(ordinal, offset, length);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
        binding_table, dynamic_ref, out_buffer_ref));
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AQL command-buffer binding kind %u is invalid",
                            kind);
  }
  return iree_hal_amdgpu_aql_resolve_buffer_ref_device_pointer(
      *out_buffer_ref, required_usage, required_access, out_device_pointer);
}

iree_status_t iree_hal_amdgpu_aql_resolve_static_binding_source_pointer(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    uint64_t* out_binding_pointer) {
  *out_binding_pointer = 0;
  iree_hal_buffer_t* buffer = iree_hal_amdgpu_aql_command_buffer_static_buffer(
      command_buffer, binding_source->slot);
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer static dispatch binding ordinal %" PRIu32
        " is invalid",
        binding_source->slot);
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  void* device_pointer =
      iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_pointer)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AQL command-buffer static dispatch binding has no staged AMDGPU "
        "backing after queue waits completed");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(buffer),
          binding_source->offset_or_pointer, &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer static dispatch binding pointer offset overflows "
        "device size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer static dispatch binding pointer offset exceeds "
        "host pointer size");
  }
  *out_binding_pointer =
      (uint64_t)((uintptr_t)device_pointer + (uintptr_t)device_offset);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_aql_resolve_indirect_parameter_source(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_binding_source_t* binding_source,
    iree_hal_buffer_ref_t* out_buffer_ref,
    const uint32_t** out_device_workgroup_count) {
  memset(out_buffer_ref, 0, sizeof(*out_buffer_ref));
  *out_device_workgroup_count = NULL;
  const uint8_t flags = binding_source->flags;
  if (flags ==
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS) {
    *out_device_workgroup_count =
        (const uint32_t*)(uintptr_t)binding_source->offset_or_pointer;
    return iree_ok_status();
  }

  iree_hal_amdgpu_command_buffer_binding_kind_t kind =
      IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_INVALID;
  if (flags ==
      (IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_DYNAMIC |
       IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS)) {
    kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_DYNAMIC;
  } else if (
      flags ==
      (IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_STATIC_BUFFER |
       IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_SOURCE_FLAG_INDIRECT_PARAMETERS)) {
    kind = IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "malformed AQL command-buffer indirect parameter source flags %u",
        binding_source->flags);
  }

  uint8_t* device_pointer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_aql_resolve_command_buffer_ref(
      command_buffer, binding_table, kind, binding_source->slot,
      binding_source->offset_or_pointer, sizeof(uint32_t[3]),
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      IREE_HAL_MEMORY_ACCESS_READ, out_buffer_ref, &device_pointer));
  *out_device_workgroup_count = (const uint32_t*)device_pointer;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_aql_try_snapshot_host_visible_indirect_parameters(
    iree_hal_buffer_ref_t buffer_ref, bool* out_available,
    uint32_t out_workgroup_count[3]) {
  *out_available = false;
  memset(out_workgroup_count, 0, sizeof(uint32_t[3]));
  if (!buffer_ref.buffer ||
      !iree_all_bits_set(iree_hal_buffer_memory_type(buffer_ref.buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    return iree_ok_status();
  }

  iree_byte_span_t host_span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_buffer_native_host_span(
      buffer_ref.buffer, buffer_ref.offset, sizeof(uint32_t[3]), &host_span));
  memcpy(out_workgroup_count, host_span.data, sizeof(uint32_t[3]));
  *out_available = true;
  return iree_ok_status();
}
