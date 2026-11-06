// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_mappings.h"

#include "loom/util/adaptive_sort.h"

static bool loom_runtime_memory_mapping_less(
    const iree_hal_memory_mapping_t* lhs,
    const iree_hal_memory_mapping_t* rhs) {
  return lhs->target != rhs->target
             ? (uintptr_t)lhs->target < (uintptr_t)rhs->target
             : lhs->target_range.byte_offset < rhs->target_range.byte_offset;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_runtime_memory_mappings_sort,
                          iree_hal_memory_mapping_t,
                          loom_runtime_memory_mapping_less)

static bool loom_runtime_memory_mappings_are_contiguous(
    const iree_hal_memory_mapping_t* previous,
    const iree_hal_memory_mapping_t* current) {
  const iree_device_size_t length = previous->target_range.byte_length;
  return previous->target == current->target &&
         previous->scope.domain == current->scope.domain &&
         previous->scope.id == current->scope.id &&
         previous->operation == current->operation &&
         previous->access == current->access &&
         previous->target_range.byte_offset + length ==
             current->target_range.byte_offset &&
         previous->source == current->source &&
         (!previous->source ||
          previous->source_offset + length == current->source_offset) &&
         previous->previous_source == current->previous_source &&
         (!previous->previous_source ||
          previous->previous_source_offset + length ==
              current->previous_source_offset);
}

iree_host_size_t loom_runtime_memory_mappings_coalesce(
    iree_host_size_t count, iree_hal_memory_mapping_t* mappings) {
  loom_runtime_memory_mappings_sort(mappings, count);
  iree_host_size_t span_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (span_count && loom_runtime_memory_mappings_are_contiguous(
                          &mappings[span_count - 1], &mappings[i])) {
      mappings[span_count - 1].target_range.byte_length +=
          mappings[i].target_range.byte_length;
    } else {
      mappings[span_count++] = mappings[i];
    }
  }
  return span_count;
}
