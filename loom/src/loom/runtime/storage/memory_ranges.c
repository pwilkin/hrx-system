// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_ranges.h"

#include "loom/runtime/storage/memory.h"
#include "loom/util/adaptive_sort.h"

static bool loom_runtime_memory_range_less(
    const loom_runtime_memory_range_t* lhs,
    const loom_runtime_memory_range_t* rhs) {
  return lhs->buffer != rhs->buffer
             ? (uintptr_t)lhs->buffer < (uintptr_t)rhs->buffer
             : lhs->offset < rhs->offset;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_runtime_memory_ranges_sort,
                          loom_runtime_memory_range_t,
                          loom_runtime_memory_range_less)

static uint64_t loom_runtime_memory_footprint_sum(uint64_t lhs, uint64_t rhs) {
  return lhs + iree_min(rhs, UINT64_MAX - lhs);
}

loom_runtime_memory_footprint_t loom_runtime_memory_ranges_footprint(
    iree_host_size_t range_count, loom_runtime_memory_range_t* ranges) {
  loom_runtime_memory_ranges_sort(ranges, range_count);
  loom_runtime_memory_footprint_t footprint = {0};
  iree_host_size_t range_index = 0;
  while (range_index < range_count) {
    const loom_runtime_memory_range_t* range = &ranges[range_index++];
    const iree_device_size_t mapping_size =
        loom_runtime_virtual_buffer_mapping_size(range->buffer);
    iree_device_size_t end = range->offset + range->length;
    while (range_index < range_count &&
           ranges[range_index].buffer == range->buffer &&
           ranges[range_index].offset / mapping_size <=
               (end - 1) / mapping_size + 1) {
      end = iree_max(end,
                     ranges[range_index].offset + ranges[range_index].length);
      ++range_index;
    }
    const loom_runtime_memory_footprint_t part =
        loom_runtime_virtual_buffer_footprint(range->buffer, range->offset,
                                              end - range->offset);
    footprint.total_bytes = loom_runtime_memory_footprint_sum(
        footprint.total_bytes, part.total_bytes);
    footprint.missing_bytes = loom_runtime_memory_footprint_sum(
        footprint.missing_bytes, part.missing_bytes);
  }
  return footprint;
}
