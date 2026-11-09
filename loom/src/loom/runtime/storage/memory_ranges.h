// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_MEMORY_RANGES_H_
#define LOOM_RUNTIME_STORAGE_MEMORY_RANGES_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct loom_runtime_virtual_buffer_t loom_runtime_virtual_buffer_t;

// One caller-validated nonempty extent in a virtual reservation. Every range
// in a commitment transaction belongs to the same physical pool.
typedef struct loom_runtime_memory_range_t {
  // Borrowed reservation whose physical backing is required.
  loom_runtime_virtual_buffer_t* buffer;
  // Byte offset within the reservation.
  iree_device_size_t offset;
  // Number of accessible bytes required from offset.
  iree_device_size_t length;
} loom_runtime_memory_range_t;

typedef struct loom_runtime_memory_footprint_t {
  // Full physical extent of the range union, including unbacked slabs.
  uint64_t total_bytes;
  // Additional commitment, excluding live or already accepted backing.
  uint64_t missing_bytes;
} loom_runtime_memory_footprint_t;

// Physical footprint of caller-validated nonempty ranges. Orders the caller's
// array in place by buffer and offset; all original ranges remain intact.
// Overlapping mapping units count once. Runs on the pool's serialized owner
// without allocating, changing backing metadata, or collecting native progress.
// Sums saturate at uint64_t. Pending backing counts until actual retirement.
loom_runtime_memory_footprint_t loom_runtime_memory_ranges_footprint(
    iree_host_size_t range_count, loom_runtime_memory_range_t* ranges);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_RUNTIME_STORAGE_MEMORY_RANGES_H_
