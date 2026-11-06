// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_STORAGE_MEMORY_MAPPINGS_H_
#define LOOM_RUNTIME_STORAGE_MEMORY_MAPPINGS_H_

#include "iree/hal/virtual_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Orders producer-owned, nonoverlapping mapping descriptors by target and
// offset, then joins contiguous spans of the same source and prior backing.
// Returns the populated prefix length. Source acquisition must have completed;
// the caller retains all independent reservation tokens until retirement.
// Only disposable descriptors change: no allocation or ownership transfer.
iree_host_size_t loom_runtime_memory_mappings_coalesce(
    iree_host_size_t count, iree_hal_memory_mapping_t* mappings);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_RUNTIME_STORAGE_MEMORY_MAPPINGS_H_
