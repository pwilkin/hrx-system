// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_STORAGE_ARGUMENTS_H_
#define LOOM_ANALYSIS_PIPELINE_STORAGE_ARGUMENTS_H_

#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// Storage supplied by one source argument. These are logical projections,
// before an execution entry chooses its physical resource ABI. Channel
// identities and scalar captures are not storage arguments.
typedef struct loom_pipeline_storage_argument_t {
  // Canonical storage root in the enclosing execution, possibly in a parent.
  loom_value_id_t root;
  // Origin of the argument's accessible span relative to root.
  loom_value_facts_t byte_offset;
  // Accessible extent; unresolved sizes retain their inferred bounds.
  loom_value_facts_t byte_length;
  // Minimum alignment of the argument's accessible origin.
  uint64_t byte_alignment;
  // Original formal argument position, independent of the physical ABI.
  uint16_t argument_index;
  // A buffer exposes the complete root. A view exposes only its own span.
  bool buffer;
} loom_pipeline_storage_argument_t;

typedef struct loom_pipeline_storage_arguments_t {
  // Arena-owned projections in source argument order, including aliases.
  const loom_pipeline_storage_argument_t* storage;
  // Number of buffer and view arguments in storage.
  iree_host_size_t storage_count;
  // Complete buffer arguments indexed by canonical root, then source position.
  // Views never grant whole-root access. Rows remain in source order above.
  const loom_pipeline_storage_argument_t* const* buffers;
  // Number of complete buffer arguments, including aliases.
  iree_host_size_t buffer_count;
} loom_pipeline_storage_arguments_t;

// Projects already-inferred argument facts without walking their producers.
// Caller facts resolve parent captures before this boundary. Unknown extents
// remain unknown; target selection decides whether specialization is required.
// The source fact context outlives the result. No IR or argument order changes.
// Status carries arena allocation failure.
iree_status_t loom_pipeline_storage_arguments_build(
    const loom_module_t* module, const loom_value_id_t* arguments,
    uint16_t argument_count, const loom_value_fact_table_t* facts,
    iree_arena_allocator_t* arena,
    loom_pipeline_storage_arguments_t* out_arguments);

// Finds an argument exposing a complete root, or NULL when none does. Equal
// roots are storage aliases, not equal channel identities or synchronization.
const loom_pipeline_storage_argument_t*
loom_pipeline_storage_arguments_find_buffer(
    const loom_pipeline_storage_arguments_t* arguments, loom_value_id_t root);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_STORAGE_ARGUMENTS_H_
