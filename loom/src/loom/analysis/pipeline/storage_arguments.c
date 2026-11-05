// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/storage_arguments.h"

#include <stdlib.h>

#include "loom/util/fact_extensions.h"

static int loom_pipeline_argument_compare(const void* lhs, const void* rhs) {
  const loom_pipeline_storage_argument_t* left =
      *(const loom_pipeline_storage_argument_t* const*)lhs;
  const loom_pipeline_storage_argument_t* right =
      *(const loom_pipeline_storage_argument_t* const*)rhs;
  if (left->root != right->root) {
    return (left->root > right->root) - (left->root < right->root);
  }
  return (int)left->argument_index - (int)right->argument_index;
}

const loom_pipeline_storage_argument_t*
loom_pipeline_storage_arguments_find_buffer(
    const loom_pipeline_storage_arguments_t* arguments, loom_value_id_t root) {
  iree_host_size_t begin = 0, end = arguments->buffer_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (arguments->buffers[middle]->root < root) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < arguments->buffer_count &&
                 arguments->buffers[begin]->root == root
             ? arguments->buffers[begin]
             : NULL;
}

iree_status_t loom_pipeline_storage_arguments_build(
    const loom_module_t* module, const loom_value_id_t* arguments,
    uint16_t argument_count, const loom_value_fact_table_t* facts,
    iree_arena_allocator_t* arena,
    loom_pipeline_storage_arguments_t* out_arguments) {
  *out_arguments = (loom_pipeline_storage_arguments_t){0};
  iree_host_size_t storage_count = 0, buffer_count = 0;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t type = loom_module_value_type(module, arguments[i]);
    buffer_count += loom_type_is_buffer(type);
    storage_count += loom_type_is_buffer(type) || loom_type_is_view(type);
  }
  if (!storage_count) {
    return iree_ok_status();
  }
  loom_pipeline_storage_argument_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, storage_count, sizeof(*storage), (void**)&storage));
  out_arguments->storage = storage;
  const loom_pipeline_storage_argument_t** buffers = NULL;
  if (buffer_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, buffer_count, sizeof(*buffers), (void**)&buffers));
    out_arguments->buffers = buffers;
  }
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_value_id_t value = arguments[i];
    const loom_type_t type = loom_module_value_type(module, value);
    const bool buffer = loom_type_is_buffer(type);
    if (!buffer && !loom_type_is_view(type)) {
      continue;
    }
    const uint32_t index = (uint32_t)out_arguments->storage_count++;
    loom_pipeline_storage_argument_t* selected = &storage[index];
    *selected = (loom_pipeline_storage_argument_t){
        .root = value,
        .byte_offset = loom_value_facts_exact_i64(0),
        .byte_length = loom_value_facts_unknown(),
        .byte_alignment = 1,
        .argument_index = i,
        .buffer = buffer,
    };
    const loom_value_facts_t value_facts =
        loom_value_fact_table_lookup(facts, value);
    if (buffer) {
      loom_value_fact_buffer_reference_t reference;
      if (loom_value_facts_query_buffer_reference(&facts->context, value_facts,
                                                  &reference)) {
        selected->root = loom_value_fact_buffer_reference_resolve_root_value(
            reference, value);
        selected->byte_length = reference.maximum_byte_extent;
        selected->byte_alignment = reference.minimum_alignment;
      }
      buffers[out_arguments->buffer_count++] = selected;
    } else {
      loom_value_fact_view_reference_t reference;
      if (loom_value_facts_query_view_reference(&facts->context, value_facts,
                                                &reference)) {
        selected->root =
            loom_value_fact_view_reference_resolve_root_value(reference, value);
        selected->byte_offset = reference.base_byte_offset;
        selected->byte_length = reference.footprint_byte_length;
        selected->byte_alignment = iree_min(reference.minimum_alignment,
                                            reference.root_minimum_alignment);
      }
    }
  }
  if (buffer_count > 1) {
    qsort(buffers, buffer_count, sizeof(*buffers),
          loom_pipeline_argument_compare);
  }
  return iree_ok_status();
}
