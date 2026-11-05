// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/source_catalog.h"

#include "loom/runtime/program/source_index.h"
#include "loomc/iree.h"

// The source index identifies admitted bytes and provider contracts. Selection
// metadata can also affect command/VM products without changing those bytes.
static void loom_runtime_source_catalog_identity(
    const loom_runtime_sources_t* sources, const loom_sha256_t* source_identity,
    loom_sha256_t* out_identity) {
  loom_sha256_state_t identity;
  loom_sha256_initialize(&identity);
  loom_sha256_update(&identity,
                     iree_make_const_byte_span(source_identity->bytes,
                                               sizeof(source_identity->bytes)));
  const iree_string_view_t fields[] = {
      sources->card ? loom_model_card_selection(sources->card)->config
                    : iree_string_view_empty(),
      sources->component ? sources->component->export_prefix
                         : iree_string_view_empty(),
  };
  uint8_t encoded[8];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(fields); ++i) {
    iree_unaligned_store_le_u64(encoded, fields[i].size);
    loom_sha256_update(&identity, iree_make_const_byte_span(encoded, 8));
    loom_sha256_update(
        &identity, iree_make_const_byte_span(fields[i].data, fields[i].size));
  }
  const uint64_t counts[] = {
      sources->count,
      sources->card ? loom_model_card_selection(sources->card)->inputs.count
                    : 0,
      sources->card ? loom_model_card_selection(sources->card)->libraries.count
                    : 0,
      sources->component ? sources->component->version : 0,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(counts); ++i) {
    iree_unaligned_store_le_u64(encoded, counts[i]);
    loom_sha256_update(&identity, iree_make_const_byte_span(encoded, 8));
  }
  loom_sha256_finalize(&identity, out_identity);
}

iree_status_t loom_runtime_source_catalog_create(
    loomc_target_environment_t* environment,
    const loom_runtime_sources_t* sources,
    loom_runtime_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_runtime_source_catalog_t** out_catalog) {
  *out_catalog = NULL;
  if (!environment) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "serving host has no compiler target configuration");
  }
  loom_runtime_source_catalog_t* catalog = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*catalog),
                                             (void**)&catalog));
  iree_atomic_ref_count_init(&catalog->ref_count);
  catalog->host_allocator = host_allocator;
  const loomc_allocator_t ca = loomc_allocator_from_iree(host_allocator);
  const loomc_context_target_options_t targets = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .target_environment = environment,
  };
  const loomc_context_options_t options = {.next = &targets};
  iree_status_t status = iree_status_from_loomc(
      loomc_context_create(&options, ca, &catalog->context));
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_compiler_create(catalog->context, NULL, ca, &catalog->compiler));
  }
  loom_sha256_t source_identity;
  if (iree_status_is_ok(status)) {
    status = loom_runtime_source_index_create(
        catalog->context, sources, NULL, report, host_allocator,
        &source_identity, &catalog->index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(loomc_cmd_compiler_create(
        catalog->index, NULL, ca, &catalog->command_compiler));
  }
  if (iree_status_is_ok(status)) {
    status = iree_status_from_loomc(
        loomc_linker_create(catalog->context, NULL, ca, &catalog->linker));
  }
  if (iree_status_is_ok(status)) {
    loom_runtime_source_catalog_identity(sources, &source_identity,
                                         &catalog->identity);
    *out_catalog = catalog;
  } else {
    loom_runtime_source_catalog_release(catalog);
  }
  return status;
}

void loom_runtime_source_catalog_retain(
    loom_runtime_source_catalog_t* catalog) {
  if (catalog) {
    iree_atomic_ref_count_inc(&catalog->ref_count);
  }
}

void loom_runtime_source_catalog_release(
    loom_runtime_source_catalog_t* catalog) {
  if (!catalog) {
    return;
  }
  if (iree_atomic_ref_count_dec(&catalog->ref_count) != 1) {
    return;
  }
  loomc_cmd_compiler_release(catalog->command_compiler);
  loomc_linker_release(catalog->linker);
  loomc_link_index_release(catalog->index);
  loomc_compiler_release(catalog->compiler);
  loomc_context_release(catalog->context);
  iree_allocator_free(catalog->host_allocator, catalog);
}
