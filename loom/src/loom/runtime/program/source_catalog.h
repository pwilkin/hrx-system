// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_SOURCE_CATALOG_H_
#define LOOM_RUNTIME_PROGRAM_SOURCE_CATALOG_H_

#include "loom/runtime/program/compilation.h"
#include "loom/runtime/program/sources.h"
#include "loom/util/digest.h"
#include "loomc/compile.h"
#include "loomc/link.h"
#include "loomc/target/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// One admitted source universe shared by a deployment's devices. All fields
// except the reference count are immutable after creation; compilation uses
// independent worker workspaces. Target profiles, pass pipelines, device loads,
// and caller reports belong to the consuming JITs, not this catalog.
typedef struct loom_runtime_source_catalog_t {
  // Construction ownership and every consuming JIT retain the catalog.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this catalog and its compiler handles.
  iree_allocator_t host_allocator;
  // Immutable grammar configured from the host's complete target environment.
  loomc_context_t* context;
  // Shared compiler bound to context; each invocation supplies its pipeline.
  loomc_compiler_t* compiler;
  // Frozen source bytes and imported snapshots with stable symbol ordinals.
  loomc_link_index_t* index;
  // Concurrent command compiler and its single native-source retention budget.
  loomc_cmd_compiler_t* command_compiler;
  // Shared immutable linker for direct dispatch roots in index.
  loomc_linker_t* linker;
  // Exact admitted source/provider bytes and immutable selection metadata.
  loom_sha256_t identity;
} loom_runtime_source_catalog_t;

// Opens/imports/indexes the selected providers once. The source namespace and
// report are borrowed only during this call. Diagnostics are appended once to
// report, including on failure. The catalog owns admitted bytes independently
// of later filesystem changes and retains the environment through its context.
iree_status_t loom_runtime_source_catalog_create(
    loomc_target_environment_t* environment,
    const loom_runtime_sources_t* sources,
    loom_runtime_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_runtime_source_catalog_t** out_catalog);

void loom_runtime_source_catalog_retain(loom_runtime_source_catalog_t* catalog);
void loom_runtime_source_catalog_release(
    loom_runtime_source_catalog_t* catalog);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PROGRAM_SOURCE_CATALOG_H_
