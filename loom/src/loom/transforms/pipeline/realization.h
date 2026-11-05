// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_PIPELINE_REALIZATION_H_
#define LOOM_TRANSFORMS_PIPELINE_REALIZATION_H_

#include "loom/analysis/pipeline/invocations.h"
#include "loom/ir/symbol_map.h"
#include "loom/pass/types.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/transforms/pipeline/worker_materialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Explicit output boundary for one occurrence. A new symbol leaves the source
// definition available to other occurrences; the original symbol replaces it.
typedef struct loom_pipeline_realization_entry_t {
  // Destination selected by the enclosing execution owner.
  loom_symbol_ref_t symbol;
  // Source location of the construction being realized.
  loom_location_id_t location;
  // Public/private contract inherited from the source construction.
  uint8_t visibility;
  // Whether the destination is an explicitly retained artifact entry.
  bool retain;
  // Exact mutable version of this output, independent of the active pass frame.
  loom_target_function_version_t* version;
} loom_pipeline_realization_entry_t;

// Shared generated-name ownership for one complete realization batch. Target
// emitters add their ordinary helper symbols here instead of taking private
// module snapshots that would miss another occurrence's generated definitions.
typedef struct loom_pipeline_realization_symbols_t {
  // Existing and newly allocated names, indexed by interned string identity.
  loom_symbol_map_t names;
  // Next helper suffix across every occurrence in the batch.
  uint32_t next_helper;
} loom_pipeline_realization_symbols_t;

// Complete immutable source occurrence presented to physical selection.
// Facts, channel identity and movement plans are established exactly once.
// Selection observes the source; emission consumes independent instances with
// their retained identities projected by the common cloning machinery.
typedef struct loom_pipeline_realization_t {
  // Construction whose public invocation contract is being implemented.
  loom_func_like_t function;
  // Selected output boundary; source identity is carried separately above.
  loom_pipeline_realization_entry_t entry;
  // Shared namespace, valid throughout worker and entry emission.
  loom_pipeline_realization_symbols_t* symbols;
  // Immutable facts for the construction's captures and allocations.
  const loom_value_fact_table_t* facts;
  // Construction-owned resources, independent of physical storage addresses.
  loom_pipeline_resources_t resources;
  // Complete static inventory, including bound children and retained control.
  loom_pipeline_invocations_t invocations;
  // Bound worker instances, in complete invocations.workers order.
  loom_pipeline_worker_t* workers;
  // Independent consuming bodies, created only after selection succeeds.
  loom_pipeline_worker_materialization_t* instances;
  // Compilation owner receiving workers and their generated helper versions.
  loom_function_version_owner_t* version_owner;
  // Source version handles during selection, independent worker versions
  // during emission, in complete invocation worker order.
  loom_target_function_version_t** worker_versions;
  // Fresh context ordinal reserved for a changed entry execution contract.
  loom_target_context_ordinal_t entry_target_context;
} loom_pipeline_realization_t;

typedef struct loom_pipeline_realization_callback_t {
  // Admits target resources and chooses mechanics from canonical source plans.
  // No IR mutation occurs here. Authored support failures are diagnostics and
  // leave out_valid false; status carries allocation and emission failures.
  iree_status_t (*select)(void* user_data, loom_module_t* module,
                          const loom_pipeline_realization_t* realization,
                          bool* out_valid);
  // Consumes one admitted worker instance. Its ordinal domain is acquired;
  // the immutable source worker's domain remains released.
  // The callback consumes channel/movement plans and emits ordinary executable
  // operations; it does not rediscover the worker body or compile it privately.
  // Original entry arguments remain in order until common capture realization
  // and pruning; mechanics may append invocation operands to that signature.
  iree_status_t (*worker)(void* user_data, loom_rewriter_t* rewriter,
                          const loom_pipeline_realization_t* realization,
                          iree_host_size_t worker_index);
  // Projects retained construction control after every worker ABI is final.
  // All source definitions remain intact through every request's preparation.
  // NULL means that emission consumes no construction bodies. Unsupported
  // authored control leaves out_valid false after emitting its diagnostic.
  iree_status_t (*prepare)(void* user_data, loom_rewriter_t* rewriter,
                           const loom_pipeline_realization_t* realization,
                           bool* out_valid);
  // Emits the execution container at entry.symbol and returns its definition.
  // Common code owns source retirement and version registration. The callback
  // updates entry.version's target facts when the execution contract changes;
  // it neither erases the source nor assumes pass.function_version is this
  // entry.
  iree_status_t (*entry)(void* user_data, loom_rewriter_t* rewriter,
                         const loom_pipeline_realization_t* realization,
                         loom_func_like_t* out_function);
  // Borrowed physical selection and emission state.
  void* user_data;
} loom_pipeline_realization_callback_t;

// One output requested from a retained occurrence. All requests are selected
// before any worker is cloned or source definition is retired.
typedef struct loom_pipeline_realization_request_t {
  // Immutable source occurrence; its enclosing source proofs are closed.
  const loom_pipeline_invocations_t* invocations;
  // Immutable worker analyses in this inventory's complete worker order.
  loom_pipeline_worker_t* workers;
  // Distinct destination, or the occurrence's original symbol for replacement.
  loom_symbol_ref_t symbol;
  // Target mechanics and their independently owned selection state.
  loom_pipeline_realization_callback_t callback;
} loom_pipeline_realization_request_t;

// Realizes a complete admitted set with one module/version observation and one
// generated-symbol namespace. Repeated source definitions may produce distinct
// entries; their source IR remains intact unless a request replaces its symbol.
// Every worker is consumed before any replaced construction is retired. Target
// mechanics consume retained analyses without source rediscovery. Authored
// rejection is diagnosed before mutation and leaves out_valid false.
iree_status_t loom_pipeline_realize_occurrences(
    loom_pass_t* pass, loom_module_t* module,
    const loom_pipeline_realization_request_t* requests,
    iree_host_size_t request_count, bool* out_valid);

// Registers an ordinary helper in its calling worker's exact execution context.
// The helper owns its own function contract; it inherits no entry retention,
// source schedules or memory proofs. Normal IR projection materializes the
// shared context at a saved compilation boundary.
iree_status_t loom_pipeline_realization_register_helper(
    loom_module_t* module, const loom_pipeline_realization_t* realization,
    iree_host_size_t worker_index, loom_func_like_t helper,
    iree_diagnostic_emitter_t diagnostic_emitter);

// Consumes an already analyzed occurrence within its enclosing execution.
// The caller has closed source-stability proofs over all concurrent children.
// Source definitions stay immutable during selection and worker emission;
// the common owner then replaces the admitted execution boundary. No resource,
// channel, movement, or source-stability analysis is repeated here.
iree_status_t loom_pipeline_realize_invocations(
    loom_pass_t* pass, loom_module_t* module,
    const loom_pipeline_invocations_t* invocations,
    loom_pipeline_worker_t* workers,
    const loom_pipeline_realization_callback_t* callback);

// Realizes one specialized pipeline after lexical loop preparation/outlining
// and before ordinary target memory legalization. Common code owns resource
// discovery, occurrence cloning, capture binding, analysis lifetime and worker
// version registration. The physical owner supplies an inventory and mechanics.
// Non-pipeline functions never enter this lifecycle. Generated workers resume
// in the existing compiler pipeline; this function starts no nested compiler.
iree_status_t loom_pipeline_realize(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    const loom_pipeline_resource_inventory_t* inventory,
    const loom_pipeline_realization_callback_t* callback);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_PIPELINE_REALIZATION_H_
