// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXTERNAL_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXTERNAL_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

// Immutable externally relocated descriptor. Its binding selects storage;
// protocol identity and reclamation stay with the enclosing input plan.
typedef struct loom_aie2p_native_external_descriptor_t {
  // Next immutable descriptor requiring invocation-time relocation.
  struct loom_aie2p_native_external_descriptor_t* next;
  // Shim owning this descriptor.
  const loom_aie2p_native_tile_t* tile;
  // Index in that shim's descriptor bank.
  uint16_t descriptor;
  // Published entry buffer ordinal.
  uint32_t binding;
  // First accessed byte within the selected buffer.
  uint64_t byte_offset;
  // Accessible envelope beginning at byte_offset.
  uint64_t byte_length;
  // Native immutable geometry, excluding the relocated address pair.
  uint32_t words[8];
} loom_aie2p_native_external_descriptor_t;

typedef struct loom_aie2p_native_input_phase_t {
  // Retained source phase; source loops are never expanded into descriptor
  // lists.
  const loom_pipeline_transport_input_phase_t* source;
  // Local destination with its independently owned record credits.
  const loom_aie2p_native_channel_t* destination;
  // Resident local DMA receiver and its startup window.
  loom_aie2p_native_distribution_endpoint_t receiver;
  // First shim descriptor for this phase. Prefixes iterate over source slots;
  // the final repeated phase has one immutable descriptor per source slot.
  uint16_t descriptor;
  // Shim executions for one complete source address period.
  uint32_t repetitions;
  // Physical source geometry before framing and binding selection.
  uint32_t words[8];
} loom_aie2p_native_input_phase_t;

typedef struct loom_aie2p_native_input_t {
  // Communication-only source worker.
  uint32_t worker;
  // Resident invocation owning receiver startup and source completion.
  const loom_pipeline_invocation_t* invocation;
  // Ancestor-owned external read cursor, or UINT32_MAX for invocation-local
  // source state. Its private frame cell stores the absolute read ordinal.
  uint32_t cursor_index;
  // Shared parent-selected payload and notification binding, or NULL when
  // common analysis proves enclosing caller storage stable for the invocation.
  const loom_memory_channel_binding_t* binding;
  // Complete lease and phase counts from common transport analysis.
  const loom_pipeline_transport_input_t* source;
  // Common exact observation sites in an existing arithmetic worker.
  loom_pipeline_continuation_t continuation;
  // Native receiver and source descriptor selection per phase.
  loom_aie2p_native_input_phase_t* phases;
  // All phase receivers share one ordered physical queue and payload route.
  bool shared_receiver;
  // Physical shim serving payload, readiness queries, and credit return.
  loom_aie2p_native_tile_t* shim;
  // One memory-to-stream engine shared by queries and admitted input phases.
  uint8_t input_engine;
  // Stream-to-memory engine returning the external source credit.
  uint8_t credit_engine;
  // First per-slot ready-query descriptor.
  uint16_t ready_descriptor;
  // First per-slot credit-return descriptor.
  uint16_t credit_descriptor;
  // Selected core-to-shim control packet class.
  uint8_t control_packet;
  // Selected core-to-shim credit payload class.
  uint8_t credit_packet;
  // Shim completion reply used only for the final source credit in the epoch.
  uint8_t completion_packet;
  // Tile-relative input task publication register.
  uint32_t input_queue;
  // Tile-relative credit task publication register.
  uint32_t credit_queue;
  // First private SSA value. Shared leases retain slot, generation, and bulk
  // position; stable enclosing storage retains only bulk position. An inherited
  // external cursor adds its terminal absolute read ordinal for frame
  // writeback. Assigned during worker materialization.
  iree_host_size_t state_index;
  // Generated admission and arrival helpers in the observing worker's context.
  loom_symbol_ref_t begin;
  // Child-entry receiver publication; root receivers start in configuration.
  loom_symbol_ref_t start;
  // Generated source reclamation helper in that same context.
  loom_symbol_ref_t complete;
  // Next input whose continuation runs on the same existing core.
  struct loom_aie2p_native_input_t* next;
} loom_aie2p_native_input_t;

// Admits native borrowed input mechanics before assigning instruction engines.
// No core or shadow payload allocation is introduced. False eligibility keeps
// ordinary execution selection; authored unsupported external use is diagnosed
// there before mutation.
iree_status_t loom_aie2p_native_select_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, iree_host_size_t worker,
    bool* out_eligible);

// Allocates native descriptors/routes from the selected common continuation.
iree_status_t loom_aie2p_native_route_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    loom_aie2p_native_input_t* input, uint32_t* next_source,
    uint8_t* next_controller, bool* out_valid);

// Emits ordinary helpers and consuming channel actions for selected inputs.
iree_status_t loom_aie2p_native_prepare_input(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input);
static inline iree_host_size_t loom_aie2p_native_input_state_count(
    const loom_aie2p_native_input_t* input) {
  return (input->binding ? 3 : 1) + (input->cursor_index != UINT32_MAX);
}
// Starts child receivers and initializes source state from its owning frame.
iree_status_t loom_aie2p_native_initialize_input(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_input_t* input, loom_value_id_t* frame_argument,
    loom_value_id_t* state, loom_value_id_t* state_views);
iree_status_t loom_aie2p_native_emit_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    iree_host_size_t action, bool after, loom_value_id_t* state);

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXTERNAL_H_
