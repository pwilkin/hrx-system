// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXTERNAL_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXTERNAL_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Externally relocated descriptor. Its binding selects storage;
// protocol identity and reclamation stay with the enclosing input plan.
typedef struct loom_aie2p_native_external_descriptor_t {
  // Next descriptor requiring invocation-time relocation.
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
  // Initial native geometry, excluding the relocated address pair.
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
  // the bulk phase either iterates slots or selects a descriptor per slot.
  uint16_t descriptor;
  // Shim executions for one complete source address period.
  uint32_t repetitions;
  // Physical source geometry before framing and binding selection.
  uint32_t words[8];
} loom_aie2p_native_input_phase_t;

// Physical descriptor realization for external ring slots. Prefix descriptors
// already iterate over slots independently of this payload/notification choice.
typedef enum loom_aie2p_native_input_slots_e {
  // One immutable payload, ready, and credit descriptor per logical slot.
  LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE,
  // Shared ready and credit descriptors; payload geometry retains per-slot BDs.
  LOOM_AIE2P_NATIVE_INPUT_SLOTS_NOTIFICATIONS,
  // Payload, ready, and credit each use one slot-iterating descriptor.
  // Replayed payload periods explicitly restore the current lease's slot.
  LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD,
} loom_aie2p_native_input_slots_t;

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
  // Existing instruction worker owning admission and source completion.
  uint32_t controller;
  // Exact demand observations, or NULL for a complete immutable input whose
  // single source admission belongs to the controller's invocation lifetime.
  const loom_pipeline_continuation_t* continuation;
  // Native receiver and source descriptor selection per phase.
  loom_aie2p_native_input_phase_t* phases;
  // All phase receivers share one ordered physical queue and payload route.
  bool shared_receiver;
  // Selected ring realization. Shared notifications select readiness only
  // after queue retirement and await each returned credit.
  loom_aie2p_native_input_slots_t slots;
  // Physical shim serving payload, readiness queries, and credit return.
  loom_aie2p_native_tile_t* shim;
  // One memory-to-stream engine shared by queries and admitted input phases.
  uint8_t input_engine;
  // Stream-to-memory engine returning the external source credit.
  uint8_t credit_engine;
  // First ready-query descriptor, or the sole descriptor for iterated slots.
  uint16_t ready_descriptor;
  // First credit-return descriptor, or the sole descriptor for iterated slots.
  uint16_t credit_descriptor;
  // Selected core-to-shim control packet class.
  uint8_t control_packet;
  // Selected core-to-shim credit payload class.
  uint8_t credit_packet;
  // Shim completion reply for terminal credits or each iterated-slot credit.
  uint8_t completion_packet;
  // Tile-relative input task publication register.
  uint32_t input_queue;
  // Tile-relative credit task publication register.
  uint32_t credit_queue;
  // Tile-relative payload iteration register for compact replayed periods.
  uint32_t payload_iteration_register;
  // Tile-relative ready descriptor iteration register for iterated slots.
  uint32_t ready_iteration_register;
  // Ready descriptor iteration stride and wrap, excluding the selected slot.
  uint32_t ready_iteration;
  // First private SSA value. Shared leases retain slot, generation, and bulk
  // position; stable enclosing storage retains only bulk position. An inherited
  // external cursor adds its terminal absolute read ordinal for frame
  // writeback. Assigned during worker materialization.
  iree_host_size_t state_index;
  // Generated source admission helper in the controller's context.
  loom_symbol_ref_t begin;
  // Child-entry receiver publication; root receivers start in configuration.
  loom_symbol_ref_t start;
  // Shared source reclamation or invocation-owned queue retirement helper.
  loom_symbol_ref_t complete;
  // Next input controlled by the same existing core.
  struct loom_aie2p_native_input_t* next;
} loom_aie2p_native_input_t;

// Whether two selected input continuations can advance the same external
// read cursor. Controller selection and common execution admission precede
// this query. Both inputs must inherit their controllers' disjoint lifetimes
// on one core; borrowing the same controller does not order two input leases.
bool loom_aie2p_native_input_cursors_compatible(
    const loom_aie2p_native_context_t* context,
    const loom_pipeline_invocations_t* invocations,
    const loom_aie2p_native_input_t* first,
    const loom_aie2p_native_input_t* second);

// Admits native phased input mechanics before assigning instruction engines.
// No core or shadow payload allocation is introduced. False eligibility keeps
// ordinary execution selection; authored unsupported external use is diagnosed
// there before mutation.
iree_status_t loom_aie2p_native_select_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, iree_host_size_t worker,
    bool* out_eligible);

// Selects slot addressing after source phase geometry has been encoded.
// The result reserves one complete descriptor slice. False means this valid
// channel geometry cannot fit the supplied native descriptor/iteration budget;
// it does not change any physical allocation cursors.
bool loom_aie2p_native_select_input_slots(const loom_xdna_dma_facts_t* dma,
                                          uint16_t available_descriptors,
                                          loom_aie2p_native_input_t* input,
                                          uint16_t* out_descriptor_count);

// Allocates native descriptors and routes from the selected controller and
// optional demand continuation.
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
  return input->continuation
             ? (input->binding ? 3 : 1) + (input->cursor_index != UINT32_MAX)
             : 0;
}
// Starts child receivers and initializes source state from its owning frame.
iree_status_t loom_aie2p_native_initialize_input(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_input_t* input, loom_value_id_t* frame_argument,
    loom_value_id_t* state, loom_value_id_t* state_views);
iree_status_t loom_aie2p_native_emit_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    iree_host_size_t action, bool after, loom_value_id_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_EXTERNAL_H_
