// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_PIPELINE_MEMORY_CHANNEL_H_
#define LOOM_TRANSFORMS_PIPELINE_MEMORY_CHANNEL_H_

#include "loom/transforms/pipeline/channel_materialization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Selected memory geometry for one FIFO. Protocol identity and backing identity
// remain separate in the enclosing plan. Every slot has this same view type;
// its dimensions and encoding may refer to SSA values visible at emission.
typedef struct loom_memory_channel_layout_t {
  // Number of simultaneously owned records; positive and representable in the
  // selected ordinal type's signed range and the target index domain. The
  // record ordinal itself uses all unsigned carrier bits. Capacity is
  // independent of epoch length.
  uint64_t capacity;
  // First slot's byte displacement from its payload backing.
  uint64_t byte_offset;
  // Byte displacement between consecutive slots. Record views may be strided;
  // their payload need not occupy the complete envelope between slot origins.
  uint64_t byte_stride;
  // Ordinary borrowed view produced by reserve, acquire, and wait.
  loom_type_t view_type;
} loom_memory_channel_layout_t;

typedef struct loom_memory_channel_endpoint_t {
  // Jointly selected layout, retained by the enclosing physical plan.
  const loom_memory_channel_layout_t* layout;
  // Ordinary buffer SSA value supplying this endpoint's payload backing.
  loom_value_id_t buffer;
  // Private read cursor in channel_materialization's threaded state.
  iree_host_size_t read_state;
  // Private write cursor in channel_materialization's threaded state.
  iree_host_size_t write_state;
} loom_memory_channel_endpoint_t;

typedef enum loom_memory_channel_notification_e {
  // Wait for the preceding consumer generation before reserving a slot.
  LOOM_MEMORY_CHANNEL_WAIT_FREE = 0,
  // Wait for the producer generation before accessing initialized payload.
  LOOM_MEMORY_CHANNEL_WAIT_READY,
  // Publish the producer generation after writes and required visibility.
  LOOM_MEMORY_CHANNEL_PUBLISH,
  // Return the consumer generation after the final source read completes.
  LOOM_MEMORY_CHANNEL_RELEASE,
} loom_memory_channel_notification_t;

// Target-selected notification and visibility mechanics. These may be ordinary
// calls, atomics, or transport instructions. A cooperative worker performs one
// protocol action for the group; its implementation owns lane participation.
typedef struct loom_memory_channel_callback_t {
  // Emits the selected action for this record. The callback owns completion
  // ordering, visibility, and any proven notification grouping. In particular,
  // a release must not reclaim a record with outstanding producer writes or
  // fanout obligations, even when accept has not yet waited for readiness.
  // A wait completes before returning to payload emission. No host handshake
  // or worker restart is implied. Status carries emission/allocation failures.
  iree_status_t (*notify)(void* user_data, loom_rewriter_t* rewriter,
                          const loom_channel_plan_action_t* action,
                          const loom_memory_channel_endpoint_t* endpoint,
                          loom_value_id_t record,
                          loom_memory_channel_notification_t notification);
  // Selected asynchronous transport and non-memory admission mechanics. The
  // enclosing selection supplies this for copy, select, bind, and calls when
  // they occur; it owns their results and obligations. In particular, copy is
  // not converted into a synchronous wait by the memory FIFO emitter.
  iree_status_t (*action)(void* user_data, loom_rewriter_t* rewriter,
                          const loom_channel_plan_action_t* action,
                          loom_value_id_t* state, iree_host_size_t state_count,
                          loom_value_id_t* results);
  // Borrowed physical plan and target emission state.
  void* user_data;
} loom_memory_channel_callback_t;

typedef struct loom_memory_channel_materialization_t {
  // Retained plan whose action ordinals index endpoints. A cloned plan keeps
  // those ordinals, independently of the source operation and SSA identities.
  const loom_channel_plan_t* plan;
  // Selected endpoint for each memory action. Entries delegated to action need
  // no endpoint. Several source actions can share the same endpoint/cursors.
  const loom_memory_channel_endpoint_t* const* endpoints;
  // Physical notification and asynchronous transport mechanics.
  loom_memory_channel_callback_t emit;
} loom_memory_channel_materialization_t;

// Builds one endpoint's counter address and expected generation. Each slot
// owns a counter: generation = record / capacity, plus one except WAIT_FREE.
// The selected control backing has ready and free planes with caller-supplied
// offsets and slot stride. Initial cursors and counters agree on their starting
// generation; resuming with retained state requires no reset. Selection bounds
// execution before unsigned ordinal/generation overflow. Addresses do not
// identify either the protocol or the owned record.
iree_status_t loom_memory_channel_notification_build(
    loom_builder_t* builder, const loom_memory_channel_layout_t* layout,
    loom_value_id_t record, loom_memory_channel_notification_t notification,
    uint64_t ready_byte_offset, uint64_t free_byte_offset,
    uint64_t slot_byte_stride, loom_location_id_t location,
    loom_value_id_t* out_byte_offset, loom_value_id_t* out_generation);

// Callback for channel_materialize. Physical selection establishes memory
// geometry, cursor ownership, carrier width, and notification safety before
// this consuming rewrite. Read/write carriers are unsigned record ordinals;
// channel carriers are erased by the enclosing callable realization. Accept
// advances only the private cursor; wait preserves ownership; fanout preserves
// the record ordinal while the selected completion protocol owns retirement.
// Loops retain their ordinary CFG and thread cursors through SSA, independent
// of how many cohorts execute during one resident epoch.
iree_status_t loom_memory_channel_materialize_action(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_channel_plan_action_t* action, loom_value_id_t* state,
    iree_host_size_t state_count, loom_value_id_t* results);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_PIPELINE_MEMORY_CHANNEL_H_
