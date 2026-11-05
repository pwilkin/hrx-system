// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_NOTIFICATIONS_H_
#define LOOM_ANALYSIS_PIPELINE_NOTIFICATIONS_H_

#include "loom/analysis/pipeline/invocations.h"

#ifdef __cplusplus
extern "C" {
#endif

// One complete consumer interval over independently owned FIFO records.
// A shared publication may make every member available together, and the last
// release may return their shared physical notification. The source reads and
// their storage lifetimes remain distinct.
typedef struct loom_pipeline_notification_interval_t {
  // Next source interval in this consumer's action order.
  struct loom_pipeline_notification_interval_t* next;
  // First acquire in the consumer's channel action table.
  iree_host_size_t first_action;
  // Last release in that same table, inclusive.
  iree_host_size_t last_action;
  // Exact visits to this interval during the resident invocation.
  uint64_t executions;
} loom_pipeline_notification_interval_t;

// Optional notification coalescing proof spanning both ends of a channel set.
// All destinations are reserved before any source is accepted or copied. Their
// sole consumer visits every member once before another channel action. A
// publication after the complete copy group and reclamation after the complete
// consumer interval therefore introduce no dependency on an earlier credit.
typedef struct loom_pipeline_notification_group_t {
  // Communication-only worker owning all destination reservations/copies.
  const loom_pipeline_worker_t* producer;
  // Sole logical consumer; cooperative participants share this cursor owner.
  const loom_pipeline_worker_t* consumer;
  // Canonical channel-set indices in producer copy order.
  const uint32_t* channels;
  // Number of independently owned records in each group.
  iree_host_size_t channel_count;
  // Exact records per member throughout this resident invocation.
  uint64_t admissions;
  // Consumer intervals, whose member order may differ from producer order.
  const loom_pipeline_notification_interval_t* intervals;
} loom_pipeline_notification_group_t;

typedef struct loom_pipeline_notification_plan_t {
  // Selected groups; source workers and channel identities remain immutable.
  const loom_pipeline_notification_group_t* groups;
  // Number of complete proofs in groups.
  iree_host_size_t group_count;
  // Group ordinal by input channel-set index; UINT32_MAX means independent.
  // NULL when the input channel set is empty.
  const uint32_t* channel_groups;
} loom_pipeline_notification_plan_t;

// Selects complete reserve/accept/copy groups from retained source plans over
// every concurrent worker occurrence. channels contains the protocol instances
// eligible for shared physical notifications; other channel actions still
// delimit consumer intervals. Call before either endpoint is consumed.
//
// Each member has one logical producer/consumer, equal capacity, exact balanced
// counts, and complete ownership intervals. Extra owners, opaque channel calls,
// unknown counts, or interposed channel actions leave notifications
// independent. Source address equality plays no part in this proof. Selection
// neither assigns control storage nor establishes payload visibility or DMA
// completion.
//
// Cost depends on source channel/action counts, never runtime record visits.
// Status carries arena allocation failure; absence of a proof is not an error.
iree_status_t loom_pipeline_notifications_analyze(
    const loom_pipeline_invocations_t* channels,
    loom_pipeline_worker_t* workers, iree_arena_allocator_t* arena,
    loom_pipeline_notification_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_NOTIFICATIONS_H_
