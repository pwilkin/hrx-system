// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_PROGRESS_H_
#define LOOM_ANALYSIS_PIPELINE_PROGRESS_H_

#include "loom/analysis/pipeline/invocations.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact endpoint counts admitted by the physical execution owner. Entries use
// the canonical invocation inventory's channel ordinals. Zero means no
// equal-record observation is admitted in that direction, including
// retained-batch replay protocols.
typedef struct loom_pipeline_progress_endpoint_t {
  // Complete read admissions during this resident execution.
  uint64_t reads;
  // Complete write admissions during this resident execution.
  uint64_t writes;
  // The physical reader can run a continuation after observing an admission.
  // Such an observation does not retire the record or its fanout obligations.
  bool observer;
} loom_pipeline_progress_endpoint_t;

typedef struct loom_pipeline_progress_callback_t {
  // Physical cost of observing source progress at an eligible reader. Both
  // arguments are canonical channel ordinals, independent of storage address.
  // Equal costs prefer the lower reader ordinal for deterministic selection.
  uint64_t (*cost)(void* user_data, uint32_t source, uint32_t reader);
  // Borrowed physical placement and continuation capabilities.
  void* user_data;
} loom_pipeline_progress_callback_t;

typedef enum loom_pipeline_progress_observation_kind_e {
  // Each reader admission observes the corresponding source record.
  LOOM_PIPELINE_PROGRESS_OBSERVATION_RECORD = 0,
  // The first reader admission observes the complete finite source batch.
  // It does not retire the source's retained reads or fanout obligations.
  LOOM_PIPELINE_PROGRESS_OBSERVATION_COMPLETE_BATCH,
} loom_pipeline_progress_observation_kind_t;

typedef struct loom_pipeline_progress_observation_t {
  // Canonical channel whose existing reader supplies the observation.
  uint32_t channel;
  // Relationship between that reader's arrivals and source publication.
  loom_pipeline_progress_observation_kind_t kind;
} loom_pipeline_progress_observation_t;

// Complete eligible observations of one channel's publication. Joint
// continuation selection intersects endpoint owners across phases while keeping
// each phase's actual observation channel; equal counts are not identities.
typedef struct loom_pipeline_progress_observations_t {
  // Observations sorted by channel and kind, unique and arena-owned.
  const loom_pipeline_progress_observation_t* values;
  // Number of eligible observations; zero for an unproved or cyclic path.
  iree_host_size_t count;
} loom_pipeline_progress_observations_t;

// Selects existing readers whose admissions prove progress through equal-record
// channel copies. A source ordinal maps to one selected observer ordinal, or
// UINT32_MAX when no observation is established. The result belongs to arena.
// Workers and channels use the complete invocation inventory order.
//
// Each node retains the cheapest observer offered by its immediate downstream
// nodes. This is a bounded greedy selection, not a global minimum over paths.
// Cycles require a record-distance proof and remain unselected. Work and memory
// scale with source channel/copy sites, never with runtime record counts.
// Selection consumes retained plans without walking source IR or inferring
// facts. Physical owners supply endpoint eligibility, placement costs, and any
// mechanism that consumes the resulting progress; no worker is launched here.
// Ordinary copies from multiple phases of the same worker compose when their
// combined coverage accounts for every source read and destination write.
// Different source identities and partial coverage never establish an edge.
//
// observation_sources requests complete eligible populations for joint
// selection. The same graph owner computes each requested source once; no
// consumer reconstructs copy edges or traverses source IR. Nonempty requests
// require out_observations, which receives channel-indexed rows with empty rows
// for unrequested sources. A zero request count allocates no population state.
// Work follows each requested source's reachable graph, with linear scratch
// and retained sparse results; epoch length does not expand either. In
// particular, intermediate channels do not acquire unused transitive sets.
// Populations also carry complete-batch observations: a retained batch acquires
// all of its source records before copying any destination. When that batch
// owns every source publication, its first downstream arrival observes the
// complete source. This proof never enters the equal-record out_readers map.
iree_status_t loom_pipeline_progress_select(
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_worker_t* workers,
    const loom_pipeline_progress_endpoint_t* endpoints,
    loom_pipeline_progress_callback_t callback,
    const uint32_t* observation_sources,
    iree_host_size_t observation_source_count, iree_arena_allocator_t* arena,
    uint32_t** out_readers,
    loom_pipeline_progress_observations_t** out_observations);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_PROGRESS_H_
