// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/progress.h"

#include <string.h>

#include "loom/util/adaptive_sort.h"

typedef struct loom_pipeline_progress_edge_t {
  // Next source depending on the same destination's progress.
  uint32_t next;
  // Upstream channel whose record ordinal this edge preserves.
  uint32_t source;
  // This edge observes a complete finite batch, not equal record ordinals.
  bool complete_batch;
} loom_pipeline_progress_edge_t;

typedef struct loom_pipeline_progress_node_t {
  // First upstream edge, or UINT32_MAX.
  uint32_t upstream;
  // Downstream channels still awaiting observer selection.
  uint32_t pending;
} loom_pipeline_progress_node_t;

// Optional forward links share the exact edges built by the progress owner.
// They are only needed when retaining complete observation populations.
typedef struct loom_pipeline_progress_downstream_t {
  // Next edge leaving the same source channel, or UINT32_MAX.
  uint32_t next;
  // Destination of this admitted equal-record copy edge.
  uint32_t destination;
  // First destination arrival observes every source publication.
  bool complete_batch;
} loom_pipeline_progress_downstream_t;

typedef struct loom_pipeline_progress_population_t {
  // First outgoing edge for each channel, or UINT32_MAX.
  uint32_t* first;
  // Forward links parallel to the existing reverse graph's edge table.
  loom_pipeline_progress_downstream_t* edges;
  // Retained channel-indexed observation sets.
  loom_pipeline_progress_observations_t* rows;
  // Visited channel/kind tags for each requested source population.
  uint32_t* marks;
  // Reusable worklist and sorted output, encoding channel*2+observation kind.
  uint64_t* worklist;
  // Unique observations; at most two kinds per channel.
  uint64_t* merged;
} loom_pipeline_progress_population_t;

typedef struct loom_pipeline_progress_contribution_t {
  // Sole FIFO reader/writer whose ordered phases contribute these records.
  uint32_t worker;
  // Canonical source channel of this ordinary, non-retained copy.
  uint32_t source;
  // Canonical destination channel preserving the copied record order.
  uint32_t destination;
  // Exact records contributed by this phase, without expanding repetitions.
  uint64_t records;
} loom_pipeline_progress_contribution_t;

static bool loom_pipeline_progress_contribution_less(
    const loom_pipeline_progress_contribution_t* lhs,
    const loom_pipeline_progress_contribution_t* rhs) {
  if (lhs->worker != rhs->worker) {
    return lhs->worker < rhs->worker;
  }
  if (lhs->source != rhs->source) {
    return lhs->source < rhs->source;
  }
  return lhs->destination < rhs->destination;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_progress_sort_contributions,
                          loom_pipeline_progress_contribution_t,
                          loom_pipeline_progress_contribution_less)

static void loom_pipeline_progress_append_edge(
    uint32_t source, uint32_t destination, bool complete_batch,
    loom_pipeline_progress_node_t* nodes, loom_pipeline_progress_edge_t* edges,
    loom_pipeline_progress_population_t* population, uint32_t* count) {
  const uint32_t index = (*count)++;
  edges[index] =
      (loom_pipeline_progress_edge_t){.next = nodes[destination].upstream,
                                      .source = source,
                                      .complete_batch = complete_batch};
  if (population->first) {
    population->edges[index] =
        (loom_pipeline_progress_downstream_t){.next = population->first[source],
                                              .destination = destination,
                                              .complete_batch = complete_batch};
    population->first[source] = index;
  }
  nodes[destination].upstream = index;
  ++nodes[source].pending;
}

static bool loom_pipeline_progress_channel_less(const uint64_t* lhs,
                                                const uint64_t* rhs) {
  return *lhs < *rhs;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_pipeline_progress_sort_channels, uint64_t,
                          loom_pipeline_progress_channel_less)

static iree_status_t loom_pipeline_progress_retain_observations(
    uint32_t source, iree_host_size_t channel_count,
    const loom_pipeline_progress_endpoint_t* endpoints,
    const loom_pipeline_progress_node_t* nodes,
    loom_pipeline_progress_population_t* population,
    iree_arena_allocator_t* arena) {
  if (population->rows[source].values || nodes[source].pending) {
    return iree_ok_status();
  }
  if (!population->marks) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, channel_count * 2,
                                                   sizeof(*population->marks),
                                                   (void**)&population->marks));
    memset(population->marks, 0xFF,
           channel_count * 2 * sizeof(*population->marks));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, channel_count * 2, sizeof(*population->worklist),
        (void**)&population->worklist));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, channel_count * 2, sizeof(*population->merged),
        (void**)&population->merged));
  }
  iree_host_size_t head = 0, tail = 0, count = 0;
  population->marks[(uint64_t)source * 2] = source;
  population->worklist[tail++] = (uint64_t)source * 2;
  while (head < tail) {
    const uint64_t visit = population->worklist[head++];
    const uint32_t channel = (uint32_t)(visit / 2);
    const bool complete_batch = visit % 2;
    if (endpoints[channel].observer && endpoints[channel].reads) {
      population->merged[count++] = visit;
    }
    for (uint32_t edge = population->first[channel]; edge != UINT32_MAX;
         edge = population->edges[edge].next) {
      const uint32_t destination = population->edges[edge].destination;
      const bool observes_batch =
          complete_batch || population->edges[edge].complete_batch;
      const uint64_t next = (uint64_t)destination * 2 + observes_batch;
      if (population->marks[next] == source || nodes[destination].pending ||
          (!observes_batch &&
           endpoints[destination].reads != endpoints[source].reads)) {
        continue;
      }
      population->marks[next] = source;
      population->worklist[tail++] = next;
    }
  }
  loom_pipeline_progress_sort_channels(population->merged, count);
  loom_pipeline_progress_observation_t* values;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*values),
                                                 (void**)&values));
  for (iree_host_size_t i = 0; i < count; ++i) {
    values[i] = (loom_pipeline_progress_observation_t){
        .channel = (uint32_t)(population->merged[i] / 2),
        .kind = population->merged[i] % 2
                    ? LOOM_PIPELINE_PROGRESS_OBSERVATION_COMPLETE_BATCH
                    : LOOM_PIPELINE_PROGRESS_OBSERVATION_RECORD};
  }
  population->rows[source] =
      (loom_pipeline_progress_observations_t){.values = values, .count = count};
  return iree_ok_status();
}

iree_status_t loom_pipeline_progress_select(
    const loom_pipeline_invocations_t* invocations,
    const loom_pipeline_worker_t* workers,
    const loom_pipeline_progress_endpoint_t* endpoints,
    loom_pipeline_progress_callback_t callback,
    const uint32_t* observation_sources,
    iree_host_size_t observation_source_count, iree_arena_allocator_t* arena,
    uint32_t** out_readers,
    loom_pipeline_progress_observations_t** out_observations) {
  *out_readers = NULL;
  if (out_observations) {
    *out_observations = NULL;
  }
  const iree_host_size_t count = invocations->channel_count;
  if (!count) {
    return iree_ok_status();
  }
  loom_pipeline_progress_node_t* nodes;
  uint32_t* ready;
  uint32_t* readers;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*nodes), (void**)&nodes));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*ready), (void**)&ready));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*readers),
                                                 (void**)&readers));
  iree_host_size_t edge_capacity = 0;
  for (iree_host_size_t i = 0; i < invocations->worker_count; ++i) {
    const loom_pipeline_transport_t* transport = &workers[i].transport;
    for (iree_host_size_t k = 0; k < transport->cohort_count; ++k) {
      edge_capacity += transport->cohorts[k].destination_count;
    }
  }
  if (edge_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline progress exceeds edge index capacity");
  }
  loom_pipeline_progress_edge_t* edges;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, edge_capacity, sizeof(*edges), (void**)&edges));
  loom_pipeline_progress_population_t population = {0};
  if (observation_source_count) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, count, sizeof(*population.first), (void**)&population.first));
    memset(population.first, 0xFF, count * sizeof(*population.first));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, edge_capacity,
                                                   sizeof(*population.edges),
                                                   (void**)&population.edges));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, count, sizeof(*population.rows), (void**)&population.rows));
    memset(population.rows, 0, count * sizeof(*population.rows));
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    nodes[i] = (loom_pipeline_progress_node_t){.upstream = UINT32_MAX};
    readers[i] =
        endpoints[i].observer && endpoints[i].reads ? (uint32_t)i : UINT32_MAX;
  }
  uint32_t edge_count = 0;
  loom_pipeline_progress_contribution_t* contributions = NULL;
  iree_host_size_t contribution_count = 0;
  for (iree_host_size_t i = 0; i < invocations->worker_count; ++i) {
    const loom_pipeline_transport_t* transport = &workers[i].transport;
    for (iree_host_size_t k = 0; k < transport->batch_count; ++k) {
      const loom_pipeline_transport_batch_t* batch = &transport->batches[k];
      const loom_pipeline_transport_cohort_t* cohort = batch->records;
      const uint32_t source = loom_pipeline_invocations_channel_index(
          invocations, cohort->admission->channel);
      const bool complete_batch =
          observation_source_count && cohort->release &&
          cohort->admissions == 1 &&
          endpoints[source].writes == batch->record_count;
      if (!complete_batch && (batch->record_count != 1 || cohort->release ||
                              cohort->admissions != cohort->transfers ||
                              !endpoints[source].reads)) {
        continue;
      }
      for (iree_host_size_t d = 0; d < cohort->destination_count; ++d) {
        const uint32_t destination = loom_pipeline_invocations_channel_index(
            invocations, cohort->destinations[d].channel);
        const uint64_t records = cohort->transfers * batch->record_count;
        if (!complete_batch && (endpoints[source].reads != records ||
                                endpoints[destination].writes != records)) {
          // Several ordered phases can partition one FIFO stream. Retain
          // only partial ordinary copies; complete edges need no join state.
          if (records > endpoints[source].reads ||
              records > endpoints[destination].writes) {
            continue;
          }
          if (!contributions) {
            IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
                arena, edge_capacity, sizeof(*contributions),
                (void**)&contributions));
          }
          contributions[contribution_count++] =
              (loom_pipeline_progress_contribution_t){
                  .worker = (uint32_t)i,
                  .source = source,
                  .destination = destination,
                  .records = records};
          continue;
        }
        if (endpoints[destination].writes != records) {
          continue;
        }
        loom_pipeline_progress_append_edge(source, destination, complete_batch,
                                           nodes, edges, &population,
                                           &edge_count);
      }
    }
  }
  loom_pipeline_progress_sort_contributions(contributions, contribution_count);
  for (iree_host_size_t begin = 0; begin < contribution_count;) {
    const loom_pipeline_progress_contribution_t* first = &contributions[begin];
    uint64_t records = first->records;
    iree_host_size_t end = begin + 1;
    while (
        end < contribution_count &&
        !loom_pipeline_progress_contribution_less(first, &contributions[end])) {
      records += contributions[end++].records;
    }
    if (records == endpoints[first->source].reads &&
        records == endpoints[first->destination].writes) {
      loom_pipeline_progress_append_edge(first->source, first->destination,
                                         false, nodes, edges, &population,
                                         &edge_count);
    }
    begin = end;
  }
  iree_host_size_t head = 0, tail = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!nodes[i].pending) {
      ready[tail++] = (uint32_t)i;
    }
  }
  while (head < tail) {
    const uint32_t destination = ready[head++];
    const uint32_t candidate = readers[destination];
    for (uint32_t edge = nodes[destination].upstream; edge != UINT32_MAX;
         edge = edges[edge].next) {
      const uint32_t source = edges[edge].source;
      const uint32_t selected = readers[source];
      if (!edges[edge].complete_batch && candidate != UINT32_MAX &&
          endpoints[candidate].reads == endpoints[source].reads) {
        if (selected == UINT32_MAX) {
          readers[source] = candidate;
        } else {
          const uint64_t candidate_cost =
              callback.cost(callback.user_data, source, candidate);
          const uint64_t selected_cost =
              callback.cost(callback.user_data, source, selected);
          if (candidate_cost < selected_cost ||
              (candidate_cost == selected_cost && candidate < selected)) {
            readers[source] = candidate;
          }
        }
      }
      if (!--nodes[source].pending) {
        ready[tail++] = source;
      }
    }
  }
  // Pending nodes participate in or depend on a cycle. An eligible reader at
  // such a node cannot establish equal-ordinal feedback through that cycle.
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (nodes[i].pending) {
      readers[i] = UINT32_MAX;
    }
  }
  // Complete only the requested transitive populations. Retaining one set at
  // every intermediate observer would make a chain quadratic even when its
  // caller needs just one phase. Repeated requests for a source share its
  // already retained result.
  for (iree_host_size_t i = 0; i < observation_source_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_progress_retain_observations(
        observation_sources[i], count, endpoints, nodes, &population, arena));
  }
  *out_readers = readers;
  if (out_observations) {
    *out_observations = population.rows;
  }
  return iree_ok_status();
}
