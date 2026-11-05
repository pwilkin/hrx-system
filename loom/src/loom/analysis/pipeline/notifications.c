// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/notifications.h"

#include "loom/ops/channel/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/util/cfg_execution.h"

typedef struct loom_pipeline_notification_owner_t {
  // First read cursor owner, or NULL until observed.
  const loom_pipeline_worker_t* reader;
  // First write cursor owner, or NULL until observed.
  const loom_pipeline_worker_t* writer;
  // All observed uses have a single logical owner in each direction.
  bool eligible;
} loom_pipeline_notification_owner_t;

typedef struct loom_pipeline_notification_candidate_t {
  // Complete source proof under construction.
  loom_pipeline_notification_group_t group;
  // Last interval, retained for constant-time append.
  loom_pipeline_notification_interval_t* tail;
  // Exact consumer visits accumulated across structural intervals.
  uint64_t consumed;
  // Every source use of these members belongs to a complete interval.
  bool eligible;
} loom_pipeline_notification_candidate_t;

static bool loom_pipeline_notification_single_owner(
    const loom_pipeline_worker_t* worker) {
  if (loom_pipeline_strand_participation(worker->source->declaration) ==
      LOOM_PIPELINE_STRAND_PARTICIPATION_COOPERATIVE) {
    return true;
  }
  for (iree_host_size_t i = 0; i < worker->rank; ++i) {
    if (worker->axes[i].count != 1) {
      return false;
    }
  }
  return true;
}

static void loom_pipeline_notification_claim(
    const loom_pipeline_invocations_t* channels,
    loom_pipeline_notification_owner_t* owners,
    const loom_channel_identity_t* identity,
    const loom_pipeline_worker_t* worker, bool writing, bool eligible) {
  const iree_host_size_t index =
      loom_pipeline_invocations_channel_index(channels, identity);
  if (index == IREE_HOST_SIZE_MAX) {
    return;
  }
  loom_pipeline_notification_owner_t* owner = &owners[index];
  const loom_pipeline_worker_t** cursor =
      writing ? &owner->writer : &owner->reader;
  owner->eligible &= eligible && (!*cursor || *cursor == worker);
  *cursor = worker;
}

static void loom_pipeline_notification_owners(
    const loom_pipeline_invocations_t* channels,
    loom_pipeline_worker_t* workers,
    loom_pipeline_notification_owner_t* owners) {
  for (iree_host_size_t i = 0; i < channels->worker_count; ++i) {
    const loom_pipeline_worker_t* worker = &workers[i];
    const bool eligible = loom_pipeline_notification_single_owner(worker);
    for (iree_host_size_t k = 0; k < worker->channels.action_count; ++k) {
      const loom_channel_plan_action_t* action = &worker->channels.actions[k];
      if (action->callable) {
        // The membership summary identifies channels but does not establish
        // where an opaque boundary consumes its particular obligations.
        for (iree_host_size_t b = 0; b < worker->binding_count; ++b) {
          loom_pipeline_notification_claim(
              channels, owners, &worker->bindings[b].channel->identity, worker,
              false, false);
        }
      } else {
        const bool writing = loom_channel_reserve_isa(action->op) ||
                             loom_channel_publish_isa(action->op);
        loom_pipeline_notification_claim(channels, owners, action->channel,
                                         worker, writing, eligible);
        if (action->destination_channel) {
          loom_pipeline_notification_claim(channels, owners,
                                           action->destination_channel, worker,
                                           true, eligible);
        }
      }
    }
  }
}

static bool loom_pipeline_notification_producer(
    const loom_pipeline_invocations_t* channels,
    const loom_pipeline_notification_owner_t* owners,
    const loom_pipeline_worker_t* worker, uint32_t* members) {
  const loom_pipeline_transport_t* transport = &worker->transport;
  const iree_host_size_t count = transport->cohort_count;
  if (count < 2 || count > transport->count / 3 ||
      transport->count != count * 3) {
    return false;
  }
  const loom_block_t* block = transport->steps[0].op->parent_block;
  const loom_pipeline_worker_t* consumer = NULL;
  uint64_t capacity = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_pipeline_transport_step_t* reserve = &transport->steps[i];
    const loom_pipeline_transport_step_t* accept = &transport->steps[count + i];
    const loom_pipeline_transport_step_t* copy =
        &transport->steps[2 * count + i];
    const loom_pipeline_transport_cohort_t* cohort = &transport->cohorts[i];
    if (!loom_channel_reserve_isa(reserve->op) ||
        !loom_channel_accept_isa(accept->op) ||
        !loom_channel_copy_isa(copy->op) ||
        reserve->op->parent_block != block ||
        accept->op->parent_block != block || copy->op->parent_block != block ||
        cohort->destination_count != 1 || cohort->release ||
        cohort->admission->op != accept->op ||
        cohort->destinations[0].reservation != reserve->op ||
        cohort->destinations[0].copy != copy->op ||
        cohort->admission_scope != cohort->copy_scope || !cohort->admissions ||
        cohort->transfers != cohort->admissions ||
        cohort->admissions != transport->cohorts[0].admissions) {
      return false;
    }
    const iree_host_size_t member = loom_pipeline_invocations_channel_index(
        channels, cohort->destinations[0].channel);
    if (member == IREE_HOST_SIZE_MAX || !owners[member].eligible ||
        owners[member].writer != worker || !owners[member].reader ||
        owners[member].reader == worker) {
      return false;
    }
    if (i == 0) {
      consumer = owners[member].reader;
      capacity = channels->channels[member].source->capacity;
    } else if (consumer != owners[member].reader ||
               capacity != channels->channels[member].source->capacity) {
      return false;
    }
    members[i] = member;
  }
  return true;
}

static bool loom_pipeline_notification_consumer_interval(
    const loom_pipeline_worker_t* worker, iree_host_size_t first,
    uint32_t candidate_index,
    const loom_pipeline_notification_candidate_t* candidate,
    const loom_pipeline_invocations_t* channels, const uint32_t* membership,
    uint64_t* seen, uint64_t sequence) {
  const loom_channel_plan_t* plan = &worker->channels;
  const iree_host_size_t count = candidate->group.channel_count;
  if (count > (plan->action_count - first) / 2) {
    return false;
  }
  const loom_block_t* block = plan->actions[first].op->parent_block;
  if (!loom_cfg_execution_count_is_exact(&worker->execution_counts,
                                         block->region_index)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_channel_plan_action_t* acquire = &plan->actions[first + i * 2];
    const loom_channel_plan_action_t* release =
        &plan->actions[first + i * 2 + 1];
    if (!loom_channel_acquire_isa(acquire->op) ||
        !loom_channel_release_isa(release->op) ||
        acquire->channel != release->channel ||
        acquire->op->parent_block != block ||
        release->op->parent_block != block ||
        loom_channel_plan_admission(plan,
                                    loom_channel_release_read(release->op)) !=
            loom_channel_acquire_read(acquire->op)) {
      return false;
    }
    const iree_host_size_t member =
        loom_pipeline_invocations_channel_index(channels, acquire->channel);
    if (member == IREE_HOST_SIZE_MAX || membership[member] != candidate_index ||
        seen[member] == sequence) {
      return false;
    }
    seen[member] = sequence;
  }
  return true;
}

iree_status_t loom_pipeline_notifications_analyze(
    const loom_pipeline_invocations_t* channels,
    loom_pipeline_worker_t* workers, iree_arena_allocator_t* arena,
    loom_pipeline_notification_plan_t* out_plan) {
  *out_plan = (loom_pipeline_notification_plan_t){0};
  if (!channels->channel_count) {
    return iree_ok_status();
  }
  const iree_host_size_t count = channels->channel_count;
  loom_pipeline_notification_owner_t* owners;
  uint32_t* membership;
  uint32_t* members;
  uint64_t* seen;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*owners),
                                                 (void**)&owners));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*membership), (void**)&membership));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, count, sizeof(*members),
                                                 (void**)&members));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*seen), (void**)&seen));
  for (iree_host_size_t i = 0; i < count; ++i) {
    owners[i] = (loom_pipeline_notification_owner_t){.eligible = true};
    membership[i] = UINT32_MAX;
    seen[i] = 0;
  }
  loom_pipeline_notification_owners(channels, workers, owners);
  loom_pipeline_notification_candidate_t* candidates;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count / 2, sizeof(*candidates), (void**)&candidates));
  uint32_t candidate_count = 0;
  uint64_t sequence = 0;
  for (iree_host_size_t i = 0; i < channels->worker_count; ++i) {
    const loom_pipeline_worker_t* worker = &workers[i];
    if (worker->transport.cohort_count > count ||
        !loom_pipeline_notification_producer(channels, owners, worker,
                                             members)) {
      continue;
    }
    const iree_host_size_t member_count = worker->transport.cohort_count;
    ++sequence;
    bool distinct = true;
    for (iree_host_size_t k = 0; k < member_count; ++k) {
      const uint32_t member = members[k];
      distinct &= seen[member] != sequence;
      seen[member] = sequence;
    }
    if (!distinct) {
      continue;
    }
    uint32_t* selected;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, member_count, sizeof(*selected), (void**)&selected));
    for (iree_host_size_t k = 0; k < member_count; ++k) {
      selected[k] = members[k];
      membership[members[k]] = candidate_count;
    }
    candidates[candidate_count++] = (loom_pipeline_notification_candidate_t){
        .group = {.producer = worker,
                  .consumer = owners[members[0]].reader,
                  .channels = selected,
                  .channel_count = member_count,
                  .admissions = worker->transport.cohorts[0].admissions},
        .eligible = true,
    };
  }
  for (iree_host_size_t i = 0; i < channels->worker_count; ++i) {
    loom_pipeline_worker_t* worker = &workers[i];
    bool domain_active = false;
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t k = 0;
         k < worker->channels.action_count && iree_status_is_ok(status);) {
      const loom_channel_plan_action_t* action = &worker->channels.actions[k];
      const iree_host_size_t channel =
          loom_pipeline_invocations_channel_index(channels, action->channel);
      const uint32_t index =
          channel == IREE_HOST_SIZE_MAX ? UINT32_MAX : membership[channel];
      if (index == UINT32_MAX || candidates[index].group.consumer != worker ||
          !candidates[index].eligible) {
        ++k;
        continue;
      }
      loom_pipeline_notification_candidate_t* candidate = &candidates[index];
      if (!domain_active) {
        loom_local_value_domain_restore(&worker->value_domain);
        domain_active = true;
      }
      if (!loom_pipeline_notification_consumer_interval(
              worker, k, index, candidate, channels, membership, seen,
              ++sequence)) {
        candidate->eligible = false;
        ++k;
        continue;
      }
      const uint64_t executions =
          worker->execution_counts
              .values[action->op->parent_block->region_index];
      if (executions > candidate->group.admissions - candidate->consumed) {
        candidate->eligible = false;
        ++k;
        continue;
      }
      loom_pipeline_notification_interval_t* interval;
      status = iree_arena_allocate(arena, sizeof(*interval), (void**)&interval);
      if (iree_status_is_ok(status)) {
        *interval = (loom_pipeline_notification_interval_t){
            .first_action = k,
            .last_action = k + candidate->group.channel_count * 2 - 1,
            .executions = executions,
        };
        if (candidate->tail) {
          candidate->tail->next = interval;
        } else {
          candidate->group.intervals = interval;
        }
        candidate->tail = interval;
        candidate->consumed += executions;
        k = interval->last_action + 1;
      }
    }
    if (domain_active) {
      loom_local_value_domain_release(&worker->value_domain);
    }
    IREE_RETURN_IF_ERROR(status);
  }
  loom_pipeline_notification_group_t* groups;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, candidate_count, sizeof(*groups), (void**)&groups));
  out_plan->groups = groups;
  out_plan->channel_groups = membership;
  for (uint32_t i = 0; i < candidate_count; ++i) {
    const loom_pipeline_notification_candidate_t* candidate = &candidates[i];
    uint32_t group = UINT32_MAX;
    if (candidate->eligible &&
        candidate->consumed == candidate->group.admissions) {
      group = (uint32_t)out_plan->group_count++;
      groups[group] = candidate->group;
    }
    for (iree_host_size_t j = 0; j < candidate->group.channel_count; ++j) {
      membership[candidate->group.channels[j]] = group;
    }
  }
  return iree_ok_status();
}
