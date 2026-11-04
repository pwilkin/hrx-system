// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/refill.h"

#include <stdlib.h>

#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"

// Reverse physical continuation graph over canonical one-to-one copy edges.
typedef struct loom_aie2p_native_progress_edge_t {
  // Next source depending on the same destination's progress.
  uint32_t next;
  // Upstream channel whose record ordinal this edge preserves.
  uint32_t source;
} loom_aie2p_native_progress_edge_t;

typedef struct loom_aie2p_native_progress_node_t {
  // First upstream edge, or UINT32_MAX.
  uint32_t upstream;
  // Downstream channels still awaiting physical reader selection.
  uint32_t pending;
  // Best reachable compute reader in the completed downstream subgraph.
  const loom_aie2p_native_channel_t* reader;
} loom_aie2p_native_progress_node_t;

static uint32_t loom_aie2p_native_progress_distance(
    const loom_aie2p_native_context_t* context,
    const loom_aie2p_native_channel_t* source,
    const loom_aie2p_native_channel_t* progress) {
  const loom_xdna_tile_coordinate_t origin =
      context->pool_tiles[source->pool_index]->coordinate;
  const loom_xdna_tile_coordinate_t target =
      context->workers[progress->cursor.reader.worker].tile->coordinate;
  return (uint32_t)abs((int)origin.column - (int)target.column) +
         (uint32_t)abs((int)origin.row - (int)target.row);
}

iree_status_t loom_aie2p_native_select_progress(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization) {
  const iree_host_size_t count = realization->invocations.channel_count;
  if (!count) {
    return iree_ok_status();
  }
  bool needs_progress = false;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    const loom_xdna_dma_facts_t* dma =
        &context->pool_tiles[channel->pool_index]->facts->dma;
    needs_progress |=
        channel->cursor.reader.maximum_admissions >
        (uint32_t)dma->maximum_task_repeat_count * dma->task_queue_depth;
  }
  if (!needs_progress) {
    return iree_ok_status();
  }
  iree_arena_allocator_t* arena = context->pass->arena;
  loom_aie2p_native_progress_node_t* nodes;
  uint32_t* ready;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*nodes), (void**)&nodes));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, count, sizeof(*ready), (void**)&ready));
  iree_host_size_t edge_capacity = 0;
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    const loom_pipeline_transport_t* transport =
        &realization->workers[i].transport;
    for (iree_host_size_t j = 0; j < transport->cohort_count; ++j) {
      edge_capacity += transport->cohorts[j].destination_count;
    }
  }
  loom_aie2p_native_progress_edge_t* edges;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, edge_capacity, sizeof(*edges), (void**)&edges));
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_aie2p_native_channel_t* channel = &context->channels[i];
    const uint32_t reader = channel->cursor.reader.worker;
    nodes[i] = (loom_aie2p_native_progress_node_t){.upstream = UINT32_MAX};
    if (!channel->replay.record_count && !channel->cursor.reader.shared &&
        !channel->cursor.writer.shared && reader != UINT32_MAX &&
        context->workers[reader].execution ==
            LOOM_AIE2P_NATIVE_EXECUTION_CORE &&
        realization->workers[reader].block_execution_counts &&
        channel->cursor.reader.maximum_admissions) {
      nodes[i].reader = channel;
    }
  }
  uint32_t edge_count = 0;
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    const loom_pipeline_transport_t* transport =
        &realization->workers[i].transport;
    for (iree_host_size_t j = 0; j < transport->batch_count; ++j) {
      const loom_pipeline_transport_batch_t* batch = &transport->batches[j];
      const loom_pipeline_transport_cohort_t* cohort = batch->records;
      if (batch->record_count != 1 || cohort->release ||
          cohort->admissions != cohort->transfers) {
        continue;
      }
      const uint32_t source_index = (uint32_t)
          loom_pipeline_invocations_channel_index(
              &realization->invocations, cohort->admission->channel);
      const loom_aie2p_native_channel_t* physical =
          &context->channels[source_index];
      if (physical->replay.record_count || physical->cursor.reader.shared ||
          physical->cursor.writer.shared ||
          physical->cursor.reader.maximum_admissions != cohort->admissions) {
        continue;
      }
      for (iree_host_size_t k = 0; k < cohort->destination_count; ++k) {
        const uint32_t destination_index = (uint32_t)
            loom_pipeline_invocations_channel_index(
                &realization->invocations, cohort->destinations[k].channel);
        if (context->channels[destination_index]
                .cursor.writer.maximum_admissions != cohort->transfers) {
          continue;
        }
        edges[edge_count] = (loom_aie2p_native_progress_edge_t){
            .next = nodes[destination_index].upstream, .source = source_index};
        nodes[destination_index].upstream = edge_count++;
        ++nodes[source_index].pending;
      }
    }
  }
  iree_host_size_t head = 0, tail = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!nodes[i].pending) {
      ready[tail++] = (uint32_t)i;
    }
  }
  // Each canonical edge is consumed once. Cycles retain no equal-ordinal
  // continuation: feedback needs its own record-distance proof.
  while (head < tail) {
    const uint32_t destination = ready[head++];
    const loom_aie2p_native_channel_t* candidate = nodes[destination].reader;
    context->channels[destination].progress = candidate;
    for (uint32_t edge = nodes[destination].upstream; edge != UINT32_MAX;
         edge = edges[edge].next) {
      const uint32_t source = edges[edge].source;
      const loom_aie2p_native_channel_t* channel = &context->channels[source];
      const loom_aie2p_native_channel_t* selected = nodes[source].reader;
      if (candidate &&
          candidate->cursor.reader.maximum_admissions ==
              channel->cursor.reader.maximum_admissions &&
          (!selected ||
           loom_aie2p_native_progress_distance(context, channel, candidate) <
               loom_aie2p_native_progress_distance(context, channel,
                                                   selected) ||
           (loom_aie2p_native_progress_distance(context, channel, candidate) ==
                loom_aie2p_native_progress_distance(context, channel,
                                                    selected) &&
            candidate < selected))) {
        nodes[source].reader = candidate;
      }
      if (!--nodes[source].pending) {
        ready[tail++] = source;
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_select_refill(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction,
    const loom_aie2p_native_channel_t* channel, uint32_t repetitions,
    uint32_t capacity, uint32_t byte_offset, uint32_t byte_stride,
    uint32_t credit_interval, uint16_t first_descriptor,
    uint16_t available_descriptors, const uint32_t words[8],
    loom_aie2p_native_dma_sequence_t* sequence, iree_string_view_t* reason,
    bool* out_valid) {
  const loom_aie2p_native_memory_layout_t* layout = tile->memory_layout;
  const loom_xdna_dma_facts_t* dma = &tile->facts->dma;
  if (!layout || !channel || !channel->progress || credit_interval != 1 ||
      channel->progress->cursor.reader.maximum_admissions != repetitions ||
      dma->task_queue_depth < 2 || capacity > dma->maximum_task_repeat_count) {
    return iree_ok_status();
  }
  const uint32_t window = dma->maximum_task_repeat_count / capacity * capacity;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_build_sequence(
      dma, layout, first_descriptor, available_descriptors, direction, window,
      capacity, byte_offset, byte_stride, 1, words, sequence, reason, out_valid,
      context->pass->arena));
  if (!*out_valid) {
    return iree_ok_status();
  }
  sequence->rewind_iteration = repetitions % capacity != 0;
  loom_aie2p_native_worker_t* worker =
      &context->workers[channel->progress->cursor.reader.worker];
  loom_aie2p_native_refill_t* group = worker->refills;
  while (group &&
         (group->progress != channel->progress || group->window != window)) {
    group = group->next;
  }
  if (!group) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena,
                                             sizeof(*group), (void**)&group));
    *group = (loom_aie2p_native_refill_t){
        .next = worker->refills,
        .progress = channel->progress,
        .window = window,
        .remaining_tasks = (repetitions - 1) / window};
    worker->refills = group;
  }
  const uint16_t queue_index = engine;
  const uint64_t base =
      (uint64_t)tile->coordinate.column << context->family->column_shift |
      (uint64_t)tile->coordinate.row << context->family->row_shift;
  const uint32_t address =
      (uint32_t)(loom_xdna_register_field_address(
                     context->family, layout->directions[direction - 1].start,
                     tile->coordinate, &queue_index) -
                 base);
  const uint32_t tail = repetitions % window;
  const uint32_t final_task =
      loom_xdna_register_field_encode(layout->directions[direction - 1].start,
                                      sequence->descriptor) |
      loom_xdna_register_field_encode(layout->directions[direction - 1].repeat,
                                      (tail ? tail : window) - 1);
  loom_aie2p_native_refill_queue_t* queue;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena, sizeof(*queue),
                                           (void**)&queue));
  *queue = (loom_aie2p_native_refill_queue_t){.next = group->queues,
                                              .tile = tile,
                                              .engine = engine,
                                              .direction = direction,
                                              .address = address,
                                              .task = sequence->tasks[0],
                                              .final_task = final_task};
  group->queues = queue;
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_route_refills(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, uint32_t* next_source,
    bool* out_valid) {
  for (iree_host_size_t i = 0; i < realization->invocations.worker_count; ++i) {
    loom_aie2p_native_worker_t* worker = &context->workers[i];
    for (loom_aie2p_native_refill_t* group = worker->refills; group;
         group = group->next) {
      for (loom_aie2p_native_refill_queue_t* queue = group->queues; queue;
           queue = queue->next) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_select_control_route(
            context, realization, i, queue->tile, false, next_source, NULL,
            out_valid));
        if (!*out_valid) {
          return iree_ok_status();
        }
        const loom_aie2p_native_worker_t* owner = &context->workers[
            context->execution.workers[context->core_engines[i]]];
        queue->packet = owner->control_packets[queue->tile - context->tiles];
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_prepare_refills(
    loom_aie2p_native_context_t* context, loom_aie2p_native_refill_t* refills) {
  loom_aie2p_worker_builder_t* code = &context->code;
  for (loom_aie2p_native_refill_t* group = refills; group;
       group = group->next) {
    for (loom_aie2p_native_refill_queue_t* queue = group->queues; queue;
         queue = queue->next) {
      loom_builder_t builder;
      loom_op_t* function;
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, &code->scalar_type, 1,
                                                    NULL, 0, &builder,
                                                    &queue->submit, &function));
      const loom_value_id_t argument =
          loom_region_entry_block(loom_low_func_def_body(function))->arg_ids[0];
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
          code, &builder, queue->packet, queue->address, &argument, 1));
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&builder, NULL, 0));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_refill_constant(
    loom_builder_t* builder, uint32_t value, loom_value_id_t* result) {
  loom_op_t* constant;
  IREE_RETURN_IF_ERROR(
      loom_scalar_constant_build(builder, loom_attr_i64((int32_t)value),
                                 loom_type_scalar(LOOM_SCALAR_TYPE_I32),
                                 LOOM_LOCATION_UNKNOWN, &constant));
  *result = loom_scalar_constant_result(constant);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_emit_refill(
    loom_builder_t* builder, const loom_aie2p_native_refill_t* refill,
    loom_value_id_t* state) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t types[] = {type, type};
  const loom_value_id_t countdown = state[refill->state_index];
  const loom_value_id_t remaining = state[refill->state_index + 1];
  loom_value_id_t zero, one, reset;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_refill_constant(builder, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_refill_constant(builder, 1, &one));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_refill_constant(builder, refill->window - 1, &reset));
  loom_op_t *boundary, *window;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, countdown,
                             zero, LOOM_LOCATION_UNKNOWN, &boundary));
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        loom_scalar_cmpi_result(boundary), types, 2, NULL, 0,
                        LOOM_LOCATION_UNKNOWN, &window));
  const loom_builder_ip_t outer = loom_builder_enter_region(
      builder, window, loom_scf_if_then_region(window));
  loom_op_t *pending, *submit;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_NE, remaining,
                             zero, LOOM_LOCATION_UNKNOWN, &pending));
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        loom_scalar_cmpi_result(pending), &type, 1, NULL, 0,
                        LOOM_LOCATION_UNKNOWN, &submit));
  const loom_builder_ip_t inner = loom_builder_enter_region(
      builder, submit, loom_scf_if_then_region(submit));
  loom_op_t* last;
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, remaining,
                             one, LOOM_LOCATION_UNKNOWN, &last));
  for (const loom_aie2p_native_refill_queue_t* queue = refill->queues; queue;
       queue = queue->next) {
    loom_value_id_t full, tail;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_refill_constant(builder, queue->task, &full));
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_refill_constant(builder, queue->final_task, &tail));
    loom_op_t* selected;
    IREE_RETURN_IF_ERROR(
        loom_scf_select_build(builder, loom_scalar_cmpi_result(last), tail,
                              full, type, LOOM_LOCATION_UNKNOWN, &selected));
    const loom_value_id_t argument = loom_scf_select_result(selected);
    loom_op_t* call;
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(builder, 0, 0, 0, queue->submit,
                                               &argument, 1, NULL, 0, NULL, 0,
                                               LOOM_LOCATION_UNKNOWN, &call));
  }
  loom_op_t *next, *yield;
  IREE_RETURN_IF_ERROR(loom_scalar_subi_build(builder, 0, remaining, one, type,
                                              LOOM_LOCATION_UNKNOWN, &next));
  const loom_value_id_t next_remaining = loom_scalar_subi_result(next);
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &next_remaining, 1,
                                            LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, inner);
  loom_builder_enter_region(builder, submit, loom_scf_if_else_region(submit));
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &remaining, 1,
                                            LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, inner);
  const loom_value_id_t updated[] = {reset, loom_op_results(submit)[0]};
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, updated, 2, LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, outer);
  loom_builder_enter_region(builder, window, loom_scf_if_else_region(window));
  IREE_RETURN_IF_ERROR(loom_scalar_subi_build(builder, 0, countdown, one, type,
                                              LOOM_LOCATION_UNKNOWN, &next));
  const loom_value_id_t continued[] = {loom_scalar_subi_result(next),
                                       remaining};
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, continued, 2,
                                            LOOM_LOCATION_UNKNOWN, &yield));
  loom_builder_restore(builder, outer);
  state[refill->state_index] = loom_op_results(window)[0];
  state[refill->state_index + 1] = loom_op_results(window)[1];
  return iree_ok_status();
}
