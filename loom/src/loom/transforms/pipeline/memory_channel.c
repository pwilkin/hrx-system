// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/memory_channel.h"

#include "loom/ops/buffer/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

static iree_status_t loom_memory_channel_constant(loom_builder_t* builder,
                                                  uint64_t value,
                                                  loom_type_t type,
                                                  loom_location_id_t location,
                                                  loom_value_id_t* out_value) {
  loom_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64((int64_t)value), type, location, &operation));
  *out_value = loom_scalar_constant_result(operation);
  return iree_ok_status();
}

static iree_status_t loom_memory_channel_increment(loom_builder_t* builder,
                                                   loom_value_id_t value,
                                                   loom_location_id_t location,
                                                   loom_value_id_t* out_value) {
  const loom_type_t type = loom_module_value_type(builder->module, value);
  loom_value_id_t one;
  IREE_RETURN_IF_ERROR(
      loom_memory_channel_constant(builder, 1, type, location, &one));
  loom_op_t* operation = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, 0, value, one, type,
                                              location, &operation));
  *out_value = loom_scalar_addi_result(operation);
  return iree_ok_status();
}

static iree_status_t loom_memory_channel_slot(
    loom_builder_t* builder, const loom_memory_channel_layout_t* layout,
    loom_value_id_t record, loom_location_id_t location,
    loom_value_id_t* out_slot) {
  const loom_type_t type = loom_module_value_type(builder->module, record);
  loom_value_id_t capacity;
  IREE_RETURN_IF_ERROR(loom_memory_channel_constant(builder, layout->capacity,
                                                    type, location, &capacity));
  loom_op_t *slot, *index;
  IREE_RETURN_IF_ERROR(loom_scalar_remui_build(builder, record, capacity, type,
                                               location, &slot));
  IREE_RETURN_IF_ERROR(loom_index_cast_build(
      builder, loom_scalar_remui_result(slot), type,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), location, &index));
  *out_slot = loom_index_cast_result(index);
  return iree_ok_status();
}

static iree_status_t loom_memory_channel_offset(loom_builder_t* builder,
                                                loom_value_id_t slot,
                                                uint64_t byte_stride,
                                                uint64_t byte_offset,
                                                loom_location_id_t location,
                                                loom_value_id_t* out_offset) {
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  loom_op_t *stride, *scaled, *origin, *offset;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64((int64_t)byte_stride),
                                offset_type, location, &stride));
  IREE_RETURN_IF_ERROR(
      loom_index_scale_build(builder, slot, loom_index_constant_result(stride),
                             offset_type, location, &scaled));
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64((int64_t)byte_offset),
                                offset_type, location, &origin));
  IREE_RETURN_IF_ERROR(loom_index_add_build(
      builder, loom_index_scale_result(scaled),
      loom_index_constant_result(origin), offset_type, location, &offset));
  *out_offset = loom_index_add_result(offset);
  return iree_ok_status();
}

static iree_status_t loom_memory_channel_payload(
    loom_builder_t* builder, const loom_memory_channel_endpoint_t* endpoint,
    loom_value_id_t record, loom_location_id_t location,
    loom_value_id_t* out_payload) {
  const loom_memory_channel_layout_t* layout = endpoint->layout;
  loom_value_id_t slot, offset;
  IREE_RETURN_IF_ERROR(
      loom_memory_channel_slot(builder, layout, record, location, &slot));
  IREE_RETURN_IF_ERROR(
      loom_memory_channel_offset(builder, slot, layout->byte_stride,
                                 layout->byte_offset, location, &offset));
  loom_op_t* view = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      builder, endpoint->buffer, offset, layout->view_type, location, &view));
  *out_payload = loom_buffer_view_result(view);
  return iree_ok_status();
}

iree_status_t loom_memory_channel_notification_build(
    loom_builder_t* builder, const loom_memory_channel_layout_t* layout,
    loom_value_id_t record, loom_memory_channel_notification_t notification,
    uint64_t ready_byte_offset, uint64_t free_byte_offset,
    uint64_t slot_byte_stride, loom_location_id_t location,
    loom_value_id_t* out_byte_offset, loom_value_id_t* out_generation) {
  loom_value_id_t slot;
  IREE_RETURN_IF_ERROR(
      loom_memory_channel_slot(builder, layout, record, location, &slot));
  const bool ready = notification == LOOM_MEMORY_CHANNEL_WAIT_READY ||
                     notification == LOOM_MEMORY_CHANNEL_PUBLISH;
  IREE_RETURN_IF_ERROR(loom_memory_channel_offset(
      builder, slot, slot_byte_stride,
      ready ? ready_byte_offset : free_byte_offset, location, out_byte_offset));
  const loom_type_t type = loom_module_value_type(builder->module, record);
  loom_value_id_t capacity;
  IREE_RETURN_IF_ERROR(loom_memory_channel_constant(builder, layout->capacity,
                                                    type, location, &capacity));
  loom_op_t* generation = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_divui_build(builder, record, capacity, type,
                                               location, &generation));
  *out_generation = loom_scalar_divui_result(generation);
  if (notification != LOOM_MEMORY_CHANNEL_WAIT_FREE) {
    IREE_RETURN_IF_ERROR(loom_memory_channel_increment(
        builder, *out_generation, location, out_generation));
  }
  return iree_ok_status();
}

iree_status_t loom_memory_channel_materialize_action(
    void* user_data, loom_rewriter_t* rewriter,
    const loom_channel_plan_action_t* action, loom_value_id_t* state,
    iree_host_size_t state_count, loom_value_id_t* results) {
  const loom_memory_channel_materialization_t* context = user_data;
  const loom_op_t* operation = action->op;
  const loom_location_id_t location = operation->location;
  loom_builder_t* builder = &rewriter->builder;
  const loom_memory_channel_endpoint_t* endpoint =
      context->endpoints[action - context->plan->actions];
  if (loom_channel_reserve_isa(operation) ||
      loom_channel_acquire_isa(operation) ||
      loom_channel_accept_isa(operation)) {
    const bool write = loom_channel_reserve_isa(operation);
    const iree_host_size_t cursor =
        write ? endpoint->write_state : endpoint->read_state;
    const loom_value_id_t record = state[cursor];
    results[0] = record;
    IREE_RETURN_IF_ERROR(loom_memory_channel_increment(
        builder, record, location, &state[cursor]));
    if (!loom_channel_accept_isa(operation)) {
      IREE_RETURN_IF_ERROR(context->emit.notify(
          context->emit.user_data, rewriter, action, endpoint, record,
          write ? LOOM_MEMORY_CHANNEL_WAIT_FREE
                : LOOM_MEMORY_CHANNEL_WAIT_READY));
      IREE_RETURN_IF_ERROR(loom_memory_channel_payload(
          builder, endpoint, record, location, &results[1]));
    }
  } else if (loom_channel_wait_isa(operation)) {
    const loom_value_id_t record = loom_channel_wait_read(operation);
    IREE_RETURN_IF_ERROR(context->emit.notify(context->emit.user_data, rewriter,
                                              action, endpoint, record,
                                              LOOM_MEMORY_CHANNEL_WAIT_READY));
    IREE_RETURN_IF_ERROR(loom_memory_channel_payload(builder, endpoint, record,
                                                     location, &results[0]));
  } else if (loom_channel_release_isa(operation) ||
             loom_channel_publish_isa(operation)) {
    IREE_RETURN_IF_ERROR(context->emit.notify(
        context->emit.user_data, rewriter, action, endpoint,
        loom_op_const_operands(operation)[0],
        loom_channel_publish_isa(operation) ? LOOM_MEMORY_CHANNEL_PUBLISH
                                            : LOOM_MEMORY_CHANNEL_RELEASE));
  } else if (loom_channel_fanout_isa(operation)) {
    for (uint16_t i = 0; i < operation->result_count; ++i) {
      results[i] = loom_channel_fanout_read(operation);
    }
  } else {
    return context->emit.action(context->emit.user_data, rewriter, action,
                                state, state_count, results);
  }
  return iree_ok_status();
}
