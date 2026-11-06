// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/external.h"

// Writes an idle descriptor's slot while preserving its address geometry.
static iree_status_t loom_aie2p_native_input_write_slot(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_input_t* input, uint32_t register_offset,
    uint32_t geometry_word, loom_value_id_t slot) {
  loom_aie2p_worker_builder_t* code = &context->code;
  const loom_xdna_register_field_info_t current = loom_xdna_register_field_info(
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_CURRENT);
  loom_value_id_t shift, current_slot, geometry, iteration;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, builder, current.least_significant_bit, &shift));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, builder, AIE2P_CORE_DESCRIPTOR_REF_LSHL_I32, slot, shift,
      &current_slot));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_constant(code, builder, geometry_word, &geometry));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_binary(code, builder, AIE2P_CORE_DESCRIPTOR_REF_OR_I32,
                               geometry, current_slot, &iteration));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, builder, input->control_packet, register_offset, &iteration, 1));
  return iree_ok_status();
}

// Selects a compact descriptor's slot after all preceding reads retire.
// Query payloads and credit completions have already drained the reply stream.
static iree_status_t loom_aie2p_native_input_select_slot(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    loom_region_t* region, const loom_aie2p_native_input_t* input,
    uint32_t register_offset, uint32_t geometry_word, loom_value_id_t slot) {
  IREE_RETURN_IF_ERROR(loom_aie2p_native_queue_retire(
      context, builder, region, input->shim, input->input_engine,
      LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM, input->control_packet,
      input->completion_packet));
  return loom_aie2p_native_input_write_slot(
      context, builder, input, register_offset, geometry_word, slot);
}

// Immutable bulk/notification descriptors select their slot through the task
// ID. Prefixes and compact descriptors instead retain hardware iteration state.
static bool loom_aie2p_native_input_initializes_slot(
    const loom_aie2p_native_input_t* input) {
  return input->binding && input->binding->protocol->payload.capacity > 1 &&
         (input->source->phase_count > 1 ||
          input->slots != LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE);
}

static iree_status_t loom_aie2p_native_input_begin(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  const loom_type_t arguments[] = {code->scalar_type, code->scalar_type,
                                   code->scalar_type};
  const iree_host_size_t argument_count =
      input->continuation ? (input->binding ? 3 : 1) : 0;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(code, arguments, argument_count,
                                                NULL, 0, &builder,
                                                &input->begin, &function));
  loom_region_t* body = loom_low_func_def_body(function);
  const loom_value_id_t* values = loom_region_entry_block(body)->arg_ids;
  loom_block_t* payload = NULL;
  if (input->continuation) {
    loom_block_t *poll = NULL, *prefix;
    if (input->binding) {
      IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &poll));
    }
    IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &prefix));
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(code->module, body, &payload));
    loom_value_id_t zero, first;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, 0, &zero));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
        code, &builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32,
        values[input->binding ? 2 : 0], zero, &first));
    loom_op_t* op;
    if (input->binding) {
      loom_value_id_t ready;
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
          code, &builder, input->ready_descriptor, &ready));
      if (input->slots == LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE) {
        IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
            code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, ready, values[0],
            &ready));
      }
      IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
          &builder, first, poll, payload, LOOM_LOCATION_UNKNOWN, &op));
      loom_builder_set_block(&builder, poll);
      if (input->slots != LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE) {
        IREE_RETURN_IF_ERROR(loom_aie2p_native_input_select_slot(
            context, &builder, body, input, input->ready_iteration_register,
            input->ready_iteration, values[0]));
      }
      IREE_RETURN_IF_ERROR(
          loom_aie2p_worker_control_write(code, &builder, input->control_packet,
                                          input->input_queue, &ready, 1));
      loom_value_id_t header, observed, available;
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
          code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
          loom_named_attr_slice_empty(), &code->scalar_type, &header));
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
          code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
          loom_named_attr_slice_empty(), &code->scalar_type, &observed));
      IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
          code, &builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32, observed,
          values[1], &available));
      IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
          &builder, available, prefix, poll, LOOM_LOCATION_UNKNOWN, &op));
    } else {
      IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
          &builder, first, prefix, payload, LOOM_LOCATION_UNKNOWN, &op));
    }
    loom_builder_set_block(&builder, prefix);
  }
  for (iree_host_size_t i = 0; i + 1 < input->source->phase_count; ++i) {
    const loom_aie2p_native_input_phase_t* phase = &input->phases[i];
    const uint32_t word =
        phase->descriptor |
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_REPEAT_COUNT,
            phase->repetitions - 1);
    loom_value_id_t task;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_worker_constant(code, &builder, word, &task));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        code, &builder, input->control_packet, input->input_queue, &task, 1));
  }
  if (payload) {
    loom_op_t* op;
    IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, payload, NULL, 0,
                                           LOOM_LOCATION_UNKNOWN, &op));
    loom_builder_set_block(&builder, payload);
  }
  const loom_aie2p_native_input_phase_t* phase =
      &input->phases[input->source->phase_count - 1];
  if (input->slots == LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD &&
      phase->source->period != phase->source->transfers) {
    // One lease may replay its address period. Automatic slot stepping alone
    // would read the next record before the current lease's last use.
    IREE_RETURN_IF_ERROR(loom_aie2p_native_input_select_slot(
        context, &builder, body, input, input->payload_iteration_register,
        phase->words[6], values[0]));
  }
  const uint32_t word =
      phase->descriptor |
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_REPEAT_COUNT,
          phase->repetitions - 1);
  loom_value_id_t task;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, word, &task));
  if (input->binding && input->slots != LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
        code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, task, values[0],
        &task));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, input->control_packet, input->input_queue, &task, 1));
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

static iree_status_t loom_aie2p_native_input_complete(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  const loom_type_t arguments[] = {code->scalar_type, code->scalar_type,
                                   code->scalar_type};
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, arguments, 3, NULL, 0, &builder, &input->complete, &function));
  loom_region_t* body = loom_low_func_def_body(function);
  const loom_value_id_t* values = loom_region_entry_block(body)->arg_ids;
  loom_value_id_t descriptor, task;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, &builder, input->credit_descriptor, &descriptor));
  if (input->slots == LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE) {
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
        code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, descriptor,
        values[0], &descriptor));
  }
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_binary(code, &builder, AIE2P_CORE_DESCRIPTOR_REF_OR_I32,
                               descriptor, values[2], &task));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, input->control_packet, input->credit_queue, &task, 1));
  const uint32_t packet =
      input->credit_packet |
      ((iree_math_count_ones_u32(input->credit_packet) & 1u) ^ 1u) << 31;
  loom_value_id_t header;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_constant(code, &builder, packet, &header));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32, &header, 1,
      loom_named_attr_slice_empty(), NULL, NULL));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32,
      &values[1], 1, loom_named_attr_slice_empty(), NULL, NULL));
  if (input->slots != LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE) {
    // Source capacity no longer bounds outstanding writes to the credit queue.
    // One completion per record bounds the queue and drains the shared reply
    // stream before the next ready-descriptor retirement query.
    loom_value_id_t completed;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
        code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
        loom_named_attr_slice_empty(), &code->scalar_type, &completed));
    return loom_aie2p_worker_return(&builder, NULL, 0);
  }
  loom_block_t *wait, *done;
  IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &wait));
  IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &done));
  loom_op_t* op;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(&builder, values[2], wait, done,
                                              LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_set_block(&builder, wait);
  loom_value_id_t completed;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
      loom_named_attr_slice_empty(), &code->scalar_type, &completed));
  IREE_RETURN_IF_ERROR(
      loom_low_br_build(&builder, done, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_set_block(&builder, done);
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

iree_status_t loom_aie2p_native_prepare_input(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input) {
  const loom_aie2p_native_worker_t* owner =
      &context->workers[context->execution
                            .workers[context->core_engines[input->controller]]];
  if (input->invocation->ordinal) {
    loom_builder_t builder;
    loom_op_t* function;
    const bool initializes_slot =
        loom_aie2p_native_input_initializes_slot(input);
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
        &context->code, &context->code.scalar_type, initializes_slot, NULL, 0,
        &builder, &input->start, &function));
    if (initializes_slot) {
      const loom_value_id_t slot =
          loom_region_entry_block(loom_low_func_def_body(function))->arg_ids[0];
      const uint64_t origin = (uint64_t)input->shim->coordinate.column
                              << context->family->column_shift;
      const iree_host_size_t phase_count =
          input->source->phase_count -
          (input->slots != LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD);
      for (iree_host_size_t i = 0; i < phase_count; ++i) {
        const loom_aie2p_native_input_phase_t* phase = &input->phases[i];
        const uint32_t offset =
            (uint32_t)(loom_xdna_register_field_address(
                           context->family,
                           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_CURRENT,
                           input->shim->coordinate, &phase->descriptor) -
                       origin);
        IREE_RETURN_IF_ERROR(loom_aie2p_native_input_write_slot(
            context, &builder, input, offset, phase->words[6], slot));
      }
      if (input->slots != LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE) {
        const uint32_t offset =
            (uint32_t)(loom_xdna_register_field_address(
                           context->family,
                           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_CURRENT,
                           input->shim->coordinate, &input->credit_descriptor) -
                       origin);
        IREE_RETURN_IF_ERROR(loom_aie2p_native_input_write_slot(
            context, &builder, input, offset, input->ready_iteration, slot));
      }
    }
    for (iree_host_size_t i = 0; i < input->source->phase_count; ++i) {
      const loom_aie2p_native_distribution_endpoint_t* receiver =
          &input->phases[i].receiver;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint_start(
          context, &builder, receiver, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
          owner->control_packets[receiver->tile - context->tiles]));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&builder, NULL, 0));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_begin(context, input));
  if (input->invocation->ordinal) {
    loom_builder_t builder;
    loom_op_t* function;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
        &context->code, NULL, 0, NULL, 0, &builder, &input->retire, &function));
    const iree_host_size_t shim = input->shim - context->tiles;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_queue_retire(
        context, &builder, loom_low_func_def_body(function), input->shim,
        input->input_engine, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
        owner->control_packets[shim], owner->control_replies[shim]));
    if (input->binding) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_queue_retire(
          context, &builder, loom_low_func_def_body(function), input->shim,
          input->credit_engine, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
          owner->control_packets[shim], owner->control_replies[shim]));
    }
    const iree_host_size_t count =
        input->shared_receiver ? 1 : input->source->phase_count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      const loom_aie2p_native_distribution_endpoint_t* receiver =
          &input->phases[i].receiver;
      const iree_host_size_t tile = receiver->tile - context->tiles;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_queue_retire(
          context, &builder, loom_low_func_def_body(function), receiver->tile,
          receiver->engine, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
          owner->control_packets[tile], owner->control_replies[tile]));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&builder, NULL, 0));
  }
  return input->binding ? loom_aie2p_native_input_complete(context, input)
                        : iree_ok_status();
}

static iree_status_t loom_aie2p_native_input_constant(loom_builder_t* builder,
                                                      uint32_t value,
                                                      loom_value_id_t* result) {
  loom_op_t* op;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64((int32_t)value),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &op));
  *result = loom_scalar_constant_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_initialize_input_state(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_input_t* input, loom_value_id_t* frame_argument,
    loom_value_id_t* state, loom_value_id_t* state_views) {
  loom_op_t* op;
  const uint32_t initial[] = {0, 1, 0};
  for (iree_host_size_t i = 0; i < (input->binding ? 3 : 1); ++i) {
    IREE_RETURN_IF_ERROR(
        loom_aie2p_native_input_constant(builder, initial[i], &state[i]));
    state_views[i] = LOOM_VALUE_ID_INVALID;
  }
  if (input->cursor_index == UINT32_MAX) {
    return iree_ok_status();
  }
  if (*frame_argument == LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_define_value(
        context->code.module, loom_type_buffer(), frame_argument));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_native_cursor_view(
      builder, *frame_argument, input->cursor_index, false, &state_views[3]));
  const int64_t origin = 0;
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_RETURN_IF_ERROR(loom_view_load_build(builder, 0, 0, state_views[3], NULL,
                                            0, &origin, 1, 0, 0, type,
                                            LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t ordinal = loom_view_load_result(op);
  const loom_value_id_t one = state[1];
  loom_value_id_t capacity, admissions;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
      builder, input->binding->protocol->payload.capacity, &capacity));
  IREE_RETURN_IF_ERROR(loom_scalar_remui_build(builder, ordinal, capacity, type,
                                               LOOM_LOCATION_UNKNOWN, &op));
  state[0] = loom_scalar_remui_result(op);
  IREE_RETURN_IF_ERROR(loom_scalar_divui_build(builder, ordinal, capacity, type,
                                               LOOM_LOCATION_UNKNOWN, &op));
  IREE_RETURN_IF_ERROR(
      loom_scalar_addi_build(builder, 0, loom_scalar_divui_result(op), one,
                             type, LOOM_LOCATION_UNKNOWN, &op));
  state[1] = loom_scalar_addi_result(op);
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
      builder, input->source->admissions, &admissions));
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(
      builder, 0, ordinal, admissions, type, LOOM_LOCATION_UNKNOWN, &op));
  state[3] = loom_scalar_addi_result(op);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_initialize_input(
    loom_aie2p_native_context_t* context, loom_builder_t* builder,
    const loom_aie2p_native_input_t* input, loom_value_id_t* frame_argument,
    loom_value_id_t* state, loom_value_id_t* state_views) {
  if (input->continuation) {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_initialize_input_state(
        context, builder, input, frame_argument, state, state_views));
  }
  loom_op_t* op;
  if (input->invocation->ordinal) {
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build(builder, 0, 0, 0, input->start, state,
                              loom_aie2p_native_input_initializes_slot(input),
                              NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  }
  if (!input->continuation) {
    return loom_low_invoke_build(builder, 0, 0, 0, input->begin, NULL, 0, NULL,
                                 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op);
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_emit_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    iree_host_size_t action, bool after, loom_value_id_t* state) {
  const iree_host_size_t tail = input->source->phase_count - 1;
  const loom_pipeline_continuation_phase_t* observation =
      &input->continuation->phases[tail];
  const bool complete_batch =
      observation->kind == LOOM_PIPELINE_PROGRESS_OBSERVATION_COMPLETE_BATCH;
  const bool bulk = action == observation->action;
  const bool first =
      action ==
      input->continuation->phases[input->continuation->first_phase].action;
  if ((!bulk && !first) || (after && !bulk)) {
    return iree_ok_status();
  }
  loom_value_id_t* values = state + input->state_index;
  const iree_host_size_t position_index = input->binding ? 2 : 0;
  const loom_value_id_t position = values[position_index];
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t types[] = {type, type, type};
  loom_value_id_t zero, one;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(builder, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(builder, 1, &one));
  loom_op_t *op, *branch;
  if (!after) {
    loom_value_id_t predicate = LOOM_VALUE_ID_INVALID;
    if (bulk) {
      loom_value_id_t phase_position = position;
      if (!complete_batch) {
        loom_value_id_t period;
        IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
            builder, input->source->phases[tail].period, &period));
        IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
            builder, position, period, type, LOOM_LOCATION_UNKNOWN, &op));
        phase_position = loom_scalar_remui_result(op);
      }
      IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
          builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, phase_position, zero,
          LOOM_LOCATION_UNKNOWN, &op));
      predicate = loom_scalar_cmpi_result(op);
      if (!first) {
        IREE_RETURN_IF_ERROR(
            loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_NE,
                                   position, zero, LOOM_LOCATION_UNKNOWN, &op));
        IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
            builder, predicate, loom_scalar_cmpi_result(op),
            loom_type_scalar(LOOM_SCALAR_TYPE_I1), LOOM_LOCATION_UNKNOWN, &op));
        predicate = loom_scalar_andi_result(op);
      }
    }
    loom_builder_ip_t saved = {0};
    if (bulk) {
      IREE_RETURN_IF_ERROR(loom_scf_if_build(builder, 0, predicate, NULL, 0,
                                             NULL, 0, LOOM_LOCATION_UNKNOWN,
                                             &branch));
      saved = loom_builder_enter_region(builder, branch,
                                        loom_scf_if_then_region(branch));
    }
    const loom_value_id_t arguments[] = {input->binding ? values[0] : zero,
                                         input->binding ? values[1] : zero,
                                         bulk ? position : zero};
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(
        builder, 0, 0, 0, input->begin,
        input->binding ? arguments : arguments + 2, input->binding ? 3 : 1,
        NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
    if (bulk) {
      IREE_RETURN_IF_ERROR(
          loom_scf_yield_build(builder, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
      loom_builder_restore(builder, saved);
    }
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, 0, position, one, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t next_position = loom_scalar_addi_result(op);
  if (!input->binding) {
    values[position_index] = next_position;
    return iree_ok_status();
  }
  loom_value_id_t limit;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
      builder, complete_batch ? 1 : observation->read_count, &limit));
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ,
                             next_position, limit, LOOM_LOCATION_UNKNOWN, &op));
  IREE_RETURN_IF_ERROR(
      loom_scf_if_build(builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION,
                        loom_scalar_cmpi_result(op), types, 3, NULL, 0,
                        LOOM_LOCATION_UNKNOWN, &branch));
  const loom_builder_ip_t saved = loom_builder_enter_region(
      builder, branch, loom_scf_if_then_region(branch));
  const uint64_t capacity = input->binding->protocol->payload.capacity;
  loom_value_id_t token;
  if (input->slots == LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE) {
    const uint64_t last = input->source->admissions - 1;
    loom_value_id_t last_slot, last_generation;
    if (input->cursor_index == UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
          builder, last % capacity, &last_slot));
      IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
          builder, last / capacity + 1, &last_generation));
    } else {
      loom_value_id_t slots;
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_input_constant(builder, capacity, &slots));
      IREE_RETURN_IF_ERROR(loom_scalar_subi_build(
          builder, 0, values[3], one, type, LOOM_LOCATION_UNKNOWN, &op));
      const loom_value_id_t last_ordinal = loom_scalar_subi_result(op);
      IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
          builder, last_ordinal, slots, type, LOOM_LOCATION_UNKNOWN, &op));
      last_slot = loom_scalar_remui_result(op);
      IREE_RETURN_IF_ERROR(loom_scalar_divui_build(
          builder, last_ordinal, slots, type, LOOM_LOCATION_UNKNOWN, &op));
      IREE_RETURN_IF_ERROR(
          loom_scalar_addi_build(builder, 0, loom_scalar_divui_result(op), one,
                                 type, LOOM_LOCATION_UNKNOWN, &op));
      last_generation = loom_scalar_addi_result(op);
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
        builder,
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_ENABLE_TOKEN_ISSUE,
            1),
        &token));
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, values[0], last_slot,
        LOOM_LOCATION_UNKNOWN, &op));
    const loom_value_id_t slot_matches = loom_scalar_cmpi_result(op);
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
        builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, values[1], last_generation,
        LOOM_LOCATION_UNKNOWN, &op));
    IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
        builder, slot_matches, loom_scalar_cmpi_result(op),
        loom_type_scalar(LOOM_SCALAR_TYPE_I1), LOOM_LOCATION_UNKNOWN, &op));
    IREE_RETURN_IF_ERROR(
        loom_scf_select_build(builder, loom_scalar_andi_result(op), token, zero,
                              type, LOOM_LOCATION_UNKNOWN, &op));
    token = loom_scf_select_result(op);
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
        builder,
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_ENABLE_TOKEN_ISSUE,
            1),
        &token));
  }
  const loom_value_id_t arguments[] = {values[0], values[1], token};
  IREE_RETURN_IF_ERROR(loom_low_invoke_build(builder, 0, 0, 0, input->complete,
                                             arguments, 3, NULL, 0, NULL, 0,
                                             LOOM_LOCATION_UNKNOWN, &op));
  loom_value_id_t slots;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_input_constant(builder, capacity, &slots));
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, 0, values[0], one, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t advanced = loom_scalar_addi_result(op);
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, advanced,
                             slots, LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t wraps = loom_scalar_cmpi_result(op);
  IREE_RETURN_IF_ERROR(loom_scf_select_build(builder, wraps, zero, advanced,
                                             type, LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t next_slot = loom_scf_select_result(op);
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, 0, values[1], one, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
  IREE_RETURN_IF_ERROR(
      loom_scf_select_build(builder, wraps, loom_scalar_addi_result(op),
                            values[1], type, LOOM_LOCATION_UNKNOWN, &op));
  // Complete-batch arrival returns the external slot, but local replay still
  // owns its SRAM. Keep advancing the local observation position so subsequent
  // replays cannot readmit the next external record.
  const loom_value_id_t next[] = {next_slot, loom_scf_select_result(op),
                                  complete_batch ? next_position : zero};
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, next, 3, LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_enter_region(builder, branch, loom_scf_if_else_region(branch));
  const loom_value_id_t advance[] = {values[0], values[1], next_position};
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, advance, 3, LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_restore(builder, saved);
  memcpy(values, loom_op_results(branch), sizeof(next));
  return iree_ok_status();
}
