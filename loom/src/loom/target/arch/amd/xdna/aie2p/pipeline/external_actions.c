// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/external.h"

static iree_status_t loom_aie2p_native_input_begin(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  const loom_type_t arguments[] = {code->scalar_type, code->scalar_type,
                                   code->scalar_type};
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
      code, arguments, 3, NULL, 0, &builder, &input->begin, &function));
  loom_region_t* body = loom_low_func_def_body(function);
  const loom_value_id_t* values = loom_region_entry_block(body)->arg_ids;
  loom_block_t *poll, *prefix, *payload;
  IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &poll));
  IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &prefix));
  IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &payload));
  loom_value_id_t zero, first, ready_base, ready;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32, values[2], zero,
      &first));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, &builder, input->ready_descriptor, &ready_base));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, ready_base, values[0],
      &ready));
  loom_op_t* op;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(&builder, first, poll, payload,
                                              LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_set_block(&builder, poll);
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, input->control_packet, input->input_queue, &ready, 1));
  loom_value_id_t header, observed, available;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
      loom_named_attr_slice_empty(), &code->scalar_type, &header));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_op(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0,
      loom_named_attr_slice_empty(), &code->scalar_type, &observed));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32, observed, values[1],
      &available));
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(&builder, available, prefix, poll,
                                              LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_set_block(&builder, prefix);
  for (iree_host_size_t i = 0; i + 1 < input->source->phase_count; ++i) {
    loom_value_id_t task;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &builder, input->phases[i].descriptor, &task));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        code, &builder, input->control_packet, input->input_queue, &task, 1));
  }
  IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, payload, NULL, 0,
                                         LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_set_block(&builder, payload);
  const loom_aie2p_native_input_phase_t* phase =
      &input->phases[input->source->phase_count - 1];
  const uint32_t word =
      phase->descriptor |
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_REPEAT_COUNT,
          phase->repetitions - 1);
  loom_value_id_t base, task;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, word, &base));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, base, values[0],
      &task));
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
  loom_value_id_t base, descriptor, task;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
      code, &builder, input->credit_descriptor, &base));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
      code, &builder, AIE2P_CORE_DESCRIPTOR_REF_ADD_I32, base, values[0],
      &descriptor));
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
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_begin(context, input));
  return loom_aie2p_native_input_complete(context, input);
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

iree_status_t loom_aie2p_native_emit_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    iree_host_size_t action, bool after, loom_value_id_t* state) {
  const iree_host_size_t tail = input->source->phase_count - 1;
  const bool bulk = action == input->continuation.phase_actions[tail];
  const bool first =
      action ==
      input->continuation.phase_actions[input->continuation.first_phase];
  if ((!bulk && !first) || (after && !bulk)) {
    return iree_ok_status();
  }
  loom_value_id_t* values = state + input->state_index;
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t types[] = {type, type, type};
  loom_value_id_t zero, one;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(builder, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(builder, 1, &one));
  loom_op_t *op, *branch;
  if (!after) {
    loom_value_id_t predicate = LOOM_VALUE_ID_INVALID;
    if (bulk) {
      loom_value_id_t period;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
          builder, input->source->phases[tail].period, &period));
      IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
          builder, values[2], period, type, LOOM_LOCATION_UNKNOWN, &op));
      IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
          builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, loom_scalar_remui_result(op),
          zero, LOOM_LOCATION_UNKNOWN, &op));
      predicate = loom_scalar_cmpi_result(op);
      if (!first) {
        IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
            builder, LOOM_SCALAR_CMPI_PREDICATE_NE, values[2], zero,
            LOOM_LOCATION_UNKNOWN, &op));
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
    const loom_value_id_t arguments[] = {values[0], values[1],
                                         bulk ? values[2] : zero};
    IREE_RETURN_IF_ERROR(loom_low_invoke_build(builder, 0, 0, 0, input->begin,
                                               arguments, 3, NULL, 0, NULL, 0,
                                               LOOM_LOCATION_UNKNOWN, &op));
    if (bulk) {
      IREE_RETURN_IF_ERROR(
          loom_scf_yield_build(builder, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
      loom_builder_restore(builder, saved);
    }
    return iree_ok_status();
  }
  loom_value_id_t limit;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
      builder, input->source->phases[tail].transfers, &limit));
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, 0, values[2], one, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t next_position = loom_scalar_addi_result(op);
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
  const uint64_t last = input->source->admissions - 1;
  loom_value_id_t last_slot, last_generation, token;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_native_input_constant(builder, last % capacity, &last_slot));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
      builder, last / capacity + 1, &last_generation));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
      builder,
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_ENABLE_TOKEN_ISSUE,
          1),
      &token));
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, values[0],
                             last_slot, LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t slot_matches = loom_scalar_cmpi_result(op);
  IREE_RETURN_IF_ERROR(
      loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, values[1],
                             last_generation, LOOM_LOCATION_UNKNOWN, &op));
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
      builder, slot_matches, loom_scalar_cmpi_result(op),
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), LOOM_LOCATION_UNKNOWN, &op));
  IREE_RETURN_IF_ERROR(
      loom_scf_select_build(builder, loom_scalar_andi_result(op), token, zero,
                            type, LOOM_LOCATION_UNKNOWN, &op));
  const loom_value_id_t arguments[] = {values[0], values[1],
                                       loom_scf_select_result(op)};
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
  const loom_value_id_t next[] = {next_slot, loom_scf_select_result(op), zero};
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
