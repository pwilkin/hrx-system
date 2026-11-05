// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/external.h"

#include "loom/analysis/storage_geometry.h"

iree_status_t loom_aie2p_native_select_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, iree_host_size_t worker,
    bool* out_eligible) {
  *out_eligible = false;
  const loom_pipeline_worker_t* source = &realization->workers[worker];
  const loom_pipeline_transport_input_t* plan = source->transport.input;
  if (!plan || !plan->acquisition || !plan->release) {
    return iree_ok_status();
  }
  const loom_aie2p_native_channel_t* external = loom_aie2p_native_channel(
      context, realization, plan->acquisition->channel);
  if (!external->external) {
    return iree_ok_status();
  }
  // Prefixes fit independently before the first bulk demand, even if the
  // observer reads a later phase first. The final phase may span many windows;
  // its exact arrivals drive additional address periods and final reclamation.
  for (iree_host_size_t i = 0; i < plan->phase_count; ++i) {
    const loom_pipeline_transport_input_phase_t* phase = &plan->phases[i];
    const loom_aie2p_native_channel_t* local = loom_aie2p_native_channel(
        context, realization, phase->reservation->channel);
    if (local->external || phase->destination_byte_offset ||
        phase->transfer->request.transferred_byte_count !=
            local->record_byte_length ||
        !iree_any_bit_set(phase->transfer->request.source.flags,
                          LOOM_MOVEMENT_ENDPOINT_STATIC_LENGTH) ||
        !phase->period || phase->period > UINT32_MAX ||
        phase->transfers > UINT32_MAX || !plan->admissions ||
        plan->admissions > UINT32_MAX / phase->transfers ||
        phase->transfers % phase->period ||
        (i + 1 < plan->phase_count && phase->transfers != 1) ||
        phase->source_byte_offset < 0 || phase->source_byte_stride < 0 ||
        phase->source_byte_stride > UINT32_MAX) {
      return iree_ok_status();
    }
  }
  loom_aie2p_native_input_t* input;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena, sizeof(*input),
                                           (void**)&input));
  *input = (loom_aie2p_native_input_t){
      .worker = (uint32_t)worker,
      .binding = external->external,
      .source = plan,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, plan->phase_count, sizeof(*input->phases),
      (void**)&input->phases));
  for (iree_host_size_t i = 0; i < plan->phase_count; ++i) {
    input->phases[i] = (loom_aie2p_native_input_phase_t){
        .source = &plan->phases[i],
        .destination = loom_aie2p_native_channel(
            context, realization, plan->phases[i].reservation->channel),
    };
  }
  context->workers[worker].input = input;
  *out_eligible = true;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_input_descriptor(
    loom_aie2p_native_context_t* context, const loom_aie2p_native_tile_t* shim,
    uint16_t descriptor, uint32_t binding, uint64_t offset, uint64_t length,
    const uint32_t words[8]) {
  loom_aie2p_native_external_descriptor_t* selected;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      context->pass->arena, sizeof(*selected), (void**)&selected));
  *selected = (loom_aie2p_native_external_descriptor_t){
      .next = context->external_descriptors,
      .tile = shim,
      .descriptor = descriptor,
      .binding = binding,
      .byte_offset = offset,
      .byte_length = length,
  };
  memcpy(selected->words, words, sizeof(selected->words));
  context->external_descriptors = selected;
  return iree_ok_status();
}

static bool loom_aie2p_native_input_route(
    loom_aie2p_native_context_t* context, uint32_t* next_source,
    uint32_t canonical, loom_xdna_tile_coordinate_t source,
    loom_xdna_stream_port_t source_port, uint8_t source_channel,
    loom_xdna_tile_coordinate_t destination,
    loom_xdna_stream_port_t destination_port, uint8_t destination_channel,
    uint8_t packet) {
  const iree_host_size_t first = context->routing.route_count;
  if (!loom_aie2p_array_route_stream(
          &context->routing, (*next_source)++, canonical, source, source_port,
          source_channel, destination, destination_port, destination_channel,
          true)) {
    return false;
  }
  loom_aie2p_native_route_label(context, first, packet);
  return true;
}

iree_status_t loom_aie2p_native_route_input(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    loom_aie2p_native_input_t* input, uint32_t* next_source,
    uint8_t* next_controller, bool* out_valid) {
  *out_valid = false;
  const loom_pipeline_worker_t* source = &realization->workers[input->worker];
  const loom_op_t* declaration = source->source->declaration;
  const loom_memory_channel_protocol_t* protocol = input->binding->protocol;
  const uint64_t capacity = protocol->payload.capacity;
  input->shim =
      &context->tiles[context->workers[input->worker].tile->coordinate.column *
                      context->family->row_count];
  loom_aie2p_native_tile_t* shim = input->shim;
  const loom_xdna_dma_facts_t* dma = &shim->facts->dma;
  const uint64_t descriptors = input->source->phase_count - 1 + capacity * 3;
  uint16_t descriptor_base, descriptor_count;
  uint16_t* used = loom_aie2p_native_descriptor_bank(
      shim, shim->next_mm2s, &descriptor_base, &descriptor_count);
  if (shim->next_mm2s == dma->channel_count_per_direction ||
      shim->next_s2mm == dma->channel_count_per_direction ||
      descriptors > descriptor_count - *used ||
      input->source->phase_count > dma->task_queue_depth ||
      capacity > dma->task_queue_depth ||
      capacity > (UINT64_C(1) << dma->iteration_bits) ||
      protocol->payload.byte_stride % dma->transfer_length_granularity ||
      protocol->payload.byte_stride / dma->transfer_length_granularity >
          (UINT64_C(1) << dma->step_size_bits)) {
    return loom_aie2p_native_reject(
        context, declaration,
        IREE_SV(
            "shim queues and immutable descriptors for shared input slots"));
  }
  input->input_engine = shim->next_mm2s++;
  input->credit_engine = shim->next_s2mm++;
  const uint16_t engine = input->input_engine;
  const uint16_t credit_engine = input->credit_engine;
  const uint64_t origin = (uint64_t)shim->coordinate.column
                          << context->family->column_shift;
  input->input_queue =
      (uint32_t)(loom_xdna_register_field_address(
                     context->family,
                     LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_START_BD_ID,
                     shim->coordinate, &engine) -
                 origin);
  input->credit_queue =
      (uint32_t)(loom_xdna_register_field_address(
                     context->family,
                     LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_S2MM_TASK_QUEUE_START_BD_ID,
                     shim->coordinate, &credit_engine) -
                 origin);
  const uint32_t canonical = *next_source;
  for (iree_host_size_t i = 0; i < input->source->phase_count; ++i) {
    loom_aie2p_native_input_phase_t* phase = &input->phases[i];
    const loom_pipeline_transport_input_phase_t* plan = phase->source;
    if (loom_movement_endpoint_minimum_byte_alignment(
            &plan->transfer->request.source) < dma->address_alignment) {
      return loom_aie2p_native_reject(
          context, plan->transfer->request.op,
          IREE_SV("aligned shared input DMA sources"));
    }
    iree_string_view_t reason;
    bool valid = false;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_select_channel_endpoint(
        context, phase->destination, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
        (uint32_t)(input->source->admissions * plan->transfers), 1, NULL,
        &phase->receiver, &reason, &valid));
    if (!valid) {
      return loom_aie2p_native_reject(context, declaration, reason);
    }
    phase->receiver.wait_for_idle = true;
    loom_storage_geometry_t geometry;
    if (!loom_storage_geometry_query(
            &source->facts.context, context->code.module,
            plan->transfer->request.source.type, &geometry) ||
        !loom_aie2p_native_encode_ingress(
            context->family, &geometry, dma,
            (uint32_t)plan->transfer->request.transferred_byte_count,
            (uint32_t)plan->source_byte_stride, (uint32_t)plan->period,
            phase->words, &phase->repetitions) ||
        phase->repetitions > dma->maximum_task_repeat_count) {
      return loom_aie2p_native_reject(
          context, declaration,
          IREE_SV(
              "a shared input address period representable by one shim task"));
    }
    phase->words[2] |=
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_ENABLE_PACKET, 1) |
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_PACKET_ID, i);
    phase->descriptor = descriptor_base + *used;
    const bool prefix = i + 1 < input->source->phase_count;
    const uint32_t count = prefix ? 1 : (uint32_t)capacity;
    *used += count;
    if (prefix && capacity > 1) {
      phase->words[6] =
          loom_xdna_register_field_encode(
              LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_WRAP,
              capacity - 1) |
          loom_xdna_register_field_encode(
              LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_STEP_SIZE,
              protocol->payload.byte_stride / dma->transfer_length_granularity -
                  1);
    }
    const uint64_t span =
        (uint64_t)plan->source_byte_stride * (plan->period - 1) +
        plan->transfer->request.source.static_byte_length;
    for (uint32_t slot = 0; slot < count; ++slot) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_input_descriptor(
          context, shim, phase->descriptor + slot,
          input->binding->payload_binding,
          protocol->payload.byte_offset + slot * protocol->payload.byte_stride +
              plan->source_byte_offset,
          span + (prefix ? (capacity - 1) * protocol->payload.byte_stride : 0),
          phase->words));
    }
    if (!loom_aie2p_native_input_route(
            context, next_source, canonical, shim->coordinate,
            LOOM_XDNA_STREAM_PORT_DMA, input->input_engine,
            phase->receiver.tile->coordinate, LOOM_XDNA_STREAM_PORT_DMA,
            phase->receiver.engine, (uint8_t)i)) {
      return loom_aie2p_native_reject(
          context, declaration,
          IREE_SV("packet routes from shared input to each local phase"));
    }
  }
  loom_aie2p_native_worker_t* observer =
      &context->workers[input->continuation.worker];
  bool valid = false;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_select_control_route(
      context, realization, input->continuation.worker, shim, true, next_source,
      next_controller, &valid));
  if (!valid) {
    return iree_ok_status();
  }
  loom_aie2p_native_worker_t* owner =
      &context
           ->workers[context->execution.workers
                         [context->core_engines[input->continuation.worker]]];
  input->control_packet = owner->control_packets[shim - context->tiles];
  input->completion_packet = owner->control_replies[shim - context->tiles];
  const iree_host_size_t core = observer->tile - context->tiles;
  if (owner->control_packet_count == 4 ||
      !loom_aie2p_native_input_route(
          context, next_source, context->control_sources[core * 2],
          observer->tile->coordinate, LOOM_XDNA_STREAM_PORT_CORE, 0,
          shim->coordinate, LOOM_XDNA_STREAM_PORT_DMA, input->credit_engine,
          owner->control_packet_count) ||
      !loom_aie2p_native_input_route(
          context, next_source, canonical, shim->coordinate,
          LOOM_XDNA_STREAM_PORT_DMA, input->input_engine,
          observer->tile->coordinate, LOOM_XDNA_STREAM_PORT_CORE, 0,
          (uint8_t)input->source->phase_count)) {
    return loom_aie2p_native_reject(
        context, declaration,
        IREE_SV(
            "shared generation query and credit routes to the observing core"));
  }
  input->credit_packet = owner->control_packet_count++;
  input->ready_descriptor = descriptor_base + *used;
  input->credit_descriptor = input->ready_descriptor + capacity;
  *used += capacity * 2;
  uint32_t words[8] = {0};
  words[0] =
      protocol->generation_byte_length / dma->transfer_length_granularity -
      dma->transfer_length_offset;
  words[4] = loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD4_BURST_LENGTH, 3);
  words[5] = loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD5_AXI_CACHE, 2);
  words[7] = loom_xdna_register_field_encode(
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD7_VALID_BD, 1);
  for (uint32_t slot = 0; slot < capacity; ++slot) {
    words[2] =
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_ENABLE_PACKET, 1) |
        loom_xdna_register_field_encode(
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD2_PACKET_ID,
            input->source->phase_count);
    IREE_RETURN_IF_ERROR(loom_aie2p_native_input_descriptor(
        context, shim, input->ready_descriptor + slot,
        input->binding->notification_binding,
        protocol->ready_byte_offset + slot * protocol->notification_byte_stride,
        protocol->generation_byte_length, words));
    words[2] = 0;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_input_descriptor(
        context, shim, input->credit_descriptor + slot,
        input->binding->notification_binding,
        protocol->free_byte_offset + slot * protocol->notification_byte_stride,
        protocol->generation_byte_length, words));
  }
  context->bindings[input->binding->payload_binding].access |= 1;
  context->bindings[input->binding->notification_binding].access |= 3;
  *out_valid = true;
  return iree_ok_status();
}
