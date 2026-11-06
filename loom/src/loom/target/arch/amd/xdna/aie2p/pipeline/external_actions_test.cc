// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <unordered_map>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/external.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/worker.h"
#include "loom/target/arch/amd/xdna/array/facts.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

namespace {

// The native emitter consumes an already selected physical input. These tests
// inspect its control packets with different entry slots, independently of
// helper names, temporary SSA numbering, and register allocation.
class ExternalInputActionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_context_initialize(iree_allocator_system(), &ir_context_);
    iree_host_size_t count;
    const auto* vtables = loom_low_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&ir_context_, LOOM_DIALECT_LOW,
                                                 vtables, count));
    IREE_ASSERT_OK(loom_context_finalize(&ir_context_));
    IREE_ASSERT_OK(loom_module_allocate(&ir_context_, IREE_SV("input_actions"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    IREE_ASSERT_OK(
        loom_aie2p_worker_builder_initialize(module_, &arena_, &context_.code));
    realization_.symbols = &symbols_;
    realization_.worker_versions = &worker_version_;
    context_.code.realization = &realization_;
    context_.family = loom_xdna_npu2_array_family();
    context_.tiles = tiles_.data();
    context_.workers = &controller_;
    context_.core_engines = &engine_;
    context_.execution.workers = &engine_;
    controller_.control_packets = packets_.data();
    controller_.control_replies = replies_.data();
    tiles_[0].coordinate = {0, 0};
    tiles_[1].coordinate = {0, 2};
    for (auto& tile : tiles_) {
      tile.facts = loom_xdna_array_tile_facts(context_.family, tile.coordinate);
    }
    tiles_[1].memory_layout = &loom_aie2p_compute_memory;
    invocation_.ordinal = 1;
    input_.invocation = &invocation_;
    input_.begin = loom_symbol_ref_null();
    input_.start = loom_symbol_ref_null();
    input_.complete = loom_symbol_ref_null();
    input_.retire = loom_symbol_ref_null();
    input_.binding = &binding_;
    input_.source = &source_;
    input_.continuation = &continuation_;
    input_.phases = phases_.data();
    input_.shim = &tiles_[0];
    input_.control_packet = packets_[0];
    input_.completion_packet = replies_[0];
    input_.credit_packet = 6;
    input_.ready_descriptor = 8;
    input_.credit_descriptor = 10;
    source_.phase_count = phases_.size();
    source_.phases = source_phases_.data();
    source_.admissions = 3;
    protocol_.payload.capacity = 5;
    protocol_.payload.byte_stride = 64;
    protocol_.notification_byte_stride = 128;
    for (size_t i = 0; i < phases_.size(); ++i) {
      source_phases_[i].period = 1;
      source_phases_[i].transfers = 1;
      phases_[i].source = &source_phases_[i];
      phases_[i].descriptor = i;
      phases_[i].repetitions = 1;
      phases_[i].words[6] = Iteration(5, 64);
      phases_[i].receiver.tile = &tiles_[1];
      phases_[i].receiver.engine = i;
      tasks_[i] |= loom_xdna_register_field_encode(
          loom_aie2p_compute_memory.directions[1].repeat, 2);
      phases_[i].receiver.sequence.tasks = &tasks_[i];
      phases_[i].receiver.sequence.task_count = 1;
    }
    input_.ready_iteration = Iteration(5, 128);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&ir_context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  static uint32_t Iteration(uint32_t capacity, uint32_t stride) {
    return loom_xdna_register_field_encode(
               LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_WRAP,
               capacity - 1) |
           loom_xdna_register_field_encode(
               LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_STEP_SIZE,
               stride / 4 - 1);
  }

  uint32_t Descriptor(uint32_t reference) const {
    return loom_low_descriptor_set_descriptor_ordinal(
        context_.code.descriptors,
        &context_.code.descriptors->descriptors[reference]);
  }

  loom_region_t* Body(loom_symbol_ref_t symbol) const {
    return loom_low_func_def_body(
        module_->symbols.entries[symbol.symbol_id].defining_op);
  }

  // Evaluate the straight-line entry packet payloads. Unexpected instructions
  // fail instead of being ignored; no execution or DMA is performed.
  void EntryWrites(uint32_t slot,
                   std::vector<std::array<uint32_t, 3>>* writes) {
    const auto* entry = loom_region_entry_block(Body(input_.start));
    std::unordered_map<loom_value_id_t, uint32_t> values;
    if (entry->arg_count) {
      values[entry->arg_ids[0]] = slot;
    }
    std::vector<uint32_t> packet;
    for (const auto* op = entry->first_op; op; op = op->next_op) {
      if (loom_low_const_isa(op)) {
        const auto attributes = loom_low_const_attrs(op);
        ASSERT_EQ(attributes.count, 1u);
        values[loom_low_const_result(op)] = attributes.entries[0].value.i64;
        continue;
      }
      if (loom_low_return_isa(op)) {
        continue;
      }
      ASSERT_TRUE(loom_low_op_isa(op));
      const auto descriptor = loom_low_op_descriptor(op);
      const auto* operands = loom_op_const_operands(op);
      if (descriptor == Descriptor(AIE2P_CORE_DESCRIPTOR_REF_LSHL_I32)) {
        values[loom_op_const_results(op)[0]] = values.at(operands[0])
                                               << values.at(operands[1]);
      } else if (descriptor == Descriptor(AIE2P_CORE_DESCRIPTOR_REF_OR_I32)) {
        values[loom_op_const_results(op)[0]] =
            values.at(operands[0]) | values.at(operands[1]);
      } else {
        ASSERT_TRUE(
            descriptor ==
                Descriptor(AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32) ||
            descriptor ==
                Descriptor(AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32));
        packet.push_back(values.at(operands[0]));
        if (descriptor ==
            Descriptor(AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32)) {
          ASSERT_EQ(packet.size(), 3u);
          writes->push_back({packet[0] & 31u, packet[1] & 0xfffffu, packet[2]});
          packet.clear();
        }
      }
    }
    EXPECT_TRUE(packet.empty());
  }

  std::array<uint32_t, 3> SlotWrite(uint16_t descriptor, uint32_t stride,
                                    uint32_t slot) const {
    return {
        packets_[0],
        static_cast<uint32_t>(loom_xdna_register_field_address(
            context_.family,
            LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_CURRENT,
            tiles_[0].coordinate, &descriptor)),
        Iteration(protocol_.payload.capacity, stride) |
            loom_xdna_register_field_encode(
                LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD6_ITERATION_CURRENT,
                slot),
    };
  }

  std::array<uint32_t, 3> ReceiverStart(uint16_t engine) const {
    return {
        packets_[1],
        static_cast<uint32_t>(loom_xdna_register_field_address(
                                  context_.family,
                                  loom_aie2p_compute_memory.directions[1].start,
                                  tiles_[1].coordinate, &engine) -
                              (uint64_t{2} << context_.family->row_shift)),
        tasks_[engine],
    };
  }

  // Arena and module own the generated helper functions.
  iree_arena_block_pool_t blocks_ = {};
  // Scratch allocation for helper symbol ownership.
  iree_arena_allocator_t arena_ = {};
  // Low operation vocabulary used by the emitter.
  loom_context_t ir_context_ = {};
  // Generated helper module, independent of a parsed source program.
  loom_module_t* module_ = nullptr;
  // One shim and one compute-memory DMA module.
  std::array<loom_aie2p_native_tile_t, 2> tiles_ = {};
  // Selected physical packet classes by tile.
  std::array<uint8_t, 2> packets_ = {2, 3};
  // Matching completion/status reply classes.
  std::array<uint8_t, 2> replies_ = {4, 5};
  // Receiver task words carrying different descriptor IDs.
  std::array<uint32_t, 2> tasks_ = {2, 3};
  // Prefix and payload source cardinalities.
  std::array<loom_pipeline_transport_input_phase_t, 2> source_phases_ = {};
  // Selected descriptor and receiver geometry for each phase.
  std::array<loom_aie2p_native_input_phase_t, 2> phases_ = {};
  // Source lease/phase counts shared by all slot realizations.
  loom_pipeline_transport_input_t source_ = {};
  // Existing observation-selected continuation; no extra core is allocated.
  loom_pipeline_continuation_t continuation_ = {};
  // Parent ring geometry supplied to the native realization.
  loom_memory_channel_protocol_t protocol_ = {};
  // Existing payload and notification buffer arguments.
  loom_memory_channel_binding_t binding_ = {&protocol_, 0, 1};
  // Child owning this input's DMA start and retirement.
  loom_pipeline_invocation_t invocation_ = {};
  // Input plan consumed by the helper emitter.
  loom_aie2p_native_input_t input_ = {};
  // Physical core representative and its control routes.
  loom_aie2p_native_worker_t controller_ = {};
  // Sole instruction engine and its representative worker.
  iree_host_size_t engine_ = 0;
  // Context-free helper registration, as supported before target refinement.
  loom_target_function_version_t* worker_version_ = nullptr;
  // Canonical helper symbol allocation state.
  loom_pipeline_realization_symbols_t symbols_ = {};
  // Realization owning the generated helpers.
  loom_pipeline_realization_t realization_ = {};
  // Native emitter and physical inventory.
  loom_aie2p_native_context_t context_ = {};
};

TEST_F(ExternalInputActionsTest, ResumesEveryMutableDescriptorAtTheParentSlot) {
  protocol_.payload.capacity = 4;
  input_.ready_iteration = Iteration(4, 128);
  for (auto& phase : phases_) {
    phase.words[6] = Iteration(4, 64);
  }
  for (const auto mode : {LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE,
                          LOOM_AIE2P_NATIVE_INPUT_SLOTS_NOTIFICATIONS,
                          LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD}) {
    input_.slots = mode;
    source_phases_[1].period = 64;
    source_phases_[1].transfers = 64;
    phases_[1].repetitions =
        mode == LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD ? 1 : 64;
    tasks_[1] =
        3 | loom_xdna_register_field_encode(
                loom_aie2p_compute_memory.directions[1].repeat, 3 * 64 - 1);
    input_.ready_descriptor =
        mode == LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD ? 2 : 5;
    input_.credit_descriptor =
        input_.ready_descriptor +
        (mode == LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE ? 4 : 1);
    IREE_ASSERT_OK(loom_aie2p_native_prepare_input(&context_, &input_));
    for (const uint32_t slot : {0u, 2u, 3u}) {
      SCOPED_TRACE(::testing::Message() << mode << ":" << slot);
      std::vector<std::array<uint32_t, 3>> actual;
      EntryWrites(slot, &actual);
      std::vector<std::array<uint32_t, 3>> expected = {SlotWrite(0, 64, slot)};
      if (mode == LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD) {
        expected.push_back(SlotWrite(1, 64, slot));
      }
      if (mode != LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE) {
        expected.push_back(SlotWrite(input_.credit_descriptor, 128, slot));
      }
      expected.push_back(ReceiverStart(0));
      expected.push_back(ReceiverStart(1));
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(ExternalInputActionsTest, FixedSlotTasksDoNotNeedIteratorWrites) {
  protocol_.payload.capacity = 4;
  input_.ready_descriptor = 4;
  input_.credit_descriptor = 8;
  source_.phase_count = 1;
  input_.slots = LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE;
  IREE_ASSERT_OK(loom_aie2p_native_prepare_input(&context_, &input_));
  EXPECT_EQ(loom_region_entry_block(Body(input_.start))->arg_count, 0u);
  std::vector<std::array<uint32_t, 3>> actual;
  EntryWrites(3, &actual);
  EXPECT_EQ(actual, (std::vector<std::array<uint32_t, 3>>{ReceiverStart(0)}));
}

TEST_F(ExternalInputActionsTest, SingleSlotPrefixesRetainConstantAddresses) {
  protocol_.payload.capacity = 1;
  input_.slots = LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE;
  input_.ready_descriptor = 2;
  input_.credit_descriptor = 3;
  for (auto& phase : phases_) {
    phase.words[6] = 0;
  }
  IREE_ASSERT_OK(loom_aie2p_native_prepare_input(&context_, &input_));
  EXPECT_EQ(loom_region_entry_block(Body(input_.start))->arg_count, 0u);
  std::vector<std::array<uint32_t, 3>> actual;
  EntryWrites(0, &actual);
  EXPECT_EQ(actual, (std::vector<std::array<uint32_t, 3>>{ReceiverStart(0),
                                                          ReceiverStart(1)}));
}

TEST_F(ExternalInputActionsTest, RootInputRetainsConfigurationInitialization) {
  invocation_.ordinal = 0;
  input_.slots = LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD;
  IREE_ASSERT_OK(loom_aie2p_native_prepare_input(&context_, &input_));
  EXPECT_FALSE(loom_symbol_ref_is_valid(input_.start));
  EXPECT_FALSE(loom_symbol_ref_is_valid(input_.retire));
  EXPECT_TRUE(loom_symbol_ref_is_valid(input_.begin));
  EXPECT_TRUE(loom_symbol_ref_is_valid(input_.complete));
}

TEST_F(ExternalInputActionsTest, ChildReturnRetiresEveryPhysicalInputQueue) {
  input_.slots = LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD;
  IREE_ASSERT_OK(loom_aie2p_native_prepare_input(&context_, &input_));
  const auto* body = Body(input_.retire);
  std::vector<uint32_t> observed_masks;
  std::vector<uint32_t> observed_addresses;
  std::unordered_map<loom_value_id_t, uint32_t> constants;
  for (uint16_t b = 0; b < body->block_count; ++b) {
    const auto* block = body->blocks[b];
    for (const auto* op = block->first_op; op; op = op->next_op) {
      if (loom_low_const_isa(op)) {
        const auto attributes = loom_low_const_attrs(op);
        ASSERT_EQ(attributes.count, 1u);
        constants[loom_low_const_result(op)] = attributes.entries[0].value.i64;
      } else if (loom_low_op_isa(op)) {
        const auto descriptor = loom_low_op_descriptor(op);
        if (descriptor == Descriptor(AIE2P_CORE_DESCRIPTOR_REF_AND_I32)) {
          observed_masks.push_back(constants.at(loom_op_const_operands(op)[1]));
        } else if (descriptor ==
                   Descriptor(
                       AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32)) {
          observed_addresses.push_back(
              constants.at(loom_op_const_operands(op)[0]) & 0xfffffu);
        }
      } else if (loom_low_cond_br_isa(op)) {
        EXPECT_EQ(loom_low_cond_br_true_dest(op), block);
        EXPECT_NE(loom_low_cond_br_false_dest(op), block);
      }
    }
  }
  std::vector<uint32_t> expected_masks;
  std::vector<uint32_t> expected_addresses;
  const std::array<loom_xdna_dma_direction_t, 4> directions = {
      LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
      LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
      LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
      LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
  };
  for (size_t i = 0; i < directions.size(); ++i) {
    const auto* tile = &tiles_[i >= 2];
    const auto status = loom_aie2p_native_queue_status(
        &context_, tile, i == 3 ? 1 : 0, directions[i]);
    expected_masks.push_back(status.pending_mask);
    expected_addresses.push_back(
        static_cast<uint32_t>(
            status.address -
            (uint64_t{tile->coordinate.row} << context_.family->row_shift)) &
        ~15u);
  }
  EXPECT_EQ(observed_addresses, expected_addresses);
  EXPECT_EQ(observed_masks, expected_masks);
}

}  // namespace
