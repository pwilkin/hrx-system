// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/external.h"

#include <array>

#include "iree/testing/gtest.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

namespace {

class InputCursorOwnershipTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Common execution admission assigned two serial child controllers to
    // one core. Communication-only workers have no instruction engine.
    for (size_t i = 0; i < inputs_.size(); ++i) {
      workers_[i * 2].invocation = &children_[i];
      workers_[i * 2 + 1].invocation = &children_[i];
      inputs_[i].worker = i * 2;
      inputs_[i].controller = i * 2 + 1;
      inputs_[i].invocation = &children_[i];
    }
    invocations_.workers = workers_.data();
    invocations_.worker_count = workers_.size();
    context_.core_engines = engines_.data();
  }

  bool Compatible() const {
    return loom_aie2p_native_input_cursors_compatible(&context_, &invocations_,
                                                      &inputs_[0], &inputs_[1]);
  }

  // Distinct serial construction identities whose controllers were admitted.
  std::array<loom_pipeline_invocation_t, 2> children_ = {};
  // Two input/controller pairs, one pair per construction.
  std::array<loom_pipeline_invocation_worker_t, 4> workers_ = {};
  // Retained worker-to-construction correspondence.
  loom_pipeline_invocations_t invocations_ = {};
  // Physical instruction-engine assignments; inputs are excluded.
  std::array<iree_host_size_t, 4> engines_ = {IREE_HOST_SIZE_MAX, 7,
                                              IREE_HOST_SIZE_MAX, 7};
  // Selected existing controllers, without additional instruction workers.
  std::array<loom_aie2p_native_input_t, 2> inputs_ = {};
  // Native realization's admitted instruction-engine map.
  loom_aie2p_native_context_t context_ = {};
};

TEST_F(InputCursorOwnershipTest, SerialContinuationsShareTheirCoreCursor) {
  EXPECT_TRUE(Compatible());
  engines_[3] = 8;
  EXPECT_FALSE(Compatible());
}

TEST_F(InputCursorOwnershipTest, IndependentInputsCannotShareAnEntrySnapshot) {
  inputs_[1].controller = inputs_[0].controller;
  inputs_[1].invocation = inputs_[0].invocation;
  EXPECT_FALSE(Compatible());
  EXPECT_TRUE(loom_aie2p_native_input_cursors_compatible(
      &context_, &invocations_, &inputs_[0], &inputs_[0]));
}

TEST_F(InputCursorOwnershipTest, AncestorInputCannotBorrowAChildLifetime) {
  // A root input may select a downstream child as its observer. That does not
  // make the input's lifetime disjoint from the child's other imported work.
  const loom_pipeline_invocation_t parent = {};
  inputs_[0].invocation = &parent;
  EXPECT_FALSE(Compatible());
  inputs_[0].invocation = &children_[0];
  inputs_[1].invocation = &parent;
  EXPECT_FALSE(Compatible());
}

class ExternalSlotsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dma_ = &loom_xdna_array_tile_kind_facts(loom_xdna_npu2_array_family(),
                                            LOOM_XDNA_TILE_KIND_SHIM_NOC)
                ->dma;
    protocol_.payload.capacity = 11;
    protocol_.payload.byte_stride = 4096;
    protocol_.notification_byte_stride = 64;
    source_.phase_count = 2;
    source_.phases = source_phases_.data();
    source_.admissions = 23;
    for (size_t i = 0; i < phases_.size(); ++i) {
      source_phases_[i].period = 1;
      source_phases_[i].transfers = 1;
      phases_[i].source = &source_phases_[i];
      phases_[i].repetitions = 1;
    }
    source_phases_[1].period = 64;
    source_phases_[1].transfers = 64;
    input_.binding = &binding_;
    input_.source = &source_;
    input_.phases = phases_.data();
  }

  bool Select(uint16_t available = 16) {
    return loom_aie2p_native_select_input_slots(dma_, available, &input_,
                                                &descriptors_);
  }

  // Physical shim descriptor and iterator bounds.
  const loom_xdna_dma_facts_t* dma_ = nullptr;
  // Complete source leases separated from their readiness generations.
  loom_memory_channel_protocol_t protocol_ = {};
  // Caller-owned payload and notification roots.
  loom_memory_channel_binding_t binding_ = {&protocol_, 0, 1};
  // Retained finite phase cardinalities, independent of ring capacity.
  std::array<loom_pipeline_transport_input_phase_t, 5> source_phases_ = {};
  // Physical per-period task counts produced by ingress geometry selection.
  std::array<loom_aie2p_native_input_phase_t, 5> phases_ = {};
  // Source lifetime containing each ordered phase.
  loom_pipeline_transport_input_t source_ = {};
  // Destination for the selected native slot realization.
  loom_aie2p_native_input_t input_ = {};
  // Complete descriptor-bank slice reserved by successful selection.
  uint16_t descriptors_ = 0;
};

TEST_F(ExternalSlotsTest, CapacityIsIndependentOfOutstandingTasks) {
  for (const uint32_t capacity : {1u, 2u, 4u, 5u, 11u, 64u}) {
    SCOPED_TRACE(capacity);
    protocol_.payload.capacity = capacity;
    ASSERT_TRUE(Select());
    if (capacity <= 4) {
      EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE);
      EXPECT_EQ(descriptors_, 1 + 3 * capacity);
    } else {
      EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD);
      EXPECT_EQ(descriptors_, 4);
    }
  }
  protocol_.payload.capacity = 65;
  EXPECT_FALSE(Select());
  EXPECT_EQ(descriptors_, 0);
}

TEST_F(ExternalSlotsTest, ReplayingOneLeaseKeepsTheCompactPayload) {
  source_phases_[1].period = 256;
  source_phases_[1].transfers = 8 * 256;
  ASSERT_TRUE(Select());
  EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD);
  EXPECT_EQ(descriptors_, 4);
}

TEST_F(ExternalSlotsTest, IntrarecordIteratorRemainsOwnedByPayloadGeometry) {
  phases_[1].repetitions = 64;
  ASSERT_TRUE(Select());
  EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_NOTIFICATIONS);
  EXPECT_EQ(descriptors_, 14);
  EXPECT_FALSE(Select(13));
  EXPECT_EQ(descriptors_, 0);
  protocol_.payload.capacity = 13;
  ASSERT_TRUE(Select());
  EXPECT_EQ(descriptors_, 16);
  protocol_.payload.capacity = 14;
  EXPECT_FALSE(Select());
}

TEST_F(ExternalSlotsTest, DescriptorPressureCanCompactASmallRing) {
  protocol_.payload.capacity = 2;
  ASSERT_TRUE(Select(7));
  EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE);
  EXPECT_EQ(descriptors_, 7);
  ASSERT_TRUE(Select(6));
  EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_PAYLOAD);
  EXPECT_EQ(descriptors_, 4);
  EXPECT_FALSE(Select(3));
  phases_[1].repetitions = 64;
  ASSERT_TRUE(Select(5));
  EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_NOTIFICATIONS);
  EXPECT_EQ(descriptors_, 5);
  EXPECT_FALSE(Select(4));
}

TEST_F(ExternalSlotsTest, EveryPrefixConsumesItsOwnDescriptorAndTask) {
  for (uint32_t phases = 1; phases <= 4; ++phases) {
    source_.phase_count = phases;
    for (auto& phase : source_phases_) {
      phase.period = 1;
      phase.transfers = 1;
    }
    source_phases_[phases - 1].period = 64;
    source_phases_[phases - 1].transfers = 64;
    ASSERT_TRUE(Select());
    EXPECT_EQ(descriptors_, phases + 2);
    EXPECT_FALSE(Select(phases + 1));
  }
  source_.phase_count = 5;
  source_phases_[3].period = 1;
  source_phases_[3].transfers = 1;
  source_phases_[4].period = 64;
  source_phases_[4].transfers = 64;
  EXPECT_FALSE(Select());
}

TEST_F(ExternalSlotsTest, SlotStridesMustFitTheSelectedAddressGenerator) {
  const uint64_t maximum_stride =
      (UINT64_C(1) << dma_->step_size_bits) * dma_->transfer_length_granularity;
  protocol_.payload.byte_stride = maximum_stride;
  protocol_.notification_byte_stride = maximum_stride;
  ASSERT_TRUE(Select());
  protocol_.payload.byte_stride += dma_->transfer_length_granularity;
  EXPECT_FALSE(Select());
  protocol_.payload.byte_stride = maximum_stride;
  protocol_.notification_byte_stride += dma_->transfer_length_granularity;
  EXPECT_FALSE(Select());
  // Immutable notification descriptors contain complete addresses, so their
  // stride need not fit an unused iterator.
  protocol_.payload.capacity = 2;
  ASSERT_TRUE(Select());
  EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE);
}

TEST_F(ExternalSlotsTest, StableCallerStorageNeedsNoNotificationDescriptors) {
  input_.binding = nullptr;
  ASSERT_TRUE(Select());
  EXPECT_EQ(input_.slots, LOOM_AIE2P_NATIVE_INPUT_SLOTS_IMMUTABLE);
  EXPECT_EQ(descriptors_, 2);
  EXPECT_FALSE(Select(1));
}

}  // namespace
