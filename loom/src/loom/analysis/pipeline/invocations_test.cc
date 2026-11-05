// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/invocations.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/pipeline/execution.h"

namespace loom {
namespace {

// A parent with one independent worker admits a wrapper. The wrapper has its
// own worker, a concurrent child, and two serial children in nested scopes.
// Only retained plan data is available: projection cannot inspect source IR.
class PipelineInvocationsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    for (size_t i = 0; i < 5; ++i) {
      entries_[i] = &occurrences_[i];
      auto& occurrence = occurrences_[i];
      occurrence.ordinal = i;
      occurrence.first_worker = i;
      occurrence.first_channel = i == 0 ? 0 : i == 1 ? 2 : 3;
      occurrence.subtree = {i < 2 ? 5 - i : 1, i < 2 ? 5 - i : 1,
                            i == 0   ? 3u
                            : i == 1 ? 1u
                                     : 0u};
      occurrence.parent = i == 0 ? nullptr : &occurrences_[i == 1 ? 0 : 1];
      occurrence.resources = &resources_[i];
      resources_[i].strands = &strands_[i];
      resources_[i].strand_count = 1;
      workers_[i] = {&occurrence, &strands_[i]};
    }
    resources_[0].composition_count = 1;
    occurrences_[0].compositions = &entries_[1];
    resources_[1].composition_count = 1;
    occurrences_[1].compositions = &entries_[2];
    resources_[0].channels = protocols_;
    resources_[0].channel_count = 2;
    resources_[1].channels = protocols_ + 2;
    resources_[1].channel_count = 1;
    for (size_t i = 0; i < 3; ++i) {
      protocols_[i].capacity = 1;
      channels_[i] = {&occurrences_[i < 2 ? 0 : 1], &protocols_[i], nullptr};
      channel_order_[i] = i;
    }
    // The wrapper imports the same protocol twice. Its unrelated sibling
    // protocol has identical storage, but must not appear in the projection.
    for (size_t i = 0; i < 2; ++i) {
      imports_[i] = {static_cast<loom_value_id_t>(i), &resources_[0],
                     &protocols_[0]};
    }
    occurrences_[1].arguments = {.channels = imports_, .channel_count = 2};
    for (size_t i = 0; i < 3; ++i) {
      leaf_imports_[i] = {static_cast<loom_value_id_t>(i),
                          &resources_[i == 0 ? 1 : 0],
                          &protocols_[i == 0 ? 2 : 0]};
    }
    occurrences_[2].arguments = occurrences_[3].arguments =
        occurrences_[4].arguments = {
            .channels = leaf_imports_,
            .channel_count = 3,
        };
    control_.graph.block_count = 1;
    steps_[0][0].kind = LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION;
    steps_[0][1] = {
        .nested = &schedules_[1],
        .kind = LOOM_PIPELINE_SCHEDULE_STEP_SERIAL,
    };
    steps_[0][2].kind = LOOM_PIPELINE_SCHEDULE_STEP_YIELD;
    steps_[1][0].kind = LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION;
    steps_[1][1].kind = LOOM_PIPELINE_SCHEDULE_STEP_YIELD;
    bindings_[0][0].child = &occurrences_[3];
    bindings_[0][1].nested = &bound_[1];
    bindings_[1][0].child = &occurrences_[4];
    for (size_t i = 0; i < 2; ++i) {
      schedule_blocks_[i] = {steps_[i], i == 0 ? 3u : 2u};
      schedules_[i].control = &control_;
      schedules_[i].blocks = &schedule_blocks_[i];
      bound_blocks_[i] = bindings_[i];
      bound_[i] = {&schedules_[i], &bound_blocks_[i]};
      occurrences_[i + 3].schedule = &schedules_[i];
    }
    resources_[1].schedule_count = 1;
    occurrences_[1].schedules = bound_;
    invocations_ = {entries_, 5, workers_, 5, channels_, 3, channel_order_};
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  // Projection storage, independent of the retained source fixture.
  iree_arena_block_pool_t blocks_;
  // Allocation owner for the consuming projection and physical execution plan.
  iree_arena_allocator_t arena_;
  // One direct worker in each invocation.
  loom_pipeline_resource_strand_t strands_[5] = {};
  // Protocol identities stay independent even when storage projections alias.
  loom_pipeline_resource_channel_t protocols_[3] = {};
  // Counts and borrowed source declarations for each occurrence.
  loom_pipeline_resources_t resources_[5] = {};
  // Parent-before-child occurrences supplied by the admitting owner.
  loom_pipeline_invocation_t occurrences_[5] = {};
  // Complete inventory indexing those occurrences.
  const loom_pipeline_invocation_t* entries_[5] = {};
  // Direct workers in occurrence order.
  loom_pipeline_invocation_worker_t workers_[5] = {};
  // Root and wrapper protocol construction rows.
  loom_pipeline_invocation_channel_t channels_[3] = {};
  // Identity-address index; protocols are contiguous in this fixture.
  iree_host_size_t channel_order_[3] = {};
  // Two formal arguments sharing one imported protocol.
  loom_pipeline_channel_binding_t imports_[2] = {};
  // Child formals importing both local and outer protocols.
  loom_pipeline_channel_binding_t leaf_imports_[3] = {};
  // Canonical single-block CFG used by both serial scopes.
  loom_value_fact_cfg_region_t control_ = {};
  // Classified steps including each scope's yield.
  loom_pipeline_schedule_step_t steps_[2][3] = {};
  // Borrowed source block descriptors.
  loom_pipeline_schedule_block_t schedule_blocks_[2] = {};
  // Source scope identities, never cloned by projection.
  loom_pipeline_schedule_t schedules_[2] = {};
  // Bound execution references parallel to the classified steps.
  loom_pipeline_invocation_step_t bindings_[2][3] = {};
  // Block pointer for each bound serial scope.
  const loom_pipeline_invocation_step_t* bound_blocks_[2] = {};
  // Scope bindings with distinct outer and nested lifetimes.
  loom_pipeline_invocation_schedule_t bound_[2] = {};
  // Complete retained inventory, including the unrelated outer worker.
  loom_pipeline_invocations_t invocations_ = {};
};

TEST_F(PipelineInvocationsTest, WholeRootBorrowsInventoryWithoutAllocation) {
  const auto used = arena_.used_allocation_size;
  loom_pipeline_invocations_t projected;
  IREE_ASSERT_OK(loom_pipeline_invocations_project(&invocations_, entries_[0],
                                                   &arena_, &projected));
  EXPECT_EQ(arena_.used_allocation_size, used);
  EXPECT_EQ(projected.entries, invocations_.entries);
  EXPECT_EQ(projected.workers, invocations_.workers);
  EXPECT_EQ(projected.channels, invocations_.channels);
  EXPECT_EQ(projected.channel_order, invocations_.channel_order);
}

TEST_F(PipelineInvocationsTest, IndependentSubtreeKeepsProtocolsAndJoins) {
  loom_pipeline_invocations_t projected;
  IREE_ASSERT_OK(loom_pipeline_invocations_project(&invocations_, entries_[1],
                                                   &arena_, &projected));
  ASSERT_EQ(projected.count, 4u);
  ASSERT_EQ(projected.worker_count, 4u);
  ASSERT_EQ(projected.channel_count, 2u);
  const auto* root = projected.entries[0];
  EXPECT_EQ(root->parent, nullptr);
  EXPECT_EQ(root->call, nullptr);
  EXPECT_EQ(root->schedule, nullptr);
  EXPECT_EQ(root->resources, &resources_[1]);
  EXPECT_EQ(root->arguments.channels, imports_);
  EXPECT_EQ(root->first_channel, 1u);
  EXPECT_EQ(root->compositions[0], projected.entries[1]);
  const auto& serial = root->schedules[0];
  EXPECT_EQ(serial.source, &schedules_[0]);
  EXPECT_EQ(serial.blocks[0][0].child, projected.entries[2]);
  EXPECT_EQ(serial.blocks[0][1].nested->blocks[0][0].child,
            projected.entries[3]);
  for (size_t i = 0; i < projected.count; ++i) {
    EXPECT_EQ(projected.entries[i]->ordinal, i);
    EXPECT_EQ(projected.entries[i]->first_worker, i);
    EXPECT_EQ(projected.workers[i].invocation, projected.entries[i]);
    EXPECT_EQ(projected.workers[i].strand, &strands_[i + 1]);
    if (i) {
      EXPECT_EQ(projected.entries[i]->parent, root);
    }
  }
  EXPECT_EQ(projected.channels[0].source, &protocols_[0]);
  EXPECT_EQ(projected.channels[0].invocation, entries_[0]);
  EXPECT_EQ(projected.channels[1].source, &protocols_[2]);
  EXPECT_EQ(projected.channels[1].invocation, root);
  for (size_t i = 0; i < projected.channel_count; ++i) {
    EXPECT_EQ(loom_pipeline_invocations_channel_index(
                  &projected, &projected.channels[i].source->identity),
              i);
  }

  // The direct worker keeps independent progress while the serial children
  // share an engine. Losing the nested binding changes this physical plan.
  const iree_host_size_t engines[] = {2, 1, 0, 0};
  loom_pipeline_execution_t execution;
  loom_pipeline_execution_conflict_t conflict;
  IREE_ASSERT_OK(loom_pipeline_execution_build(&projected, engines, 3, &arena_,
                                               &execution, &conflict));
  EXPECT_EQ(conflict.second_worker, IREE_HOST_SIZE_MAX);
  EXPECT_EQ(execution.workers[0], 2u);
  EXPECT_EQ(execution.workers[1], 1u);
  EXPECT_EQ(execution.workers[2], 0u);

  // A second projection imports the first projection's local protocol. New
  // execution ordinals must not change either channel's construction identity.
  loom_pipeline_invocations_t leaf;
  IREE_ASSERT_OK(loom_pipeline_invocations_project(
      &projected, projected.entries[3], &arena_, &leaf));
  EXPECT_EQ(leaf.count, 1u);
  EXPECT_EQ(leaf.worker_count, 1u);
  ASSERT_EQ(leaf.channel_count, 2u);
  EXPECT_EQ(leaf.channels[0].source, &protocols_[0]);
  EXPECT_EQ(leaf.channels[1].source, &protocols_[2]);
  EXPECT_EQ(leaf.channels[1].invocation, root);
  EXPECT_EQ(leaf.entries[0]->first_channel, 2u);
  EXPECT_EQ(leaf.entries[0]->subtree.channels, 0u);
}

}  // namespace
}  // namespace loom
