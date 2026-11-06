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
#include "loom/analysis/pipeline/lifetime_layout.h"
#include "loom/analysis/source_storage_packing.h"

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
    IREE_ASSERT_OK(loom_pipeline_lifetimes_build(&invocations_, &arena_,
                                                 &invocations_.lifetimes));
    for (size_t i = 0; i < 5; ++i) {
      occurrences_[i].lifetime = invocations_.lifetimes.owners[i];
    }
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
  EXPECT_EQ(projected.lifetimes.nodes, invocations_.lifetimes.nodes);
}

TEST_F(PipelineInvocationsTest, IndependentSubtreeKeepsProtocolsAndJoins) {
  loom_pipeline_invocations_t projected;
  IREE_ASSERT_OK(loom_pipeline_invocations_project(&invocations_, entries_[1],
                                                   &arena_, &projected));
  ASSERT_EQ(projected.count, 4u);
  ASSERT_EQ(projected.worker_count, 4u);
  ASSERT_EQ(projected.channel_count, 2u);
  EXPECT_EQ(projected.lifetimes.nodes, invocations_.lifetimes.nodes);
  const auto* root = projected.entries[0];
  EXPECT_EQ(root->parent, nullptr);
  EXPECT_EQ(root->call, nullptr);
  EXPECT_EQ(root->schedule, nullptr);
  EXPECT_EQ(root->lifetime, occurrences_[1].lifetime);
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

TEST_F(PipelineInvocationsTest, ResourceRangesComposeAcrossPoolsAndLifetimes) {
  // Root and wrapper state coexist with the concurrent child and whichever
  // serial child is active. Distinct requests within one owner also coexist.
  const uint64_t lengths[] = {3, 5, 7, 11, 13};
  loom_pipeline_lifetime_request_t requests[9] = {};
  for (size_t i = 0; i < 5; ++i) {
    requests[i] = {
        .lifetime = occurrences_[i].lifetime,
        .pool = 0,
        .length = lengths[i],
        .alignment = 1,
    };
  }
  requests[5] = {
      .lifetime = occurrences_[3].lifetime,
      .pool = 0,
      .length = 2,
      .alignment = 1,
  };
  requests[6] = {
      .lifetime = occurrences_[0].lifetime,
      .pool = 1,
      .length = 1,
      .alignment = 1,
  };
  requests[7] = {
      .lifetime = occurrences_[3].lifetime,
      .pool = 1,
      .length = 3,
      .alignment = 8,
  };
  requests[8] = {
      .lifetime = occurrences_[4].lifetime,
      .pool = 1,
      .length = 17,
      .alignment = 16,
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(
      &invocations_.lifetimes, requests, IREE_ARRAYSIZE(requests), 2, &arena_,
      &extents));
  const uint64_t offsets[] = {0, 3, 8, 15, 15, 26, 0, 16, 16};
  for (size_t i = 0; i < IREE_ARRAYSIZE(requests); ++i) {
    EXPECT_EQ(requests[i].offset, offsets[i]) << i;
  }
  EXPECT_EQ(extents[0].length, 28u);
  EXPECT_EQ(extents[0].alignment, 1u);
  EXPECT_EQ(extents[1].length, 33u);
  EXPECT_EQ(extents[1].alignment, 16u);
}

TEST_F(PipelineInvocationsTest, ProjectedResourceRangesRetainOuterOwnership) {
  loom_pipeline_invocations_t projected;
  IREE_ASSERT_OK(loom_pipeline_invocations_project(&invocations_, entries_[1],
                                                   &arena_, &projected));
  // One imported parent resource remains live around the projected wrapper.
  // The two serial child resources can share, but neither aliases that import.
  loom_pipeline_lifetime_request_t requests[] = {
      {
          .lifetime = projected.channels[0].invocation->lifetime,
          .pool = 0,
          .length = 1,
          .alignment = 1,
      },
      {
          .lifetime = projected.entries[2]->lifetime,
          .pool = 0,
          .length = 1,
          .alignment = 1,
      },
      {
          .lifetime = projected.entries[3]->lifetime,
          .pool = 0,
          .length = 1,
          .alignment = 1,
      },
  };
  const loom_pipeline_lifetime_extent_t* extents = nullptr;
  IREE_ASSERT_OK(loom_pipeline_lifetime_layout(&projected.lifetimes, requests,
                                               IREE_ARRAYSIZE(requests), 1,
                                               &arena_, &extents));
  EXPECT_EQ(requests[0].offset, 0u);
  EXPECT_EQ(requests[1].offset, 1u);
  EXPECT_EQ(requests[2].offset, 1u);
  EXPECT_EQ(extents[0].length, 2u);
}

TEST_F(PipelineInvocationsTest, StorageRespectsOwnersAndCanonicalBackings) {
  const loom_source_storage_packing_range_t fixed[] = {{0, 64}};
  loom_pipeline_resource_pool_t pools[2] = {};
  IREE_ASSERT_OK(loom_source_storage_packing_allocate({}, fixed, 1, &arena_,
                                                      &pools[0].packing));
  IREE_ASSERT_OK(loom_source_storage_packing_allocate({}, nullptr, 0, &arena_,
                                                      &pools[1].packing));
  // A different inventory ordering must still identify the same two backings.
  const loom_pipeline_resource_pool_t reversed[] = {pools[1], pools[0]};
  const uint64_t lengths[] = {32, 33, 48, 96, 24};
  const uint64_t alignments[] = {16, 32, 16, 128, 64};
  loom_pipeline_resource_allocation_t allocations[5][2] = {};
  for (size_t i = 0; i < 5; ++i) {
    resources_[i].pools = i % 2 ? reversed : pools;
    resources_[i].pool_count = 2;
    resources_[i].allocations = allocations[i];
    resources_[i].allocation_count = 2;
    allocations[i][0] = {
        .pool_index = static_cast<uint32_t>(i % 2),
        .byte_length = lengths[i],
        .byte_alignment = alignments[i],
    };
    allocations[i][1] = {
        .pool_index = static_cast<uint32_t>(1 - i % 2),
        .construction_order = 1,
        .byte_length = i == 3 ? 8u : 0u,
        .byte_alignment = 8,
    };
  }
  const loom_pipeline_storage_allocation_t** placed = nullptr;
  IREE_ASSERT_OK(loom_pipeline_construction_storage_build(&invocations_,
                                                          &arena_, &placed));
  const uint64_t offsets[] = {128, 256, 304, 384, 384};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(placed[i][0].source, &allocations[i][0]);
    EXPECT_EQ(placed[i][0].byte_offset, offsets[i]);
    EXPECT_EQ(placed[i][0].byte_offset % alignments[i], 0u);
    occurrences_[i].storage = placed[i];
    for (size_t j = 0; j < i; ++j) {
      if (i == 4 && j == 3) {
        continue;
      }
      EXPECT_TRUE(offsets[i] >= offsets[j] + lengths[j] ||
                  offsets[j] >= offsets[i] + lengths[i]);
    }
  }
  EXPECT_EQ(
      loom_source_storage_packing_requirement(pools[0].packing).byte_length,
      480u);
  EXPECT_EQ(
      loom_source_storage_packing_requirement(pools[1].packing).byte_length,
      8u);
  uint64_t service_offset = UINT64_MAX;
  IREE_ASSERT_OK(loom_source_storage_packing_reserve(
      pools[0].packing, 64, 64, nullptr, 0, &service_offset));
  EXPECT_EQ(service_offset, 64u);

  const auto before_projection =
      loom_source_storage_packing_requirement(pools[0].packing).byte_length;
  loom_pipeline_invocations_t projected;
  IREE_ASSERT_OK(loom_pipeline_invocations_project(&invocations_, entries_[1],
                                                   &arena_, &projected));
  for (size_t i = 0; i < projected.count; ++i) {
    EXPECT_EQ(projected.entries[i]->storage, placed[i + 1]);
  }
  EXPECT_EQ(
      loom_source_storage_packing_requirement(pools[0].packing).byte_length,
      before_projection);
}

TEST_F(PipelineInvocationsTest,
       ConcurrentStorageUsesConstructionOrderAndFixedReservationHoles) {
  // Keep only the root's concurrent wrapper and its concurrent child.
  resources_[1].schedule_count = 0;
  invocations_.count = 3;
  IREE_ASSERT_OK(loom_pipeline_lifetimes_build(&invocations_, &arena_,
                                               &invocations_.lifetimes));
  const loom_source_storage_packing_range_t fixed[] = {{64, 64}, {192, 64}};
  loom_pipeline_resource_pool_t pool = {};
  IREE_ASSERT_OK(loom_source_storage_packing_allocate({}, fixed, 2, &arena_,
                                                      &pool.packing));
  loom_pipeline_resource_allocation_t allocations[3] = {};
  for (size_t i = 0; i < 3; ++i) {
    resources_[i].pools = &pool;
    resources_[i].pool_count = 1;
    resources_[i].allocations = &allocations[i];
    resources_[i].allocation_count = 1;
    allocations[i].byte_length = 64;
    allocations[i].byte_alignment = 64;
  }
  // Lookup order differs from construction order. The first allocation fits
  // below the first fixed range, while the second needs the next complete hole.
  const loom_pipeline_resource_allocation_t root_allocations[] = {
      {
          .root_value_id = 1,
          .construction_order = 1,
          .byte_length = 64,
          .byte_alignment = 64,
      },
      {
          .root_value_id = 2,
          .construction_order = 0,
          .byte_length = 32,
          .byte_alignment = 32,
      },
  };
  resources_[0].allocations = root_allocations;
  resources_[0].allocation_count = IREE_ARRAYSIZE(root_allocations);
  const loom_pipeline_storage_allocation_t** placed = nullptr;
  IREE_ASSERT_OK(loom_pipeline_construction_storage_build(&invocations_,
                                                          &arena_, &placed));
  EXPECT_EQ(placed[0][0].byte_offset, 128u);
  EXPECT_EQ(placed[0][1].byte_offset, 0u);
  EXPECT_EQ(placed[1][0].byte_offset, 256u);
  EXPECT_EQ(placed[2][0].byte_offset, 320u);
  EXPECT_EQ(loom_source_storage_packing_requirement(pool.packing).byte_length,
            384u);
}

}  // namespace
}  // namespace loom
