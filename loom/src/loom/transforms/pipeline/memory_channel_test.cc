// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/memory_channel.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/channel/ops.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/util/fact_table.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class MemoryChannelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&context_));
    Register(LOOM_DIALECT_BUFFER, loom_buffer_dialect_vtables);
    Register(LOOM_DIALECT_CFG, loom_cfg_dialect_vtables);
    Register(LOOM_DIALECT_CHANNEL, loom_channel_dialect_vtables);
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    Register(LOOM_DIALECT_INDEX, loom_index_dialect_vtables);
    Register(LOOM_DIALECT_SCALAR, loom_scalar_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("memory_channel"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_rewriter_initialize(&rewriter_, module_, &arena_);
    loom_builder_set_block(&rewriter_.builder, loom_module_block(module_));
    view_type_ = loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I16,
                                     loom_dim_pack_static(2),
                                     loom_dim_pack_static(3), 0);
    loom_type_id_t payload;
    IREE_ASSERT_OK(loom_module_intern_type_id(
        module_,
        loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I16,
                            loom_dim_pack_static(2), loom_dim_pack_static(3),
                            0),
        &payload));
    loom_type_t channel_type;
    IREE_ASSERT_OK(loom_channel_type_make(module_, payload, &channel_type));
    IREE_ASSERT_OK(loom_read_type_make(module_, 0, payload,
                                       (loom_read_type_mode_t)0, &read_type_));
    IREE_ASSERT_OK(loom_write_type_make(module_, payload, &write_type_));
    loom_string_id_t name;
    loom_symbol_id_t symbol;
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("worker"), &name));
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    const loom_type_t arguments[] = {channel_type, loom_type_buffer()};
    IREE_ASSERT_OK(loom_func_def_build(
        &rewriter_.builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), {0, symbol},
        arguments, IREE_ARRAYSIZE(arguments), nullptr, 0, nullptr, 0, nullptr,
        0, LOOM_LOCATION_UNKNOWN, &function_));
    loom_builder_enter_region(&rewriter_.builder, function_,
                              loom_func_def_body(function_));
    entry_ = loom_region_entry_block(loom_func_def_body(function_));
  }

  void TearDown() override {
    loom_rewriter_deinitialize(&rewriter_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, dialect, vtables, count));
  }

  loom_value_id_t Constant(uint64_t value, loom_scalar_type_t scalar_type) {
    const auto facts = loom_value_facts_make_signed_raw_bits(
        value, loom_scalar_type_bitwidth(scalar_type));
    loom_op_t* operation;
    IREE_CHECK_OK(loom_scalar_constant_build(
        &rewriter_.builder, loom_attr_i64(facts.range_lo),
        loom_type_scalar(scalar_type), LOOM_LOCATION_UNKNOWN, &operation));
    return loom_scalar_constant_result(operation);
  }

  void Return() {
    loom_op_t* operation;
    IREE_ASSERT_OK(loom_func_return_build(&rewriter_.builder, nullptr, 0,
                                          LOOM_LOCATION_UNKNOWN, &operation));
  }

  void Verify() {
    const loom_verify_options_t options = {
        .sink = {.fn = loom_diagnostic_stderr_sink},
    };
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    ASSERT_EQ(result.error_count, 0u);
  }

  void ComputeFacts(loom_value_fact_table_t* facts) {
    IREE_ASSERT_OK(loom_value_fact_table_initialize(facts, &arena_, 0));
    loom_type_registry_configure_fact_context(&facts->context);
    IREE_ASSERT_OK(loom_value_fact_table_compute(
        facts, module_, loom_func_like_cast(module_, function_)));
  }

  int64_t Exact(const loom_value_fact_table_t& facts, loom_value_id_t value) {
    int64_t result = 0;
    EXPECT_TRUE(loom_value_facts_as_exact_i64(
        loom_value_fact_table_lookup(&facts, value), &result));
    return result;
  }

  // Test sink for the selected visibility mechanism. The memory emitter owns
  // neither that mechanism nor fanout completion; record each requested event.
  struct Notification {
    // Source operation, used only while materializing the action.
    uint32_t operation_kind;
    // Selected readiness/publication direction.
    loom_memory_channel_notification_t kind;
    // Exact private ordinal supplied by the ownership carrier.
    loom_value_id_t record;
  };

  static iree_status_t Notify(void* user_data, loom_rewriter_t*,
                              const loom_channel_plan_action_t* action,
                              const loom_memory_channel_endpoint_t*,
                              loom_value_id_t record,
                              loom_memory_channel_notification_t kind) {
    auto* self = static_cast<MemoryChannelTest*>(user_data);
    self->notifications_.push_back({action->op->kind, kind, record});
    return iree_ok_status();
  }

  static iree_status_t Transfer(void* user_data, loom_rewriter_t*,
                                const loom_channel_plan_action_t* action,
                                loom_value_id_t*, iree_host_size_t,
                                loom_value_id_t*) {
    auto* self = static_cast<MemoryChannelTest*>(user_data);
    EXPECT_TRUE(loom_channel_copy_isa(action->op));
    self->transfers_.push_back({loom_channel_copy_source(action->op),
                                loom_channel_copy_destination(action->op)});
    return iree_ok_status();
  }

  void Materialize(loom_value_id_t initial_read, loom_value_id_t initial_write,
                   loom_memory_channel_callback_t callbacks) {
    Verify();
    const loom_channel_identity_t identity = {entry_->arg_ids[0]};
    const loom_memory_channel_layout_t layout = {3, 96, 32, view_type_};
    const loom_memory_channel_endpoint_t endpoint = {&layout,
                                                     entry_->arg_ids[1], 0, 1};
    loom_local_value_domain_t domain = {};
    IREE_ASSERT_OK(loom_local_value_domain_acquire_for_region(
        module_, loom_func_def_body(function_), &arena_, &domain));
    const loom_channel_plan_binding_t binding = {entry_->arg_ids[0], &identity};
    loom_channel_plan_t plan;
    loom_channel_plan_rejection_t rejection;
    IREE_ASSERT_OK(loom_channel_plan_build(&domain, &binding, 1, nullptr,
                                           &arena_, &plan, &rejection));
    ASSERT_EQ(rejection.kind, LOOM_CHANNEL_PLAN_REJECTION_NONE);
    loom_cfg_graph_t graph;
    IREE_ASSERT_OK(loom_cfg_graph_build(module_, loom_func_def_body(function_),
                                        &arena_, &graph));
    std::vector<loom_type_t> carriers(plan.value_count);
    std::vector<uint8_t> erased(plan.value_count);
    for (loom_value_ordinal_t i = 0; i < plan.value_count; ++i) {
      const auto type = loom_module_value_type(module_, domain.value_ids[i]);
      if (loom_channel_type_isa(type)) {
        erased[i] = 1;
      } else if (loom_read_type_isa(type) || loom_write_type_isa(type)) {
        carriers[i] = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
      }
    }
    std::vector<const loom_memory_channel_endpoint_t*> endpoints(
        plan.action_count, &endpoint);
    loom_memory_channel_materialization_t memory = {&plan, endpoints.data(),
                                                    callbacks};
    const loom_value_id_t state[] = {initial_read, initial_write};
    const loom_channel_materialization_options_t options = {
        carriers.data(),
        erased.data(),
        state,
        IREE_ARRAYSIZE(state),
        {loom_memory_channel_materialize_action, nullptr, &memory}};
    IREE_ASSERT_OK(
        loom_channel_materialize(&rewriter_, &plan, &graph, &options));
    loom_local_value_domain_release(&domain);
    IREE_ASSERT_OK(loom_block_remove_arg(module_, entry_, 0));
    Verify();
  }

  iree_arena_block_pool_t pool_ = {};
  iree_arena_allocator_t arena_ = {};
  loom_context_t context_ = {};
  loom_module_t* module_ = nullptr;
  loom_rewriter_t rewriter_ = {};
  loom_op_t* function_ = nullptr;
  loom_block_t* entry_ = nullptr;
  loom_type_t view_type_ = {};
  loom_type_t read_type_ = {};
  loom_type_t write_type_ = {};
  std::vector<Notification> notifications_;
  std::vector<std::array<loom_value_id_t, 2>> transfers_;
};

TEST_F(MemoryChannelTest, NotificationAddressAndGenerationAcrossRingReuse) {
  struct Case {
    // Fixed-width unsigned ordinal represented by signed scalar payload bits.
    loom_scalar_type_t type;
    // Dynamic record number, independent of the number of live slots.
    uint64_t record;
    // Number of slots in the selected ring.
    uint64_t capacity;
    // Expected selected slot address in the ready counter plane.
    int64_t ready_offset;
    // Expected selected slot address in the free counter plane.
    int64_t free_offset;
    // Last completed generation before this record becomes ready.
    int64_t prior_generation;
  };
  const Case cases[] = {
      {LOOM_SCALAR_TYPE_I8, 0, 3, 64, 1024, 0},
      {LOOM_SCALAR_TYPE_I16, 8, 3, 320, 1280, 2},
      {LOOM_SCALAR_TYPE_I32, 41, 1, 64, 1024, 41},
      {LOOM_SCALAR_TYPE_I32, 1024, 512, 64, 1024, 2},
      {LOOM_SCALAR_TYPE_I32, UINT32_C(4294967294), 3, 320, 1280, 1431655764},
      {LOOM_SCALAR_TYPE_I64, UINT64_C(8589934597), 512, 704, 1664, 16777216},
  };
  const loom_memory_channel_notification_t notifications[] = {
      LOOM_MEMORY_CHANNEL_WAIT_FREE, LOOM_MEMORY_CHANNEL_WAIT_READY,
      LOOM_MEMORY_CHANNEL_PUBLISH, LOOM_MEMORY_CHANNEL_RELEASE};
  std::vector<loom_value_id_t> values;
  for (const auto& item : cases) {
    const loom_memory_channel_layout_t layout = {item.capacity, 0, 0, {}};
    const auto record = Constant(item.record, item.type);
    for (auto notification : notifications) {
      loom_value_id_t offset, generation;
      IREE_ASSERT_OK(loom_memory_channel_notification_build(
          &rewriter_.builder, &layout, record, notification, 64, 1024, 128,
          LOOM_LOCATION_UNKNOWN, &offset, &generation));
      values.push_back(offset);
      values.push_back(generation);
    }
  }
  Return();
  Verify();
  loom_value_fact_table_t facts;
  ComputeFacts(&facts);
  size_t i = 0;
  for (const auto& item : cases) {
    SCOPED_TRACE(::testing::Message() << item.record << "/" << item.capacity);
    for (auto notification : notifications) {
      const bool ready = notification == LOOM_MEMORY_CHANNEL_WAIT_READY ||
                         notification == LOOM_MEMORY_CHANNEL_PUBLISH;
      EXPECT_EQ(Exact(facts, values[i++]),
                ready ? item.ready_offset : item.free_offset);
      EXPECT_EQ(Exact(facts, values[i++]),
                item.prior_generation +
                    (notification != LOOM_MEMORY_CHANNEL_WAIT_FREE));
    }
  }
}

TEST_F(MemoryChannelTest, AcceptWaitAndFanoutPreserveTheOwnedRecord) {
  const auto initial_read = Constant(5, LOOM_SCALAR_TYPE_I32);
  const auto initial_write = Constant(1, LOOM_SCALAR_TYPE_I32);
  loom_op_t *accept, *fanout, *wait, *release0, *release1, *acquire, *release2,
      *reserve, *publish;
  IREE_ASSERT_OK(loom_channel_accept_build(&rewriter_.builder, 0, 0,
                                           entry_->arg_ids[0], read_type_,
                                           LOOM_LOCATION_UNKNOWN, &accept));
  const loom_type_t reads[] = {read_type_, read_type_};
  IREE_ASSERT_OK(loom_channel_fanout_build(
      &rewriter_.builder, loom_channel_accept_read(accept), reads, 2, nullptr,
      0, LOOM_LOCATION_UNKNOWN, &fanout));
  IREE_ASSERT_OK(loom_channel_wait_build(&rewriter_.builder,
                                         loom_op_results(fanout)[1], view_type_,
                                         LOOM_LOCATION_UNKNOWN, &wait));
  IREE_ASSERT_OK(loom_channel_release_build(&rewriter_.builder,
                                            loom_op_results(fanout)[0],
                                            LOOM_LOCATION_UNKNOWN, &release0));
  IREE_ASSERT_OK(loom_channel_release_build(&rewriter_.builder,
                                            loom_op_results(fanout)[1],
                                            LOOM_LOCATION_UNKNOWN, &release1));
  IREE_ASSERT_OK(loom_channel_acquire_build(
      &rewriter_.builder, 0, 0, entry_->arg_ids[0], read_type_, view_type_,
      LOOM_LOCATION_UNKNOWN, &acquire));
  IREE_ASSERT_OK(loom_channel_release_build(&rewriter_.builder,
                                            loom_channel_acquire_read(acquire),
                                            LOOM_LOCATION_UNKNOWN, &release2));
  IREE_ASSERT_OK(loom_channel_reserve_build(
      &rewriter_.builder, entry_->arg_ids[0], write_type_, view_type_,
      LOOM_LOCATION_UNKNOWN, &reserve));
  IREE_ASSERT_OK(loom_channel_publish_build(&rewriter_.builder,
                                            loom_channel_reserve_write(reserve),
                                            LOOM_LOCATION_UNKNOWN, &publish));
  Return();
  Materialize(initial_read, initial_write, {Notify, nullptr, this});

  ASSERT_EQ(notifications_.size(), 7u);
  EXPECT_EQ(notifications_[0].operation_kind, LOOM_OP_CHANNEL_WAIT);
  EXPECT_EQ(notifications_[0].kind, LOOM_MEMORY_CHANNEL_WAIT_READY);
  EXPECT_EQ(notifications_[1].kind, LOOM_MEMORY_CHANNEL_RELEASE);
  EXPECT_EQ(notifications_[2].kind, LOOM_MEMORY_CHANNEL_RELEASE);
  EXPECT_EQ(notifications_[3].kind, LOOM_MEMORY_CHANNEL_WAIT_READY);
  EXPECT_EQ(notifications_[4].kind, LOOM_MEMORY_CHANNEL_RELEASE);
  EXPECT_EQ(notifications_[5].kind, LOOM_MEMORY_CHANNEL_WAIT_FREE);
  EXPECT_EQ(notifications_[6].kind, LOOM_MEMORY_CHANNEL_PUBLISH);
  // Accept and fanout emit no notification. Both releases carry the same
  // record, allowing selected completion mechanics to return one final credit.
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(notifications_[i].record, initial_read);
  }
  for (size_t i = 5; i < 7; ++i) {
    EXPECT_EQ(notifications_[i].record, initial_write);
  }
  loom_value_fact_table_t facts;
  ComputeFacts(&facts);
  // Accept advanced once; wait and the two fanout retirements did not advance
  // the cursor. The next acquire owns record 6, whose slot wraps back to zero.
  EXPECT_EQ(Exact(facts, notifications_[3].record), 6);
  EXPECT_EQ(notifications_[3].record, notifications_[4].record);
  std::vector<int64_t> origins;
  for (auto* operation = entry_->first_op; operation;
       operation = operation->next_op) {
    if (!loom_buffer_view_isa(operation)) {
      continue;
    }
    EXPECT_TRUE(loom_type_equal(
        loom_module_value_type(module_, loom_buffer_view_result(operation)),
        view_type_));
    EXPECT_EQ(loom_buffer_view_buffer(operation), entry_->arg_ids[0]);
    origins.push_back(Exact(facts, loom_buffer_view_byte_offset(operation)));
  }
  EXPECT_EQ(origins, (std::vector<int64_t>{160, 96, 128}));
}

TEST_F(MemoryChannelTest, CopyTransfersOwnershipWithoutAnImplicitDrain) {
  const auto initial_read = Constant(0, LOOM_SCALAR_TYPE_I32);
  const auto initial_write = Constant(1, LOOM_SCALAR_TYPE_I32);
  loom_op_t *accept, *reserve, *copy;
  IREE_ASSERT_OK(loom_channel_accept_build(&rewriter_.builder, 0, 0,
                                           entry_->arg_ids[0], read_type_,
                                           LOOM_LOCATION_UNKNOWN, &accept));
  IREE_ASSERT_OK(loom_channel_reserve_build(
      &rewriter_.builder, entry_->arg_ids[0], write_type_, view_type_,
      LOOM_LOCATION_UNKNOWN, &reserve));
  IREE_ASSERT_OK(loom_channel_copy_build(
      &rewriter_.builder, loom_channel_accept_read(accept),
      loom_channel_reserve_write(reserve), LOOM_LOCATION_UNKNOWN, &copy));
  Return();
  Materialize(initial_read, initial_write, {Notify, Transfer, this});
  ASSERT_EQ(notifications_.size(), 1u);
  EXPECT_EQ(notifications_[0].kind, LOOM_MEMORY_CHANNEL_WAIT_FREE);
  EXPECT_EQ(notifications_[0].record, initial_write);
  ASSERT_EQ(transfers_.size(), 1u);
  EXPECT_EQ(transfers_[0][0], initial_read);
  EXPECT_EQ(transfers_[0][1], initial_write);
  // Transport owns readiness, source retirement, and destination publication.
  // The memory emitter introduces none of those effects ahead of completion.
}

}  // namespace
}  // namespace loom
