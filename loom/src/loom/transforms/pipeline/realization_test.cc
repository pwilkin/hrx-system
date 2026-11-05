// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/pipeline/realization.h"

#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/pass/value_facts.h"
#include "loom/target/pass_environment.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

// The subject is occurrence ownership, not a physical target. The callbacks
// emit ordinary empty entry functions and observe the admitted worker clones.
// Native artifact tests separately exercise the complete physical mechanics.
class PipelineRealizationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    loom_pass_value_fact_owner_initialize(&blocks_, &fact_owner_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_type_registry_register_common_types(&context_));
    Register(LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
    Register(LOOM_DIALECT_PIPELINE, loom_pipeline_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("realization"),
                                        &blocks_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    const auto worker = Symbol("worker");
    worker_version_.base.function = Function(&builder_, worker);
    worker_version_.base.type = &loom_target_function_version_type;
    worker_version_.target_context_ordinal = 3;
    const auto source = Symbol("source");
    loom_op_t* definition;
    IREE_ASSERT_OK(loom_pipeline_def_build(
        &builder_, 0, 0, 0, 0, loom_symbol_ref_null(), source, nullptr, 0,
        nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &definition));
    source_version_.base.type = &loom_target_function_version_type;
    source_version_.base.function = loom_func_like_cast(module_, definition);
    source_version_.base.flags = LOOM_FUNCTION_VERSION_FLAG_RETAIN;
    source_version_.target_context_ordinal = 2;
    loom_builder_enter_region(&builder_, definition,
                              loom_pipeline_def_body(definition));
    const int64_t origin = 0, count = 1;
    loom_op_t* strand;
    IREE_ASSERT_OK(loom_pipeline_strand_build(
        &builder_, 0, 0, loom_symbol_ref_null(), nullptr, 0, &origin, 1,
        nullptr, 0, &count, 1, nullptr, 0, &count, 1, LOOM_LOCATION_UNKNOWN,
        &strand));
    loom_builder_enter_region(&builder_, strand,
                              loom_pipeline_strand_body(strand));
    loom_op_t* call;
    IREE_ASSERT_OK(loom_func_call_build(&builder_, 0, 0, 0, 0, worker, nullptr,
                                        0, nullptr, 0, nullptr, 0,
                                        LOOM_LOCATION_UNKNOWN, &call));
    loom_op_t* terminator;
    IREE_ASSERT_OK(
        loom_pipeline_end_build(&builder_, LOOM_LOCATION_UNKNOWN, &terminator));
    loom_builder_enter_region(&builder_, definition,
                              loom_pipeline_def_body(definition));
    IREE_ASSERT_OK(loom_pipeline_finish_build(&builder_, LOOM_LOCATION_UNKNOWN,
                                              &terminator));
    loom_builder_set_block(&builder_, loom_module_block(module_));
    builder_.ip.parent_op = nullptr;
    IREE_ASSERT_OK(loom_value_fact_table_initialize(&facts_, &arena_, 0));
    loom_type_registry_configure_fact_context(&facts_.context);
    IREE_ASSERT_OK(loom_value_fact_table_compute(
        &facts_, module_, source_version_.base.function));
    bool valid = false;
    IREE_ASSERT_OK(loom_pipeline_resources_build(
        module_, source_version_.base.function, &facts_, nullptr, 0, nullptr, 0,
        nullptr, 0, {}, &arena_, &resources_, &valid));
    ASSERT_TRUE(valid);
    ASSERT_EQ(resources_.strand_count, 1u);
    IREE_ASSERT_OK(loom_pipeline_invocations_build(
        module_, source_version_.base.function, &facts_, &resources_, nullptr,
        0, {}, {}, &arena_, &invocations_, &valid));
    ASSERT_TRUE(valid);
    IREE_ASSERT_OK(loom_pipeline_invocations_analyze_workers(
        module_, &invocations_, nullptr, {}, &arena_, &source_workers_,
        &valid));
    ASSERT_TRUE(valid);
    loom_function_version_owner_initialize(&arena_, &versions_);
    IREE_ASSERT_OK(
        loom_function_version_owner_append(&versions_, &source_version_.base));
    IREE_ASSERT_OK(
        loom_function_version_owner_append(&versions_, &worker_version_.base));
    capability_ =
        loom_target_pass_capability_make_mutable(nullptr, &versions_, nullptr);
    capabilities_[0] = &capability_.base;
    environment_ = loom_pass_environment_make(capabilities_, 1);
    pass_.arena = &arena_;
    pass_.value_facts = &fact_owner_;
    pass_.environment = &environment_;
    // Deliberately not the entry: a composing module pass has no child frame.
    pass_.function_version = &worker_version_.base;
    Verify();
  }

  void TearDown() override {
    loom_pass_value_fact_owner_deinitialize(&fact_owner_);
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  void Register(uint8_t dialect,
                const loom_op_vtable_t* const* (*get)(iree_host_size_t*)) {
    iree_host_size_t count;
    const auto* vtables = get(&count);
    IREE_ASSERT_OK(
        loom_context_register_dialect(&context_, dialect, vtables, count));
  }

  loom_symbol_ref_t Symbol(const char* name) {
    loom_string_id_t string;
    loom_symbol_id_t symbol;
    IREE_CHECK_OK(loom_module_intern_string(
        module_, iree_make_cstring_view(name), &string));
    IREE_CHECK_OK(loom_module_add_symbol(module_, string, &symbol));
    return {0, symbol};
  }

  loom_func_like_t Function(loom_builder_t* builder, loom_symbol_ref_t symbol) {
    loom_builder_set_block(builder, loom_module_block(module_));
    builder->ip.parent_op = nullptr;
    loom_op_t* function;
    IREE_CHECK_OK(loom_func_def_build(
        builder, 0, 0, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(),
        loom_attr_absent(), 0, loom_named_attr_slice_empty(),
        LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), symbol, nullptr,
        0, nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN,
        &function));
    loom_builder_enter_region(builder, function, loom_func_def_body(function));
    loom_op_t* terminator;
    IREE_CHECK_OK(loom_func_return_build(builder, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &terminator));
    loom_builder_set_block(builder, loom_module_block(module_));
    builder->ip.parent_op = nullptr;
    return loom_func_like_cast(module_, function);
  }

  void Verify() {
    const loom_verify_options_t options = {
        .sink = {.fn = loom_diagnostic_stderr_sink},
    };
    loom_verify_result_t result;
    IREE_ASSERT_OK(loom_verify_module(module_, &options, &result));
    EXPECT_EQ(result.error_count, 0u);
  }

  static iree_status_t Select(void* data, loom_module_t*,
                              const loom_pipeline_realization_t* realization,
                              bool* valid) {
    auto* self = static_cast<PipelineRealizationTest*>(data);
    EXPECT_TRUE(self->workers_.empty());
    EXPECT_NE(realization->function.op->parent_block, nullptr);
    EXPECT_EQ(realization->workers, self->source_workers_);
    ++self->selected_;
    *valid = self->selected_ != self->reject_;
    return iree_ok_status();
  }

  static iree_status_t Worker(void* data, loom_rewriter_t*,
                              const loom_pipeline_realization_t* realization,
                              iree_host_size_t index) {
    auto* self = static_cast<PipelineRealizationTest*>(data);
    EXPECT_EQ(self->selected_, 2u);
    EXPECT_NE(realization->function.op->parent_block, nullptr);
    auto function = realization->instances[index].channels.function;
    EXPECT_NE(function.op, realization->workers[index].function.op);
    self->workers_.push_back(function.op);
    EXPECT_EQ(realization->worker_versions[index]->base.function.op,
              function.op);
    EXPECT_EQ(realization->worker_versions[index]->target_context_ordinal, 3);
    return iree_ok_status();
  }

  static bool HasWorkerBody(const void* data, iree_host_size_t) {
    return static_cast<const PipelineRealizationTest*>(data)->has_body_;
  }

  static iree_status_t Prepare(void* data, loom_rewriter_t*,
                               const loom_pipeline_realization_t* realization,
                               bool* valid) {
    auto* self = static_cast<PipelineRealizationTest*>(data);
    EXPECT_EQ(self->workers_.size(), self->has_body_ ? 2u : 0u);
    EXPECT_NE(realization->function.op->parent_block, nullptr);
    EXPECT_EQ(realization->invocations.entries[0]->function.op,
              self->source_version_.base.function.op);
    ++self->prepared_;
    *valid = true;
    return iree_ok_status();
  }

  static iree_status_t Entry(void* data, loom_rewriter_t* rewriter,
                             const loom_pipeline_realization_t* realization,
                             loom_func_like_t* output) {
    auto* self = static_cast<PipelineRealizationTest*>(data);
    EXPECT_EQ(self->workers_.size(), self->has_body_ ? 2u : 0u);
    EXPECT_EQ(self->prepared_, 2u);
    if (!self->has_body_) {
      EXPECT_EQ(realization->instances[0].channels.function.op, nullptr);
      EXPECT_EQ(realization->worker_versions[0], nullptr);
    }
    *output = self->Function(&rewriter->builder, realization->entry.symbol);
    realization->entry.version->target_context_ordinal =
        realization->entry_target_context;
    self->outputs_.push_back(realization->entry.version);
    return iree_ok_status();
  }

  loom_pipeline_realization_request_t Request(loom_symbol_ref_t symbol) {
    return {&invocations_,
            source_workers_,
            symbol,
            {Select, HasWorkerBody, Worker, Prepare, Entry, this}};
  }

  // Arena and module own all source and consuming instances throughout the
  // test.
  iree_arena_block_pool_t blocks_;
  // Retained analysis and compiler-version storage.
  iree_arena_allocator_t arena_;
  // Dialect vocabulary for the ordinary source and emitted entry functions.
  loom_context_t context_;
  // Source definitions remain available until every occurrence is consumed.
  loom_module_t* module_ = nullptr;
  // Ordinary source construction helper.
  loom_builder_t builder_;
  // Immutable source facts.
  loom_value_fact_table_t facts_;
  // Immutable source placements.
  loom_pipeline_resources_t resources_;
  // Retained analysis owner with closed source proofs.
  loom_pipeline_invocations_t invocations_;
  // Shared source occurrence requested at two independent destinations.
  loom_pipeline_worker_t* source_workers_ = nullptr;
  // Stable source entry identity.
  loom_target_function_version_t source_version_ = {};
  // Stable source worker identity.
  loom_target_function_version_t worker_version_ = {};
  // Compilation owner that receives emitted worker and entry versions.
  loom_function_version_owner_t versions_;
  // Normal mutable target capability supplied by the compilation owner.
  loom_target_pass_capability_t capability_;
  // Borrowed capability list for the pass frame.
  const loom_pass_environment_capability_t* capabilities_[1];
  // Explicit environment, independent of a child-specific function frame.
  loom_pass_environment_t environment_;
  // Normal pass fact cache, invalidated at the consuming rewrite boundary.
  loom_pass_value_fact_owner_t fact_owner_;
  // Normal pass frame consumed by realization.
  loom_pass_t pass_ = {};
  // Number of physical selections observed before mutation.
  size_t selected_ = 0;
  // Entry preparations completed while all source constructions remain live.
  size_t prepared_ = 0;
  // Optional one-based selection rejected by the target admission callback.
  size_t reject_ = 0;
  // Selected mechanics consume source plans without an instruction body when
  // false, just as native descriptor-only workers do.
  bool has_body_ = true;
  // Independently materialized workers in emission order.
  std::vector<loom_op_t*> workers_;
  // Distinct output entry versions, or the source version for replacement.
  std::vector<loom_target_function_version_t*> outputs_;
};

TEST_F(PipelineRealizationTest, IndependentEntriesPreserveSourceAndVersions) {
  const auto source = source_version_.base.function;
  const auto worker = worker_version_.base.function;
  const auto second = Symbol("second");
  const auto first = Symbol("first");
  const loom_pipeline_realization_request_t requests[] = {Request(second),
                                                          Request(first)};
  bool valid = false;
  IREE_ASSERT_OK(
      loom_pipeline_realize_occurrences(&pass_, module_, requests, 2, &valid));
  ASSERT_TRUE(valid);
  ASSERT_EQ(outputs_.size(), 2u);
  EXPECT_EQ(source_version_.base.function.op, source.op);
  EXPECT_NE(source.op->parent_block, nullptr);
  EXPECT_EQ(worker_version_.base.function.op, worker.op);
  EXPECT_NE(workers_[0], workers_[1]);
  EXPECT_NE(outputs_[0], outputs_[1]);
  EXPECT_NE(outputs_[0], &source_version_);
  EXPECT_EQ(outputs_[0]->base.flags, 0u);
  EXPECT_EQ(outputs_[0]->target_context_ordinal, 4);
  EXPECT_EQ(outputs_[1]->target_context_ordinal, 5);
  EXPECT_EQ(loom_func_like_callee(outputs_[0]->base.function).symbol_id,
            second.symbol_id);
  EXPECT_EQ(loom_func_like_callee(outputs_[1]->base.function).symbol_id,
            first.symbol_id);
  EXPECT_EQ(versions_.list.count, 6u);
  Verify();
}

TEST_F(PipelineRealizationTest, ReplacementWaitsForEverySourceConsumer) {
  const auto source = source_version_.base.function;
  const loom_pipeline_realization_request_t requests[] = {
      Request(loom_func_like_callee(source)), Request(Symbol("independent"))};
  bool valid = false;
  IREE_ASSERT_OK(
      loom_pipeline_realize_occurrences(&pass_, module_, requests, 2, &valid));
  ASSERT_TRUE(valid);
  ASSERT_EQ(outputs_.size(), 2u);
  EXPECT_EQ(outputs_[0], &source_version_);
  EXPECT_NE(source_version_.base.function.op, source.op);
  EXPECT_NE(outputs_[1], &source_version_);
  EXPECT_EQ(outputs_[1]->base.flags, 0u);
  EXPECT_EQ(versions_.list.count, 5u);
  Verify();
}

TEST_F(PipelineRealizationTest, OffloadedWorkersNeedNoBodiesOrVersions) {
  has_body_ = false;
  const auto worker = worker_version_.base.function;
  const loom_pipeline_realization_request_t requests[] = {
      Request(Symbol("first")), Request(Symbol("second"))};
  const auto symbols = module_->symbols.count;
  bool valid = false;
  IREE_ASSERT_OK(
      loom_pipeline_realize_occurrences(&pass_, module_, requests, 2, &valid));
  ASSERT_TRUE(valid);
  EXPECT_TRUE(workers_.empty());
  EXPECT_EQ(outputs_.size(), 2u);
  EXPECT_EQ(module_->symbols.count, symbols);
  EXPECT_EQ(versions_.list.count, 4u);
  EXPECT_EQ(worker_version_.base.function.op, worker.op);
  EXPECT_NE(worker.op->parent_block, nullptr);
  Verify();
}

TEST_F(PipelineRealizationTest, RejectedSelectionLeavesAllSourcesIntact) {
  const auto source = source_version_.base.function;
  const loom_pipeline_realization_request_t requests[] = {
      Request(Symbol("first")), Request(Symbol("second"))};
  const auto symbols = module_->symbols.count;
  reject_ = 2;
  bool valid = true;
  IREE_ASSERT_OK(
      loom_pipeline_realize_occurrences(&pass_, module_, requests, 2, &valid));
  EXPECT_FALSE(valid);
  EXPECT_TRUE(workers_.empty());
  EXPECT_TRUE(outputs_.empty());
  EXPECT_EQ(module_->symbols.count, symbols);
  EXPECT_EQ(source_version_.base.function.op, source.op);
  EXPECT_EQ(versions_.list.count, 2u);
  Verify();
}

}  // namespace
}  // namespace loom
