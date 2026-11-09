// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/hal/testing/deferred_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Owns one real reservation transaction through its explicit completion.
struct Acquisition {
  explicit Acquisition(iree_hal_pool_t* source, iree_device_size_t length)
      : pool(source),
        request(iree_hal_make_pool_reservation_request(length, 16)) {}
  ~Acquisition() {
    EXPECT_TRUE(completed);
    if (iree_status_is_ok(status) && result != IREE_HAL_POOL_ACQUIRE_NONE &&
        result < IREE_HAL_POOL_ACQUIRE_EXHAUSTED) {
      iree_hal_pool_release_reservations(pool, 1, &reservation, nullptr);
    }
    iree_status_free(status);
  }
  void Submit() {
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool, 1, &request, nullptr,
        {[](void* user_data, iree_status_t status,
            iree_hal_pool_acquire_result_t result) {
           auto* self = static_cast<Acquisition*>(user_data);
           EXPECT_FALSE(self->completed);
           self->status = status;
           self->result = result;
           self->completed = true;
         },
         this},
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &reservation, &info));
  }
  // Borrowed pool outliving this transaction and its returned reservation.
  iree_hal_pool_t* pool;
  // Stable request borrowed until completion.
  iree_hal_pool_reservation_request_t request;
  // Owned reservation after successful completion.
  iree_hal_pool_reservation_t reservation = {};
  // Borrowed acquisition history for the returned reservation.
  iree_hal_pool_acquire_info_t info = {};
  // Owned completion status.
  iree_status_t status = iree_ok_status();
  // Capacity outcome, meaningful only after successful completion.
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_NONE;
  // Exactly one callback has completed the transaction.
  bool completed = false;
};

class TLSFPoolGrowthTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    const auto allocator = iree_allocator_system();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), allocator, &proactor_));
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator, &tracker_));
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(0, allocator, &provider_));
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, provider_, notification_, tracker_, allocator, &native_));
    IREE_ASSERT_OK(
        iree::hal::testing::DeferredBackingPool::Create(native_, &backing_));
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = 4096;
    options.tlsf_options.alignment = 16;
    IREE_ASSERT_OK(
        iree_hal_tlsf_pool_create(backing_, &options, allocator, &pool_));
  }
  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_hal_pool_release(backing_);
    iree_hal_pool_release(native_);
    iree_hal_slab_provider_release(provider_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
  }
  // Owns capacity observations for all pools in this test.
  iree_async_proactor_t* proactor_ = nullptr;
  // Owned native capacity notification.
  iree_async_notification_t* notification_ = nullptr;
  // Owned causal domain for native and offset storage.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Owned native host-storage provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Owned real source whose allocations and bytes are observed.
  iree_hal_pool_t* native_ = nullptr;
  // Owned dependency gate; all successful backing comes from native_.
  iree::hal::testing::DeferredBackingPool* backing_ = nullptr;
  // Owned production suballocator under test.
  iree_hal_pool_t* pool_ = nullptr;
};

TEST_P(TLSFPoolGrowthTest, PendingParentUsesCapacityObservation) {
  backing_->defer = true;
  Acquisition first(pool_, 128);
  first.Submit();
  EXPECT_FALSE(first.completed);
  // Only the captured request stays deferred. Any duplicate growth would
  // allocate real extra backing and complete immediately.
  backing_->defer = false;
  const uint32_t epoch =
      iree_async_notification_query_epoch(pool_->notification);
  Acquisition second(pool_, 128);
  second.Submit();
  EXPECT_TRUE(second.completed);
  IREE_EXPECT_OK(iree_status_clone(second.status));
  EXPECT_EQ(second.result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(second.reservation.block_handle, 0u);
  EXPECT_EQ(iree_async_notification_query_epoch(pool_->notification), epoch);

  // Dedicated allocations do not compete for shared-parent growth.
  {
    Acquisition dedicated(pool_, 8192);
    dedicated.Submit();
    EXPECT_TRUE(dedicated.completed);
    IREE_EXPECT_OK(iree_status_clone(dedicated.status));
    EXPECT_EQ(dedicated.result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }
  if (GetParam()) {
    backing_->FailOne();
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                          iree_status_clone(first.status));
  } else {
    backing_->CompleteOne();
    IREE_EXPECT_OK(iree_status_clone(first.status));
    EXPECT_EQ(first.result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }
  EXPECT_TRUE(first.completed);
  EXPECT_NE(iree_async_notification_query_epoch(pool_->notification), epoch);

  Acquisition retry(pool_, 128);
  retry.Submit();
  EXPECT_TRUE(retry.completed);
  IREE_EXPECT_OK(iree_status_clone(retry.status));
  EXPECT_EQ(retry.result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_stats_t statistics;
  iree_hal_pool_query_stats(native_, &statistics);
  EXPECT_EQ(statistics.bytes_committed, 4096u);
}

INSTANTIATE_TEST_SUITE_P(ParentCompletion, TLSFPoolGrowthTest,
                         ::testing::Bool());

}  // namespace
