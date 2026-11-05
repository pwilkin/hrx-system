// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/sharing.h"

#include "iree/testing/gtest.h"

namespace {

TEST(Aie2pSharingTest, IndependentCoresAgreeOnSharedPacketTerminals) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {0, 1}};
  const loom_aie2p_native_tile_t shim = {.coordinate = {0, 0}};
  loom_aie2p_native_delivery_t delivery = {};
  // One core reaches SRAM before the shim; another reaches the shim first.
  // Their independent source ordinals cannot define the shared packet IDs.
  const uint8_t memory_packet = loom_aie2p_native_delivery_packet(
      &delivery, 0, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  const uint8_t shim_packet = loom_aie2p_native_delivery_packet(
      &delivery, UINT32_C(1) << memory_packet, &shim,
      LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  ASSERT_NE(memory_packet, shim_packet);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                &delivery, 0, &shim, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            shim_packet);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                &delivery, UINT32_C(1) << shim_packet, &memory,
                LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            memory_packet);

  const uint32_t occupied =
      (UINT32_C(1) << memory_packet) | (UINT32_C(1) << shim_packet);
  const uint8_t credit_packet = loom_aie2p_native_delivery_packet(
      &delivery, occupied, &shim, LOOM_XDNA_STREAM_PORT_DMA, 0);
  EXPECT_NE(credit_packet, memory_packet);
  EXPECT_NE(credit_packet, shim_packet);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(&delivery, 0, &shim,
                                              LOOM_XDNA_STREAM_PORT_DMA, 0),
            credit_packet);
  EXPECT_NE(loom_aie2p_native_delivery_packet(&delivery, 0, &shim,
                                              LOOM_XDNA_STREAM_PORT_DMA, 1),
            credit_packet);
}

TEST(Aie2pSharingTest, UnsharedCorePacketsKeepIndependentIdentities) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {0, 1}};
  const loom_aie2p_native_tile_t shim = {.coordinate = {0, 0}};
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                nullptr, 0, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            0);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                nullptr, 0, &shim, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            0);
  EXPECT_EQ(loom_aie2p_native_delivery_packet(
                nullptr, 1, &shim, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
            1);
}

}  // namespace
