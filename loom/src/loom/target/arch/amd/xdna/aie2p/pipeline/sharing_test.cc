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

TEST(Aie2pSharingTest, CollectivesPreserveSparseControlIdentities) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {7, 1}};
  loom_aie2p_native_tile_t terminals[30] = {};
  loom_aie2p_native_delivery_t delivery = {};
  // Independent core streams populate the shared path's destination namespace.
  // The next core's first control route therefore receives a sparse high ID.
  for (uint8_t i = 0; i < 30; ++i) {
    terminals[i].coordinate = {static_cast<uint16_t>(i / 6),
                               static_cast<uint16_t>(i % 6)};
    ASSERT_NE(
        loom_aie2p_native_delivery_packet(
            &delivery, 0, &terminals[i], LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
        UINT8_MAX);
  }
  const uint8_t control = loom_aie2p_native_delivery_packet(
      &delivery, 0, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  ASSERT_EQ(control, 31);
  uint32_t occupied = UINT32_C(1) << control;
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), 30);
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), 29);
  EXPECT_EQ(occupied, UINT32_C(0xe0000000));
}

TEST(Aie2pSharingTest, CollectivesReserveEveryNonPayloadIdentityOnce) {
  const uint32_t payload = UINT32_C(1) << LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID;
  uint32_t occupied = payload;
  uint32_t selected = 0;
  for (uint8_t i = 0; i < 31; ++i) {
    const uint8_t packet = loom_aie2p_native_collective_packet(&occupied);
    ASSERT_LT(packet, 32);
    EXPECT_NE(packet, LOOM_AIE2P_NATIVE_DELIVERY_PACKET_ID);
    const uint32_t bit = UINT32_C(1) << packet;
    EXPECT_EQ(selected & bit, 0u);
    selected |= bit;
    EXPECT_EQ(occupied, selected | payload);
  }
  EXPECT_EQ(selected, UINT32_C(0xfffeffff));
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), UINT8_MAX);
  EXPECT_EQ(occupied, selected | payload);
}

TEST(Aie2pSharingTest, UnsharedCollectivesCanUseEveryPacketIdentity) {
  uint32_t occupied = 0;
  for (uint8_t i = 0; i < 32; ++i) {
    const uint32_t previous = occupied;
    const uint8_t packet = loom_aie2p_native_collective_packet(&occupied);
    ASSERT_LT(packet, 32);
    EXPECT_EQ(previous & (UINT32_C(1) << packet), 0u);
    EXPECT_EQ(occupied, previous | (UINT32_C(1) << packet));
  }
  EXPECT_EQ(occupied, UINT32_MAX);
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), UINT8_MAX);
  EXPECT_EQ(occupied, UINT32_MAX);
}

TEST(Aie2pSharingTest, ControlSelectionPreservesCollectiveIdentities) {
  const loom_aie2p_native_tile_t memory = {.coordinate = {0, 1}};
  // The shared payload and all but two other IDs are assigned on this core.
  uint32_t occupied = UINT32_C(0x7ffffffe);
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), 31);
  const uint8_t control = loom_aie2p_native_delivery_packet(
      nullptr, occupied, &memory, LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0);
  ASSERT_EQ(control, 0);
  occupied |= UINT32_C(1) << control;
  EXPECT_EQ(loom_aie2p_native_collective_packet(&occupied), UINT8_MAX);
  EXPECT_EQ(
      loom_aie2p_native_delivery_packet(nullptr, occupied, &memory,
                                        LOOM_XDNA_STREAM_PORT_TILE_CONTROL, 0),
      UINT8_MAX);
}

}  // namespace
