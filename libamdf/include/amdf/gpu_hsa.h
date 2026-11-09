// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_GPU_HSA_H_
#define AMDF_GPU_HSA_H_

#include "amdf/gpu_target.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Packet encodings used by the HSA vendor-packet protocol. These describe
/// possible encodings, not a live runtime's support for submitting them.
typedef uint32_t amdf_gpu_hsa_packet_features_t;
enum amdf_gpu_hsa_packet_feature_e {
  // AMD vendor AQL PM4-IB packets can jump to device-visible PM4 programs.
  AMDF_GPU_HSA_PACKET_AQL_PM4_IB = 1u << 0,
  // AMD vendor AQL BARRIER_VALUE packets can wait on arbitrary signal values.
  AMDF_GPU_HSA_PACKET_AQL_BARRIER_VALUE = 1u << 1,
  // PM4 WAIT_REG_MEM64 packets can perform 64-bit memory comparisons.
  AMDF_GPU_HSA_PACKET_WAIT_REG_MEM64 = 1u << 2,
  // PM4 EVENT_WRITE can emit compute-pipeline events.
  AMDF_GPU_HSA_PACKET_PM4_EVENT_WRITE = 1u << 3,
  // PM4 SET_SH_REG can program persistent shader registers.
  AMDF_GPU_HSA_PACKET_PM4_SET_SH_REG = 1u << 4,
  // PM4 SET_UCONFIG_REG can program user configuration registers.
  AMDF_GPU_HSA_PACKET_PM4_SET_UCONFIG_REG = 1u << 5,
  // PM4 COPY_DATA can read register values into memory.
  AMDF_GPU_HSA_PACKET_PM4_REGISTER_READBACK = 1u << 6,
  // PM4 COPY_DATA can read performance-counter values into memory.
  AMDF_GPU_HSA_PACKET_PM4_PERFCOUNTER_READBACK = 1u << 7,
  // PM4 COPY_DATA can write immediate values into registers/perfcounters.
  AMDF_GPU_HSA_PACKET_PM4_IMMEDIATE_WRITE = 1u << 8,
  // PM4 WRITE_DATA can write immediate values into memory through TC_L2.
  AMDF_GPU_HSA_PACKET_PM4_WRITE_DATA_MEMORY = 1u << 9,
  // PM4 COPY_DATA can copy memory through TC_L2 into memory through TC_L2.
  AMDF_GPU_HSA_PACKET_PM4_COPY_DATA_MEMORY = 1u << 10,
  // PM4 ACQUIRE_MEM can perform explicit cache-management operations.
  AMDF_GPU_HSA_PACKET_PM4_ACQUIRE_MEM = 1u << 11,
  // PM4 DISPATCH_DIRECT can launch compute work from a PM4 command stream.
  AMDF_GPU_HSA_PACKET_PM4_COMPUTE_DISPATCH_DIRECT = 1u << 12,
  // PM4 DISPATCH_INDIRECT can launch indirect compute work from a PM4 command
  // stream.
  AMDF_GPU_HSA_PACKET_PM4_COMPUTE_DISPATCH_INDIRECT = 1u << 13,
  // PM4 ACQUIRE_MEM uses the gfx9 CP_COHER_CNTL packet layout.
  AMDF_GPU_HSA_PACKET_PM4_ACQUIRE_MEM_COHERENCY = 1u << 14,
  // PM4 ACQUIRE_MEM uses the gfx10+ GCR_CNTL packet layout.
  AMDF_GPU_HSA_PACKET_PM4_ACQUIRE_MEM_GLOBAL_CACHE = 1u << 15,
  // PM4 WAIT_REG_MEM/WAIT_REG_MEM64 can implement 32/64-bit HAL atomic waits.
  AMDF_GPU_HSA_PACKET_PM4_ATOMIC_WAIT = 1u << 16,
  // PM4 ATOMIC_MEM swap can implement 32/64-bit HAL atomic stores.
  AMDF_GPU_HSA_PACKET_PM4_ATOMIC_STORE = 1u << 17,
  // Command-processor memory accesses bypass GL2 and access MALL directly.
  AMDF_GPU_HSA_PACKET_PM4_CP_MEMORY_BYPASSES_GL2 = 1u << 18,
};

/// COPY_DATA clock destinations and cache policy for queue-local timestamps.
typedef uint32_t amdf_gpu_hsa_clock_copy_t;
enum amdf_gpu_hsa_clock_copy_e {
  AMDF_GPU_HSA_CLOCK_COPY_NONE = 0,
  AMDF_GPU_HSA_CLOCK_COPY_MEMORY_STREAM = 1,
  AMDF_GPU_HSA_CLOCK_COPY_TC_L2_LRU = 2,
  AMDF_GPU_HSA_CLOCK_COPY_TC_L2_LAST_USE = 3,
};

/// Event-number table encodings used by the profiling ABI. These are indexes
/// into immutable event tables; clients do not infer instruction capabilities.
typedef uint32_t amdf_gpu_hsa_counter_encoding_t;
enum amdf_gpu_hsa_counter_encoding_e {
  AMDF_GPU_HSA_COUNTER_ENCODING_NONE = 0,
  AMDF_GPU_HSA_COUNTER_ENCODING_GFX9 = 1,
  AMDF_GPU_HSA_COUNTER_ENCODING_GFX90A = 2,
  AMDF_GPU_HSA_COUNTER_ENCODING_GFX940 = 3,
  AMDF_GPU_HSA_COUNTER_ENCODING_GFX10 = 4,
  AMDF_GPU_HSA_COUNTER_ENCODING_GFX11 = 5,
  AMDF_GPU_HSA_COUNTER_ENCODING_COUNT = 6,
};

/// Target-dependent HSA descriptor and reporting facts. Execution-unit counts
/// use HSA agent units, which need not equal native physical-CU counts. Queries
/// perform no allocation or driver IO and do not qualify a live allocation,
/// memory mapping, or submission path.
typedef struct amdf_gpu_hsa_target_capabilities_t {
  /// Common shader descriptor encodings, shared with native preparation.
  amdf_gpu_shader_capabilities_t shader;
  /// Nonzero when exact resource accounting is defined for this processor.
  uint32_t shader_resource_model;
  /// HSA compute units per scheduling domain, indexed by CU/WGP mode.
  uint32_t compute_units_per_domain[2];
  /// LDS bytes per HSA scheduling domain, indexed by CU/WGP mode.
  uint32_t local_memory_byte_length[2];
  /// Standard barriers per HSA scheduling domain, indexed by CU/WGP mode.
  uint32_t barrier_count[2];
  /// HSA compute units represented by one execution-mask resource.
  uint32_t compute_units_per_resource;
  /// RSRC2 bit declaring a descriptor's trap SGPR reservation.
  uint32_t trap_handler_mask;
  /// Nonzero when the target's HDP flush protocol can publish host VRAM stores;
  /// runtime registers, CPU access, and mapping guarantees remain required.
  uint32_t host_data_path_flush;
  /// Nonzero when the HSA uncached allocation flag has target support.
  uint32_t uncached_allocations;
  /// GWS cooperative scheduling granule in HSA compute units, or zero.
  uint32_t cooperative_compute_unit_granule;
  /// Compute units withheld from GWS scheduling for forward progress.
  uint32_t cooperative_reserved_compute_units;
  /// Hardware/vendor packet forms intersected with the live runtime contract.
  amdf_gpu_hsa_packet_features_t packet_features;
  /// COPY_DATA destination and cache policy for the GPU clock.
  amdf_gpu_hsa_clock_copy_t clock_copy;
  /// Nonzero for the published XCC/SE/CU execution-unit ordinal encoding.
  uint32_t xcc_compute_unit_id;
  /// Profiling ABI event-number table, or NONE when unmodeled.
  amdf_gpu_hsa_counter_encoding_t counter_encoding;
  /// Register address of COMPUTE_PGM_RSRC3, or zero when unknown.
  uint32_t resource3_register;
  /// LDS bytes represented by one descriptor allocation unit.
  uint32_t local_memory_granule;
  /// RSRC2 user-SGPR count mask after shifting right by one.
  uint32_t user_data_count_mask;
  /// Descriptor input properties accepted by a scratch-free PM4 launch.
  uint32_t kernel_code_properties;
} amdf_gpu_hsa_target_capabilities_t;

/// Exact processors with a complete HSA scheduling resource model.
static inline int amdf_gpu_hsa_has_shader_resource_model(uint32_t major,
                                                         uint32_t minor,
                                                         uint32_t stepping) {
  switch (major) {
    case 9:
      if (minor == 0) {
        switch (stepping) {
          case 0:   // gfx900
          case 2:   // gfx902
          case 4:   // gfx904
          case 6:   // gfx906
          case 8:   // gfx908
          case 9:   // gfx909
          case 10:  // gfx90a
          case 12:  // gfx90c
            return 1;
          default:
            return 0;
        }
      } else if (minor == 4) {
        return stepping <= 2;  // gfx940-gfx942
      } else if (minor == 5) {
        return stepping == 0;  // gfx950
      }
      return 0;
    case 10:
      return (minor == 1 && stepping <= 3) || (minor == 3 && stepping <= 6);
    case 11:
      return (minor == 0 && stepping <= 3) || (minor == 5 && stepping <= 3) ||
             (minor == 7 && stepping <= 2);
    case 12:
      return ((minor == 0 || minor == 5) && stepping <= 1);
    default:
      return 0;
  }
}

/// Queries HSA ABI facts without initializing or consulting a runtime.
static inline amdf_gpu_hsa_target_capabilities_t amdf_gpu_hsa_query_target(
    uint32_t major, uint32_t minor, uint32_t stepping) {
  const amdf_gpu_target_capabilities_t target =
      amdf_gpu_target_query_capabilities(major, minor, stepping);
  amdf_gpu_hsa_target_capabilities_t result = {{0}};
  result.shader = target.shader;
  result.shader_resource_model =
      amdf_gpu_hsa_has_shader_resource_model(major, minor, stepping);
  if (major >= 9 && major <= 12) {
    result.compute_units_per_domain[0] = 1;
    result.compute_units_per_domain[1] = 2;
    result.local_memory_byte_length[0] = target.shader.local_memory_byte_length;
    result.local_memory_byte_length[1] =
        target.shader.local_memory_byte_length * 2;
    result.barrier_count[0] = 16;
    result.barrier_count[1] = 32;
    result.compute_units_per_resource = major == 9 ? 1 : 2;
    result.trap_handler_mask = major == 9 ? 1u << 6 : 0;
    result.resource3_register = major == 9 ? 0x2e2d : 0x2e28;
    result.local_memory_granule = target.shader.local_memory_granule;
    result.user_data_count_mask = 31;
    result.kernel_code_properties =
        (1u << 3) | (major <= 10 ? 1u : 0) | (major >= 10 ? 1u << 10 : 0);
  }
  if (major == 12) {
    if (minor == 5) {
      // HSA exposes two execution units for this CU-mode scheduling domain.
      result.compute_units_per_domain[0] = 2;
      result.local_memory_byte_length[0] = 320u * 1024;
      result.resource3_register = 0x2e23;
      result.user_data_count_mask = 63;
    } else if (minor != 0) {
      result.compute_units_per_resource = 0;
    }
  }
  result.host_data_path_flush = (major == 9 && minor >= 4 && stepping <= 2) ||
                                (major == 12 && (minor == 0 || minor >= 5));
  result.uncached_allocations = major >= 12;
  result.xcc_compute_unit_id = major == 9 && minor == 4 && stepping == 2;
  if (major == 9 && minor == 0 && stepping == 10) {
    result.cooperative_compute_unit_granule = 8;
    result.cooperative_reserved_compute_units = 8;
  }
  enum {
    // PM4 packet families using the CP_COHER_CNTL cache layout.
    coherency_packets = AMDF_GPU_HSA_PACKET_AQL_PM4_IB |
                        AMDF_GPU_HSA_PACKET_PM4_EVENT_WRITE |
                        AMDF_GPU_HSA_PACKET_PM4_SET_SH_REG |
                        AMDF_GPU_HSA_PACKET_PM4_ACQUIRE_MEM |
                        AMDF_GPU_HSA_PACKET_PM4_ACQUIRE_MEM_COHERENCY |
                        AMDF_GPU_HSA_PACKET_PM4_COMPUTE_DISPATCH_DIRECT,
    // PM4 packet families using the GCR_CNTL cache layout.
    global_cache_packets = AMDF_GPU_HSA_PACKET_AQL_PM4_IB |
                           AMDF_GPU_HSA_PACKET_WAIT_REG_MEM64 |
                           AMDF_GPU_HSA_PACKET_PM4_WRITE_DATA_MEMORY |
                           AMDF_GPU_HSA_PACKET_PM4_COPY_DATA_MEMORY |
                           AMDF_GPU_HSA_PACKET_PM4_EVENT_WRITE |
                           AMDF_GPU_HSA_PACKET_PM4_SET_SH_REG |
                           AMDF_GPU_HSA_PACKET_PM4_SET_UCONFIG_REG |
                           AMDF_GPU_HSA_PACKET_PM4_REGISTER_READBACK |
                           AMDF_GPU_HSA_PACKET_PM4_PERFCOUNTER_READBACK |
                           AMDF_GPU_HSA_PACKET_PM4_IMMEDIATE_WRITE |
                           AMDF_GPU_HSA_PACKET_PM4_ACQUIRE_MEM |
                           AMDF_GPU_HSA_PACKET_PM4_ACQUIRE_MEM_GLOBAL_CACHE |
                           AMDF_GPU_HSA_PACKET_PM4_COMPUTE_DISPATCH_DIRECT |
                           AMDF_GPU_HSA_PACKET_PM4_COMPUTE_DISPATCH_INDIRECT |
                           AMDF_GPU_HSA_PACKET_PM4_ATOMIC_WAIT |
                           AMDF_GPU_HSA_PACKET_PM4_ATOMIC_STORE,
  };

  if (major >= 9 && major <= 12) {
    result.packet_features = AMDF_GPU_HSA_PACKET_AQL_PM4_IB;
  }
  if (major == 9 && ((minor == 0 && stepping >= 8) || minor >= 4)) {
    result.packet_features |= coherency_packets;
  } else if (major >= 10 && major <= 12) {
    result.packet_features |= global_cache_packets;
    if (major == 12) {
      result.packet_features |= AMDF_GPU_HSA_PACKET_PM4_CP_MEMORY_BYPASSES_GL2;
    }
  }
  if (major == 9 &&
      ((minor == 0 && stepping == 10) || (minor >= 4 && stepping <= 2))) {
    result.packet_features |= AMDF_GPU_HSA_PACKET_AQL_BARRIER_VALUE;
  }
  if (major == 9) {
    result.clock_copy = AMDF_GPU_HSA_CLOCK_COPY_MEMORY_STREAM;
  } else if (major == 10 || major == 11) {
    result.clock_copy = AMDF_GPU_HSA_CLOCK_COPY_TC_L2_LRU;
  } else if (major == 12) {
    result.clock_copy = AMDF_GPU_HSA_CLOCK_COPY_TC_L2_LAST_USE;
  }
  if (major == 9 && minor == 0) {
    result.counter_encoding = stepping == 10
                                  ? AMDF_GPU_HSA_COUNTER_ENCODING_GFX90A
                                  : AMDF_GPU_HSA_COUNTER_ENCODING_GFX9;
  } else if (major == 9 && minor == 4 && stepping <= 2) {
    result.counter_encoding = AMDF_GPU_HSA_COUNTER_ENCODING_GFX940;
  } else if (major == 10 &&
             ((minor == 1 && stepping == 0) || (minor == 3 && stepping <= 2))) {
    result.counter_encoding = AMDF_GPU_HSA_COUNTER_ENCODING_GFX10;
  } else if (major == 11 && minor == 0 && stepping <= 2) {
    result.counter_encoding = AMDF_GPU_HSA_COUNTER_ENCODING_GFX11;
  }
  return result;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_GPU_HSA_H_
