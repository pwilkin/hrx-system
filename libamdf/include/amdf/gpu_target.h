// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_GPU_TARGET_H_
#define AMDF_GPU_TARGET_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Instruction sets used to describe immutable, relocation-free GPU programs.
/// These describe the bytes supplied by a compiler, not queue availability.
typedef uint32_t amdf_gpu_instruction_set_t;
enum amdf_gpu_instruction_set_e {
  AMDF_GPU_INSTRUCTION_SET_NONE = 0,
  AMDF_GPU_INSTRUCTION_SET_GFX9 = 1u << 0,
  AMDF_GPU_INSTRUCTION_SET_GFX90A = 1u << 1,
  AMDF_GPU_INSTRUCTION_SET_GFX94 = 1u << 2,
  AMDF_GPU_INSTRUCTION_SET_GFX101 = 1u << 3,
  AMDF_GPU_INSTRUCTION_SET_GFX103 = 1u << 4,
  AMDF_GPU_INSTRUCTION_SET_GFX11 = 1u << 5,
  AMDF_GPU_INSTRUCTION_SET_GFX115 = 1u << 6,
  AMDF_GPU_INSTRUCTION_SET_GFX117 = 1u << 7,
  AMDF_GPU_INSTRUCTION_SET_GFX12 = 1u << 8,
  AMDF_GPU_INSTRUCTION_SET_GFX125 = 1u << 9,
};

/// Compiler device-library grid barrier protocol, independently of whether a
/// live queue supports cooperative scheduling or the memory supports atomics.
typedef uint32_t amdf_gpu_grid_barrier_t;
enum amdf_gpu_grid_barrier_e {
  AMDF_GPU_GRID_BARRIER_NONE = 0,
  AMDF_GPU_GRID_BARRIER_MEMORY = 1,
  AMDF_GPU_GRID_BARRIER_GWS = 2,
};

/// Immutable shader resource encodings. Counts for a wave mode use index zero
/// for wave32 and index one for wave64. A zero wave size mask is unsupported.
typedef struct amdf_gpu_shader_capabilities_t {
  /// Nonzero when DS_GWS_INIT is available to initialize a grid semaphore.
  uint32_t global_wave_sync;
  /// Supported wave sizes: bit 5 for wave32 and bit 6 for wave64.
  uint32_t wave_size_mask;
  /// Nonzero when the descriptor can select a two-CU workgroup processor.
  uint32_t workgroup_processor_mode;
  /// RSRC1 bit selecting workgroup-processor mode, or zero for another
  /// encoding.
  uint32_t workgroup_processor_mask;
  /// Independently allocated SIMDs per physical compute unit.
  uint32_t simds_per_compute_unit;
  /// Vector registers per SIMD in each wave mode's register units.
  uint32_t vector_register_count[2];
  /// Registers represented by one encoded VGPR block in each wave mode.
  uint32_t vector_encoding_granule[2];
  /// Registers allocated to a wave after rounding in each wave mode.
  uint32_t vector_allocation_granule[2];
  /// Scalar registers per SIMD, or zero when scalar allocation does not limit
  /// occupancy. Native trap reservation is a separate live-queue fact.
  uint32_t scalar_register_count;
  /// Scalar registers represented by one encoded SGPR block.
  uint32_t scalar_encoding_granule;
  /// Scalar registers allocated to a wave before native trap reservation.
  uint32_t scalar_allocation_granule;
  /// Bytes in one workgroup-local memory allocation unit.
  uint32_t local_memory_granule;
  /// Architectural LDS capacity in CU mode, before native qualification.
  uint32_t local_memory_byte_length;
  /// Standard barrier slots in one CU-mode scheduling domain.
  uint32_t barrier_count;
  /// RSRC3 mask containing a shared-VGPR count in eight-register units.
  uint32_t shared_vector_register_mask;
  /// RSRC3 features whose residency requires additional scheduling facts.
  uint32_t unmodeled_resource3_mask;
  /// RSRC2 bit enabling dynamic VGPR allocation, or zero when absent.
  uint32_t dynamic_vector_resource2_mask;
  /// RSRC3 bit enabling dynamic VGPR allocation, or zero when absent.
  uint32_t dynamic_vector_resource3_mask;
} amdf_gpu_shader_capabilities_t;

/// PM4 compute encodings, separate from a live family's publication contract.
typedef struct amdf_gpu_pm4_capabilities_t {
  /// Nonzero when the prepared compute register encoding is described here.
  uint32_t compute_registers;
  /// Register address for COMPUTE_PGM_RSRC3.
  uint32_t resource3_register;
  /// RSRC2 user-SGPR count mask after shifting right by one.
  uint32_t user_data_count_mask;
  /// Log2 bytes represented by one LDS_SIZE register unit.
  uint32_t group_segment_unit_log2;
  /// Register address for COMPUTE_TMPRING_SIZE.
  uint32_t scratch_size_register;
  /// Maximum private bytes per wave represented by the scratch size field.
  uint32_t maximum_scratch_wave_byte_length;
  /// GCR actions making prior shader writes available to later shaders.
  uint32_t shader_cache_control;
  /// GCR actions releasing dirty data to system-visible memory.
  uint32_t release_cache_control;
  /// GCR actions acquiring system-visible memory before later shader reads.
  uint32_t acquire_cache_control;
  /// Nonzero when CP scalar accesses bypass shader L2.
  uint32_t command_processor_bypasses_l2;
  /// Nonzero when instruction-cache publication uses the coherency-control
  /// ACQUIRE_MEM format instead of GCR.
  uint32_t instruction_cache_uses_coherency_control;
} amdf_gpu_pm4_capabilities_t;

/// Pure ISA/ABI facts for cold program preparation and offline inspection.
///
/// These facts do not authorize native services, memory operations, or queue
/// admission. Live endpoint, queue-family, and memory queries establish those
/// independent contracts. Queries allocate nothing and initialize no driver.
typedef struct amdf_gpu_target_capabilities_t {
  /// Instruction sets accepted for embedded compiler-produced programs.
  amdf_gpu_instruction_set_t instruction_sets;
  /// Device-library algorithm used by ordinary cooperative grid barriers.
  amdf_gpu_grid_barrier_t grid_barrier;
  /// Shader descriptor interpretation and architectural resource units.
  amdf_gpu_shader_capabilities_t shader;
  /// Native compute-register and cache-command encoding facts.
  amdf_gpu_pm4_capabilities_t pm4;
  /// RSRC3 instruction-prefetch mask after shifting right by four.
  uint32_t instruction_prefetch_mask;
  /// Exact processor artifacts require explicit silicon-revision metadata.
  uint32_t requires_revision_metadata;
} amdf_gpu_target_capabilities_t;

/// Queries immutable target facts. Unknown fields remain zero; this never
/// substitutes a nearby target for an unknown instruction or register encoding.
/// The version is identity supplied by native discovery or an artifact parser.
static inline amdf_gpu_target_capabilities_t amdf_gpu_target_query_capabilities(
    uint32_t major, uint32_t minor, uint32_t stepping) {
  amdf_gpu_target_capabilities_t result = {0};
  if (major == 9) {
    result.instruction_sets = minor < 4 ? AMDF_GPU_INSTRUCTION_SET_GFX9 : 0;
    if (minor == 0 && stepping == 10) {
      result.instruction_sets |= AMDF_GPU_INSTRUCTION_SET_GFX90A;
    } else if (minor == 4 || minor == 5) {
      result.instruction_sets = AMDF_GPU_INSTRUCTION_SET_GFX94;
    }
  } else if (major == 10) {
    result.instruction_sets = minor == 1   ? AMDF_GPU_INSTRUCTION_SET_GFX101
                              : minor == 3 ? AMDF_GPU_INSTRUCTION_SET_GFX103
                                           : 0;
  } else if (major == 11 && (minor == 0 || minor == 5 || minor == 7)) {
    result.instruction_sets = AMDF_GPU_INSTRUCTION_SET_GFX11;
    if (minor == 5) {
      result.instruction_sets |= AMDF_GPU_INSTRUCTION_SET_GFX115;
    }
    if (minor == 7) {
      result.instruction_sets |= AMDF_GPU_INSTRUCTION_SET_GFX117;
    }
  } else if (major == 12) {
    result.instruction_sets = minor == 0   ? AMDF_GPU_INSTRUCTION_SET_GFX12
                              : minor == 5 ? AMDF_GPU_INSTRUCTION_SET_GFX125
                                           : 0;
  }
  if ((major == 9 && minor == 4 && stepping == 2) ||
      (major == 9 && minor == 5 && stepping == 0) || major == 11 ||
      major == 12) {
    result.grid_barrier = AMDF_GPU_GRID_BARRIER_MEMORY;
  } else if (major == 9 || major == 10) {
    result.grid_barrier = AMDF_GPU_GRID_BARRIER_GWS;
  }
  result.shader.global_wave_sync = major == 9 || major == 10;
  if (major == 9) {
    const uint32_t unified =
        (minor == 0 && stepping == 10) || minor == 4 || minor == 5;
    result.shader.wave_size_mask = 1u << 6;
    result.shader.simds_per_compute_unit = 4;
    result.shader.vector_register_count[1] = unified ? 512 : 256;
    result.shader.vector_encoding_granule[1] = unified ? 8 : 4;
    result.shader.vector_allocation_granule[1] = unified ? 8 : 4;
    result.shader.scalar_register_count = 800;
    result.shader.scalar_encoding_granule = 8;
    result.shader.scalar_allocation_granule = 16;
    result.shader.local_memory_granule = minor == 5 ? 1280 : 512;
    result.shader.local_memory_byte_length = (minor == 5 ? 160u : 64u) * 1024;
    result.shader.barrier_count = 16;
    result.shader.unmodeled_resource3_mask = unified ? 1u << 16 : 0;
  } else if (major >= 10 && major <= 12) {
    const uint32_t wave32_only = major == 12 && minor == 5;
    const uint32_t large_register_file =
        major == 12 || (major == 11 && ((minor == 0 && stepping <= 1) ||
                                        (minor == 5 && stepping == 1)));
    result.shader.wave_size_mask = (1u << 5) | (wave32_only ? 0 : 1u << 6);
    result.shader.workgroup_processor_mode = !wave32_only;
    result.shader.workgroup_processor_mask = 1u << 29;
    result.shader.simds_per_compute_unit = wave32_only ? 4 : 2;
    result.shader.vector_register_count[0] = large_register_file ? 1536 : 1024;
    result.shader.vector_register_count[1] =
        result.shader.vector_register_count[0] / 2;
    result.shader.vector_encoding_granule[0] = wave32_only ? 16 : 8;
    result.shader.vector_encoding_granule[1] = 4;
    result.shader.vector_allocation_granule[0] = major == 10 && minor == 1 ? 8
                                                 : large_register_file     ? 24
                                                                           : 16;
    result.shader.vector_allocation_granule[1] =
        result.shader.vector_allocation_granule[0] / 2;
    result.shader.local_memory_granule = wave32_only ? 2048 : 512;
    result.shader.local_memory_byte_length = (wave32_only ? 320u : 64u) * 1024;
    result.shader.barrier_count = 16;
    result.shader.shared_vector_register_mask = major <= 11 ? 0xf : 0;
    result.shader.unmodeled_resource3_mask =
        major == 12 ? (1u << 13) | (wave32_only ? (7u << 14) | (7u << 18) : 0)
                    : 0;
    result.shader.dynamic_vector_resource2_mask =
        major == 12 && !wave32_only ? 1u << 6 : 0;
    result.shader.dynamic_vector_resource3_mask = wave32_only ? 1u << 17 : 0;
  }
  result.pm4.instruction_cache_uses_coherency_control = major == 9;
  const uint32_t pm4_supported =
      (major == 11 && (minor == 0 || minor == 5 || minor == 7)) ||
      (major == 12 && (minor == 0 || minor == 5));
  if (pm4_supported) {
    const uint32_t compact_registers = major == 12 && minor == 5;
    result.pm4.compute_registers = 1;
    result.pm4.resource3_register = compact_registers ? 0x2e23 : 0x2e28;
    result.pm4.user_data_count_mask = compact_registers ? 63 : 31;
    result.pm4.group_segment_unit_log2 = compact_registers ? 11 : 9;
    result.pm4.scratch_size_register = compact_registers ? 0x2e16 : 0x2e18;
    result.pm4.maximum_scratch_wave_byte_length =
        (major >= 12 ? 0x3ffffu : 0x7fffu) << 8;
    const uint32_t lower_writeback = (1u << 6) | (1u << 16);
    const uint32_t shader = lower_writeback | (1u << 7) | (1u << 8) |
                            (major == 11 ? (1u << 5) | (1u << 9) : 0);
    const uint32_t system_scope = compact_registers ? 2u << 4 : 0;
    result.pm4.shader_cache_control = shader;
    result.pm4.release_cache_control =
        lower_writeback | (1u << 15) | system_scope;
    result.pm4.acquire_cache_control =
        shader | (1u << 15) | (1u << 14) | system_scope;
    result.pm4.command_processor_bypasses_l2 = major != 11;
  }
  result.instruction_prefetch_mask = major == 11   ? 0x3f
                                     : major == 12 ? 0xff
                                                   : 0;
  result.requires_revision_metadata =
      major == 12 && minor == 5 && stepping == 0;
  return result;
}

/// Selects the first compatible immutable instruction variant, or UINT32_MAX.
/// Each candidate begins with an amdf_gpu_instruction_set_t and is separated
/// by candidate_stride bytes. Candidate metadata is caller-owned and remains
/// readable through the call. Ordering expresses the owner's preference among
/// compatible artifacts; this function owns compatibility interpretation.
static inline uint32_t amdf_gpu_target_select_instructions(
    amdf_gpu_instruction_set_t supported, uint32_t candidate_count,
    const void* candidates, uint32_t candidate_stride) {
  for (uint32_t i = 0; i < candidate_count; ++i) {
    const amdf_gpu_instruction_set_t required =
        *(const amdf_gpu_instruction_set_t*)((const uint8_t*)candidates +
                                             (uint64_t)i * candidate_stride);
    if (required && (supported & required) == required) {
      return i;
    }
  }
  return UINT32_MAX;
}

/// Parses the numeric suffix of an exact AMDGPU processor (for example,
/// "90a" or "1250"). The input span need not be terminated. Returns zero for
/// malformed spelling and leaves output storage unchanged. Qualifiers and
/// generic target syntax belong to the surrounding artifact grammar.
static inline int amdf_gpu_target_parse_version(const char* data,
                                                uint64_t length,
                                                uint32_t* out_major,
                                                uint32_t* out_minor,
                                                uint32_t* out_stepping) {
  if (length != 3 && length != 4) {
    return 0;
  }
  if (length == 4 && data[0] != '1') {
    return 0;
  }
  const uint32_t first = (uint32_t)(data[length - 3] - '0');
  const uint32_t minor = (uint32_t)(data[length - 2] - '0');
  const char digit = data[length - 1];
  const uint32_t stepping = digit >= '0' && digit <= '9'   ? digit - '0'
                            : digit >= 'a' && digit <= 'f' ? digit - 'a' + 10
                            : digit >= 'A' && digit <= 'F' ? digit - 'A' + 10
                                                           : UINT32_MAX;
  if (first > 9 || minor > 9 || stepping > 15) {
    return 0;
  }
  *out_major = first + (length == 4 ? 10 : 0);
  *out_minor = minor;
  *out_stepping = stepping;
  return 1;
}

/// Compares exact native/artifact identities without deriving capabilities.
static inline int amdf_gpu_target_equal(uint32_t lhs_major, uint32_t lhs_minor,
                                        uint32_t lhs_stepping,
                                        uint32_t rhs_major, uint32_t rhs_minor,
                                        uint32_t rhs_stepping) {
  return lhs_major == rhs_major && lhs_minor == rhs_minor &&
         lhs_stepping == rhs_stepping;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_GPU_TARGET_H_
