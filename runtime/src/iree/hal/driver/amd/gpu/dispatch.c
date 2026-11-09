// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/dispatch.h"

#include "amdf/gpu_target.h"
#include "iree/hal/driver/amd/gpu/image/program.h"

// Relocation-free grid_sync.c instructions and complete descriptor facts are
// checked against configured source compilation. GFX9.4 has no private-segment
// buffer SGPRs; earlier GFX9 and GFX10 receive kernargs in s[4:5].
static const uint32_t iree_hal_amd_gpu_grid_sync_gfx9[] = {
    0xc00a0002u, 0x00000000u, 0xc0020182u, 0x00000010u, 0x7e020280u,
    0x7e080281u, 0x7e040301u, 0x7e060301u, 0xbf8cc07fu, 0xdc7c8000u,
    0x00000101u, 0x7e000206u, 0x7e060202u, 0x7e080203u, 0xdc7c8010u,
    0x00000101u, 0xbefc0080u, 0x7e080301u, 0x7e060300u, 0xdc7c8020u,
    0x00000101u, 0x8100c106u, 0x7e000200u, 0xd9330000u, 0x00000000u,
    0xbf8c0000u, 0xbf810000u};
static const uint32_t iree_hal_amd_gpu_grid_sync_gfx90a[] = {
    0xc00a0002u, 0x00000000u, 0xc0020182u, 0x00000010u, 0x7e000280u,
    0x7e060281u, 0x7e020300u, 0x7e040300u, 0xbf8cc07fu, 0xdc7c8000u,
    0x00000000u, 0xbefc0080u, 0x7e040202u, 0x7e060203u, 0xdc7c8010u,
    0x00000000u, 0xbf800000u, 0x7e040206u, 0x7e060300u, 0x8106c106u,
    0xdc7c8020u, 0x00000000u, 0xbf800000u, 0x7e000206u, 0xd9330000u,
    0x00000000u, 0xbf8c0000u, 0xbf810000u};
static const uint32_t iree_hal_amd_gpu_grid_sync_gfx94[] = {
    0xc00a0100u, 0x00000000u, 0xc0020080u, 0x00000010u, 0x7e000280u,
    0x7e060281u, 0x7e020300u, 0x7e040300u, 0xbf8cc07fu, 0xdc7c8000u,
    0x00040000u, 0x7e080202u, 0x7e0a0300u, 0x7e040206u, 0x7e060207u,
    0xdc7c8010u, 0x00040000u, 0x8102c102u, 0xbefc0080u, 0x7e047080u,
    0xdc7c8020u, 0x00040200u, 0x7e000202u, 0xd9330000u, 0x00000000u,
    0xbf8c0000u, 0xbf810000u};
static const uint32_t iree_hal_amd_gpu_grid_sync_gfx101[] = {
    0xbfa10001u, 0xf4080002u, 0xfa000000u, 0xf4000182u, 0xfa000010u,
    0x7e020280u, 0x7e080281u, 0xbefc0380u, 0x7e040301u, 0x7e060301u,
    0xbf8cc07fu, 0xdc788000u, 0x00000101u, 0x7e060202u, 0x7e080203u,
    0x7e000206u, 0x8102c106u, 0xdc788010u, 0x00000101u, 0x7e080301u,
    0x7e060300u, 0x7e000202u, 0xdc788020u, 0x00000101u, 0xd8660000u,
    0x00000000u, 0xbf8c0000u, 0xbf810000u};
static const uint32_t iree_hal_amd_gpu_grid_sync_gfx103[] = {
    0xbfa10001u, 0xf4080002u, 0xfa000000u, 0xf4000102u, 0xfa000010u,
    0x7e020280u, 0x7e080281u, 0xbefc0380u, 0x7e040301u, 0x7e060301u,
    0xbf8cc07fu, 0xdc788000u, 0x00000101u, 0x7e060202u, 0x7e080203u,
    0x7e000204u, 0x8102c104u, 0xdc788010u, 0x00000101u, 0x7e080301u,
    0x7e060300u, 0x7e000202u, 0xdc788020u, 0x00000101u, 0xd8660000u,
    0x00000000u, 0xbf8c0000u, 0xbf810000u};

iree_status_t iree_hal_amd_gpu_grid_sync_program_query(
    iree_hal_amd_gpu_gfxip_version_t gfxip, iree_const_byte_span_t* out_code,
    iree_hal_amd_gpu_image_kernel_t* out_kernel) {
  static const iree_hal_amd_gpu_program_variant_t variants[] = {
      {.instructions = AMDF_GPU_INSTRUCTION_SET_GFX90A,
       .code = {(const uint8_t*)iree_hal_amd_gpu_grid_sync_gfx90a,
                sizeof(iree_hal_amd_gpu_grid_sync_gfx90a)},
       .kernel = {.compute_pgm_rsrc1 = 0xaf0040u,
                  .compute_pgm_rsrc2 = 0x8cu,
                  .compute_pgm_rsrc3 = 0x0u,
                  .code_properties = 0x9u,
                  .kernarg_preload = 0x0u}},
      {.instructions = AMDF_GPU_INSTRUCTION_SET_GFX9,
       .code = {(const uint8_t*)iree_hal_amd_gpu_grid_sync_gfx9,
                sizeof(iree_hal_amd_gpu_grid_sync_gfx9)},
       .kernel = {.compute_pgm_rsrc1 = 0xaf0041u,
                  .compute_pgm_rsrc2 = 0x8cu,
                  .compute_pgm_rsrc3 = 0x0u,
                  .code_properties = 0x9u,
                  .kernarg_preload = 0x0u}},
      {.instructions = AMDF_GPU_INSTRUCTION_SET_GFX94,
       .code = {(const uint8_t*)iree_hal_amd_gpu_grid_sync_gfx94,
                sizeof(iree_hal_amd_gpu_grid_sync_gfx94)},
       .kernel = {.compute_pgm_rsrc1 = 0xaf0040u,
                  .compute_pgm_rsrc2 = 0x84u,
                  .compute_pgm_rsrc3 = 0x1u,
                  .code_properties = 0x8u,
                  .kernarg_preload = 0x0u}},
      {.instructions = AMDF_GPU_INSTRUCTION_SET_GFX101,
       .code = {(const uint8_t*)iree_hal_amd_gpu_grid_sync_gfx101,
                sizeof(iree_hal_amd_gpu_grid_sync_gfx101)},
       .kernel = {.compute_pgm_rsrc1 = 0xe0af0000u,
                  .compute_pgm_rsrc2 = 0x8cu,
                  .compute_pgm_rsrc3 = 0x0u,
                  .code_properties = 0x409u,
                  .kernarg_preload = 0x0u}},
      {.instructions = AMDF_GPU_INSTRUCTION_SET_GFX103,
       .code = {(const uint8_t*)iree_hal_amd_gpu_grid_sync_gfx103,
                sizeof(iree_hal_amd_gpu_grid_sync_gfx103)},
       .kernel = {.compute_pgm_rsrc1 = 0xe0af0000u,
                  .compute_pgm_rsrc2 = 0x8cu,
                  .compute_pgm_rsrc3 = 0x0u,
                  .code_properties = 0x409u,
                  .kernarg_preload = 0x0u}},
  };
  const amdf_gpu_target_capabilities_t target =
      amdf_gpu_target_query_capabilities(gfxip.major, gfxip.minor,
                                         gfxip.stepping);
  return iree_hal_amd_gpu_program_select(target.instruction_sets,
                                         IREE_ARRAYSIZE(variants), variants,
                                         out_code, out_kernel);
}

void iree_hal_amd_gpu_dispatch_emplace_grid_sync(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    const iree_hal_amd_gpu_dispatch_geometry_t* dispatch,
    uint64_t grid_sync_address, void* arguments,
    iree_hal_amd_gpu_grid_sync_t* grid_sync) {
  uint64_t workgroup_count = 1;
  uint64_t workitem_count = 1;
  for (uint32_t i = 0; i < 3; ++i) {
    workgroup_count *= dispatch->workgroup_count[i];
    const uint64_t count = dispatch->workgroup_count[i];
    const uint64_t size = dispatch->workgroup_size[i];
    workitem_count *= count * size - (count && dispatch->partial_size[i]
                                          ? size - dispatch->partial_size[i]
                                          : 0);
  }
  *grid_sync = (iree_hal_amd_gpu_grid_sync_t){
      .grid_count = workgroup_count ? 1u : 0u,
      .total_workitem_count = workitem_count,
      .workgroup_count = (uint32_t)workgroup_count,
  };
  if (layout->requirements & IREE_HAL_AMD_GPU_KERNARG_REQUIREMENT_GRID_SYNC) {
    memcpy((uint8_t*)arguments + layout->grid_sync_offset, &grid_sync_address,
           sizeof(grid_sync_address));
  }
}

// Resource accounting uses the kernel descriptor's encoded allocation, not
// optional compiler register-use annotations. GFX9's unified VGPR/AGPR count
// and RDNA's distinct encoding/allocation granules are therefore preserved.
typedef struct iree_hal_amd_gpu_dispatch_residency_t {
  // Lanes selected by the kernel descriptor.
  uint32_t wavefront_size;
  // Physical compute units cooperating within one scheduling domain.
  uint32_t compute_units_per_domain;
  // Independently allocated register files per physical compute unit.
  uint32_t simds_per_compute_unit;
  // Vector registers per SIMD, expressed in this kernel's wave width.
  uint32_t vector_register_count;
  // Vector registers represented by one descriptor block.
  uint32_t vector_encoding_granule;
  // Vector registers reserved per wave after hardware allocation rounding.
  uint32_t vector_allocation_granule;
  // Half-wave shared registers, expressed in the kernel's wave-width units.
  uint32_t shared_vector_register_count;
  // Bytes reserved for each workgroup-local memory allocation unit.
  uint32_t local_memory_granule;
  // Standard barrier slots available to multi-wave workgroups in a domain.
  uint32_t barrier_count;
  // Scalar registers per SIMD, or zero when scalar allocation is not limiting.
  uint32_t scalar_register_count;
  // Scalar registers represented by one descriptor block.
  uint32_t scalar_encoding_granule;
  // Scalar registers allocated before native trap reservation.
  uint32_t scalar_allocation_granule;
  // Dynamic allocation removes the static vector-register residency bound.
  bool dynamic_vector_registers;
} iree_hal_amd_gpu_dispatch_residency_t;

static iree_status_t iree_hal_amd_gpu_dispatch_residency(
    const amdf_gpu_endpoint_info_t* target,
    const iree_hal_amd_gpu_image_kernel_t* kernel,
    iree_hal_amd_gpu_dispatch_residency_t* out_residency) {
  const amdf_gpu_shader_capabilities_t capabilities =
      amdf_gpu_target_query_capabilities(
          target->gfx_ip.major, target->gfx_ip.minor, target->gfx_ip.stepping)
          .shader;
  if (!capabilities.wave_size_mask) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "native dispatch resource encoding is unavailable");
  }
  const bool wave32 = (kernel->code_properties & (1u << 10)) != 0;
  const uint32_t wave_size = wave32 ? 32 : 64;
  if (!(capabilities.wave_size_mask & wave_size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target does not support the kernel wave size");
  }
  const bool wgp =
      (kernel->compute_pgm_rsrc1 & capabilities.workgroup_processor_mask) != 0;
  if (wgp && !capabilities.workgroup_processor_mode) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target requires CU-mode execution");
  }
  const uint32_t shared_registers =
      (kernel->compute_pgm_rsrc3 & capabilities.shared_vector_register_mask) *
      8;
  if (wave32 && shared_registers) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shared vector registers require wave64 execution");
  }
  if (kernel->compute_pgm_rsrc3 & capabilities.unmodeled_resource3_mask) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel residency requires native scheduling resource facts");
  }
  const uint32_t mode = wave32 ? 0 : 1;
  const iree_hal_amd_gpu_dispatch_residency_t residency = {
      .wavefront_size = wave_size,
      .compute_units_per_domain = wgp ? 2 : 1,
      .simds_per_compute_unit = capabilities.simds_per_compute_unit,
      .vector_register_count = capabilities.vector_register_count[mode],
      .vector_encoding_granule = capabilities.vector_encoding_granule[mode],
      .vector_allocation_granule = capabilities.vector_allocation_granule[mode],
      .shared_vector_register_count = shared_registers / 2,
      .local_memory_granule = capabilities.local_memory_granule,
      .barrier_count = capabilities.barrier_count * (wgp ? 2 : 1),
      .scalar_register_count = capabilities.scalar_register_count,
      .scalar_encoding_granule = capabilities.scalar_encoding_granule,
      .scalar_allocation_granule = capabilities.scalar_allocation_granule,
      .dynamic_vector_registers =
          (kernel->compute_pgm_rsrc2 &
           capabilities.dynamic_vector_resource2_mask) ||
          (kernel->compute_pgm_rsrc3 &
           capabilities.dynamic_vector_resource3_mask),
  };
  *out_residency = residency;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_dispatch_query_concurrency(
    const amdf_gpu_endpoint_info_t* target, uint32_t compute_unit_count,
    const iree_hal_amd_gpu_image_kernel_t* kernel,
    const iree_hal_amd_gpu_metadata_kernel_t* metadata,
    uint32_t scalar_register_reservation,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  uint64_t invocation_count = 1;
  for (int i = 0; i < 3; ++i) {
    if (params.workgroup_size[i] > 1024) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "workgroup dimension %d exceeds native capacity",
                              i);
    }
    if (metadata->has_required_workgroup_size &&
        params.workgroup_size[i] != metadata->required_workgroup_size[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "workgroup dimension %d violates the function "
                              "requirement",
                              i);
    }
    invocation_count *= params.workgroup_size[i];
  }
  if (invocation_count > 1024 ||
      (metadata->max_flat_workgroup_size &&
       invocation_count > metadata->max_flat_workgroup_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workgroup exceeds function or native capacity");
  }
  if (metadata->has_workgroup_cluster_size) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "cluster residency requires native cluster "
                            "placement admission");
  }
  iree_hal_amd_gpu_dispatch_residency_t residency;
  IREE_RETURN_IF_ERROR(
      iree_hal_amd_gpu_dispatch_residency(target, kernel, &residency));
  const uint32_t domain_units = residency.compute_units_per_domain;
  const uint32_t simds = residency.simds_per_compute_unit * domain_units;
  uint32_t waves_per_simd =
      target->compute.maximum_wave_count_per_compute_unit /
      residency.simds_per_compute_unit;
  if (!residency.dynamic_vector_registers) {
    const uint32_t registers = ((kernel->compute_pgm_rsrc1 & 0x3Fu) + 1) *
                               residency.vector_encoding_granule;
    const uint32_t granule = residency.vector_allocation_granule;
    const uint32_t allocated = ((registers + granule - 1) / granule) * granule +
                               residency.shared_vector_register_count;
    waves_per_simd =
        iree_min(waves_per_simd, residency.vector_register_count / allocated);
  }
  uint32_t maximum_waves_per_simd = waves_per_simd;
  if (residency.scalar_register_count) {
    // GFX9 reserves scalar registers in 16-register units. Trap allocation
    // belongs to the native context, not the compiler descriptor's trap bit.
    const uint32_t registers = (((kernel->compute_pgm_rsrc1 >> 6) & 0xFu) + 1) *
                               residency.scalar_encoding_granule;
    const uint32_t granule = residency.scalar_allocation_granule;
    const uint32_t allocated = ((registers + granule - 1) / granule) * granule;
    const bool known = scalar_register_reservation != UINT32_MAX;
    waves_per_simd =
        iree_min(waves_per_simd,
                 residency.scalar_register_count /
                     (allocated + (known ? scalar_register_reservation : 16u)));
    maximum_waves_per_simd =
        known ? waves_per_simd
              : iree_min(maximum_waves_per_simd,
                         residency.scalar_register_count / allocated);
  }
  const uint32_t waves =
      ((uint32_t)invocation_count + residency.wavefront_size - 1) /
      residency.wavefront_size;
  uint32_t group_limit = waves > 1 ? residency.barrier_count : UINT32_MAX;
  const uint64_t local_bytes = (uint64_t)metadata->group_segment_fixed_size +
                               params.dynamic_workgroup_local_memory;
  if (local_bytes > target->compute.local_data_share_byte_length) {
    group_limit = 0;
  } else if (local_bytes) {
    const uint64_t granule = residency.local_memory_granule;
    const uint64_t allocated =
        ((local_bytes + granule - 1) / granule) * granule;
    group_limit = iree_min(
        group_limit, (uint32_t)(target->compute.local_data_share_byte_length *
                                domain_units / allocated));
  }
  const uint32_t groups = iree_min(waves_per_simd * simds / waves, group_limit);
  if (groups != iree_min(maximum_waves_per_simd * simds / waves, group_limit)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "native trap SGPR reservation is unavailable and changes residency");
  }
  *out_concurrency = (iree_hal_queue_dispatch_concurrency_t){
      .scheduling_domain_count = compute_unit_count / domain_units,
      .maximum_concurrent_workgroup_count_per_domain = groups,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_dispatch_prepare(
    const iree_hal_amd_gpu_function_t* function,
    uint32_t maximum_group_segment_byte_length, bool workitem_grid,
    iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_host_size_t binding_count, iree_hal_dispatch_flags_t flags,
    iree_hal_amd_gpu_dispatch_geometry_t* out_dispatch) {
  const bool indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  const iree_hal_amd_gpu_metadata_kernel_t* metadata = function->metadata;
  const bool indirect_arguments =
      iree_hal_dispatch_uses_indirect_arguments(flags);
  const bool native_arguments =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  if (native_arguments && indirect_arguments) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native direct and indirect arguments conflict");
  }
  if (indirect_arguments) {
    if (binding_count != 1 || constants.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "indirect arguments require one binding and "
                              "no constants");
    }
  } else if (native_arguments) {
    if (binding_count ||
        constants.data_length < function->kernarg->native_byte_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "native argument extent does not match the "
                              "function ABI");
    }
  } else if (binding_count != function->kernarg->binding_count ||
             constants.data_length != function->kernarg->constant_byte_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dense arguments do not match function reflection");
  }
  iree_hal_amd_gpu_dispatch_geometry_t dispatch = {0};
  const bool default_size =
      !(config.workgroup_size[0] | config.workgroup_size[1] |
        config.workgroup_size[2]);
  const bool exact =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_EXACT_WORKITEM_COUNT);
  if (exact && indirect_parameters) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "exact work-item counts conflict with indirect parameters");
  }
  uint64_t flat_size = 1;
  bool partial = false;
  for (int i = 0; i < 3; ++i) {
    const uint32_t size = default_size
                              ? (metadata->has_required_workgroup_size
                                     ? metadata->required_workgroup_size[i]
                                     : 1)
                              : config.workgroup_size[i];
    if (!size || size > 1024 ||
        (metadata->has_required_workgroup_size &&
         size != metadata->required_workgroup_size[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "workgroup dimension %d violates function or "
                              "native geometry",
                              i);
    }
    flat_size *= size;
    dispatch.workgroup_size[i] = size;
    if (exact) {
      if (!config.workitem_count[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "exact work-item dimensions must be nonzero");
      }
      const uint32_t remainder = config.workitem_count[i] % size;
      dispatch.workgroup_count[i] =
          config.workitem_count[i] / size + (remainder != 0);
      dispatch.partial_size[i] = remainder ? remainder : size;
      partial |= remainder != 0;
    } else {
      if (config.workitem_count[i]) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "work-item dimensions require the exact flag");
      }
      dispatch.workgroup_count[i] =
          indirect_parameters ? 0 : config.workgroup_count[i];
      if (workitem_grid && dispatch.workgroup_count[i] > UINT32_MAX / size) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "dispatch packet grid dimension %d exceeds "
                                "32-bit workitem counts",
                                i);
      }
    }
    if (metadata->max_workgroup_count[i] &&
        dispatch.workgroup_count[i] > metadata->max_workgroup_count[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "workgroup count exceeds function metadata");
    }
  }
  if (flat_size > 1024 || (metadata->max_flat_workgroup_size &&
                           flat_size > metadata->max_flat_workgroup_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "workgroup exceeds function or native invocation "
                            "capacity");
  }
  if (partial && metadata->uniform_workgroup_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "function requires complete workgroups");
  }
  if (!partial) {
    memset(dispatch.partial_size, 0, sizeof(dispatch.partial_size));
  }
  const uint64_t group_bytes = (uint64_t)metadata->group_segment_fixed_size +
                               config.dynamic_workgroup_local_memory;
  if (group_bytes > maximum_group_segment_byte_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch LDS exceeds the native queue capacity");
  }
  dispatch.group_segment_byte_length = (uint32_t)group_bytes;
  *out_dispatch = dispatch;
  return iree_ok_status();
}

iree_status_t iree_hal_amd_gpu_dispatch_validate_direct_bindings(
    const iree_hal_amd_gpu_kernarg_layout_t* layout,
    iree_hal_buffer_ref_t parameters, iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    if (!parameters.buffer || (parameters.offset & 3)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "indirect parameters require a DWORD-aligned buffer");
    }
    if (parameters.length != IREE_HAL_WHOLE_BUFFER && parameters.length < 12) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "indirect parameters require three DWORD counts");
    }
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(parameters.buffer,
                                                        parameters.offset, 12));
  }
  const bool indirect_arguments =
      iree_hal_dispatch_uses_indirect_arguments(flags);
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    const iree_hal_buffer_ref_t reference = bindings.values[i];
    if (!reference.buffer) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "direct dispatch binding %" PRIhsz " is null", i);
    }
    // The remaining range is valid whenever its starting offset is valid.
    // Native arguments consume only the address, so no length is materialized.
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
        reference.buffer, reference.offset,
        reference.length == IREE_HAL_WHOLE_BUFFER ? 0 : reference.length));
  }
  if (indirect_arguments) {
    const iree_hal_buffer_ref_t reference = bindings.values[0];
    const iree_device_size_t length =
        reference.length == IREE_HAL_WHOLE_BUFFER
            ? iree_hal_buffer_byte_length(reference.buffer) - reference.offset
            : reference.length;
    if (length < layout->byte_length) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "indirect arguments do not cover the native ABI");
    }
  }
  return iree_ok_status();
}

void iree_hal_amd_gpu_dispatch_emplace_arguments(
    const iree_hal_amd_gpu_function_t* function,
    const iree_hal_amd_gpu_dispatch_geometry_t* dispatch,
    iree_const_byte_span_t constants, const uint64_t* binding_addresses,
    iree_hal_dispatch_flags_t flags, void* target) {
  if (!iree_hal_dispatch_uses_indirect_arguments(flags)) {
    iree_hal_amd_gpu_kernarg_inputs_t inputs = {
        .dynamic_lds_size = dispatch->group_segment_byte_length -
                            function->metadata->group_segment_fixed_size,
        .grid_dimensions = 1,
    };
    for (int i = 0; i < 3; ++i) {
      const uint32_t partial = dispatch->partial_size[i];
      const bool remainder = partial && partial != dispatch->workgroup_size[i];
      inputs.block_count[i] = dispatch->workgroup_count[i] - remainder;
      inputs.group_size[i] = (uint16_t)dispatch->workgroup_size[i];
      inputs.remainder[i] = remainder ? (uint16_t)partial : 0;
      if (dispatch->workgroup_count[i] != 1 ||
          dispatch->workgroup_size[i] != 1) {
        inputs.grid_dimensions = (uint16_t)(i + 1);
      }
    }
    if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
      // HAL supplies a three-axis tuple without a separate rank. Device copies
      // populate its counts; unused dimensions have size and count one.
      inputs.grid_dimensions = 3;
    }
    if (iree_any_bit_set(flags,
                         IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
      iree_hal_amd_gpu_kernarg_layout_emplace_native(
          function->kernarg, constants, &inputs, target);
    } else {
      iree_hal_amd_gpu_kernarg_layout_emplace(function->kernarg, constants.data,
                                              binding_addresses, &inputs,
                                              target);
    }
  }
}

iree_status_t iree_hal_amd_gpu_dispatch_resolve_address(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    iree_hal_buffer_ref_t reference, iree_hal_buffer_usage_t usage,
    iree_hal_memory_access_t access, uint64_t* out_address) {
  const iree_hal_memory_contract_t* contract =
      reference.buffer->memory.contract;
  const uint64_t scope_id = (uint64_t)family->memory.queue_scope_id +
                            (site - IREE_HAL_MEMORY_SITE_QUEUE);
  if (!contract || !family->memory.domain ||
      family->memory.domain != contract->domain ||
      scope_id >= contract->scope_count) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "buffer is outside the dispatch family domain");
  }
  const iree_hal_memory_scope_access_t* scope = &contract->scopes[scope_id];
  const iree_hal_buffer_interface_t interface =
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  const uint16_t index = scope->bindings[interface];
  if (!iree_any_bit_set(scope->interfaces, UINT32_C(1) << interface) ||
      index == IREE_HAL_BUFFER_NATIVE_BINDING_INDEX_NONE) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "GPU address access is not prepared for this "
                            "dispatch family");
  }
  if (access) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
        iree_hal_buffer_allowed_access(reference.buffer), access));
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage_any(
      scope->usage & iree_hal_buffer_allowed_usage(reference.buffer), usage));
  const uint64_t base =
      iree_hal_buffer_native_binding(reference.buffer,
                                     (iree_hal_buffer_native_binding_slot_t){
                                         .index = index, .type = interface})
          .device_address;
  if (!base) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dispatch buffer has no committed GPU address");
  }
  if (!iree_checked_add_u64(base, reference.offset, out_address)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch buffer address overflows");
  }
  return iree_ok_status();
}
