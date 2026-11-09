// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/dispatch_concurrency.h"

#include "amdf/gpu_hsa.h"

// Architectural resources available to workgroups in one scheduling domain.
typedef struct iree_hal_amdgpu_dispatch_concurrency_profile_t {
  // Wavefront size selected by the loaded kernel descriptor.
  uint32_t wavefront_size;
  // Compute units participating in each scheduling domain.
  uint32_t compute_units_per_domain;
  // Workgroup-local memory available in each scheduling domain, in bytes.
  uint32_t local_memory_size_per_domain;
  // Workgroup-local memory allocation granule, in bytes.
  uint32_t local_memory_allocation_granule;
  // Vector registers available to waves on each SIMD.
  uint32_t vector_register_count_per_simd;
  // Vector register count represented by one descriptor block.
  uint32_t vector_register_encoding_granule;
  // Vector register allocation granule for one wave.
  uint32_t vector_register_allocation_granule;
  // Standard workgroup barriers available in each scheduling domain.
  uint32_t standard_barrier_count_per_domain;
  // Scalar registers available to waves on each SIMD, or zero if unbounded.
  uint32_t scalar_register_count_per_simd;
  // Scalar register count represented by one descriptor block.
  uint32_t scalar_register_encoding_granule;
  // Scalar register allocation granule for one wave.
  uint32_t scalar_register_allocation_granule;
  // Additional scalar registers reserved for each wave by a trap handler.
  uint32_t trap_scalar_register_count_per_wave;
  // True when dynamic vector register allocation removes the static limit.
  uint32_t uses_dynamic_vector_register_allocation : 1;
} iree_hal_amdgpu_dispatch_concurrency_profile_t;

static uint32_t iree_hal_amdgpu_dispatch_concurrency_align_up(
    uint32_t value, uint32_t alignment) {
  return ((value + alignment - 1u) / alignment) * alignment;
}

static iree_status_t iree_hal_amdgpu_dispatch_concurrency_select_profile(
    const iree_hal_amdgpu_dispatch_concurrency_inputs_t* inputs,
    iree_hal_amdgpu_dispatch_concurrency_profile_t* out_profile) {
  const iree_hal_amdgpu_dispatch_concurrency_capabilities_t* capabilities =
      inputs->capabilities;
  const iree_hal_amdgpu_kernel_descriptor_t* descriptor =
      inputs->kernel_descriptor;
  if (IREE_UNLIKELY(!descriptor)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "loaded AMDGPU function has no host-readable kernel descriptor");
  }
  const iree_hal_amdgpu_gfxip_version_t version = capabilities->gfxip_version;
  const amdf_gpu_hsa_target_capabilities_t target =
      amdf_gpu_hsa_query_target(version.major, version.minor, version.stepping);
  if (IREE_UNLIKELY(capabilities->target_kind !=
                        IREE_HAL_AMDGPU_TARGET_KIND_EXACT ||
                    !target.shader_resource_model)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "dispatch concurrency is not defined for AMDGPU target gfx%u.%u.%u",
        capabilities->gfxip_version.major, capabilities->gfxip_version.minor,
        capabilities->gfxip_version.stepping);
  }

  const bool wave32 = iree_any_bit_set(
      descriptor->kernel_code_properties,
      IREE_HAL_AMDGPU_KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  const uint32_t wave_size = wave32 ? 32 : 64;
  const bool wgp = (descriptor->compute_pgm_rsrc1 &
                    target.shader.workgroup_processor_mask) != 0;
  if (!(target.shader.wave_size_mask & wave_size) ||
      (wgp && !target.shader.workgroup_processor_mode)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel wave size or workgroup mode has no native HSA resource model");
  }
  if (descriptor->compute_pgm_rsrc3 &
      (target.shader.unmodeled_resource3_mask |
       target.shader.shared_vector_register_mask)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "kernel scheduling features require additional "
                            "native HSA resource facts");
  }
  const uint32_t mode = wave32 ? 0 : 1;
  const iree_hal_amdgpu_dispatch_concurrency_profile_t profile = {
      .wavefront_size = wave_size,
      .compute_units_per_domain = target.compute_units_per_domain[wgp],
      .local_memory_size_per_domain = target.local_memory_byte_length[wgp],
      .local_memory_allocation_granule = target.shader.local_memory_granule,
      .vector_register_count_per_simd =
          target.shader.vector_register_count[mode],
      .vector_register_encoding_granule =
          target.shader.vector_encoding_granule[mode],
      .vector_register_allocation_granule =
          target.shader.vector_allocation_granule[mode],
      .standard_barrier_count_per_domain = target.barrier_count[wgp],
      .scalar_register_count_per_simd = target.shader.scalar_register_count,
      .scalar_register_encoding_granule = target.shader.scalar_encoding_granule,
      .scalar_register_allocation_granule =
          target.shader.scalar_allocation_granule,
      .trap_scalar_register_count_per_wave =
          descriptor->compute_pgm_rsrc2 & target.trap_handler_mask ? 16u : 0u,
      .uses_dynamic_vector_register_allocation =
          (descriptor->compute_pgm_rsrc2 &
           target.shader.dynamic_vector_resource2_mask) ||
          (descriptor->compute_pgm_rsrc3 &
           target.shader.dynamic_vector_resource3_mask),
  };

  *out_profile = profile;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_dispatch_concurrency_calculate_domain_count(
    const iree_hal_amdgpu_dispatch_concurrency_inputs_t* inputs,
    const iree_hal_amdgpu_dispatch_concurrency_profile_t* profile,
    uint32_t* out_domain_count) {
  const iree_hal_amdgpu_queue_execution_resource_topology_t* topology =
      inputs->execution_resource_topology;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_queue_execution_resource_topology_verify(topology));

  const uint32_t expected_execution_units_per_resource =
      amdf_gpu_hsa_query_target(inputs->capabilities->gfxip_version.major,
                                inputs->capabilities->gfxip_version.minor,
                                inputs->capabilities->gfxip_version.stepping)
          .compute_units_per_resource;
  if (IREE_UNLIKELY(topology->execution_units_per_resource !=
                    expected_execution_units_per_resource)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU queue resources contain %u execution units on a target "
        "requiring %u",
        topology->execution_units_per_resource,
        expected_execution_units_per_resource);
  }

  const iree_host_size_t resource_count =
      iree_hal_amdgpu_queue_execution_resource_count(topology);
  const iree_hal_queue_execution_resource_list_t selected_resources =
      inputs->execution_resources;
  if (IREE_UNLIKELY(selected_resources.count && !selected_resources.ordinals)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU queue has execution resources without ordinal storage");
  }
  for (iree_host_size_t i = 0; i < selected_resources.count; ++i) {
    const iree_hal_queue_execution_resource_ordinal_t ordinal =
        selected_resources.ordinals[i];
    if (IREE_UNLIKELY(ordinal >= resource_count)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU queue execution resource %u exceeds resource count %" PRIhsz,
          ordinal, resource_count);
    }
    if (IREE_UNLIKELY(i != 0 &&
                      ordinal <= selected_resources.ordinals[i - 1])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU queue execution resources are not sorted and unique");
    }
  }

  uint32_t selected_compute_unit_count = 0;
  const bool is_cooperative = iree_any_bit_set(
      inputs->queue_features, IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH);
  if (is_cooperative) {
    const iree_hal_amdgpu_dispatch_concurrency_capabilities_t* capabilities =
        inputs->capabilities;
    if (IREE_UNLIKELY(selected_resources.count != 0 ||
                      !capabilities->has_cooperative_compute_unit_count ||
                      capabilities->cooperative_compute_unit_count == 0 ||
                      capabilities->cooperative_compute_unit_count >
                          topology->execution_unit_count)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "AMDGPU cooperative queue does not have an exact compute-unit "
          "count");
    }
    selected_compute_unit_count = capabilities->cooperative_compute_unit_count;
  } else {
    const iree_host_size_t selected_resource_count =
        selected_resources.count ? selected_resources.count : resource_count;
    if (IREE_UNLIKELY(selected_resource_count >
                      UINT32_MAX / topology->execution_units_per_resource)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU queue execution-resource count is too "
                              "large to model");
    }
    selected_compute_unit_count = (uint32_t)selected_resource_count *
                                  topology->execution_units_per_resource;
  }

  if (IREE_UNLIKELY(selected_compute_unit_count == 0 ||
                    selected_compute_unit_count %
                            profile->compute_units_per_domain !=
                        0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue compute-unit count %u cannot be partitioned into "
        "%u-unit scheduling domains",
        selected_compute_unit_count, profile->compute_units_per_domain);
  }
  *out_domain_count =
      selected_compute_unit_count / profile->compute_units_per_domain;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_calculate_dispatch_concurrency(
    const iree_hal_amdgpu_dispatch_concurrency_inputs_t* inputs,
    iree_hal_queue_dispatch_concurrency_params_t params,
    iree_hal_queue_dispatch_concurrency_t* out_concurrency) {
  if (IREE_UNLIKELY(!inputs->capabilities ||
                    !inputs->execution_resource_topology)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue has no dispatch concurrency capabilities");
  }
  const iree_hal_amdgpu_dispatch_concurrency_capabilities_t* capabilities =
      inputs->capabilities;
  if (IREE_UNLIKELY(capabilities->simd_count_per_compute_unit == 0 ||
                    capabilities->maximum_waves_per_compute_unit == 0 ||
                    capabilities->maximum_waves_per_compute_unit %
                            capabilities->simd_count_per_compute_unit !=
                        0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU physical device does not have consistent SIMD occupancy "
        "facts");
  }
  if (IREE_UNLIKELY(params.workgroup_size[0] == 0 ||
                    params.workgroup_size[1] == 0 ||
                    params.workgroup_size[2] == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU workgroup dimensions must be non-zero");
  }

  iree_hal_amdgpu_dispatch_concurrency_profile_t profile;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_dispatch_concurrency_select_profile(inputs, &profile));
  if (IREE_UNLIKELY(inputs->workgroup_cluster_size[0] != 0 ||
                    inputs->workgroup_cluster_size[1] != 0 ||
                    inputs->workgroup_cluster_size[2] != 0)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU clustered dispatch concurrency is not implemented");
  }

  uint32_t scheduling_domain_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_dispatch_concurrency_calculate_domain_count(
          inputs, &profile, &scheduling_domain_count));

  uint64_t workgroup_invocation_count = 0;
  uint64_t workgroup_xy = 0;
  if (IREE_UNLIKELY(
          !iree_checked_mul_u64(params.workgroup_size[0],
                                params.workgroup_size[1], &workgroup_xy) ||
          !iree_checked_mul_u64(workgroup_xy, params.workgroup_size[2],
                                &workgroup_invocation_count))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU workgroup invocation count overflows");
  }
  uint64_t rounded_workgroup_invocation_count = 0;
  if (IREE_UNLIKELY(!iree_checked_add_u64(
          workgroup_invocation_count, profile.wavefront_size - 1u,
          &rounded_workgroup_invocation_count))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU rounded workgroup invocation count overflows");
  }
  const uint64_t wave_count =
      rounded_workgroup_invocation_count / profile.wavefront_size;
  if (IREE_UNLIKELY(capabilities->simd_count_per_compute_unit >
                    UINT32_MAX / profile.compute_units_per_domain)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU scheduling-domain SIMD count overflows");
  }
  const uint32_t simd_count_per_domain =
      capabilities->simd_count_per_compute_unit *
      profile.compute_units_per_domain;
  const uint32_t maximum_waves_per_simd =
      capabilities->maximum_waves_per_compute_unit /
      capabilities->simd_count_per_compute_unit;
  uint64_t maximum_workgroups_per_domain =
      ((uint64_t)maximum_waves_per_simd * simd_count_per_domain) / wave_count;

  // Single-wave workgroups require no standard barrier slot.
  if (wave_count > 1) {
    maximum_workgroups_per_domain =
        iree_min(maximum_workgroups_per_domain,
                 (uint64_t)profile.standard_barrier_count_per_domain);
  }

  const iree_hal_amdgpu_kernel_descriptor_t* descriptor =
      inputs->kernel_descriptor;
  if (params.dynamic_workgroup_local_memory >
      inputs->maximum_dynamic_workgroup_local_memory_size) {
    maximum_workgroups_per_domain = 0;
  } else {
    const uint64_t required_local_memory =
        (uint64_t)descriptor->group_segment_fixed_size +
        params.dynamic_workgroup_local_memory;
    if (required_local_memory != 0) {
      const uint64_t rounded_local_memory =
          ((required_local_memory + profile.local_memory_allocation_granule -
            1u) /
           profile.local_memory_allocation_granule) *
          profile.local_memory_allocation_granule;
      maximum_workgroups_per_domain =
          iree_min(maximum_workgroups_per_domain,
                   profile.local_memory_size_per_domain / rounded_local_memory);
    }
  }

  if (!profile.uses_dynamic_vector_register_allocation) {
    const uint32_t encoded_vector_register_blocks =
        (descriptor->compute_pgm_rsrc1 &
         IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_MASK) >>
        IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT;
    const uint32_t vector_register_count =
        (encoded_vector_register_blocks + 1u) *
        profile.vector_register_encoding_granule;
    const uint32_t allocated_vector_register_count =
        iree_hal_amdgpu_dispatch_concurrency_align_up(
            vector_register_count, profile.vector_register_allocation_granule);
    const uint32_t vector_limited_waves_per_simd = iree_min(
        maximum_waves_per_simd, profile.vector_register_count_per_simd /
                                    allocated_vector_register_count);
    maximum_workgroups_per_domain = iree_min(
        maximum_workgroups_per_domain,
        ((uint64_t)vector_limited_waves_per_simd * simd_count_per_domain) /
            wave_count);
  }

  if (profile.scalar_register_count_per_simd != 0) {
    const uint32_t encoded_scalar_register_blocks =
        (descriptor->compute_pgm_rsrc1 &
         IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_MASK) >>
        IREE_HAL_AMDGPU_COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT;
    const uint32_t scalar_register_count =
        (encoded_scalar_register_blocks + 1u) *
        profile.scalar_register_encoding_granule;
    const uint32_t allocated_scalar_register_count =
        iree_hal_amdgpu_dispatch_concurrency_align_up(
            scalar_register_count, profile.scalar_register_allocation_granule) +
        profile.trap_scalar_register_count_per_wave;
    const uint32_t scalar_limited_waves_per_simd = iree_min(
        maximum_waves_per_simd, profile.scalar_register_count_per_simd /
                                    allocated_scalar_register_count);
    maximum_workgroups_per_domain = iree_min(
        maximum_workgroups_per_domain,
        ((uint64_t)scalar_limited_waves_per_simd * simd_count_per_domain) /
            wave_count);
  }

  if (IREE_UNLIKELY(maximum_workgroups_per_domain > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU workgroup concurrency exceeds the HAL representation");
  }
  const iree_hal_queue_dispatch_concurrency_t concurrency = {
      .scheduling_domain_count = scheduling_domain_count,
      .maximum_concurrent_workgroup_count_per_domain =
          (uint32_t)maximum_workgroups_per_domain,
  };
  *out_concurrency = concurrency;
  return iree_ok_status();
}
