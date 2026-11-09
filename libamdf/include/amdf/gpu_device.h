// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_GPU_DEVICE_H_
#define AMDF_GPU_DEVICE_H_

// Compile-time native capabilities for freestanding AMDGPU device programs.
// Host preparation uses gpu_target.h with the discovered or parsed identity.
#if defined(__GFX9__) || defined(__GFX10__)
#define AMDF_GPU_DEVICE_GLOBAL_WAVE_SYNC 1
#else
#define AMDF_GPU_DEVICE_GLOBAL_WAVE_SYNC 0
#endif

// Bit width accepted by the interrupt message's event identifier operand.
#if defined(__GFX10__)
#define AMDF_GPU_DEVICE_EVENT_ID_MASK 0x7FFFFFu
#elif defined(__GFX9__) || defined(__GFX11__) || defined(__GFX12__)
#define AMDF_GPU_DEVICE_EVENT_ID_MASK 0xFFFFFFu
#else
#define AMDF_GPU_DEVICE_EVENT_ID_MASK 0xFFu
#endif

// Whether HW_ID and XCC_ID expose the published XCC/SE/CU ordinal encoding.
#if defined(__gfx942__) || defined(__gfx9_4_generic__)
#define AMDF_GPU_DEVICE_XCC_COMPUTE_UNIT_ID 1
#else
#define AMDF_GPU_DEVICE_XCC_COMPUTE_UNIT_ID 0
#endif

#endif  // AMDF_GPU_DEVICE_H_
