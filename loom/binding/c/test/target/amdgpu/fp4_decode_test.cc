// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sstream>
#include <string>

#include "iree/testing/gtest.h"
#include "loomc/emit.h"
#include "target_test_util.h"

namespace loomc::testing::amdgpu {
namespace {

class Fp4DecodeTest : public ::testing::TestWithParam<const char*> {};

// Packed scale bytes must survive lowering, allocation, and executable emission
// for every payload width and pair-aligned power-of-two group. These profiles
// cover portable lookup, native two-element conversion, and native pk8
// conversion (which must retain the portable route for groups below eight).
TEST_P(Fp4DecodeTest, HonorsPairGroupBoundaryForEveryPayloadWidth) {
  auto environment = CreateAmdgpuTargetEnvironment();
  auto context = CreateAmdgpuContext(environment.get());
  auto workspace = CreateWorkspace();
  auto compiler = CreateCompiler(context.get());
  auto profile = CreateTargetProfile(environment.get(), GetParam());
  const loomc_string_view_t root = loomc_make_cstring_view("decode");
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .identifier = loomc_make_cstring_view("decode.hsaco"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_compile_artifact_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(options),
      .roots = &root,
      .root_count = 1,
      .target_profile = profile.get(),
      .emit_options = &emit_options,
  };
  for (int payload_words = 1; payload_words <= 8; ++payload_words) {
    const int elements = payload_words * 8;
    for (int group_elements = 1; group_elements <= elements;
         group_elements *= 2) {
      if (elements % group_elements != 0) {
        continue;
      }
      SCOPED_TRACE(::testing::Message()
                   << GetParam() << " elements=" << elements
                   << " group=" << group_elements);
      const int scale_words = (elements / group_elements + 3) / 4;
      std::ostringstream source_text;
      source_text
          << "kernel.def @decode() {\n"
             "  %one = index.constant 1 : index\n"
             "  kernel.launch.config workgroups(%one, %one, %one) "
             "workgroup_size(%one, %one, %one) : index\n"
             "} launch(%payload_buffer: buffer, %scale_buffer: buffer, "
             "%output_buffer: buffer) {\n"
             "  %base = index.constant 0 : offset\n"
             "  %origin = index.constant 0 : index\n"
             "  %payload_view = buffer.view %payload_buffer[%base] : buffer "
             "-> view<"
          << payload_words << "xi32>\n"
          << "  %scale_view = buffer.view %scale_buffer[%base] : buffer -> "
             "view<"
          << scale_words << "xi32>\n"
          << "  %output_view = buffer.view %output_buffer[%base] : buffer -> "
             "view<"
          << elements << "xbf16>\n"
          << "  %schema = encoding.define #encoding.f4e2m1<affine=scale_only, "
             "payload_elements="
          << elements
          << ", payload_packing=little_endian_nibbles, payload_registers="
          << payload_words
          << ", scale_format=e8m0, scale_group_elements=" << group_elements
          << ", scale_operands=1, scale_topology=block_1d, "
             "zero_scale_fallback=true> : encoding<schema>\n"
             "  %payload = vector.load %payload_view[%origin] : view<"
          << payload_words << "xi32> -> vector<" << payload_words << "xi32>\n"
          << "  %scale = vector.load %scale_view[%origin] : view<"
          << scale_words << "xi32> -> vector<" << scale_words << "xi32>\n"
          << "  %decoded = vector.decode %payload using %schema {scale = "
             "%scale : vector<"
          << scale_words << "xi32>} : vector<" << payload_words
          << "xi32>, encoding<schema> -> vector<" << elements << "xbf16>\n"
          << "  vector.store %decoded, %output_view[%origin] : vector<"
          << elements << "xbf16>, view<" << elements << "xbf16>\n"
          << "  kernel.return\n}\n";
      auto source = CreateTextSource("decode.loom", source_text.str().c_str());
      auto module =
          DeserializeModule(context.get(), workspace.get(), source.get());
      ASSERT_NE(module, nullptr);
      loomc_result_t* result = nullptr;
      LOOMC_ASSERT_OK(loomc_compile_artifact(
          compiler.get(), workspace.get(), /*pass_program=*/nullptr,
          module.get(), &options, loomc_allocator_system(), &result));
      ResultPtr owned_result(result);
      if (group_elements == 1) {
        // Neither the portable pair recipe nor a native packet may silently
        // apply one scale to independently scaled nibbles.
        EXPECT_FALSE(loomc_result_succeeded(result));
        ASSERT_EQ(loomc_result_diagnostic_count(result), 1u);
        const auto* diagnostic = loomc_result_diagnostic_at(result, 0);
        EXPECT_EQ(ToString(diagnostic->code), "TARGET/001");
        EXPECT_NE(ToString(diagnostic->message).find("vector.decode"),
                  std::string::npos);
        continue;
      }
      ExpectSucceededResult(result);
      ASSERT_NE(FindArtifact(result, LOOMC_ARTIFACT_KIND_EXECUTABLE,
                             LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
                nullptr);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DescriptorFamilies, Fp4DecodeTest,
                         ::testing::Values("gfx942", "gfx1100", "gfx1151",
                                           "gfx1200", "gfx950", "gfx1250"),
                         [](const ::testing::TestParamInfo<const char*>& info) {
                           return info.param;
                         });

}  // namespace
}  // namespace loomc::testing::amdgpu
