// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/function_projection_reader.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class FunctionProjectionReaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("projected"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Backing storage shared by the output module and invocation arenas.
  iree_arena_block_pool_t pool_;
  // Retained caller storage containing the opaque reader object.
  iree_arena_allocator_t arena_;
  // Empty finalized dialect registry for allocation-only API coverage.
  loom_context_t context_;
  // Output module whose ownership remains independent of the reader.
  loom_module_t* module_ = nullptr;
  // Empty source metadata; these tests do not decode symbol payloads.
  loom_bytecode_module_metadata_t metadata_ = {};
};

TEST_F(FunctionProjectionReaderTest, UsesCallerArenaWithoutOwningOutput) {
  void* caller_storage = nullptr;
  IREE_ASSERT_OK(
      iree_arena_allocate(&arena_, sizeof(uint32_t), &caller_storage));
  auto* sentinel = static_cast<uint32_t*>(caller_storage);
  *sentinel = 42;
  const iree_host_size_t used_before = arena_.used_allocation_size;
  const iree_host_size_t allocated_before = arena_.total_allocation_size;
  loom_bytecode_function_projection_reader_t* reader = nullptr;
  IREE_ASSERT_OK(loom_bytecode_function_projection_reader_allocate(
      iree_const_byte_span_empty(), IREE_SV("empty.loombc"), &metadata_,
      module_, nullptr, &arena_, &reader));
  ASSERT_NE(reader, nullptr);
  EXPECT_GT(arena_.used_allocation_size, used_before);
  EXPECT_EQ(arena_.total_allocation_size, allocated_before);
  const iree_host_size_t used_with_reader = arena_.used_allocation_size;

  loom_bytecode_function_projection_reader_deinitialize(reader);
  EXPECT_EQ(arena_.used_allocation_size, used_with_reader);
  EXPECT_EQ(*sentinel, 42u);
  iree_arena_reset(&arena_);
  EXPECT_EQ(arena_.used_allocation_size, 0u);

  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(module_, IREE_SV("after_reader"), &name));
  loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
  EXPECT_EQ(module_->symbols.count, 1u);
}

TEST_F(FunctionProjectionReaderTest, FailedBackingAllocationReturnsNoReader) {
  const iree_allocator_t failing_allocator = {
      nullptr,
      [](void*, iree_allocator_command_t, const void*, void**) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      },
  };
  iree_arena_block_pool_t failing_pool;
  iree_arena_block_pool_initialize(4096, failing_allocator, &failing_pool);
  iree_arena_allocator_t failing_arena;
  iree_arena_initialize(&failing_pool, &failing_arena);
  loom_bytecode_function_projection_reader_t* reader = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      loom_bytecode_function_projection_reader_allocate(
          iree_const_byte_span_empty(), IREE_SV("empty.loombc"), &metadata_,
          module_, nullptr, &failing_arena, &reader));
  EXPECT_EQ(reader, nullptr);
  EXPECT_EQ(failing_arena.used_allocation_size, 0u);
  loom_bytecode_function_projection_reader_deinitialize(reader);
  iree_arena_deinitialize(&failing_arena);
  iree_arena_block_pool_deinitialize(&failing_pool);
}

}  // namespace
}  // namespace loom
