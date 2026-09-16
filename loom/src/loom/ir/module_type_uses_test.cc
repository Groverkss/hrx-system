// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ModuleTypeUsesTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<ModuleTypeUsesTest*>(self);
    if (test->fail_allocations_ && command != IREE_ALLOCATOR_COMMAND_FREE) {
      ++test->failed_allocations_;
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(1024, {this, Allocate}, &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("references"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    width_ = AddArgument(loom_type_scalar(LOOM_SCALAR_TYPE_INDEX));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_value_id_t AddArgument(loom_type_t type) {
    loom_value_id_t value = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module_, type, &value));
    IREE_CHECK_OK(
        loom_block_add_arg(module_, loom_module_block(module_), value));
    return value;
  }

  void CheckEdges(iree_host_size_t expected_count) {
    EXPECT_EQ(module_->type_uses.active_count, expected_count);
    iree_host_size_t count = 0;
    loom_type_use_id_t previous = LOOM_TYPE_USE_ID_INVALID;
    for (auto id = loom_module_value_first_incoming_type_use(module_, width_);
         id != LOOM_TYPE_USE_ID_INVALID;
         id = module_->type_uses.records[id].next_incoming_use_id) {
      ASSERT_LT(count++, expected_count);
      const auto& record = module_->type_uses.records[id];
      EXPECT_EQ(record.referenced_value_id, width_);
      EXPECT_EQ(record.previous_incoming_use_id, previous);
      EXPECT_EQ(loom_module_value_first_outgoing_type_use(module_,
                                                          record.user_value_id),
                id);
      EXPECT_EQ(record.next_outgoing_use_id, LOOM_TYPE_USE_ID_INVALID);
      EXPECT_EQ(record.previous_outgoing_use_id, LOOM_TYPE_USE_ID_INVALID);
      previous = id;
    }
    EXPECT_EQ(count, expected_count);
  }

  // Failure injection affects backing allocations, not the subject API.
  bool fail_allocations_ = false;
  // Number of rejected backing allocation requests.
  uint32_t failed_allocations_ = 0;
  // Backing storage with deliberately small blocks for forced-growth tests.
  iree_arena_block_pool_t pool_ = {};
  // Minimal context; these fixtures exercise value/type APIs, not operations.
  loom_context_t context_ = {};
  // Module owning the reference table under test.
  loom_module_t* module_ = nullptr;
  // Index-typed argument referenced by dependent argument types.
  loom_value_id_t width_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(ModuleTypeUsesTest, RebuildReusesFullAndFragmentedCapacity) {
  std::array<loom_value_id_t, 64> arguments;
  const auto type = loom_type_pool(loom_dim_pack_dynamic(width_));
  for (auto& argument : arguments) {
    argument = AddArgument(type);
  }
  const auto* records = module_->type_uses.records;
  const auto capacity = module_->type_uses.record_capacity;
  ASSERT_EQ(capacity, arguments.size());
  for (int iteration = 0; iteration < 3; ++iteration) {
    IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
    EXPECT_EQ(module_->type_uses.records, records);
    EXPECT_EQ(module_->type_uses.record_capacity, capacity);
    CheckEdges(arguments.size());
  }

  for (size_t i = 0; i < arguments.size(); i += 2) {
    IREE_ASSERT_OK(loom_module_set_value_type(
        module_, arguments[i], loom_type_pool(loom_dim_pack_static(4))));
  }
  ASSERT_EQ(module_->type_uses.free_count, arguments.size() / 2);
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  EXPECT_EQ(module_->type_uses.records, records);
  EXPECT_EQ(module_->type_uses.record_capacity, capacity);
  CheckEdges(arguments.size() / 2);
  for (size_t i = 0; i < arguments.size(); i += 2) {
    EXPECT_EQ(loom_module_value_first_outgoing_type_use(module_, arguments[i]),
              LOOM_TYPE_USE_ID_INVALID);
    IREE_ASSERT_OK(loom_module_set_value_type(module_, arguments[i], type));
  }
  EXPECT_EQ(module_->type_uses.records, records);
  CheckEdges(arguments.size());
}

TEST_F(ModuleTypeUsesTest, FailedRebuildPreservesEdgesAndCanRetry) {
  const auto type = loom_type_pool(loom_dim_pack_dynamic(width_));
  const auto first = AddArgument(type);
  // Bulk readers assign canonical types before reconstructing side tables.
  // Additional arguments initially have static types and no outgoing records.
  constexpr iree_host_size_t kArgumentCount = 128;
  for (iree_host_size_t i = 1; i < kArgumentCount; ++i) {
    auto argument = AddArgument(loom_type_pool(loom_dim_pack_static(4)));
    loom_module_value(module_, argument)->type =
        loom_module_value_type(module_, first);
  }
  const auto table = module_->type_uses;
  const auto incoming =
      loom_module_value_first_incoming_type_use(module_, width_);
  const auto outgoing =
      loom_module_value_first_outgoing_type_use(module_, first);
  ASSERT_GT(kArgumentCount * sizeof(loom_type_use_t),
            iree_arena_block_pool_max_allocation_size(&pool_));
  fail_allocations_ = true;
  auto status = loom_module_recompute_type_uses(module_);
  fail_allocations_ = false;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_EQ(failed_allocations_, 1u);
  EXPECT_EQ(module_->type_uses.records, table.records);
  EXPECT_EQ(module_->type_uses.record_capacity, table.record_capacity);
  EXPECT_EQ(module_->type_uses.record_count, table.record_count);
  EXPECT_EQ(module_->type_uses.free_count, table.free_count);
  EXPECT_EQ(module_->type_uses.first_free_use_id, table.first_free_use_id);
  EXPECT_EQ(loom_module_value_first_incoming_type_use(module_, width_),
            incoming);
  EXPECT_EQ(loom_module_value_first_outgoing_type_use(module_, first),
            outgoing);
  CheckEdges(1);

  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  EXPECT_EQ(module_->type_uses.record_capacity, kArgumentCount);
  CheckEdges(kArgumentCount);
  const auto* records = module_->type_uses.records;
  IREE_ASSERT_OK(loom_module_recompute_type_uses(module_));
  EXPECT_EQ(module_->type_uses.records, records);
  CheckEdges(kArgumentCount);
}

}  // namespace
}  // namespace loom
