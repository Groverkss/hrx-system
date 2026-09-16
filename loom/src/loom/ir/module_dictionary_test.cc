// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdio>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class ModuleDictionaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32768, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("dictionary"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  std::vector<loom_named_attr_t> Entries(uint32_t count) {
    std::vector<loom_named_attr_t> entries(count);
    // Intern in reverse spelling order so comparing IDs cannot sort the keys.
    for (uint32_t i = count; i > 0; --i) {
      char key[32];
      std::snprintf(key, sizeof(key), "parameter_%05u", i - 1);
      IREE_CHECK_OK(loom_module_intern_string(
          module_, iree_make_cstring_view(key), &entries[i - 1].name_id));
      entries[i - 1].value = loom_attr_i64(i - 1);
    }
    return entries;
  }

  void Check(const std::vector<loom_named_attr_t>& entries) {
    loom_attribute_t dictionary = loom_attr_absent();
    IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
        module_, loom_make_named_attr_slice(entries.data(), entries.size()),
        &dictionary));
    ASSERT_EQ(dictionary.kind, LOOM_ATTR_DICT);
    ASSERT_EQ(dictionary.count, entries.size());
    if (!entries.empty()) {
      EXPECT_NE(dictionary.dict_entries, entries.data());
    }
    for (iree_host_size_t i = 0; i < entries.size(); ++i) {
      const auto& entry = dictionary.dict_entries[i];
      EXPECT_EQ(entry.value.kind, LOOM_ATTR_I64);
      EXPECT_EQ(entry.value.i64, i);
      EXPECT_EQ(entry.reserved, 0u);
      char key[32];
      std::snprintf(key, sizeof(key), "parameter_%05u",
                    static_cast<unsigned>(i));
      EXPECT_TRUE(
          iree_string_view_equal(module_->strings.entries[entry.name_id],
                                 iree_make_cstring_view(key)));
    }
    IREE_EXPECT_OK(loom_module_verify_canonical_attr_dict(module_, dictionary));
  }

  // Backing pool for canonical module-owned storage.
  iree_arena_block_pool_t pool_ = {};
  // Minimal context; attribute APIs need no registered operations.
  loom_context_t context_ = {};
  // Module owning interned key identities and copied dictionary payloads.
  loom_module_t* module_ = nullptr;
};

TEST_F(ModuleDictionaryTest, OrderedReversedAndShuffledAcrossSortBoundary) {
  for (uint32_t count : {0, 1, 2, 63, 64, 65, 256, 4096, UINT16_MAX}) {
    SCOPED_TRACE(count);
    auto entries = Entries(count);
    Check(entries);
    std::reverse(entries.begin(), entries.end());
    Check(entries);
    uint32_t random = 0x91441u;
    for (uint32_t i = count; i > 1; --i) {
      random = random * 1664525u + 1013904223u;
      std::swap(entries[i - 1], entries[random % i]);
    }
    Check(entries);
  }
}

TEST_F(ModuleDictionaryTest, SeparatedDuplicatesAreRejectedAfterReordering) {
  for (uint32_t count : {2, 64, 65, 4096}) {
    SCOPED_TRACE(count);
    auto entries = Entries(count);
    std::reverse(entries.begin(), entries.end());
    entries.back().name_id = entries.front().name_id;
    loom_attribute_t dictionary = loom_attr_absent();
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_module_make_canonical_attr_dict(
            module_, loom_make_named_attr_slice(entries.data(), entries.size()),
            &dictionary));
    EXPECT_EQ(dictionary.kind, LOOM_ATTR_ABSENT);
  }
}

TEST_F(ModuleDictionaryTest, ReorderedPayloadsAreCopiedOnce) {
  auto entries = Entries(65);
  int64_t payload[] = {3, 5, 8};
  for (auto& entry : entries) {
    entry.value = loom_attr_i64_array(payload, IREE_ARRAYSIZE(payload));
  }
  std::reverse(entries.begin(), entries.end());
  const auto used_before = module_->arena.used_allocation_size;
  loom_attribute_t dictionary = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_, loom_make_named_attr_slice(entries.data(), entries.size()),
      &dictionary));
  const auto used = module_->arena.used_allocation_size - used_before;
  // One output array and one aligned payload copy per entry, with no sort
  // scratch or repeated canonicalization of values during comparisons.
  const auto expected =
      iree_host_align(entries.size() * sizeof(loom_named_attr_t),
                      iree_max_align_t) +
      entries.size() * iree_host_align(sizeof(payload), iree_max_align_t);
  EXPECT_EQ(used, expected);
  payload[0] = 99;
  for (uint16_t i = 0; i < dictionary.count; ++i) {
    const auto& value = dictionary.dict_entries[i].value;
    EXPECT_EQ(value.kind, LOOM_ATTR_I64_ARRAY);
    EXPECT_EQ(value.count, IREE_ARRAYSIZE(payload));
    EXPECT_NE(value.i64_array, payload);
    EXPECT_EQ(value.i64_array[0], 3);
    EXPECT_EQ(value.i64_array[1], 5);
    EXPECT_EQ(value.i64_array[2], 8);
  }
}

TEST_F(ModuleDictionaryTest, InvalidKeysAreRejectedBeforeSorting) {
  auto entries = Entries(65);
  std::reverse(entries.begin(), entries.end());
  for (uint32_t index : {0, 32, 64}) {
    SCOPED_TRACE(index);
    const auto valid_id = entries[index].name_id;
    for (loom_string_id_t invalid_id :
         {static_cast<loom_string_id_t>(module_->strings.count),
          LOOM_STRING_ID_INVALID}) {
      entries[index].name_id = invalid_id;
      loom_attribute_t dictionary = loom_attr_absent();
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_INVALID_ARGUMENT,
          loom_module_make_canonical_attr_dict(
              module_,
              loom_make_named_attr_slice(entries.data(), entries.size()),
              &dictionary));
      EXPECT_EQ(dictionary.kind, LOOM_ATTR_ABSENT);
    }
    entries[index].name_id = valid_id;
  }
}

}  // namespace
}  // namespace loom
