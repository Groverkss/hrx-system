// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/adaptive_sort.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

struct SortCounters {
  // Number of comparator invocations during one sort.
  iree_host_size_t comparison_count = 0;
  // Number of element copies and assignments during one sort.
  iree_host_size_t movement_count = 0;
};

struct CountedValue {
  CountedValue(uint32_t key, SortCounters* counters)
      : key(key), counters(counters) {}

  CountedValue(const CountedValue& other)
      : key(other.key), counters(other.counters) {
    ++counters->movement_count;
  }

  CountedValue& operator=(const CountedValue& other) {
    key = other.key;
    counters = other.counters;
    ++counters->movement_count;
    return *this;
  }

  // Inline key or index into the context's external key table.
  uint32_t key;
  // Caller-owned movement counters shared by all elements.
  SortCounters* counters;
};

static bool CountedValueLess(const CountedValue* lhs, const CountedValue* rhs) {
  ++lhs->counters->comparison_count;
  return lhs->key < rhs->key;
}

LOOM_DEFINE_ADAPTIVE_SORT(SortCountedValues, CountedValue, CountedValueLess)

static bool IndexedValueLess(const uint32_t* keys, const CountedValue* lhs,
                             const CountedValue* rhs) {
  ++lhs->counters->comparison_count;
  return keys[lhs->key] < keys[rhs->key];
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(SortIndexedValues, CountedValue,
                                       const uint32_t*, IndexedValueLess)

static bool PointerValueLess(const uint32_t* const* lhs,
                             const uint32_t* const* rhs) {
  return **lhs < **rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(SortPointerValues, const uint32_t*, PointerValueLess)

TEST(AdaptiveSortTest, PointerElementsPreservePointeeConstQualification) {
  const uint32_t keys[] = {4, 1, 3, 2};
  const uint32_t* values[] = {&keys[0], &keys[1], &keys[2], &keys[3]};
  SortPointerValues(values, IREE_ARRAYSIZE(values));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    EXPECT_EQ(*values[i], i + 1);
  }
}

static iree_host_size_t ComparisonBound(iree_host_size_t count) {
  iree_host_size_t log2_ceil = 0;
  for (iree_host_size_t power = 1; power < count; power *= 2) {
    ++log2_ceil;
  }
  return 8 * count * log2_ceil + 2 * count;
}

TEST(AdaptiveSortTest, ContextKeysCoverInsertionAndHeapPaths) {
  for (iree_host_size_t count : {0, 1, 2, 63, 64, 65, 256, 4096}) {
    SCOPED_TRACE(count);
    SortCounters counters;
    std::vector<uint32_t> ascending_keys(count);
    std::vector<uint32_t> descending_keys(count);
    std::iota(ascending_keys.begin(), ascending_keys.end(), 0u);
    std::iota(descending_keys.rbegin(), descending_keys.rend(), 0u);
    std::vector<CountedValue> values;
    values.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      values.emplace_back(i, &counters);
    }

    // The same element IDs sort differently with each caller's key table.
    // Reverse order crosses the movement budget for counts above 64.
    counters = {};
    SortIndexedValues(descending_keys.data(), values.data(), count);
    for (iree_host_size_t i = 0; i < count; ++i) {
      EXPECT_EQ(values[i].key, count - i - 1);
    }
    EXPECT_LE(counters.comparison_count, ComparisonBound(count));
    EXPECT_LE(counters.movement_count, 3 * ComparisonBound(count));

    counters = {};
    SortIndexedValues(ascending_keys.data(), values.data(), count);
    for (iree_host_size_t i = 0; i < count; ++i) {
      EXPECT_EQ(values[i].key, i);
    }
    EXPECT_LE(counters.comparison_count, ComparisonBound(count));
    EXPECT_LE(counters.movement_count, 3 * ComparisonBound(count));

    counters = {};
    SortIndexedValues(ascending_keys.data(), values.data(), count);
    EXPECT_LE(counters.comparison_count, count);
    EXPECT_EQ(counters.movement_count, 0u);
  }
}

static SortCounters SortAndVerify(const std::vector<uint32_t>& keys) {
  SortCounters counters;
  std::vector<CountedValue> values;
  values.reserve(keys.size());
  for (uint32_t key : keys) {
    values.emplace_back(key, &counters);
  }

  counters = {};
  SortCountedValues(values.data(), values.size());

  std::vector<uint32_t> expected = keys;
  std::sort(expected.begin(), expected.end());
  for (iree_host_size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i].key, expected[i]);
  }
  return counters;
}

TEST(AdaptiveSortTest, SortedAndNearlySortedInputsStayLinear) {
  constexpr iree_host_size_t kCount = 4096;
  std::vector<uint32_t> keys(kCount);
  std::iota(keys.begin(), keys.end(), 0u);
  SortCounters counters = SortAndVerify(keys);
  EXPECT_LE(counters.comparison_count, kCount);
  EXPECT_EQ(counters.movement_count, 0u);

  for (iree_host_size_t i = 63; i < kCount; i += 64) {
    std::swap(keys[i - 1], keys[i]);
  }
  counters = SortAndVerify(keys);
  EXPECT_LE(counters.comparison_count, 2 * kCount);
  EXPECT_LE(counters.movement_count, kCount);
}

TEST(AdaptiveSortTest, SmallWorstCaseInputRemainsBounded) {
  constexpr iree_host_size_t kCount =
      LOOM_ADAPTIVE_SORT_INSERTION_COUNT_THRESHOLD;
  std::vector<uint32_t> keys(kCount);
  std::iota(keys.rbegin(), keys.rend(), 0u);
  EXPECT_LE(SortAndVerify(keys).comparison_count, kCount * kCount);
}

TEST(AdaptiveSortTest, AdversarialInputsHaveNLogNComparisonBound) {
  constexpr iree_host_size_t kCount = 4096;
  const iree_host_size_t comparison_bound = ComparisonBound(kCount);

  std::vector<uint32_t> rotated(kCount);
  for (iree_host_size_t i = 0; i < kCount; ++i) {
    rotated[i] = (uint32_t)((i + kCount / 2) % kCount);
  }
  EXPECT_LE(SortAndVerify(rotated).comparison_count, comparison_bound);

  std::vector<uint32_t> reversed(kCount);
  std::iota(reversed.rbegin(), reversed.rend(), 0u);
  EXPECT_LE(SortAndVerify(reversed).comparison_count, comparison_bound);

  std::vector<uint32_t> organ_pipe;
  organ_pipe.reserve(kCount);
  for (uint32_t i = 0; i < kCount; i += 2) organ_pipe.push_back(i);
  for (uint32_t i = kCount - 1;; i -= 2) {
    organ_pipe.push_back(i);
    if (i < 2) break;
  }
  EXPECT_LE(SortAndVerify(organ_pipe).comparison_count, comparison_bound);

  std::vector<uint32_t> duplicate_heavy(kCount);
  for (iree_host_size_t i = 0; i < kCount; ++i) {
    duplicate_heavy[i] = (uint32_t)(((i * 17) ^ (i >> 2)) & 7);
  }
  EXPECT_LE(SortAndVerify(duplicate_heavy).comparison_count, comparison_bound);

  std::vector<uint32_t> generated_order(kCount);
  uint32_t state = 0x9E3779B9u;
  for (uint32_t& key : generated_order) {
    state = state * 1664525u + 1013904223u;
    key = state;
  }
  EXPECT_LE(SortAndVerify(generated_order).comparison_count, comparison_bound);
}

}  // namespace
