// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/writer/catalog.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

static const loom_attr_descriptor_t kParameters[] = {
    {/*.name=*/LOOM_BSTRING_REF(5, "block"),
     /*.attr_kind=*/LOOM_ATTR_I64},
    {/*.name=*/LOOM_BSTRING_REF(4, "left"),
     /*.attr_kind=*/LOOM_ATTR_ENCODING,
     /*.flags=*/LOOM_ATTR_OPTIONAL},
    {/*.name=*/LOOM_BSTRING_REF(8, "metadata"),
     /*.attr_kind=*/LOOM_ATTR_DICT,
     /*.flags=*/LOOM_ATTR_OPTIONAL},
    {/*.name=*/LOOM_BSTRING_REF(5, "right"),
     /*.attr_kind=*/LOOM_ATTR_ENCODING,
     /*.flags=*/LOOM_ATTR_OPTIONAL},
};
static const loom_encoding_family_descriptor_t kDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(12, "test.catalog"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/IREE_ARRAYSIZE(kParameters),
    /*.parameter_descriptors=*/kParameters,
};
static const loom_encoding_vtable_t kVtable = {/*.descriptor=*/&kDescriptor};

class CatalogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_register_encoding_vtable(&context_, &kVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("catalog"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  loom_string_id_t Intern(iree_string_view_t name) {
    loom_string_id_t id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &id));
    return id;
  }

  uint16_t AddEncoding(loom_named_attr_slice_t parameters,
                       loom_string_id_t alias = LOOM_STRING_ID_INVALID) {
    const loom_encoding_t encoding = {
        /*.name_id=*/Intern(IREE_SV("test.catalog")),
        /*.alias_id=*/alias,
        /*.attribute_count=*/static_cast<uint8_t>(parameters.count),
        /*.family=*/{},
        /*.attributes=*/parameters.entries,
    };
    uint16_t id = 0;
    IREE_CHECK_OK(loom_module_add_encoding(module_, &encoding, &id));
    EXPECT_TRUE(
        loom_encoding_static_is_valid(loom_module_encoding(module_, id)));
    return id;
  }

  uint16_t AddDependentEncoding(
      uint16_t child, uint8_t fanout, int64_t block_size,
      loom_string_id_t alias = LOOM_STRING_ID_INVALID) {
    const loom_named_attr_t parameters[] = {
        {Intern(IREE_SV("block")), {}, loom_attr_i64(block_size)},
        {Intern(IREE_SV("left")), {}, loom_attr_encoding(child)},
        {Intern(IREE_SV("right")), {}, loom_attr_encoding(child)},
    };
    return AddEncoding(loom_make_named_attr_slice(parameters, 1 + fanout),
                       alias);
  }

  iree_status_t NumberEncodings(loom_bytecode_numbering_t* numbering) {
    IREE_RETURN_IF_ERROR(
        loom_bytecode_numbering_initialize(numbering, module_, &arena_));
    for (iree_host_size_t i = 0; i < module_->encodings.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_bytecode_number_encoding(
          numbering, static_cast<uint16_t>(i + 1)));
    }
    return iree_ok_status();
  }

  // Blocks shared by source IR and invocation scratch.
  iree_arena_block_pool_t pool_;
  // Invocation-owned catalog storage.
  iree_arena_allocator_t arena_;
  // Registry containing the encoding family used by the source module.
  loom_context_t context_;
  // Immutable source module after catalog initialization.
  loom_module_t* module_ = nullptr;
};

TEST_F(CatalogTest, EmptyEncodingTable) {
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  EXPECT_EQ(numbering.strings.count, 1u);
  EXPECT_EQ(numbering.types.count, 0u);
}

TEST_F(CatalogTest, EncodingOrderPreservesFirstUseAndAliases) {
  uint16_t child = AddDependentEncoding(0, 0, 4, Intern(IREE_SV("root")));
  child = AddDependentEncoding(child, 1, 8, Intern(IREE_SV("tail")));
  child = AddDependentEncoding(child, 2, 16);
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  const iree_string_view_t expected[] = {
      IREE_SV(""),      IREE_SV("test.catalog"), IREE_SV("root"),
      IREE_SV("block"), IREE_SV("tail"),         IREE_SV("left"),
      IREE_SV("right"),
  };
  ASSERT_EQ(numbering.strings.count, IREE_ARRAYSIZE(expected));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(expected); ++i) {
    EXPECT_TRUE(
        iree_string_view_equal(numbering.strings.values[i], expected[i]));
  }
  IREE_ASSERT_OK(loom_bytecode_number_attr_value(
      &numbering, loom_attr_encoding(child), nullptr));
  EXPECT_EQ(numbering.strings.count, IREE_ARRAYSIZE(expected));
  EXPECT_EQ(numbering.types.count, 0u);
}

TEST_F(CatalogTest, SharedEncodingDependenciesStayBounded) {
  uint16_t child = AddDependentEncoding(0, 0, 1);
  for (int64_t block_size = 2; block_size <= 257; ++block_size) {
    child = AddDependentEncoding(child, 2, block_size);
  }
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  EXPECT_EQ(module_->encodings.count, 257u);
  EXPECT_EQ(numbering.strings.count, 5u);
  EXPECT_EQ(numbering.types.count, 0u);
}

TEST_F(CatalogTest, DeepEncodingDependenciesNeedNoRecursiveStack) {
  uint16_t child = AddDependentEncoding(0, 0, 1);
  for (int64_t block_size = 2; block_size <= 8192; ++block_size) {
    child = AddDependentEncoding(child, 1, block_size);
  }
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  EXPECT_EQ(module_->encodings.count, 8192u);
  EXPECT_EQ(numbering.strings.count, 4u);
}

TEST_F(CatalogTest, EncodingPayloadsNumberNestedTypesAndStrings) {
  const uint16_t base = AddDependentEncoding(0, 0, 4);
  loom_type_id_t element = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &element));
  const loom_string_id_t label = Intern(IREE_SV("element_label"));
  const loom_named_attr_t metadata_entries[] = {
      {Intern(IREE_SV("element")), {}, loom_attr_type(element)},
      {Intern(IREE_SV("label")), {}, loom_attr_string(label)},
  };
  loom_attribute_t metadata;
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      module_, loom_make_named_attr_slice(metadata_entries, 2), &metadata));
  const loom_named_attr_t parameters[] = {
      {Intern(IREE_SV("block")), {}, loom_attr_i64(8)},
      {Intern(IREE_SV("left")), {}, loom_attr_encoding(base)},
      {Intern(IREE_SV("metadata")), {}, metadata},
  };
  AddEncoding(
      loom_make_named_attr_slice(parameters, IREE_ARRAYSIZE(parameters)));
  loom_bytecode_numbering_t numbering;
  IREE_ASSERT_OK(NumberEncodings(&numbering));
  ASSERT_EQ(numbering.types.count, 1u);
  EXPECT_EQ(numbering.types.module_indices_by_writer_id[0], element);
  const uint32_t label_id = numbering.strings.writer_ids_by_module_id[label];
  ASSERT_LT(label_id, numbering.strings.count);
  EXPECT_TRUE(iree_string_view_equal(numbering.strings.values[label_id],
                                     IREE_SV("element_label")));
}

}  // namespace
}  // namespace loom
