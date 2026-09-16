// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "amdf/gpu.h"
#include "experimental/xdna/executable.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "libamdf/cts/xdna/xdna_device_fixture.h"

namespace {

// Both canonical compiler fixtures multiply sixteen low-32-bit integer pairs.
constexpr size_t kElementCount = 16;
constexpr size_t kBindingByteLength = kElementCount * sizeof(uint32_t);
constexpr size_t kBindingByteOffset = kBindingByteLength;
constexpr size_t kBindingStorageByteLength = 3 * kBindingByteLength;
constexpr uint8_t kGuardValue = 0xA5;
using BindingValues = std::array<uint32_t, kElementCount>;
using ResolvedBindings = std::array<iree_hal_amd_xdna_executable_binding_t, 3>;

// Unsigned arithmetic gives exact low bits for signed and overflowing products.
constexpr BindingValues kValues = {
    0,          1,          2,          3,          7,          31,
    65535,      65536,      0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFD,
    0xFFFFFFFE, 0xFFFFFFFF, 0x12345678, 0x87654321};

class XdnaExecutionTest
    : public XdnaDeviceFixture,
      public ::testing::WithParamInterface<amdf_memory_profile_roles_t> {
 protected:
  struct MappedMemory {
    // Case-owned allocation; its context and device outlive it.
    amdf_memory_t* memory = nullptr;
    // Explicit host mapping, destroyed before the allocation.
    amdf_host_mapping_t* mapping = nullptr;
    // Borrowed host view of the mapped range.
    uint8_t* pointer = nullptr;
  };

  struct Execution {
    // Case-owned context borrowing the shared ordinary device.
    amdf_xdna_context_t* context = nullptr;
    // Private instruction allocation, released before its exact context.
    MappedMemory instructions;
    // Mapped extent covering initialization and execution ranges.
    iree_host_size_t byte_length = 0;
    // Resolved establishing command over caller-owned instruction backing.
    amdf_xdna_kernel_command_t initialization = {};
    // Resolved continuation with the same live context and bindings.
    amdf_xdna_kernel_command_t continuation = {};
    // Native transport lease borrowing this execution's context.
    amdf_kernel_queue_t* queue = nullptr;
  };

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaDeviceFixture::SetUp());
    amdf_xdna_endpoint_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_XDNA_ENDPOINT_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(xdna_api_->endpoint_query_info(endpoint_, &info), AMDF_STATUS_OK);
    amdf_xdna_device_info_t device_info = {};
    device_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_INFO;
    device_info.structure_size = sizeof(device_info);
    ASSERT_EQ(xdna_api_->device_query_info(device_, &device_info),
              AMDF_STATUS_OK);
    ASSERT_NE(device_info.context.scheduling_modes &
                  AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED,
              0u)
        << "required time-sliced XDNA contexts are unavailable";
    instruction_alignment_ = device_info.instruction.address_alignment;

    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        iree_make_cstring_view(info.target_id), 1, &target));
    const iree_file_toc_t* image = nullptr;
    if (target.identity.device_profile_id == UINT64_C(0x5354524958000001)) {
      image = iree_hal_amd_xdna_test_mul_i32_npu4_create();
    } else if (target.identity.device_profile_id ==
               UINT64_C(0x535848414C4F0001)) {
      image = iree_hal_amd_xdna_test_mul_i32_create();
    } else {
      FAIL() << "no canonical multiplication fixture for " << info.target_id;
    }
    amdf_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint_info),
              AMDF_STATUS_OK);
    uint32_t family_ordinal = UINT32_MAX;
    for (uint32_t i = 0; i < endpoint_info.queue_family_count; ++i) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      ASSERT_EQ(api_->endpoint_query_queue_family_info(endpoint_, i, &family),
                AMDF_STATUS_OK);
      if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) !=
              0) {
        family_ordinal = i;
        break;
      }
    }
    ASSERT_NE(family_ordinal, UINT32_MAX);
    queue_family_ordinal_ = family_ordinal;
    const auto* image_bytes = reinterpret_cast<const uint8_t*>(image->data);
    auto sequence = iree::hal::amd::xdna::testing::MakeOwnedByteSequence(
        std::vector<uint8_t>(image_bytes, image_bytes + image->size));
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
        sequence.get(), &target, iree_allocator_system(), &executable_));
    IREE_ASSERT_OK(iree_hal_amd_xdna_image_find_entry(
        executable_, IREE_SV("mul_i32"), &entry_ordinal_));
  }

  void DestroyMemory(MappedMemory* memory) {
    if (memory->mapping) {
      ASSERT_EQ(api_->host_mapping_destroy(memory->mapping), AMDF_STATUS_OK);
      memory->mapping = nullptr;
      memory->pointer = nullptr;
    }
    if (memory->memory) {
      ASSERT_EQ(api_->memory_destroy(memory->memory), AMDF_STATUS_OK);
      memory->memory = nullptr;
    }
  }

  void DestroyExecution(Execution* execution) {
    if (execution->queue) {
      ASSERT_EQ(api_->kernel_queue_destroy(execution->queue), AMDF_STATUS_OK);
      execution->queue = nullptr;
    }
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&execution->instructions));
    if (execution->context) {
      ASSERT_EQ(xdna_api_->context_destroy(execution->context), AMDF_STATUS_OK);
      execution->context = nullptr;
    }
  }

  void TearDown() override {
    ASSERT_NO_FATAL_FAILURE(DestroyExecution(&first_));
    ASSERT_NO_FATAL_FAILURE(DestroyExecution(&second_));
    for (auto& binding : bindings_) {
      iree_hal_buffer_release(binding.buffer);
      binding.buffer = nullptr;
      ASSERT_NO_FATAL_FAILURE(DestroyMemory(&binding.storage));
    }
    if (external_memory_.type != AMDF_EXTERNAL_MEMORY_TYPE_NONE) {
      api_->external_memory_release(&external_memory_);
    }
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&export_source_));
    iree_hal_amd_xdna_image_destroy(executable_);
    executable_ = nullptr;
    XdnaDeviceFixture::TearDown();
  }

  void MapMemory(uint64_t byte_length, MappedMemory* memory) {
    amdf_memory_map_info_t map = {};
    map.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map.structure_size = sizeof(map);
    map.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    map.byte_length = byte_length;
    ASSERT_EQ(api_->memory_map(memory->memory, &map, &memory->mapping),
              AMDF_STATUS_OK);
    amdf_host_mapping_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->host_mapping_query_info(memory->mapping, &info),
              AMDF_STATUS_OK);
    memory->pointer = static_cast<uint8_t*>(info.pointer);
  }

  void CreateBindings() {
    memory_access_.requirements.address_kinds = uint64_t{1}
                                                << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    if (GetParam() != AMDF_MEMORY_PROFILE_ROLE_CREATE) {
      amdf_memory_scope_info_t scope_info = {};
      scope_info.type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO;
      scope_info.structure_size = sizeof(scope_info);
      ASSERT_EQ(api_->memory_scope_query_info(system_scope_, &scope_info),
                AMDF_STATUS_OK);
      amdf_memory_profile_roles_t available_roles = 0;
      for (uint32_t ordinal = 0; ordinal < scope_info.memory_profile_count;
           ++ordinal) {
        amdf_memory_profile_t profile = {};
        profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
        profile.structure_size = sizeof(profile);
        amdf_memory_access_capabilities_t capabilities = {};
        capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capabilities.structure_size = sizeof(capabilities);
        const amdf_status_t status =
            QueryMemoryProfile(ordinal, &profile, &capabilities);
        if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
          continue;
        ASSERT_EQ(status, AMDF_STATUS_OK);
        available_roles |= profile.roles;
      }
      if ((available_roles & GetParam()) == 0) {
        GTEST_SKIP() << "XDNA memory role " << GetParam()
                     << " is not advertised";
      }
    }
    const uint32_t profile_ordinal =
        FindMemoryProfileOrdinal(GetParam() | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
                                 AMDF_MEMORY_FLAG_HOST_VISIBLE);
    ASSERT_NE(profile_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    for (size_t i = 0; i < bindings_.size(); ++i) {
      auto& binding = bindings_[i];
      if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_IMPORT) {
        ASSERT_NO_FATAL_FAILURE(
            ImportMemory(profile_ordinal, &binding.storage));
        if (IsSkipped()) return;
      } else {
        amdf_memory_create_info_t create = {};
        create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
        create.structure_size = sizeof(create);
        create.memory_profile_ordinal = profile_ordinal;
        create.access_count = 1;
        create.accesses = &memory_access_;
        create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
        create.byte_length = kBindingStorageByteLength;
        if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
          create.registered_host_pointer =
              binding.caller_storage.data() + kBindingByteLength;
          create.registered_host_cacheability =
              AMDF_HOST_CACHEABILITY_WRITE_BACK;
          create.minimum_alignment = kBindingByteLength;
        }
        ASSERT_EQ(api_->memory_create(system_scope_, &create,
                                      &binding.storage.memory),
                  AMDF_STATUS_OK);
        ASSERT_NO_FATAL_FAILURE(
            MapMemory(kBindingStorageByteLength, &binding.storage));
        if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
          ASSERT_EQ(binding.storage.pointer, create.registered_host_pointer);
        }
      }
      std::memset(binding.storage.pointer, kGuardValue,
                  kBindingStorageByteLength);
      ASSERT_NO_FATAL_FAILURE(WrapBinding(i));
      ASSERT_NO_FATAL_FAILURE(QueryHostCacheOperations(i));
    }
  }

  void WrapBinding(size_t ordinal) {
    auto& binding = bindings_[ordinal];
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        iree_hal_buffer_placement_undefined(),
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE |
            IREE_HAL_MEMORY_ACCESS_UNALIGNED,
        IREE_HAL_BUFFER_USAGE_STORAGE, kBindingByteLength,
        iree_make_byte_span(binding.storage.pointer + kBindingByteOffset,
                            kBindingByteLength),
        iree_hal_buffer_release_callback_null(), iree_allocator_system(),
        &binding.buffer));
    resolved_bindings_[ordinal].buffer_ref =
        iree_hal_make_buffer_ref(binding.buffer, 0, kBindingByteLength);
    resolved_bindings_[ordinal].memory = binding.storage.memory;
    resolved_bindings_[ordinal].memory_byte_offset = kBindingByteOffset;
    ASSERT_EQ(api_->memory_query_address(
                  binding.storage.memory, 0, AMDF_MEMORY_ADDRESS_XDNA_DMA,
                  &resolved_bindings_[ordinal].device_address),
              AMDF_STATUS_OK);
    resolved_bindings_[ordinal].device_address += kBindingByteOffset;
  }

  void QueryHostCacheOperations(size_t ordinal) {
    auto& binding = bindings_[ordinal];
    amdf_memory_site_t host = {};
    host.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
    host.structure_size = sizeof(host);
    host.kind = AMDF_MEMORY_SITE_KIND_HOST;
    host.value.host_mapping = binding.storage.mapping;
    amdf_memory_site_t device = {};
    device.type = AMDF_STRUCTURE_TYPE_MEMORY_SITE;
    device.structure_size = sizeof(device);
    device.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    device.value.device.memory = binding.storage.memory;
    device.value.device.queue_family_ordinal = queue_family_ordinal_;
    amdf_memory_pair_info_t pair = {};
    pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair.structure_size = sizeof(pair);
    ASSERT_EQ(api_->memory_query_pair_info(&host, &device, &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    ASSERT_EQ(pair.release.executor,
              AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
    ASSERT_EQ(pair.release.host_operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
    ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    ASSERT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    binding.host_cache.publish = pair.release.host_operation;

    ASSERT_EQ(api_->memory_query_pair_info(&device, &host, &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    ASSERT_EQ(pair.acquire.executor,
              AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
    ASSERT_EQ(pair.acquire.host_operation,
              AMDF_HOST_CACHE_OPERATION_INVALIDATE);
    ASSERT_EQ(pair.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
    ASSERT_EQ(pair.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
    binding.host_cache.acquire = pair.acquire.host_operation;
  }

  void RequireDirectHostTransport(
      const amdf_memory_profile_t& profile,
      amdf_external_memory_support_flags_t required_flags) {
    for (uint32_t i = 0; i < profile.external_memory_support_count; ++i) {
      const auto& support = profile.external_memory_support[i];
      if (support.type == AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD &&
          (support.flags & required_flags) == required_flags) {
        return;
      }
    }
    GTEST_SKIP() << "XDNA direct-host external ranges are not advertised";
  }

  void ImportMemory(uint32_t profile_ordinal, MappedMemory* memory) {
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    ASSERT_EQ(QueryMemoryProfile(profile_ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    RequireDirectHostTransport(
        profile, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT |
                     AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
    if (IsSkipped()) return;
    const auto source_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE;
    const uint32_t source_ordinal = FindMemoryProfileOrdinal(
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_EXPORT |
            AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        source_flags);
    ASSERT_NE(source_ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    ASSERT_EQ(QueryMemoryProfile(source_ordinal, &profile, &capabilities),
              AMDF_STATUS_OK);
    RequireDirectHostTransport(
        profile, AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT |
                     AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET);
    if (IsSkipped()) return;
    const uint64_t source_offset =
        profile.allocation.native_byte_length_granularity + kBindingByteLength;
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = source_ordinal;
    create.access_count = 1;
    create.accesses = &memory_access_;
    create.required_flags = source_flags;
    create.byte_length = source_offset + kBindingStorageByteLength;
    ASSERT_EQ(
        api_->memory_create(system_scope_, &create, &export_source_.memory),
        AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(MapMemory(create.byte_length, &export_source_));
    std::memset(export_source_.pointer, kGuardValue, create.byte_length);
    std::memset(export_source_.pointer + source_offset, 0x3C,
                kBindingStorageByteLength);
    ASSERT_EQ(api_->host_mapping_cache_control(export_source_.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, create.byte_length),
              AMDF_STATUS_OK);
    amdf_memory_export_info_t export_info = {};
    export_info.type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO;
    export_info.structure_size = sizeof(export_info);
    export_info.external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD;
    export_info.byte_offset = source_offset;
    export_info.byte_length = kBindingStorageByteLength;
    ASSERT_EQ(api_->memory_export(export_source_.memory, &export_info,
                                  &external_memory_),
              AMDF_STATUS_OK);
    const auto identity = external_memory_.physical_backing_id;
    ASSERT_TRUE(amdf_physical_memory_id_is_valid(&identity));
    ASSERT_EQ(external_memory_.source_byte_offset, source_offset);
    amdf_memory_import_info_t import_info = {};
    import_info.type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO;
    import_info.structure_size = sizeof(import_info);
    import_info.memory_profile_ordinal = profile_ordinal;
    import_info.access_count = 1;
    import_info.accesses = &memory_access_;
    import_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    import_info.minimum_alignment = kBindingByteLength;
    ASSERT_EQ(api_->memory_import(system_scope_, &import_info,
                                  &external_memory_, &memory->memory),
              AMDF_STATUS_OK);
    const amdf_external_memory_t empty = {};
    ASSERT_EQ(std::memcmp(&external_memory_, &empty, sizeof(empty)), 0);
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&export_source_));
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(memory->memory, &info), AMDF_STATUS_OK);
    ASSERT_TRUE(
        amdf_physical_memory_id_is_equal(&info.physical_backing_id, &identity));
    ASSERT_EQ(info.source_byte_offset, source_offset);
    ASSERT_EQ(info.byte_length, kBindingStorageByteLength);
    ASSERT_NO_FATAL_FAILURE(MapMemory(kBindingStorageByteLength, memory));
    for (size_t i = 0; i < kBindingStorageByteLength; ++i) {
      ASSERT_EQ(memory->pointer[i], 0x3C) << "imported byte " << i;
    }
  }

  void PrepareExecution(const ResolvedBindings& bindings,
                        Execution* execution) {
    amdf_xdna_context_create_info_t context_create = {};
    context_create.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
    context_create.structure_size = sizeof(context_create);
    context_create.logical_column_count = 1;
    context_create.physical_column_origin =
        AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    context_create.acceptable_scheduling_modes =
        AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
    ASSERT_EQ(xdna_api_->context_create(device_, &context_create,
                                        &execution->context),
              AMDF_STATUS_OK);
    const auto* tables = iree_hal_amd_xdna_image_tables(executable_);
    const auto entry =
        iree_hal_amd_xdna_image_tables_entry(tables, entry_ordinal_);
    ASSERT_EQ(entry.allocation_use_count, 1u);
    const auto requirement = iree_hal_amd_xdna_image_tables_allocation(
        tables, iree_hal_amd_xdna_image_tables_allocation_use(
                    tables, entry.first_allocation_use));
    ASSERT_EQ(requirement.domain, IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND);
    execution->byte_length = requirement.byte_length;
    amdf_memory_scope_t* scope = nullptr;
    uint32_t count = 0;
    ASSERT_EQ(xdna_api_->context_enumerate_memory_scopes(execution->context, 1,
                                                         &scope, &count),
              AMDF_STATUS_OK);
    amdf_memory_device_access_t access = {};
    access.device = device_;
    access.requirements.access = AMDF_MEMORY_ACCESS_READ |
                                 AMDF_MEMORY_ACCESS_WRITE |
                                 AMDF_MEMORY_ACCESS_EXECUTE;
    access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    access.requirements.address_kinds = uint64_t{1}
                                        << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
    amdf_memory_profile_t profile = {};
    profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
    profile.structure_size = sizeof(profile);
    amdf_memory_access_capabilities_t capabilities = {};
    capabilities.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capabilities.structure_size = sizeof(capabilities);
    ASSERT_EQ(api_->memory_scope_query_device_profile(scope, 0, 1, &access,
                                                      &profile, &capabilities),
              AMDF_STATUS_OK);
    const uint64_t granularity = profile.allocation.byte_length_granularity;
    ASSERT_GT(granularity, 0u);
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = profile.ordinal;
    create.access_count = 1;
    create.accesses = &access;
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length =
        ((execution->byte_length + granularity - 1) / granularity) *
        granularity;
    create.minimum_alignment = profile.allocation.minimum_alignment;
    auto& instructions = execution->instructions;
    ASSERT_EQ(api_->memory_create(scope, &create, &instructions.memory),
              AMDF_STATUS_OK);
    uint64_t firmware_address = 0;
    ASSERT_EQ(api_->memory_query_address(instructions.memory, 0,
                                         AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
                                         &firmware_address),
              AMDF_STATUS_OK);
    ASSERT_EQ(firmware_address % instruction_alignment_, 0u);
    ASSERT_NO_FATAL_FAILURE(MapMemory(execution->byte_length, &instructions));
    iree_hal_amd_xdna_executable_storage_t storage = {};
    storage.memory = instructions.memory;
    storage.mapping =
        iree_make_byte_span(instructions.pointer, execution->byte_length);
    storage.device_address = firmware_address;
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_load(
        executable_, entry_ordinal_, 1, &storage));
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_bind(
        executable_, entry_ordinal_, 1, &storage, bindings.size(),
        bindings.data()));
    uint32_t continuation = 0;
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_invocation(
        executable_, entry_ordinal_, 0, 1, &storage, &execution->initialization,
        &continuation));
    ASSERT_EQ(continuation, 1u);
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_invocation(
        executable_, entry_ordinal_, continuation, 1, &storage,
        &execution->continuation, &continuation));
    ASSERT_EQ(continuation, 1u);
    ASSERT_EQ(api_->host_mapping_cache_control(instructions.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, execution->byte_length),
              AMDF_STATUS_OK);
    amdf_xdna_kernel_queue_create_info_t queue_create = {};
    queue_create.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO;
    queue_create.structure_size = sizeof(queue_create);
    queue_create.queue_family_ordinal = queue_family_ordinal_;
    ASSERT_EQ(xdna_api_->kernel_queue_create(execution->context, &queue_create,
                                             &execution->queue),
              AMDF_STATUS_OK);
    amdf_kernel_queue_info_t queue_info = {};
    queue_info.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO;
    queue_info.structure_size = sizeof(queue_info);
    ASSERT_EQ(api_->kernel_queue_query_info(execution->queue, &queue_info),
              AMDF_STATUS_OK);
    ASSERT_EQ(queue_info.queue_family_ordinal, queue_family_ordinal_);
    ASSERT_EQ(queue_info.command_type, AMDF_QUEUE_COMMAND_TYPE_XDNA);
    ASSERT_GE(queue_info.maximum_pending_submission_count, 1u);
    ASSERT_GE(queue_info.maximum_command_count, 1u);
  }

  void RunExecution(const Execution& execution,
                    const amdf_xdna_kernel_command_t* command) {
    amdf_xdna_kernel_queue_submission_info_t submit = {};
    submit.type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO;
    submit.structure_size = sizeof(submit);
    submit.command_count = 1;
    submit.commands = command;
    uint64_t submission = 0;
    ASSERT_EQ(
        xdna_api_->kernel_queue_submit(execution.queue, &submit, &submission),
        AMDF_STATUS_OK);
    ASSERT_EQ(api_->kernel_queue_wait(execution.queue, submission,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    amdf_kernel_queue_status_t status = {};
    status.type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS;
    status.structure_size = sizeof(status);
    ASSERT_EQ(api_->kernel_queue_query_status(execution.queue, &status),
              AMDF_STATUS_OK);
    ASSERT_EQ(status.retired_submission, submission);
    ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
  }

  void WriteBinding(size_t ordinal, const BindingValues& values) {
    auto& storage = bindings_[ordinal].storage;
    for (size_t i = 0; i < kElementCount; ++i) {
      iree_unaligned_store_le_u32(storage.pointer + kBindingByteOffset + i * 4,
                                  values[i]);
    }
    ASSERT_EQ(api_->host_mapping_cache_control(
                  storage.mapping, bindings_[ordinal].host_cache.publish, 0,
                  kBindingStorageByteLength),
              AMDF_STATUS_OK);
  }

  void VerifyBindings(const std::array<BindingValues, 3>& expected) {
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      SCOPED_TRACE(ordinal);
      const auto& storage = bindings_[ordinal].storage;
      ASSERT_EQ(api_->host_mapping_cache_control(
                    storage.mapping, bindings_[ordinal].host_cache.acquire, 0,
                    kBindingStorageByteLength),
                AMDF_STATUS_OK);
      for (size_t i = 0; i < kBindingByteLength; ++i) {
        ASSERT_EQ(storage.pointer[i], kGuardValue) << "prefix " << i;
        ASSERT_EQ(storage.pointer[2 * kBindingByteLength + i], kGuardValue)
            << "suffix " << i;
      }
      for (size_t i = 0; i < kElementCount; ++i) {
        ASSERT_EQ(iree_unaligned_load_le_u32(storage.pointer +
                                             kBindingByteOffset + i * 4),
                  expected[ordinal][i])
            << "element " << i;
      }
    }
  }

  // Native kernel queue family selected from the libamdf endpoint.
  uint32_t queue_family_ordinal_ = UINT32_MAX;
  // Case-owned immutable decoded compiler image.
  iree_hal_amd_xdna_image_t* executable_ = nullptr;
  // Indexed multiplication entry in the immutable image.
  uint32_t entry_ordinal_ = 0;
  // Storage requirements derived from the endpoint and executable.
  iree_host_size_t instruction_alignment_ = 0;
  // Primary execution owner used by every case.
  Execution first_;
  // Independently prepared execution sharing only ordinary data and device.
  Execution second_;
  // Temporary export source, released before imported bindings are used.
  MappedMemory export_source_;
  // Move-owned DMA-BUF value, consumed by import or released at teardown.
  amdf_external_memory_t external_memory_ = {};
  // Native and HAL owners for the three canonical bindings.
  struct Binding {
    // Optional caller-owned backing. Registration begins inside this array
    // and remains live until the native memory handle is destroyed.
    alignas(kBindingByteLength)
        std::array<uint8_t, kBindingByteLength +
                                kBindingStorageByteLength> caller_storage = {};
    // Native memory and its explicit host view.
    MappedMemory storage;
    // HAL wrapper borrowing storage through all command completion.
    iree_hal_buffer_t* buffer = nullptr;
    // Directional operations selected once from the concrete host/device pair.
    struct {
      // Releases CPU writes before the program reads through DMA.
      amdf_host_cache_operation_t publish = 0;
      // Makes completed DMA writes visible before CPU verification.
      amdf_host_cache_operation_t acquire = 0;
    } host_cache;
  };
  // Lhs, rhs and output backing, independent of instruction storage.
  std::array<Binding, 3> bindings_;
  // DMA addresses resolved through the public memory handles.
  ResolvedBindings resolved_bindings_ = {};
};

TEST_P(XdnaExecutionTest, ReusesImmutableInstructionsWithChangingInputs) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings());
  if (IsSkipped()) return;
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    std::array<BindingValues, 3> expected;
    BindingValues poisoned;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = kValues[(i * 3 + iteration + 5) % kElementCount];
      expected[2][i] = expected[0][i] * expected[1][i];
      // Every output must change; neither zero-fill nor stale output can pass.
      poisoned[i] = ~expected[2][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
    const auto* command =
        iteration == 0 ? &first_.initialization : &first_.continuation;
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_, command));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  }
}

TEST_P(XdnaExecutionTest, SharesDataAcrossIndependentContextLifetimes) {
  ASSERT_NO_FATAL_FAILURE(CreateBindings());
  if (IsSkipped()) return;
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  // A: (lhs, rhs) -> intermediate; B: (intermediate, rhs) -> lhs.
  const ResolvedBindings consumer_bindings = {
      resolved_bindings_[2], resolved_bindings_[1], resolved_bindings_[0]};
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(consumer_bindings, &second_));
  ASSERT_NE(first_.context, second_.context);
  ASSERT_NE(first_.instructions.memory, second_.instructions.memory);

  std::array<amdf_memory_info_t, 3> original_info = {};
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    auto& info = original_info[ordinal];
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(bindings_[ordinal].storage.memory, &info),
              AMDF_STATUS_OK);
  }

  std::array<BindingValues, 3> expected;
  BindingValues poisoned;
  for (size_t i = 0; i < kElementCount; ++i) {
    expected[0][i] = kValues[i];
    // Odd nonunit factors keep the second computation distinct modulo 2^32.
    expected[1][i] = static_cast<uint32_t>(i * 2 + 3);
    expected[2][i] = expected[0][i] * expected[1][i];
    poisoned[i] = ~expected[2][i];
  }
  ASSERT_NO_FATAL_FAILURE(WriteBinding(0, expected[0]));
  ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
  ASSERT_NO_FATAL_FAILURE(WriteBinding(2, poisoned));
  ASSERT_NO_FATAL_FAILURE(RunExecution(first_, &first_.initialization));
  // The consumer sees the producer's bytes directly. There is no CPU payload
  // access or cache transition between these fully retired finite commands.
  ASSERT_NO_FATAL_FAILURE(RunExecution(second_, &second_.initialization));
  for (size_t i = 0; i < kElementCount; ++i) {
    expected[0][i] = expected[2][i] * expected[1][i];
  }
  ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  ASSERT_NO_FATAL_FAILURE(DestroyExecution(&first_));

  // Shared backing borrows the ordinary device, not the departed producer.
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    amdf_memory_info_t info = {};
    info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    info.structure_size = sizeof(info);
    ASSERT_EQ(api_->memory_query_info(bindings_[ordinal].storage.memory, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(info.byte_length, original_info[ordinal].byte_length);
    ASSERT_EQ(info.source_byte_offset,
              original_info[ordinal].source_byte_offset);
    ASSERT_TRUE(amdf_physical_memory_id_is_equal(
        &info.physical_backing_id,
        &original_info[ordinal].physical_backing_id));
    uint64_t address = 0;
    ASSERT_EQ(
        api_->memory_query_address(bindings_[ordinal].storage.memory, 0,
                                   AMDF_MEMORY_ADDRESS_XDNA_DMA, &address),
        AMDF_STATUS_OK);
    ASSERT_EQ(address + kBindingByteOffset,
              resolved_bindings_[ordinal].device_address);
  }
  for (uint32_t iteration = 0; iteration < 2; ++iteration) {
    SCOPED_TRACE(iteration);
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[1][i] += 2;
      expected[0][i] = expected[2][i] * expected[1][i];
      poisoned[i] = ~expected[0][i];
    }
    ASSERT_NO_FATAL_FAILURE(WriteBinding(0, poisoned));
    ASSERT_NO_FATAL_FAILURE(WriteBinding(1, expected[1]));
    ASSERT_NO_FATAL_FAILURE(RunExecution(second_, &second_.continuation));
    ASSERT_NO_FATAL_FAILURE(VerifyBindings(expected));
  }
}

// Exercises the queue visibility contract with real producers and consumers.
// COPY_DATA uses TC L2; it does not qualify shader-side PROGRAM transitions.
class XdnaPoolVisibilityTest : public XdnaExecutionTest {
 protected:
  static constexpr uint64_t kStagingByteLength = 4096;
  static constexpr uint64_t kReadbackByteOffset = 1024;
  static constexpr uint64_t kCompletionByteOffset = 2048;
  static constexpr uint64_t kRingByteLength = 4096;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(XdnaExecutionTest::SetUp());
    const void* extension = nullptr;
    const auto extension_status =
        api_->query_extension(AMDF_EXTENSION_GPU, AMDF_GPU_EXTENSION_VERSION_1,
                              AMDF_GPU_EXTENSION_VERSION_LATEST, &extension);
    if (extension_status ==
        amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "the GPU provider is not enabled";
    }
    ASSERT_EQ(extension_status, AMDF_STATUS_OK);
    gpu_api_ = static_cast<const amdf_gpu_api_t*>(extension);
    uint32_t count = 0;
    ASSERT_EQ(api_->endpoint_enumerate(instance_, 0, nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> endpoints(count);
    ASSERT_EQ(
        api_->endpoint_enumerate(instance_, count, endpoints.data(), &count),
        AMDF_STATUS_OK);
    amdf_endpoint_t* gpu_endpoint = nullptr;
    for (const auto& summary : endpoints) {
      if (summary.engine_kind != AMDF_ENGINE_KIND_GPU) continue;
      amdf_endpoint_t* endpoint = nullptr;
      ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint),
                AMDF_STATUS_OK);
      amdf_endpoint_info_t info = {};
      info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
      info.structure_size = sizeof(info);
      ASSERT_EQ(api_->endpoint_query_info(endpoint, &info), AMDF_STATUS_OK);
      for (uint32_t ordinal = 0; ordinal < info.queue_family_count; ++ordinal) {
        amdf_queue_family_info_t family = {};
        family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
        family.structure_size = sizeof(family);
        ASSERT_EQ(
            api_->endpoint_query_queue_family_info(endpoint, ordinal, &family),
            AMDF_STATUS_OK);
        if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
            family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 &&
            (family.format_features &
             AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR) != 0 &&
            (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) !=
                0 &&
            (family.user_queue_capabilities &
             AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER) != 0 &&
            family.maximum_ring_byte_length >= kRingByteLength) {
          gpu_family_ = family;
          gpu_endpoint = endpoint;
          break;
        }
      }
      if (gpu_endpoint) break;
    }
    if (!gpu_endpoint) {
      GTEST_SKIP() << "GPU PM4 user publication with GCR is not advertised";
    }
    ASSERT_EQ(GetCtsDeviceCache().GetGpuDevice(gpu_endpoint, &gpu_device_),
              AMDF_STATUS_OK);
    pool_accesses_[0] = memory_access_;
    pool_accesses_[0].requirements.address_kinds =
        UINT64_C(1) << AMDF_MEMORY_ADDRESS_XDNA_DMA;
    pool_accesses_[1].device = gpu_device_;
    pool_accesses_[1].requirements.access =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    pool_accesses_[1].requirements.flags =
        AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    pool_accesses_[1].requirements.address_kinds = UINT64_C(1)
                                                   << AMDF_MEMORY_ADDRESS_GPU;
  }

  void TearDown() override {
    if (gpu_mapping_) {
      ASSERT_EQ(api_->user_queue_mapping_destroy(gpu_mapping_), AMDF_STATUS_OK);
      gpu_mapping_ = nullptr;
    }
    if (gpu_queue_) {
      ASSERT_EQ(api_->user_queue_destroy(gpu_queue_), AMDF_STATUS_OK);
      gpu_queue_ = nullptr;
    }
    ASSERT_NO_FATAL_FAILURE(DestroyMemory(&staging_));
    XdnaExecutionTest::TearDown();
  }

  void FindProfile(uint32_t access_count,
                   const amdf_memory_device_access_t* accesses,
                   amdf_memory_profile_roles_t role,
                   amdf_memory_profile_t* out_profile) {
    out_profile->ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t ordinal = 0;; ++ordinal) {
      amdf_memory_profile_t profile = {};
      profile.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE;
      profile.structure_size = sizeof(profile);
      std::array<amdf_memory_access_capabilities_t, 2> capabilities = {};
      for (auto& capability : capabilities) {
        capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
        capability.structure_size = sizeof(capability);
      }
      const auto status = api_->memory_scope_query_device_profile(
          system_scope_, ordinal, access_count, accesses, &profile,
          capabilities.data());
      if (amdf_status_code(status) == AMDF_STATUS_CODE_OUT_OF_RANGE) break;
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED))
        continue;
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile.roles & role) != 0 &&
          (profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) != 0 &&
          (profile.supported_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE) != 0) {
        *out_profile = profile;
        return;
      }
    }
  }

  void CreatePool() {
    amdf_memory_profile_t profile = {};
    ASSERT_NO_FATAL_FAILURE(FindProfile(
        pool_accesses_.size(), pool_accesses_.data(), GetParam(), &profile));
    if (profile.ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN &&
        GetParam() == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
      GTEST_SKIP() << "joint host registration is not advertised";
    }
    ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    std::array<amdf_memory_profile_site_t, 3> sites = {};
    sites[0].kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    sites[0].value.device.queue_family_ordinal = queue_family_ordinal_;
    sites[1].kind = AMDF_MEMORY_SITE_KIND_DEVICE;
    sites[1].value.device.access_ordinal = 1;
    sites[1].value.device.queue_family_ordinal = gpu_family_.ordinal;
    sites[2].kind = AMDF_MEMORY_SITE_KIND_HOST;
    sites[2].value.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    amdf_memory_profile_pair_query_t query = {};
    query.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY;
    query.structure_size = sizeof(query);
    query.memory_profile_ordinal = profile.ordinal;
    query.access_count = pool_accesses_.size();
    query.accesses = pool_accesses_.data();
    query.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    query.registered_host_cacheability =
        profile.registration.registered_host_cacheability;
    // All recipes are obtained before any pool backing exists, and retained
    // across independent allocations, queue submissions and data generations.
    std::array<amdf_memory_pair_info_t, 3> pairs = {};
    const uint32_t edges[][2] = {{2, 0}, {1, 0}, {0, 1}};
    for (size_t i = 0; i < pairs.size(); ++i) {
      query.producer = sites[edges[i][0]];
      query.consumer = sites[edges[i][1]];
      pairs[i].type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
      pairs[i].structure_size = sizeof(pairs[i]);
      ASSERT_EQ(
          api_->memory_scope_query_pair_info(system_scope_, &query, &pairs[i]),
          AMDF_STATUS_OK);
      ASSERT_NE(pairs[i].flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
    }
    const auto& publish = pairs[0].release;
    ASSERT_EQ(publish.kind, AMDF_CACHE_TRANSITION_KIND_RANGE);
    ASSERT_EQ(publish.executor, AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT);
    ASSERT_EQ(pairs[0].acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pairs[1].acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(pairs[2].release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    gpu_release_ = pairs[1].release;
    gpu_acquire_ = pairs[2].acquire;
    ASSERT_EQ(gpu_release_.operation, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
    ASSERT_EQ(gpu_acquire_.operation, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);

    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      auto& storage = bindings_[ordinal].storage;
      amdf_memory_create_info_t create = {};
      create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
      create.structure_size = sizeof(create);
      create.memory_profile_ordinal = profile.ordinal;
      create.access_count = query.access_count;
      create.accesses = query.accesses;
      create.required_flags = query.required_flags;
      create.byte_length = kBindingStorageByteLength;
      create.minimum_alignment = kBindingByteLength;
      create.registered_host_cacheability = query.registered_host_cacheability;
      if (GetParam() == AMDF_MEMORY_PROFILE_ROLE_REGISTER) {
        create.registered_host_pointer =
            registered_pages_[ordinal].data() + kBindingByteLength;
      }
      ASSERT_EQ(api_->memory_create(system_scope_, &create, &storage.memory),
                AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(MapMemory(create.byte_length, &storage));
      std::memset(storage.pointer, kGuardValue, create.byte_length);
      ASSERT_EQ(
          api_->host_mapping_cache_control(
              storage.mapping, publish.host_operation, 0, create.byte_length),
          AMDF_STATUS_OK);
      ASSERT_NO_FATAL_FAILURE(WrapBinding(ordinal));
      ASSERT_EQ(
          api_->memory_query_address(storage.memory, 1, AMDF_MEMORY_ADDRESS_GPU,
                                     &gpu_addresses_[ordinal]),
          AMDF_STATUS_OK);
    }
  }

  void CreateGpuQueue() {
    amdf_memory_profile_t profile = {};
    ASSERT_NO_FATAL_FAILURE(FindProfile(
        1, &pool_accesses_[1], AMDF_MEMORY_PROFILE_ROLE_CREATE, &profile));
    ASSERT_NE(profile.ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_create_info_t create = {};
    create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    create.structure_size = sizeof(create);
    create.memory_profile_ordinal = profile.ordinal;
    create.access_count = 1;
    create.accesses = &pool_accesses_[1];
    create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    create.byte_length = kStagingByteLength;
    ASSERT_EQ(api_->memory_create(system_scope_, &create, &staging_.memory),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(MapMemory(create.byte_length, &staging_));
    ASSERT_EQ(
        api_->memory_query_address(staging_.memory, 0, AMDF_MEMORY_ADDRESS_GPU,
                                   &staging_address_),
        AMDF_STATUS_OK);
    amdf_gpu_user_queue_create_info_t queue = {};
    queue.type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO;
    queue.structure_size = sizeof(queue);
    queue.queue_family_ordinal = gpu_family_.ordinal;
    queue.priority = AMDF_QUEUE_PRIORITY_NORMAL;
    queue.producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE;
    queue.required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER;
    queue.ring_byte_length =
        std::max(gpu_family_.minimum_ring_byte_length, kRingByteLength);
    ASSERT_EQ(gpu_api_->user_queue_create(gpu_device_, &queue, &gpu_queue_),
              AMDF_STATUS_OK);
    ASSERT_EQ(api_->user_queue_map(gpu_queue_, nullptr, &gpu_mapping_),
              AMDF_STATUS_OK);
    gpu_mapping_info_.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO;
    gpu_mapping_info_.structure_size = sizeof(gpu_mapping_info_);
    ASSERT_EQ(
        api_->user_queue_mapping_query_info(gpu_mapping_, &gpu_mapping_info_),
        AMDF_STATUS_OK);
    ASSERT_EQ(gpu_mapping_info_.index_bits, 64u);
    ASSERT_EQ(gpu_mapping_info_.doorbell_bits, 64u);
  }

  static uint32_t Pm4Header(uint32_t opcode, uint32_t count) {
    return (3u << 30) | (opcode << 8) | ((count - 2) << 16);
  }

  static void AppendGpuTransition(const amdf_cache_transition_t& transition,
                                  std::vector<uint32_t>* words) {
    ASSERT_EQ(transition.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(transition.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    // CS_PARTIAL_FLUSH followed by a conservative full-range ACQUIRE_MEM GCR
    // implements both system release and system acquire on this format.
    constexpr uint32_t kGcr = (3 << 0) | (1 << 4) | (1 << 5) | (1 << 7) |
                              (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15);
    words->insert(words->end(),
                  {Pm4Header(0x46, 2), 7 | (4 << 8), Pm4Header(0x58, 8), 0,
                   UINT32_MAX, 0xff, 0, 0, 0x0a, kGcr});
  }

  static void AppendGpuCopy(uint64_t source, uint64_t target,
                            size_t byte_length, std::vector<uint32_t>* words) {
    for (size_t offset = 0; offset < byte_length; offset += sizeof(uint32_t)) {
      // COPY_DATA between TC L2 addresses, with write confirmation.
      words->insert(words->end(),
                    {Pm4Header(0x40, 6), 2 | (2 << 8) | (1 << 20),
                     static_cast<uint32_t>(source + offset),
                     static_cast<uint32_t>((source + offset) >> 32),
                     static_cast<uint32_t>(target + offset),
                     static_cast<uint32_t>((target + offset) >> 32)});
    }
  }

  void RunGpu(std::vector<uint32_t> words, uint32_t completion_value) {
    auto* completion = reinterpret_cast<volatile uint32_t*>(
        staging_.pointer + kCompletionByteOffset);
    ASSERT_NE(*completion, completion_value);
    const uint64_t completion_address =
        staging_address_ + kCompletionByteOffset;
    // WRITE_DATA confirms a separate coherent completion line after the
    // ordered copies/barriers. Ring consumption alone is not completion.
    words.insert(words.end(), {Pm4Header(0x37, 5), (2 << 8) | (1 << 20),
                               static_cast<uint32_t>(completion_address),
                               static_cast<uint32_t>(completion_address >> 32),
                               completion_value});
    auto* ring = reinterpret_cast<uint32_t*>(
        static_cast<uintptr_t>(gpu_mapping_info_.ring_address));
    const uint64_t capacity =
        gpu_mapping_info_.ring_byte_length / sizeof(*ring);
    std::vector<uint32_t> publication;
    auto append_padding = [&](size_t count) {
      publication.push_back(Pm4Header(0x10, count));
      publication.resize(publication.size() + count - 1, 0);
    };
    // Preserve packet boundaries across ring wrap and leave space for a
    // complete type-3 NOP instead of stranding one dword at the tail.
    for (size_t i = 0; i < words.size();) {
      const size_t count = ((words[i] >> 16) & 0x3fff) + 2;
      const size_t tail =
          capacity - (published_index_ + publication.size()) % capacity;
      if (tail < count || tail == count + 1) append_padding(tail);
      publication.insert(publication.end(), words.begin() + i,
                         words.begin() + i + count);
      i += count;
    }
    size_t padding = 8 - publication.size() % 8;
    if (padding == 1) padding += 8;
    append_padding(padding);
    // All prior batches have retired. The publication still reserves the
    // native PM4 empty/full discriminator by remaining smaller than the ring.
    ASSERT_LT(publication.size(), capacity);
    for (size_t i = 0; i < publication.size(); ++i) {
      ring[(published_index_ + i) % capacity] = publication[i];
    }
    published_index_ += publication.size();
    auto* write_index = reinterpret_cast<volatile uint64_t*>(
        static_cast<uintptr_t>(gpu_mapping_info_.write_index_address));
    auto* doorbell = reinterpret_cast<volatile uint64_t*>(
        static_cast<uintptr_t>(gpu_mapping_info_.doorbell_address));
    std::atomic_thread_fence(std::memory_order_release);
    *write_index = published_index_;
    std::atomic_thread_fence(std::memory_order_release);
    *doorbell = published_index_;
    ASSERT_EQ(api_->user_queue_wait_consumed(gpu_queue_, published_index_,
                                             AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    while (*completion != completion_value) {
      amdf_user_queue_status_t status = {};
      status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
      status.structure_size = sizeof(status);
      ASSERT_EQ(api_->user_queue_query_status(gpu_queue_, &status),
                AMDF_STATUS_OK);
      ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
      std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
  }

  // GPU extension table and device borrowed from the shared CTS provider.
  const amdf_gpu_api_t* gpu_api_ = nullptr;
  // Ordinary GPU address domain retained by the CTS cache.
  amdf_device_t* gpu_device_ = nullptr;
  // Exact PM4 family used by qualification and publication.
  amdf_queue_family_info_t gpu_family_ = {};
  // Complete pool contract, ordered XDNA then GPU.
  std::array<amdf_memory_device_access_t, 2> pool_accesses_ = {};
  // GPU release selected before any backing exists.
  amdf_cache_transition_t gpu_release_ = {};
  // GPU acquire selected before any backing exists.
  amdf_cache_transition_t gpu_acquire_ = {};
  // Independent pages prevent native registration ranges from overlapping.
  alignas(4096) std::array<std::array<uint8_t, 4096>, 3> registered_pages_ = {};
  // Cold GPU addresses corresponding to the three XDNA bindings.
  std::array<uint64_t, 3> gpu_addresses_ = {};
  // GPU-only host staging, readback and a separate completion cache line.
  MappedMemory staging_;
  // GPU address of staging_; never used by XDNA.
  uint64_t staging_address_ = 0;
  // Case-owned GPU queue, destroyed before any reachable backing.
  amdf_user_queue_t* gpu_queue_ = nullptr;
  // Host producer mapping, destroyed before the queue.
  amdf_user_queue_mapping_t* gpu_mapping_ = nullptr;
  // Cold publication operands for the host producer.
  amdf_user_queue_mapping_info_t gpu_mapping_info_ = {};
  // Monotonic dword index; each batch completes before the next overwrites it.
  uint64_t published_index_ = 0;
};

TEST_P(XdnaPoolVisibilityTest, ReplaysQualifiedGpuXdnaGpuTransitions) {
  ASSERT_NO_FATAL_FAILURE(CreatePool());
  if (IsSkipped()) return;
  ASSERT_NO_FATAL_FAILURE(CreateGpuQueue());
  ASSERT_NO_FATAL_FAILURE(PrepareExecution(resolved_bindings_, &first_));
  std::vector<uint32_t> ingress;
  std::vector<uint32_t> egress;
  ASSERT_NO_FATAL_FAILURE(AppendGpuTransition(gpu_acquire_, &egress));
  for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
    AppendGpuCopy(staging_address_ + ordinal * kBindingByteLength,
                  gpu_addresses_[ordinal] + kBindingByteOffset,
                  kBindingByteLength, &ingress);
    AppendGpuCopy(gpu_addresses_[ordinal],
                  staging_address_ + kReadbackByteOffset +
                      ordinal * kBindingStorageByteLength,
                  kBindingStorageByteLength, &egress);
  }
  ASSERT_NO_FATAL_FAILURE(AppendGpuTransition(gpu_release_, &ingress));
  // Retire the GPU readback writes before its host-visible completion marker.
  ASSERT_NO_FATAL_FAILURE(AppendGpuTransition(gpu_release_, &egress));
  for (uint32_t iteration = 0; iteration < 8; ++iteration) {
    SCOPED_TRACE(iteration);
    std::memset(staging_.pointer, 0, kStagingByteLength);
    std::array<BindingValues, 3> expected;
    for (size_t i = 0; i < kElementCount; ++i) {
      expected[0][i] = kValues[(i + iteration) % kElementCount];
      expected[1][i] = kValues[(i * 3 + iteration + 5) % kElementCount];
      expected[2][i] = expected[0][i] * expected[1][i];
      for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
        iree_unaligned_store_le_u32(
            staging_.pointer + ordinal * kBindingByteLength + i * 4,
            ordinal == 2 ? ~expected[2][i] : expected[ordinal][i]);
      }
    }
    ASSERT_EQ(api_->host_mapping_cache_control(staging_.mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, kStagingByteLength),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(RunGpu(ingress, iteration * 2 + 1));
    const auto* command =
        iteration == 0 ? &first_.initialization : &first_.continuation;
    ASSERT_NO_FATAL_FAILURE(RunExecution(first_, command));
    ASSERT_NO_FATAL_FAILURE(RunGpu(egress, iteration * 2 + 2));
    // The CPU has not touched or maintained the shared payload since setup.
    // Readback covers guards as well as products; poison prevents stale output
    // from passing and GPU writes make the consumer's L2 lines nontrivial.
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            staging_.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
            kReadbackByteOffset, bindings_.size() * kBindingStorageByteLength),
        AMDF_STATUS_OK);
    for (size_t ordinal = 0; ordinal < bindings_.size(); ++ordinal) {
      SCOPED_TRACE(ordinal);
      const auto* readback = staging_.pointer + kReadbackByteOffset +
                             ordinal * kBindingStorageByteLength;
      for (size_t i = 0; i < kBindingByteLength; ++i) {
        ASSERT_EQ(readback[i], kGuardValue) << "prefix " << i;
        ASSERT_EQ(readback[2 * kBindingByteLength + i], kGuardValue)
            << "suffix " << i;
      }
      for (size_t i = 0; i < kElementCount; ++i) {
        ASSERT_EQ(
            iree_unaligned_load_le_u32(readback + kBindingByteOffset + i * 4),
            expected[ordinal][i])
            << "element " << i;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    PoolBacking, XdnaPoolVisibilityTest,
    ::testing::Values(AMDF_MEMORY_PROFILE_ROLE_CREATE,
                      AMDF_MEMORY_PROFILE_ROLE_REGISTER),
    [](const ::testing::TestParamInfo<amdf_memory_profile_roles_t>& info) {
      return info.param == AMDF_MEMORY_PROFILE_ROLE_CREATE ? "Allocated"
                                                           : "Registered";
    });

INSTANTIATE_TEST_SUITE_P(
    MemoryBacking, XdnaExecutionTest,
    ::testing::Values(AMDF_MEMORY_PROFILE_ROLE_CREATE,
                      AMDF_MEMORY_PROFILE_ROLE_REGISTER,
                      AMDF_MEMORY_PROFILE_ROLE_IMPORT),
    [](const ::testing::TestParamInfo<amdf_memory_profile_roles_t>& info) {
      switch (info.param) {
        case AMDF_MEMORY_PROFILE_ROLE_CREATE:
          return "Allocated";
        case AMDF_MEMORY_PROFILE_ROLE_REGISTER:
          return "Registered";
        default:
          return "Imported";
      }
    });

}  // namespace
