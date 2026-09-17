// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/consumption.h"

#include <string.h>

#include "loom/ir/module.h"

static void loom_consumption_bitset_set(uint64_t* bits,
                                        iree_host_size_t word_count,
                                        iree_host_size_t bit_index) {
  const iree_host_size_t word_index = bit_index / 64u;
  IREE_ASSERT_LT(word_index, word_count);
  bits[word_index] |= ((uint64_t)1) << (bit_index % 64u);
}

static bool loom_consumption_bitset_test(const uint64_t* bits,
                                         iree_host_size_t word_count,
                                         iree_host_size_t bit_index) {
  const iree_host_size_t word_index = bit_index / 64u;
  IREE_ASSERT_LT(word_index, word_count);
  return (bits[word_index] & (((uint64_t)1) << (bit_index % 64u))) != 0;
}

static iree_host_size_t loom_consumption_bitset_word_count(
    iree_host_size_t bit_count) {
  return (bit_count + 63u) / 64u;
}

static bool loom_consumption_region_is_cfg(const loom_region_t* region) {
  return region &&
         iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG);
}

void loom_consumption_region_query_initialize(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_consumption_region_query_t* out_query) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(region);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_query);
  memset(out_query, 0, sizeof(*out_query));
  out_query->module = module;
  out_query->region = region;
  out_query->arena = arena;
}

void loom_consumption_region_query_initialize_with_cfg_graph(
    const loom_module_t* module, const loom_region_t* region,
    const loom_cfg_graph_t* cfg_graph, const loom_liveness_analysis_t* liveness,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_consumption_region_query_t* out_query) {
  loom_consumption_region_query_initialize(module, region, arena, out_query);
  IREE_ASSERT_ARGUMENT(cfg_graph);
  IREE_ASSERT(cfg_graph->module == module);
  IREE_ASSERT(cfg_graph->region == region);
  IREE_ASSERT_EQ(cfg_graph->block_count, region->block_count);
  out_query->cfg_graph = *cfg_graph;
  out_query->cfg_graph_ready = true;
  out_query->liveness = liveness;
  out_query->value_domain = value_domain;
}

static iree_status_t loom_consumption_region_query_cfg_graph(
    loom_consumption_region_query_t* query,
    const loom_cfg_graph_t** out_graph) {
  *out_graph = NULL;
  if (!loom_consumption_region_is_cfg(query->region)) {
    return iree_ok_status();
  }
  if (!query->cfg_graph_ready) {
    IREE_RETURN_IF_ERROR(loom_cfg_graph_build(query->module, query->region,
                                              query->arena, &query->cfg_graph));
    query->cfg_graph_ready = true;
  }
  *out_graph = &query->cfg_graph;
  return iree_ok_status();
}

static iree_status_t loom_consumption_region_query_prepare_cfg_search(
    loom_consumption_region_query_t* query, iree_host_size_t block_count,
    iree_host_size_t* out_visited_word_count) {
  iree_host_size_t visited_word_count =
      loom_consumption_bitset_word_count(block_count);
  for (iree_host_size_t i = 0; i < query->visited_block_count; ++i) {
    const uint16_t block_index = query->visited_blocks[i];
    const uint64_t mask = ((uint64_t)1) << (block_index % 64u);
    query->visited_bits[block_index / 64u] &= ~mask;
    query->reachable_bits[block_index / 64u] &= ~mask;
  }
  query->visited_block_count = 0;
  query->pending_block_count = 0;
  if (visited_word_count > query->visited_word_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, visited_word_count, sizeof(*query->visited_bits),
        &query->visited_word_capacity, (void**)&query->visited_bits));
    memset(query->visited_bits, 0,
           visited_word_count * sizeof(query->visited_bits[0]));
  }
  if (visited_word_count > query->reachable_word_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, visited_word_count, sizeof(*query->reachable_bits),
        &query->reachable_word_capacity, (void**)&query->reachable_bits));
    memset(query->reachable_bits, 0,
           visited_word_count * sizeof(query->reachable_bits[0]));
  }
  if (block_count > query->visited_block_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, block_count, sizeof(*query->visited_blocks),
        &query->visited_block_capacity, (void**)&query->visited_blocks));
  }
  if (block_count > query->block_heap_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        query->arena, 0, block_count, sizeof(*query->block_heap),
        &query->block_heap_capacity, (void**)&query->block_heap));
  }
  *out_visited_word_count = visited_word_count;
  return iree_ok_status();
}

static void loom_consumption_cfg_search_push(
    loom_consumption_region_query_t* query, uint16_t block_index,
    iree_host_size_t word_count) {
  const loom_cfg_graph_t* graph = &query->cfg_graph;
  if (!loom_cfg_graph_block_is_reachable(graph, block_index) ||
      loom_consumption_bitset_test(query->visited_bits, word_count,
                                   block_index)) {
    return;
  }
  loom_consumption_bitset_set(query->visited_bits, word_count, block_index);
  query->visited_blocks[query->visited_block_count++] = block_index;
  iree_host_size_t position = query->pending_block_count++;
  const uint16_t component = graph->blocks[block_index].component;
  while (position != 0) {
    const iree_host_size_t parent = (position - 1) / 2;
    const uint16_t parent_block = query->block_heap[parent];
    if (graph->blocks[parent_block].component >= component) break;
    query->block_heap[position] = parent_block;
    position = parent;
  }
  query->block_heap[position] = block_index;
}

static uint16_t loom_consumption_cfg_search_pop(
    loom_consumption_region_query_t* query) {
  const loom_cfg_graph_t* graph = &query->cfg_graph;
  const uint16_t block_index = query->block_heap[0];
  const uint16_t last = query->block_heap[--query->pending_block_count];
  iree_host_size_t position = 0;
  while (position * 2 + 1 < query->pending_block_count) {
    iree_host_size_t child = position * 2 + 1;
    if (child + 1 < query->pending_block_count &&
        graph->blocks[query->block_heap[child + 1]].component >
            graph->blocks[query->block_heap[child]].component) {
      ++child;
    }
    if (graph->blocks[last].component >=
        graph->blocks[query->block_heap[child]].component) {
      break;
    }
    query->block_heap[position] = query->block_heap[child];
    position = child;
  }
  query->block_heap[position] = last;
  return block_index;
}

static bool loom_consumption_search_cfg_reachability(
    loom_consumption_use_after_query_t* query, uint16_t target_index) {
  loom_consumption_region_query_t* region_query = query->region_query;
  const loom_cfg_graph_t* graph = &region_query->cfg_graph;
  const iree_host_size_t word_count = query->reachable_word_count;
  if (!query->search_initialized) {
    const loom_block_t* consuming_block = query->consuming_op->parent_block;
    const uint16_t consuming_index =
        (uint16_t)loom_cfg_graph_block_index(graph, consuming_block);
    if (query->recreation_block == consuming_block) {
      loom_consumption_bitset_set(region_query->visited_bits, word_count,
                                  consuming_index);
      region_query->visited_blocks[region_query->visited_block_count++] =
          consuming_index;
    }
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, consuming_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      loom_consumption_cfg_search_push(region_query, successors.values[i],
                                       word_count);
    }
    query->search_initialized = true;
  }
  if (loom_consumption_bitset_test(region_query->reachable_bits, word_count,
                                   target_index)) {
    return true;
  }

  // Inter-component edges only decrease the retained component ordinal. Once
  // the highest pending component is below the target, no unexpanded path can
  // reach it. Keep that frontier for subsequent queries of later components.
  const uint16_t target_component = graph->blocks[target_index].component;
  while (region_query->pending_block_count != 0) {
    if (graph->blocks[region_query->block_heap[0]].component <
        target_component) {
      break;
    }
    const uint16_t block_index = loom_consumption_cfg_search_pop(region_query);
    if (graph->blocks[block_index].block == query->recreation_block) continue;
    loom_consumption_bitset_set(region_query->reachable_bits, word_count,
                                block_index);
    const loom_cfg_block_index_span_t successors =
        loom_cfg_graph_successors(graph, block_index);
    for (iree_host_size_t i = 0; i < successors.count; ++i) {
      loom_consumption_cfg_search_push(region_query, successors.values[i],
                                       word_count);
    }
    if (block_index == target_index) return true;
  }
  return false;
}

static const loom_op_t* loom_consumption_region_anchor_op(
    const loom_region_t* region, const loom_op_t* op) {
  const loom_op_t* anchor_op = op;
  while (anchor_op != NULL && anchor_op->parent_block != NULL &&
         anchor_op->parent_block->parent_region != region) {
    anchor_op = anchor_op->parent_op;
  }
  if (anchor_op == NULL || anchor_op->parent_block == NULL ||
      anchor_op->parent_block->parent_region != region) {
    return NULL;
  }
  return anchor_op;
}

iree_status_t loom_consumption_use_after_query_prepare(
    loom_consumption_region_query_t* region_query,
    const loom_op_t* consuming_op, loom_value_id_t value_id,
    loom_consumption_use_after_query_t* out_query) {
  if (!region_query || !consuming_op || !out_query) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "consumption use-after query requires region query, operation, and "
        "output query");
  }
  *out_query = (loom_consumption_use_after_query_t){
      .region_query = region_query,
      .consuming_op = consuming_op,
      .value_id = value_id,
  };
  if (!region_query->module || !region_query->region || !region_query->arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "consumption region query is not initialized");
  }
  if (!consuming_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "consuming op must belong to a block");
  }
  if (consuming_op->parent_block->parent_region != region_query->region) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "consumption query must describe the consuming op region");
  }
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= region_query->module->values.count ||
      !loom_consumption_region_is_cfg(region_query->region)) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(region_query->module, value_id);
  if (loom_value_is_block_arg(value)) {
    out_query->recreation_block = loom_value_def_block(value);
  } else {
    const loom_op_t* defining_op = loom_value_def_op(value);
    out_query->recreation_block =
        defining_op != NULL ? defining_op->parent_block : NULL;
  }
  const loom_cfg_graph_t* cfg_graph = NULL;
  IREE_RETURN_IF_ERROR(
      loom_consumption_region_query_cfg_graph(region_query, &cfg_graph));
  if (cfg_graph->malformed) return iree_ok_status();
  if (region_query->liveness != NULL) {
    const loom_liveness_analysis_t* liveness = region_query->liveness;
    const loom_liveness_segment_range_t segments =
        loom_liveness_segment_range_for_value_ordinal(
            liveness, loom_local_value_domain_ordinal(
                          region_query->value_domain, value_id));
    const iree_host_size_t block_index =
        loom_cfg_graph_block_index(cfg_graph, consuming_op->parent_block);
    // Liveness already proves whether any path observes this dynamic value
    // after block exit. Without a live-out, only later same-block uses remain.
    if (!loom_liveness_segment_range_contains(
            liveness->segments, segments,
            liveness->blocks[block_index].end_point)) {
      return iree_ok_status();
    }
  }
  return loom_consumption_region_query_prepare_cfg_search(
      region_query, cfg_graph->block_count, &out_query->reachable_word_count);
}

bool loom_consumption_use_after_query_contains(
    loom_consumption_use_after_query_t* query, loom_use_t use) {
  IREE_ASSERT_ARGUMENT(query);
  IREE_ASSERT_ARGUMENT(query->region_query);
  IREE_ASSERT_ARGUMENT(query->consuming_op);
  loom_consumption_region_query_t* region_query = query->region_query;
  const loom_op_t* use_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  IREE_ASSERT_LT(operand_index, use_op->operand_count);
  IREE_ASSERT_EQ(loom_op_const_operands(use_op)[operand_index],
                 query->value_id);
  const loom_op_t* anchor_op =
      loom_consumption_region_anchor_op(region_query->region, use_op);
  if (anchor_op == NULL) {
    return false;
  }
  const loom_block_t* anchor_block = anchor_op->parent_block;
  const loom_block_t* consuming_block = query->consuming_op->parent_block;
  if (anchor_block == consuming_block &&
      anchor_op->block_ordinal > query->consuming_op->block_ordinal) {
    return true;
  }
  if (query->reachable_word_count == 0) {
    return false;
  }
  const loom_cfg_graph_t* graph = &region_query->cfg_graph;
  const iree_host_size_t block_index =
      loom_cfg_graph_block_index(graph, anchor_block);
  if (!loom_cfg_graph_block_is_reachable(graph, (uint16_t)block_index)) {
    return false;
  }
  if (anchor_block == query->recreation_block) return false;

  const iree_host_size_t consuming_index =
      loom_cfg_graph_block_index(graph, consuming_block);
  const loom_cfg_block_info_t* consuming_info = &graph->blocks[consuming_index];
  const loom_cfg_block_info_t* anchor_info = &graph->blocks[block_index];
  if (!consuming_info->reachable) {
    return loom_consumption_search_cfg_reachability(query,
                                                    (uint16_t)block_index);
  }
  if (anchor_info->component > consuming_info->component) return false;
  const iree_host_size_t cut_index =
      loom_cfg_graph_block_index(graph, query->recreation_block);
  // Blocks in one component reach each other without leaving the component.
  // A simple path does not reenter its source, but another definition block in
  // the component can cut that path. Same-block uses require a real cycle.
  if (anchor_info->component == consuming_info->component &&
      (block_index != consuming_index ||
       (consuming_info->component_is_cyclic && cut_index != consuming_index)) &&
      (cut_index == IREE_HOST_SIZE_MAX || cut_index == consuming_index ||
       graph->blocks[cut_index].component != consuming_info->component)) {
    return true;
  }
  // The earliest reachable DFS node also reaches every node in its subtree.
  // Component order proves when the path to that root cannot cross the value's
  // definition. Otherwise only the consuming block's own tree path is known.
  const loom_cfg_block_info_t* path_root = consuming_info;
  const loom_cfg_block_info_t* reachable_root =
      &graph->blocks[consuming_info->reachability_root];
  if (cut_index == IREE_HOST_SIZE_MAX ||
      graph->blocks[cut_index].component > consuming_info->component ||
      graph->blocks[cut_index].component < reachable_root->component) {
    path_root = reachable_root;
  }
  if (block_index != consuming_index &&
      path_root->preorder <= anchor_info->preorder &&
      anchor_info->preorder < path_root->preorder_end) {
    if (cut_index == IREE_HOST_SIZE_MAX) return true;
    const loom_cfg_block_info_t* cut_info = &graph->blocks[cut_index];
    if (cut_info->preorder <= path_root->preorder ||
        cut_info->preorder > anchor_info->preorder ||
        anchor_info->preorder >= cut_info->preorder_end) {
      return true;
    }
  }
  return loom_consumption_search_cfg_reachability(query, (uint16_t)block_index);
}

iree_status_t loom_consumption_find_use_after(
    loom_consumption_region_query_t* query, const loom_op_t* consuming_op,
    loom_value_id_t value_id, loom_consumption_use_t* out_use,
    bool* out_found) {
  if (!out_found) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "consumption query requires an output flag");
  }
  *out_found = false;
  if (!query || !consuming_op || !out_use) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "consumption query requires query, consuming op, and output use");
  }
  loom_consumption_use_after_query_t use_after_query = {0};
  IREE_RETURN_IF_ERROR(loom_consumption_use_after_query_prepare(
      query, consuming_op, value_id, &use_after_query));
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= query->module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(query->module, value_id);
  const loom_use_t* use_ptr = NULL;
  loom_value_for_each_use(value, use_ptr) {
    if (!loom_consumption_use_after_query_contains(&use_after_query,
                                                   *use_ptr)) {
      continue;
    }
    *out_use = (loom_consumption_use_t){
        .op = loom_use_user_op(*use_ptr),
        .operand_index = loom_use_operand_index(*use_ptr),
    };
    *out_found = true;
    return iree_ok_status();
  }
  return iree_ok_status();
}
