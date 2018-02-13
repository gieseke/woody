/*
 * tree.h
 *
 *  Created on: 25.10.2014
 *      Author: fgieseke
 */

#ifndef FOREST_STANDARD_INCLUDE_TREE_H_
#define FOREST_STANDARD_INCLUDE_TREE_H_

#include "global.h"

/* --------------------------------------------------------------------------------
 * Initializes a forest
 * --------------------------------------------------------------------------------
 */
inline void init_forest(FOREST *forest, int n_trees);

/* --------------------------------------------------------------------------------
 * Frees memory allocated for a forest
 * --------------------------------------------------------------------------------
 */
inline void free_forest(FOREST *forest, int free_trees, int free_forest);

/* --------------------------------------------------------------------------------
 * Initializes a single tree
 * --------------------------------------------------------------------------------
 */
inline void init_tree(TREE *tree, int n_allocated);

/* --------------------------------------------------------------------------------
 * Frees memory allocated for a single tree
 * --------------------------------------------------------------------------------
 */
inline void free_tree(TREE *tree);

/* --------------------------------------------------------------------------------
 * Returns a node based on given node id
 * --------------------------------------------------------------------------------
 */
inline TREE_NODE* get_node(TREE *tree, int node_id);

/* --------------------------------------------------------------------------------
 * Adds a node to a given tree
 * --------------------------------------------------------------------------------
 */
int add_node_to_tree(TREE *tree, int parent_id, int is_left_child);

/* --------------------------------------------------------------------------------
 * Attaches tree to a leaf of another tree
 * --------------------------------------------------------------------------------
 */
int attach_tree(TREE *tree, TREE *subtree, int leaf_id);

/* --------------------------------------------------------------------------------
 * Generates an internal node
 * --------------------------------------------------------------------------------
 */
inline int generate_internal_tree_node(TREE *tree, int parent_id, int is_left_child,
		int is_leaf, int feature, FLOAT_TYPE threshold, int node_samples);

/* --------------------------------------------------------------------------------
 * Generates a single leaf
 * --------------------------------------------------------------------------------
 */
inline int generate_tree_leaf(TREE *tree, int parent_id, int is_left_child,
		FLOAT_TYPE leaf_value, unsigned int leaf_criterion);

/* --------------------------------------------------------------------------------
 * Initializes node entries for internal node
 * --------------------------------------------------------------------------------
 */
void init_internal_tree_node(TREE_NODE *node, int parent_id, int feature,
		FLOAT_TYPE threshold, int node_samples);

/* --------------------------------------------------------------------------------
 * Initializes node entries for leaf node
 * --------------------------------------------------------------------------------
 */
void init_tree_leaf(TREE_NODE *node, int parent_id, FLOAT_TYPE leaf_value, unsigned int leaf_criterion);

#endif /* FOREST_STANDARD_INCLUDE_TREE_H_ */
