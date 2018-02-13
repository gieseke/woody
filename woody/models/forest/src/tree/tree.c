/*
 * tree.c
 *
 *  Created on: 25.10.2014
 *      Author: fgieseke
 */

#include "include/tree.h"

/* --------------------------------------------------------------------------------
 * Initializes a forest
 * --------------------------------------------------------------------------------
 */
inline void init_forest(FOREST *forest, int n_trees) {

	forest->n_trees = n_trees;
	forest->trees = (TREE*) malloc(n_trees * sizeof(TREE));

}

/* --------------------------------------------------------------------------------
 * Frees memory allocated for a forest
 * --------------------------------------------------------------------------------
 */
inline void free_forest(FOREST *forest, int free_trees, int free_forest) {

	if (forest != NULL) {

		if (free_trees) {

			int b;
			for (b = 0; b < forest->n_trees; b++) {
				free_tree(forest->trees + b);
			}

		}

		free(forest->trees);
		if (free_forest) {
			free(forest);
		}
	}

}

/* --------------------------------------------------------------------------------
 * Initializes a single tree
 * --------------------------------------------------------------------------------
 */
inline void init_tree(TREE *tree, int n_allocated) {

	tree->n_allocated = n_allocated;
	tree->node_counter = 0;
	tree->root = (TREE_NODE*) malloc(tree->n_allocated * sizeof(TREE_NODE));

}

/* --------------------------------------------------------------------------------
 * Frees memory allocated for a single tree
 * --------------------------------------------------------------------------------
 */
inline void free_tree(TREE *tree) {

	free(tree->root);

}

/* --------------------------------------------------------------------------------
 * Returns a node based on given node id
 * --------------------------------------------------------------------------------
 */
inline TREE_NODE* get_node(TREE *tree, int node_id) {

	if (node_id >= tree->node_counter || node_id < 0) {
		printf("Error: node id out of bounds!\n");
		exit(0);
	}

	return tree->root + node_id;

}

/* --------------------------------------------------------------------------------
 * Adds a node to a given tree
 * --------------------------------------------------------------------------------
 */
int add_node_to_tree(TREE *tree, int parent_id, int is_left_child) {

	// get current node counter
	int node_id = tree->node_counter;

	// reallocate space if needed and increment counter
	if (node_id >= tree->n_allocated) {
		tree->n_allocated *= 2;
		tree->root = (TREE_NODE*) realloc(tree->root,
				tree->n_allocated * sizeof(TREE_NODE));
	}

	// set parent's ids accordingly
	if (parent_id != -1) {

		TREE_NODE *parental_node = tree->root + parent_id;

		if (is_left_child) {
			parental_node->left_id = node_id;
		} else {
			parental_node->right_id = node_id;
		}
	}

	// increase node counter
	tree->node_counter += 1;

	return node_id;
}

/* --------------------------------------------------------------------------------
 * Attaches tree to a leaf of another tree
 * --------------------------------------------------------------------------------
 */
int attach_tree(TREE *tree, TREE *subtree, int leaf_id) {

	int i;

	if (subtree->node_counter <= 0) {
		printf("ERROR: Subtree contains no nodes!\n");
		return -1;
	}

	int start_id = tree->node_counter;
	int subtree_size = subtree->node_counter;

	// we replace node leaf_id with the root of the
	// subtree, hence, we need one node less ...
	// reallocate space if needed
	if (tree->node_counter + subtree_size - 1 >= tree->n_allocated) {
		tree->n_allocated += (subtree_size - 1);
		tree->root = (TREE_NODE*) realloc(tree->root,
				tree->n_allocated * sizeof(TREE_NODE));
	}

	// copy nodes (NOT the first one of the subtree)
	memcpy(tree->root + start_id, subtree->root + 1,
			(subtree_size - 1) * sizeof(TREE_NODE));

	// transform internal pointers
	for (i = 0; i < subtree_size - 1; i++) {
		if (tree->root[start_id + i].left_id != TREE_CHILD_ID_NOT_SET){
			tree->root[start_id + i].left_id += start_id - 1;
		}
		if (tree->root[start_id + i].right_id != TREE_CHILD_ID_NOT_SET){
			tree->root[start_id + i].right_id += start_id - 1;
		}
	}

	// adapt leaf node (based on root of subtree)
	TREE_NODE *node = tree->root + leaf_id;
	if (subtree->root[0].left_id != TREE_CHILD_ID_NOT_SET){
		node->left_id = subtree->root[0].left_id + start_id - 1;
	} else {
		node->left_id = TREE_CHILD_ID_NOT_SET;
	}
	if (subtree->root[0].right_id != TREE_CHILD_ID_NOT_SET) {
		node->right_id = subtree->root[0].right_id + start_id - 1;
	} else {
		node->right_id = TREE_CHILD_ID_NOT_SET;
	}
	node->feature = subtree->root[0].feature;
	node->thres_or_leaf = subtree->root[0].thres_or_leaf;
	node->leaf_criterion =  subtree->root[0].leaf_criterion;

	// increment overall counter
	tree->node_counter += subtree_size - 1;

	return 0;

}

/* --------------------------------------------------------------------------------
 * Generates an internal node
 * --------------------------------------------------------------------------------
 */
inline int generate_internal_tree_node(TREE *tree, int parent_id,
		int is_left_child, int is_leaf, int feature, FLOAT_TYPE threshold,
		int node_samples) {

	int node_id = add_node_to_tree(tree, parent_id, is_left_child);
	TREE_NODE *node = get_node(tree, node_id);

	// set node values
	init_internal_tree_node(node, parent_id, feature, threshold, node_samples);

	return node_id;

}

/* --------------------------------------------------------------------------------
 * Generates a single leaf
 * --------------------------------------------------------------------------------
 */
inline int generate_tree_leaf(TREE *tree, int parent_id, int is_left_child,
FLOAT_TYPE leaf_value, unsigned int leaf_criterion) {

	int node_id = add_node_to_tree(tree, parent_id, is_left_child);
	TREE_NODE *node = get_node(tree, node_id);

	// set node values
	init_tree_leaf(node, parent_id, leaf_value, leaf_criterion);

	return node_id;

}

/* --------------------------------------------------------------------------------
 * Initializes node entries for internal node
 * --------------------------------------------------------------------------------
 */
void init_internal_tree_node(TREE_NODE *node, int parent_id, int feature,
FLOAT_TYPE threshold, int node_samples) {

	node->feature = feature;
	node->thres_or_leaf = threshold;
	node->left_id = TREE_CHILD_ID_NOT_SET;
	node->right_id = TREE_CHILD_ID_NOT_SET;
	node->leaf_criterion = LEAF_CRIT_NO_LEAF;

}

/* --------------------------------------------------------------------------------
 * Initializes node entries for leaf node
 * --------------------------------------------------------------------------------
 */
void init_tree_leaf(TREE_NODE *node, int parent_id, FLOAT_TYPE leaf_value,
		unsigned int leaf_criterion) {

	node->thres_or_leaf = leaf_value;
	node->left_id = TREE_CHILD_ID_NOT_SET;
	node->right_id = TREE_CHILD_ID_NOT_SET;
	node->leaf_criterion = leaf_criterion;

}
