/*
 * base.h
 */

#ifndef FORESTS_STANDARD_INCLUDE_BASE_H_
#define FORESTS_STANDARD_INCLUDE_BASE_H_

#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "util.h"
#include "global.h"
#include "cpu.h"
#include "../cpu/include/base.h"

#if USE_GPU > 0
#include "../gpu/include/base.h"
#endif


/* --------------------------------------------------------------------------------
 * Initialize (extern)
 * --------------------------------------------------------------------------------
 */
void init_extern(int seed,
		int n_estimators,
		int min_samples_split,
		int max_features,
		int bootstrap,
		int max_depth,
		int min_samples_leaf,
		int learning_type,
		int criterion,
		int tree_traversal_mode,
		int leaf_stopping_mode,
		int tree_type,
		int num_threads,
		int verbosity_level,
		int patterns_transposed,
		PARAMETERS *params,
		FOREST *forest);

/* --------------------------------------------------------------------------------
 * Fit forest (extern)
 * --------------------------------------------------------------------------------
 */
void fit_extern(FLOAT_TYPE *Xtrain,
		int nXtrain,
		int dXtrain,
		FLOAT_TYPE *Ytrain,
		int nYtrain,
		int *bootstrap_indices,
		int nbootstrap_indices,
		int dbootstrap_indices,
		int *bootstrap_indices_weights,
		int nbootstrap_indices_weights,
		int dbootstrap_indices_weights,
		int use_bindices,
		PARAMETERS *params,
		FOREST *forest);

/* --------------------------------------------------------------------------------
 * Compute predictions (extern)
 * --------------------------------------------------------------------------------
 */
void predict_extern(FLOAT_TYPE *Xtest,
		int nXtest,
		int dXtest,
		FLOAT_TYPE *predictions,
		int npredictions,
		int *indices,
		int nindices,
		int dindices,
		PARAMETERS *params,
		FOREST *forest);

void predict_all_extern(FLOAT_TYPE *Xtest,
		int nXtest,
		int dXtest,
		FLOAT_TYPE *preds,
		int npreds,
		int dpreds,
		int *indices,
		int nindices,
		int dindices,
		PARAMETERS *params,
		FOREST *forest);
		
/* --------------------------------------------------------------------------------
 * Frees resources (extern)
 * --------------------------------------------------------------------------------
 */
void free_resources_extern(PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Returns number of bytes used for forest (extern)
 * --------------------------------------------------------------------------------
 */
long get_num_bytes_forest_extern(PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Returns forest as array of integers (extern)
 * --------------------------------------------------------------------------------
 */
void get_forest_as_array_extern(PARAMETERS *params, FOREST *forest, int *aforest, int naforest);

/* --------------------------------------------------------------------------------
 * Restores a forest based on an array of integers (extern)
 * --------------------------------------------------------------------------------
 */
void restore_forest_from_array_extern(PARAMETERS *params, FOREST *forest, int *aforest, int naforest);

/* --------------------------------------------------------------------------------
 * Stores a forest to a file (extern)
 * --------------------------------------------------------------------------------
 */
void save_forest_extern(PARAMETERS *params, FOREST *forest, char *fname);

/* --------------------------------------------------------------------------------
 * Loads a forest from a file (extern)
 * --------------------------------------------------------------------------------
 */
void load_forest_extern(PARAMETERS *params, FOREST *forest, char *fname);

/* --------------------------------------------------------------------------------
 * Returns a single tree of the forest (extern)
 * --------------------------------------------------------------------------------
 */
void get_tree_extern(TREE *tree, unsigned int index, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Returns a single node of a given tree (extern)
 * --------------------------------------------------------------------------------
 */
void get_tree_node_extern(TREE *tree, int node_id, TREE_NODE *node);

/* --------------------------------------------------------------------------------
 * Attaches tree to a leaf of another tree
 * --------------------------------------------------------------------------------
 */
void attach_tree_extern(unsigned int index, FOREST *forest, TREE *subtree, int leaf_id);

/* --------------------------------------------------------------------------------
 * Prints parameters (extern)
 * --------------------------------------------------------------------------------
 */
void print_parameters_extern(PARAMETERS *params);

#endif /* FORESTS_STANDARD_INCLUDE_BASE_H_ */

