/*
 * base.c
 */
#include "include/base.h"

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
		FOREST *forest) {

	int i;

	set_default_parameters(params);

	params->n_estimators = n_estimators;
	params->min_samples_split = min_samples_split;
	params->max_features = max_features;
	params->bootstrap = bootstrap;
	params->max_depth = max_depth;
	params->min_samples_leaf = min_samples_leaf;
	params->tree_traversal_mode = tree_traversal_mode;
	params->leaf_stopping_mode = leaf_stopping_mode;
	params->num_threads = num_threads;
	params->verbosity_level = verbosity_level;
	params->seed = seed;
	params->criterion = criterion;
	params->learning_type = learning_type;
	params->tree_type = tree_type;
	params->n_subset_check = -1;
	params->patterns_transposed = patterns_transposed;


	check_parameters(params);

	srand(params->seed);

	// initialize timers and certain parameters
	INIT(params, forest);
	for (i = 0; i < 10; i++) {
		INIT_MY_TIMER(params->timers + i);
	}
	params->max_ytrain_value = MIN_FLOAT_TYPE;

}

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
		FOREST *forest) {

	PRINT(params)("\nFitting forest ...\n");
	START_MY_TIMER(params->timers + 3);

	fit_forest(Xtrain, nXtrain, dXtrain, Ytrain, bootstrap_indices,
			bootstrap_indices_weights, nbootstrap_indices, dbootstrap_indices,
			use_bindices, params, forest);
	INIT_AFTER_FITTING(params, forest);

	STOP_MY_TIMER(params->timers + 3);
	PRINT(params)("Fitting time (extern): \t\t\t\t\t\t\t\t\t%2.10f\n",
	GET_MY_TIMER(params->timers + 3));

}

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
		FOREST *forest) {

	PRINT(params)("Computing predictions ...\n");

	PREDICT(Xtest, nXtest, dXtest, predictions, indices, dindices, params, forest);

	PRINT(params)("Prediction time (extern): \t\t\t\t\t\t\t\t%2.10f\n",
	GET_MY_TIMER(params->timers + 1));
	PRINT(params)(" -> Tree queries: \t\t\t\t\t\t\t\t\t%2.10f\n",
	GET_MY_TIMER(params->timers + 2));

}

/* --------------------------------------------------------------------------------
 * Compute predictions (extern)
 * --------------------------------------------------------------------------------
 */
void predict_all_extern(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
		FLOAT_TYPE *preds, int npreds, int dpreds, int *indices, int nindices,
		int dindices, PARAMETERS *params, FOREST *forest) {

	PRINT(params)("Computing all predictions ...\n");

	cpu_query_forest_all_preds(Xtest, nXtest, dXtest, preds, npreds, dpreds, indices, dindices, params, forest);

	PRINT(params)("Prediction time (extern): \t\t\t\t\t\t\t\t%2.10f\n",
	GET_MY_TIMER(params->timers + 1));
	PRINT(params)(" -> Tree queries: \t\t\t\t\t\t\t\t\t%2.10f\n",
	GET_MY_TIMER(params->timers + 2));

#if USE_GPU > 0
	PRINT(params)(" --> Kernel query trees: \t\t\t\t\t\t\t\t%2.10f\n", GET_MY_TIMER(params->timers + 3));
#endif

}

/* --------------------------------------------------------------------------------
 * Frees resources (extern)
 * --------------------------------------------------------------------------------
 */
void free_resources_extern(PARAMETERS *params, FOREST *forest) {

	FREE_RESOURCES(params, forest);

}

/* --------------------------------------------------------------------------------
 * Returns number of bytes used for forest (extern)
 * --------------------------------------------------------------------------------
 */
long get_num_bytes_forest_extern(PARAMETERS *params, FOREST *forest) {

	int b, n_bytes;

	n_bytes = 0;
	n_bytes += sizeof(PARAMETERS);
	n_bytes += sizeof(FOREST);

	for (b = 0; b < forest->n_trees; b++) {
		n_bytes += sizeof(TREE);
		n_bytes += forest->trees[b].n_allocated * sizeof(TREE_NODE);
	}

	return n_bytes;

}

/* --------------------------------------------------------------------------------
 * Returns forest as array of integers (extern)
 * --------------------------------------------------------------------------------
 */
void get_forest_as_array_extern(PARAMETERS *params, FOREST *forest,
		int *aforest, int naforest) {

	int b;
	int counter = 0;

	// cast pointer
	void *aforest_casted = (void*) aforest;

	memcpy(aforest_casted + counter, params, sizeof(PARAMETERS));
	counter += sizeof(PARAMETERS);

	memcpy(aforest_casted + counter, forest, sizeof(FOREST));
	counter += sizeof(FOREST);

	// store all trees
	for (b = 0; b < forest->n_trees; b++) {

		memcpy(aforest_casted + counter, forest->trees + b, sizeof(TREE));
		counter += sizeof(TREE);

		memcpy(aforest_casted + counter, forest->trees[b].root,
				forest->trees[b].n_allocated * sizeof(TREE_NODE));
		counter += forest->trees[b].n_allocated * sizeof(TREE_NODE);

	}

}

/* --------------------------------------------------------------------------------
 * Restores a forest based on an array of integers (extern)
 * --------------------------------------------------------------------------------
 */
void restore_forest_from_array_extern(PARAMETERS *params, FOREST *forest,
		int *aforest, int naforest) {

	int b;
	int counter = 0;

	// cast pointer
	void *aforest_casted = (void*) aforest;

	memcpy(params, aforest_casted + counter, sizeof(PARAMETERS));
	counter += sizeof(PARAMETERS);

	memcpy(forest, aforest_casted + counter, sizeof(FOREST));
	counter += sizeof(FOREST);

	// load all trees
	forest->trees = (TREE*) malloc(forest->n_trees * sizeof(TREE));
	for (b = 0; b < forest->n_trees; b++) {

		memcpy(forest->trees + b, aforest_casted + counter, sizeof(TREE));
		counter += sizeof(TREE);

		forest->trees[b].root = (TREE_NODE*) malloc(
				forest->trees[b].n_allocated * sizeof(TREE_NODE));
		memcpy(forest->trees[b].root, aforest_casted + counter,
				forest->trees[b].n_allocated * sizeof(TREE_NODE));
		counter += forest->trees[b].n_allocated * sizeof(TREE_NODE);

	}

}

/* --------------------------------------------------------------------------------
 * Stores a forest to a file (extern)
 * --------------------------------------------------------------------------------
 */
void save_forest_extern(PARAMETERS *params, FOREST *forest, char *fname) {

	int b;

	FILE *ofile = fopen(fname, "w");

	if (ofile != NULL) {

		fwrite(params, sizeof(PARAMETERS), 1, ofile);
		fwrite(forest, sizeof(FOREST), 1, ofile);

		// store all trees
		for (b = 0; b < forest->n_trees; b++) {
			fwrite(forest->trees + b, sizeof(TREE), 1, ofile);
			// store all nodes
			fwrite(forest->trees[b].root,
					forest->trees[b].n_allocated * sizeof(TREE_NODE), 1, ofile);
		}

		fclose(ofile);

	}

}

/* --------------------------------------------------------------------------------
 * Loads a forest from a file (extern)
 * --------------------------------------------------------------------------------
 */
void load_forest_extern(PARAMETERS *params, FOREST *forest, char *fname) {

	int b;
	int retvals = 0;

	FILE *ifile = fopen(fname, "r");

	if (ifile != NULL) {

		retvals = fread(params, sizeof(PARAMETERS), 1, ifile);
		retvals = fread(forest, sizeof(FOREST), 1, ifile);

		// load all trees
		for (b = 0; b < forest->n_trees; b++) {
			retvals = fread(forest->trees + b, sizeof(TREE), 1, ifile);
			// load all nodes
			retvals = fread(forest->trees[b].root,
					forest->trees[b].n_allocated * sizeof(TREE_NODE), 1, ifile);

		}

		fclose(ifile);
	}

	if (retvals < 0) {
		printf("Error while loading forest from file!\n");
		exit(EXIT_FAILURE);
	}
}

/* --------------------------------------------------------------------------------
 * Returns a single tree of the forest (extern)
 * --------------------------------------------------------------------------------
 */
void get_tree_extern(TREE *tree, unsigned int index, FOREST *forest) {

	TREE *t = forest->trees + index;
	tree->root = t->root;
	tree->n_allocated = t->n_allocated;
	tree->node_counter = t->node_counter;

}

/* --------------------------------------------------------------------------------
 * Returns a single node of a given tree (extern)
 * --------------------------------------------------------------------------------
 */
void get_tree_node_extern(TREE *tree, int node_id, TREE_NODE *node) {

	node->feature = tree->root[node_id].feature;
	node->left_id = tree->root[node_id].left_id;
	node->right_id = tree->root[node_id].right_id;
	node->thres_or_leaf = tree->root[node_id].thres_or_leaf;
	node->leaf_criterion = tree->root[node_id].leaf_criterion;

}

/* --------------------------------------------------------------------------------
 * Attaches tree to a leaf of another tree
 * --------------------------------------------------------------------------------
 */
void attach_tree_extern(unsigned int index, FOREST *forest, TREE *subtree,
		int leaf_id) {

	TREE *tree = forest->trees + index;
	attach_tree(tree, subtree, leaf_id);

}

/* --------------------------------------------------------------------------------
 * 	s parameters (extern)
 * --------------------------------------------------------------------------------
 */
void print_parameters_extern(PARAMETERS *params) {

	printf(
			"=================================== Parameter Settings ===================================\n");
	printf("Initial seed (seed): %i\n", params->seed);
	printf("Number of estimators (n_estimators): %i\n", params->n_estimators);
	printf("Number of minimum samples per split (min_samples_split): %i\n",
			params->min_samples_split);
	printf("Number of maximum features (max_features): %i\n",
			params->max_features);
	printf("Make use of bootstrap indices (bootstrap): %i\n",
			params->bootstrap);
	printf("Maximum depth of tree (max_depth): %i\n", params->max_depth);
	printf("Minimum number of samples per leaf (min_samples_leaf): %i\n",
			params->min_samples_leaf);
	printf("Number of threads for CPU (num_threads): %i\n",
			params->num_threads);
	printf("Level of verbosity (verbosity_level): %i\n",
			params->verbosity_level);
	printf("Tree traversal mode (tree_traversal_mode): %i\n",
			params->tree_traversal_mode);
	printf("Criterion used for computing splits: %i\n", params->criterion);
	printf("Learning type: %i\n", params->learning_type);
	printf("Tree type (tree_type): %i\n", params->tree_type);
	printf("Double precision? (USE_DOUBLE): %i\n", USE_DOUBLE);
	printf(
			"==========================================================================================\n");

}
