/*
 * cpu.c
 *
 *  Created on: 17.04.2015
 *      Author: fgieseke
 */

#include "include/cpu.h"

/* --------------------------------------------------------------------------------
 * Fits a model given the training data (and parameters)
 * --------------------------------------------------------------------------------
 */
void fit_forest(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain,
		FLOAT_TYPE *Ytrain, int *bootstrap_indices, int *bootstrap_indices_weights,
		int n_bootstrap_indices, int d_bootstrap_indices, int use_bindices,
		PARAMETERS *params, FOREST *forest) {

	int b;

	// set max features if needed
	if (params->max_features < 0 || params->max_features > dXtrain) {
		PRINT(params)("Warning: Setting max_features to dXtrain!\n");
		params->max_features = dXtrain;
	}

	params->Xtrain = Xtrain;
	params->nXtrain = nXtrain;
	params->dXtrain = dXtrain;

	// initialize forest and training data
	init_forest(forest, params->n_estimators);

	omp_set_dynamic(0);
	omp_set_num_threads(params->num_threads);

	unsigned int *rstates = (unsigned int*) malloc(params->n_estimators * sizeof(unsigned int));
	for (b = 0; b < params->n_estimators; b++) {
		rstates[b] = rand();
	}

#pragma omp parallel for
	for (b = 0; b < params->n_estimators; b++) {

		TRAINING_DATA *train_data = INIT_TRAINING_DATA(Xtrain, nXtrain, dXtrain, Ytrain, params);

		if (use_bindices) {
			INIT_BINDICES(train_data, nXtrain, &rstates[b],
					bootstrap_indices + b * d_bootstrap_indices,
					bootstrap_indices_weights + b * d_bootstrap_indices,
					d_bootstrap_indices, params);
		} else {
			INIT_BINDICES(train_data, nXtrain, &rstates[b], NULL, NULL, -1, params);
		}

		// compute mapped labels
		train_data->Ytrain_mapped = (FLOAT_TYPE*) malloc(
				train_data->bindices->n_indices * sizeof(FLOAT_TYPE));
		compute_mapped_labels(params, train_data);

		build_single_tree(forest->trees + b, train_data, params, &rstates[b]);

		FREE_BINDICES(train_data, use_bindices);
		free(train_data->Ytrain_mapped);
		FREE_TRAINING_DATA(train_data, params);

	}

	free(rstates);

}

/* --------------------------------------------------------------------------------
 * Builds a single tree.
 * --------------------------------------------------------------------------------
 */
void build_single_tree(TREE *tree, TRAINING_DATA *train_data,
		PARAMETERS *params, unsigned int *rstate) {

	init_tree(tree, 128);
	PQUEUE *traversal_queue = pqueue_new(10);

	TRAVERSAL_RECORD *root_record = init_traversal_record(0,
			train_data->bindices->n_indices, train_data->dXtrain, 0,
			TREE_ROOT_PARENT_ID, NO_LEFT_CHILD, NO_LEAF);
	init_const_features_array(root_record->const_features, train_data->dXtrain);

	pqueue_push(traversal_queue, (void*) root_record, 0);
	process_all_nodes(tree, train_data, traversal_queue, params, rstate);

	free(traversal_queue);

}

/* --------------------------------------------------------------------------------
 * Process huge nodes
 * --------------------------------------------------------------------------------
 */
void process_all_nodes(TREE *tree, TRAINING_DATA *train_data,
		PQUEUE *traversal_queue, PARAMETERS *params, unsigned int *rstate) {

	int j, n_trecords;

	while (!pqueue_is_empty(traversal_queue)) {

		// get and remove single record
		TRAVERSAL_RECORD **trecords = get_chunk_trecords(traversal_queue, &n_trecords, 1);

		// compute best splits (results are stored in trecords)
		COMPUTE_SPLITS(trecords, n_trecords, train_data, params, rstate);

		// generate internal nodes/leaves
		generate_leaves_nodes(tree, trecords, n_trecords, train_data, params, traversal_queue);

		for (j = 0; j < n_trecords; j++) {
			free_traversal_record(trecords[j]);
		}
		free(trecords);

	}

}

/* --------------------------------------------------------------------------------
 * Returns a chunk of traversal records.
 * --------------------------------------------------------------------------------
 */
TRAVERSAL_RECORD **get_chunk_trecords(PQUEUE *traversal_queue, int *n_trecords,
		int n_to_be_removed) {

	int j;

	// determine number of records that should be taken
	int n_remove = n_to_be_removed;

	// get traversal records from queue
	TRAVERSAL_RECORD **trecords = (TRAVERSAL_RECORD**) malloc(
			n_remove * sizeof(TRAVERSAL_RECORD*));
	for (j = 0; j < n_remove; j++) {
		trecords[j] = pqueue_pop(traversal_queue, NULL);
	}

	// update flag and return pointer
	*n_trecords = n_remove;

	return trecords;

}

/* --------------------------------------------------------------------------------
 * Generate traversal records for the children of a given traversal record
 * --------------------------------------------------------------------------------
 */
void generate_traversal_records_children(TREE *tree, TRAINING_DATA *train_data,
		PQUEUE *huge_traversal_queue, TRAVERSAL_RECORD *trecord,
		PARAMETERS *params, int node_id) {

	int start = trecord->start;
	int end = trecord->end;
	int depth = trecord->depth;
	SPLIT_RECORD *split_record = trecord->split_record;

	// split bootstrap indices
	cpu_partition_bindinces(train_data->bindices->indices,
			train_data->bindices->indices_wmappings, train_data->Ytrain_mapped,
			start, end, split_record->threshold, split_record->feature, params,
			train_data);

	// compute priorites based on traversal mode
	int prio_left, prio_right;

	if (params->tree_traversal_mode == TREE_TRAVERSAL_MODE_DFS) {

		prio_right = -depth;
		prio_left = -(depth + 1);

	} else if (params->tree_traversal_mode == TREE_TRAVERSAL_MODE_NODE_SIZE) {

		prio_right = -(end - split_record->pos);
		prio_left = -(split_record->pos - start);

	} else if (params->tree_traversal_mode == TREE_TRAVERSAL_MODE_PROB) {

		if (split_record->prob_left > split_record->prob_right) {
			prio_left = -(depth + 1);
			prio_right = -depth;
		} else {
			prio_left = -depth;
			prio_right = -(depth + 1);
		}

	} else {

		printf("Error: Unknown traversal mode! Exiting ...");
		exit(-1);

	}

	// early leaf check
	unsigned int right_leaf_criterion = LEAF_CRIT_NO_LEAF;
	int right_n_samples = end - split_record->pos;

	if (right_n_samples < params->min_samples_split){
		right_leaf_criterion = LEAF_CRIT_MIN_SAMPLES_SPLIT;
	}
	if (right_n_samples < 2 * params->min_samples_leaf){
		right_leaf_criterion = LEAF_CRIT_MIN_SAMPLES_LEAF;
	}
	if (split_record->impurity_right <= MIN_IMPURITY_SPLIT){
		if (params->leaf_stopping_mode != LEAF_MODE_STOP_IGNORE_IMPURITY){
			right_leaf_criterion = LEAF_CRIT_MIN_IMPURITY;
		}
	}

	// early leaf check
	unsigned int left_leaf_criterion = LEAF_CRIT_NO_LEAF;
	int left_n_samples = split_record->pos - start;

	if (left_n_samples < params->min_samples_split){
		left_leaf_criterion = LEAF_CRIT_MIN_SAMPLES_SPLIT;
	}
	if (left_n_samples < 2 * params->min_samples_leaf){
		left_leaf_criterion = LEAF_CRIT_MIN_SAMPLES_LEAF;
	}
	if (split_record->impurity <= MIN_IMPURITY_SPLIT){
		if (params->leaf_stopping_mode != LEAF_MODE_STOP_IGNORE_IMPURITY){
			left_leaf_criterion = LEAF_CRIT_MIN_IMPURITY;
		}
	}

	if (split_record->prob_left > split_record->prob_right) {
		generate_next_leaf_node(node_id, start, split_record->pos, left_leaf_criterion,
				depth, prio_left, LEFT_CHILD, tree, huge_traversal_queue,
				trecord, train_data, params);
		generate_next_leaf_node(node_id, split_record->pos, end, right_leaf_criterion,
				depth, prio_right, NO_LEFT_CHILD, tree, huge_traversal_queue,
				trecord, train_data, params);
	} else {

		generate_next_leaf_node(node_id, split_record->pos, end, right_leaf_criterion,
				depth, prio_right, NO_LEFT_CHILD, tree, huge_traversal_queue,
				trecord, train_data, params);
		generate_next_leaf_node(node_id, start, split_record->pos, left_leaf_criterion,
				depth, prio_left, LEFT_CHILD, tree, huge_traversal_queue,
				trecord, train_data, params);
	}

}

void generate_next_leaf_node(int node_id, int start, int end, unsigned int leaf_criterion,
		int depth, int prio, int child_flag, TREE *tree,
		PQUEUE *huge_traversal_queue, TRAVERSAL_RECORD *trecord,
		TRAINING_DATA *train_data, PARAMETERS *params) {

	if (leaf_criterion != LEAF_CRIT_NO_LEAF) {

		generate_leaf(tree, start, end, node_id, child_flag, leaf_criterion, train_data, params);

	} else {

		TRAVERSAL_RECORD *record = init_traversal_record(start, end,
				train_data->dXtrain, depth + 1, node_id, child_flag, NO_LEAF);
		pqueue_push(huge_traversal_queue, (void*) record, prio);
		update_const_features_array(record->const_features,
				trecord->const_features, train_data->dXtrain);

	}

}

/* --------------------------------------------------------------------------------
 * Generates leaves and nodes
 * --------------------------------------------------------------------------------
 */
void generate_leaves_nodes(TREE *tree, TRAVERSAL_RECORD **trecords,
		int n_trecords, TRAINING_DATA *train_data, PARAMETERS *params,
		PQUEUE *huge_traversal_queue) {

	int i;

	for (i = 0; i < n_trecords; i++) {

		int n_node_samples = trecords[i]->end - trecords[i]->start;
		unsigned int leaf_criterion = LEAF_CRIT_NO_LEAF;

		if (trecords[i]->split_record->leaf_detected) {
			leaf_criterion = LEAF_CRIT_DETECTED;
		}

		if (trecords[i]->depth >= params->max_depth){
			leaf_criterion = LEAF_CRIT_MAX_DEPTH;
		}

		if (trecords[i]->end - trecords[i]->start < params->min_samples_split){
			leaf_criterion = LEAF_CRIT_MIN_SAMPLES_SPLIT;
		}

		if (trecords[i]->end - trecords[i]->start < 2 * params->min_samples_leaf){
			leaf_criterion = LEAF_CRIT_MIN_SAMPLES_LEAF;
		}

		if (trecords[i]->split_record->impurity <= MIN_IMPURITY_SPLIT){
			if (params->leaf_stopping_mode != LEAF_MODE_STOP_IGNORE_IMPURITY){
				leaf_criterion = LEAF_CRIT_MIN_IMPURITY;
			}
		}

		if (trecords[i]->split_record->pos >= trecords[i]->end){
			leaf_criterion = LEAF_CRIT_POS_END;
		}

		if (leaf_criterion != LEAF_CRIT_NO_LEAF) {

			// Generate a new leaf
			generate_leaf(tree, trecords[i]->start, trecords[i]->end,
					trecords[i]->parent_id, trecords[i]->is_left_child,
					leaf_criterion, train_data, params);

		} else {

			// Otherwise, generate two new traversal records for the two children
			int node_id = generate_internal_tree_node(tree,
					trecords[i]->parent_id, trecords[i]->is_left_child,
					NO_LEAF,
					trecords[i]->split_record->feature,
					trecords[i]->split_record->threshold,
					n_node_samples);
			generate_traversal_records_children(tree, train_data,
					huge_traversal_queue, trecords[i], params, node_id);

		}

	}

}

/* --------------------------------------------------------------------------------
 * Generates a single leaf node
 * --------------------------------------------------------------------------------
 */
void generate_leaf(TREE *tree, int start, int end, int parent_id,
		int is_left_child, unsigned int leaf_criterion, TRAINING_DATA *train_data, PARAMETERS *params) {

	// compute leaf values from scratch (otherwise, one has to store all class counts per split)
	FLOAT_TYPE leaf_value = cpu_criterion_leaf(start, end, train_data, params);

	if (params->learning_type == LEARNING_PROBLEM_TYPE_CLASSIFICATION) {
		// store original label here
		leaf_value = train_data->classes[(int) leaf_value];
	}

	generate_tree_leaf(tree, parent_id, is_left_child, leaf_value, leaf_criterion);

}

