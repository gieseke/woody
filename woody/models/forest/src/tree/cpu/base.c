/*
 * base.c
 */

#include "include/base.h"

/* --------------------------------------------------------------------------------
 * Initializes components on host system
 * --------------------------------------------------------------------------------
 */
void cpu_init(PARAMETERS *params, FOREST *forest) {

}

/* --------------------------------------------------------------------------------
 * Initializes components after training phase
 * --------------------------------------------------------------------------------
 */
void cpu_init_after_fitting(PARAMETERS *params, FOREST *forest) {

}

/* --------------------------------------------------------------------------------
 * Frees resources
 * --------------------------------------------------------------------------------
 */
void cpu_free_resources(PARAMETERS *params, FOREST *forest) {

	free_forest(forest, 1, 0);

}

/* --------------------------------------------------------------------------------
 * Computes the predictions using the CPU (for test patterns)
 * --------------------------------------------------------------------------------
 */
void cpu_predict(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
FLOAT_TYPE *predictions, int *indices, int dindices, PARAMETERS *params,
		FOREST *forest) {

	cpu_query_forest(Xtest, nXtest, dXtest, predictions, indices, dindices,
			params, forest);

}

/* --------------------------------------------------------------------------------
 * Query forest (via individual trees)
 * --------------------------------------------------------------------------------
 */
void cpu_query_forest(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
FLOAT_TYPE *predictions, int *indices, int dindices, PARAMETERS *params,
		FOREST *forest) {

	START_MY_TIMER(params->timers + 1);

	int i, b, j;

	int n_preds = nXtest;
	if (dindices > 0) {
		n_preds = dindices;
	}
	// get all individual predictions (from all trees)
	FLOAT_TYPE *preds = (FLOAT_TYPE*) malloc(
			params->n_estimators * n_preds * sizeof(FLOAT_TYPE));

	START_MY_TIMER(params->timers + 2);

	omp_set_dynamic(0);
	omp_set_num_threads(params->num_threads);

#pragma omp parallel for
	for (b = 0; b < params->n_estimators; b++) {
		cpu_query_tree(forest->trees[b], Xtest, nXtest, dXtest,
				preds + b * n_preds, indices, dindices,
				params->prediction_type);
	}
	STOP_MY_TIMER(params->timers + 2);

	if (params->prediction_type == PREDICTION_TYPE_NORMAL) {

		// combine all predictions
		if (params->learning_type == LEARNING_PROBLEM_TYPE_REGRESSION) {

			for (i = 0; i < n_preds; i++) {

				predictions[i] = 0.0;

				for (b = 0; b < params->n_estimators; b++) {
					predictions[i] += preds[i + b * n_preds];
				}
				predictions[i] /= params->n_estimators;
			}

		} else if (params->learning_type == LEARNING_PROBLEM_TYPE_CLASSIFICATION) {

			int max_possible_label = ((int) params->max_ytrain_value) + 1;

			int *class_counts = (int*) calloc(max_possible_label, sizeof(int));

			for (i = 0; i < n_preds; i++) {

				for (j = 0; j < max_possible_label; j++) {
					class_counts[j] = 0;
				}

				for (b = 0; b < params->n_estimators; b++) {
					FLOAT_TYPE pred = (int) preds[i + b * n_preds];
					class_counts[(int) pred] += 1;
				}
				int max_index = find_max_class(class_counts,
						max_possible_label);
				predictions[i] = max_index;

			}

			free(class_counts);

		}

		else {

			printf("Error: Unknown learning type. Exiting ...\n");
			exit(EXIT_FAILURE);

		}
	} else if (params->prediction_type == PREDICTION_TYPE_LEAVES_IDS) {

		for (b = 0; b < params->n_estimators; b++) {
			for (i = 0; i < n_preds; i++) {
				predictions[b * n_preds + i] = *(preds + b * n_preds + i);
			}
		}

	} else {
		printf("Error: Unknown prediction type: %i ", params->prediction_type);
		exit(EXIT_FAILURE);
	}

	free(preds);

	STOP_MY_TIMER(params->timers + 1);

}

/* --------------------------------------------------------------------------------
 * Queries the forest (all raw predictions)
 * --------------------------------------------------------------------------------
 */
void cpu_query_forest_all_preds(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
FLOAT_TYPE *preds, int n_preds, int d_preds, int *indices,
int dindices, PARAMETERS *params, FOREST *forest) {

	START_MY_TIMER(params->timers + 1);

	int i, b;

	if (dindices > 0) {
		n_preds = dindices;
	}
	// get all individual predictions (from all trees)
	FLOAT_TYPE *preds_tmp = (FLOAT_TYPE*) malloc(
			params->n_estimators * n_preds * sizeof(FLOAT_TYPE));

	START_MY_TIMER(params->timers + 2);

	omp_set_dynamic(0);
	omp_set_num_threads(params->num_threads);

#pragma omp parallel for
	for (b = 0; b < params->n_estimators; b++) {
		cpu_query_tree(forest->trees[b], Xtest, nXtest, dXtest,
				preds_tmp + b * n_preds, indices, dindices,
				params->prediction_type);
	}
	STOP_MY_TIMER(params->timers + 2);

	for (i = 0; i < n_preds; i++) {
		for (b = 0; b < params->n_estimators; b++) {
			preds[i*params->n_estimators + b] = preds_tmp[i + b * n_preds];
		}
	}


	free(preds_tmp);

	STOP_MY_TIMER(params->timers + 1);

}


/* --------------------------------------------------------------------------------
 * Queries a single tree
 * --------------------------------------------------------------------------------
 */
void cpu_query_tree(TREE tree, FLOAT_TYPE *Xtest, int nXtest, int dXtest,
FLOAT_TYPE *predictions, int *indices, int dindices, int prediction_type) {

	register TREE_NODE *node = tree.root;
	register FLOAT_TYPE *tpatt;

	register unsigned int i, node_id, idx;

	int n_preds = nXtest;
	if (dindices > 0) {
		n_preds = dindices;
	}
	for (i = 0; i < n_preds; i++) {

		if (dindices > 0) {
			idx = indices[i];
		} else {
			idx = i;
		}

		tpatt = Xtest + idx * dXtest;
		node_id = TREE_ROOT_ID;

		while (node[node_id].left_id != TREE_CHILD_ID_NOT_SET) {
			if (tpatt[node[node_id].feature] <= node[node_id].thres_or_leaf) {
				node_id = node[node_id].left_id;
			} else {
				node_id = node[node_id].right_id;
			}
		}

		if (prediction_type == PREDICTION_TYPE_NORMAL) {
			predictions[i] = node[node_id].thres_or_leaf;
		} else if (prediction_type == PREDICTION_TYPE_LEAVES_IDS) {
			predictions[i] = (FLOAT_TYPE) node_id;
		} else {
			printf("Error: Unknown prediction type: %i ", prediction_type);
			exit(EXIT_FAILURE);
		}

	}

}

/* --------------------------------------------------------------------------------
 * Initializes the training data
 * --------------------------------------------------------------------------------
 */
TRAINING_DATA *cpu_init_training_data(FLOAT_TYPE *Xtrain, int nXtrain,
		int dXtrain, FLOAT_TYPE *Ytrain, PARAMETERS *params) {

	TRAINING_DATA *train_data = (TRAINING_DATA*) malloc(sizeof(TRAINING_DATA));

	train_data->Xtrain = Xtrain;
	train_data->Ytrain = Ytrain;
	train_data->nXtrain = nXtrain;
	train_data->dXtrain = dXtrain;

	return train_data;

}

/* --------------------------------------------------------------------------------
 * Frees space allocated for training data
 * --------------------------------------------------------------------------------
 */
void cpu_free_training_data(TRAINING_DATA *train_data, PARAMETERS *params) {

	free(train_data);

}

/* --------------------------------------------------------------------------------
 * Computes, for each traversal record, the corresponding split record
 * that contains the splitting feature and threshold
 * --------------------------------------------------------------------------------
 */
void cpu_compute_splits(TRAVERSAL_RECORD **trecords, int n_trecords,
		TRAINING_DATA *train_data, PARAMETERS *params, unsigned int *rstate) {

	int i, j;

	// generate random features (without replacement)
	int *random_features = (int*) malloc(
			n_trecords * train_data->dXtrain * sizeof(int));

	for (i = 0; i < n_trecords; i++) {
		generate_random_feature_subset(
				random_features + i * train_data->dXtrain, train_data->dXtrain,
				rstate);
	}

	for (i = 0; i < n_trecords; i++) {

		// Initialize best split record: multiple features
		// are tested below and the current best result (improvement)
		// can be used to do some "early stopping"
		//
		// The split record is also updated in the course of the
		// tests, i.e., it contains all the necessary information
		// for the follow-up splitting process
		SPLIT_RECORD *best_split = init_split_record();

		// A flag that indicates how many non-constant features have
		// been checked. In case no constant feature was checked,
		// the current record corresponds to a leaf (see below)
		int n_non_constant_features_checked = 0;

		// Check at most train_data->dXtrain features
		for (j = 0; j < train_data->dXtrain; j++) {

			// Check single feature
			int F = get_feature_dimension(
					random_features + i * train_data->dXtrain, j);
			cpu_check_single_feature(trecords[i], F, train_data, params, rstate,
					&n_non_constant_features_checked, best_split);

			// Break as soon as max_features or a non_constant feature was check
			if (j >= params->max_features - 1
					&& n_non_constant_features_checked > 0) {
				break;
			}

		}

		// If all features are constant: leaf detected!
		if (j == train_data->dXtrain && n_non_constant_features_checked == 0) {
			best_split->leaf_detected = LEAF;
		}

		// Copy best result for the i-th record
		copy_split_record(best_split, trecords[i]->split_record);
		free_split_record(best_split);

	}

	free(random_features);

}

void gather_feature_values(PATTERN_LABEL_WEIGHT *XF_Y_W, int F, int start,
		int end, TRAINING_DATA *train_data, PARAMETERS *params) {

	int i, idx;
	int trans = params->patterns_transposed == TRANSPOSED;

	for (i = start; i < end; i++) {

		idx = train_data->bindices->indices[i];

		XF_Y_W[i - start].label = *(train_data->Ytrain_mapped + i);
		XF_Y_W[i - start].weight = *(train_data->bindices->indices_wmappings + i);

		if (trans) {
			XF_Y_W[i - start].pattern = *(train_data->Xtrain
					+ F * train_data->nXtrain + idx);
		} else {
			XF_Y_W[i - start].pattern = train_data->Xtrain[idx
					* train_data->dXtrain + F];
		}

	}

}

/* --------------------------------------------------------------------------------
 * Checks a single feature
 * --------------------------------------------------------------------------------
 */
void cpu_check_single_feature(TRAVERSAL_RECORD *trecord, int F,
		TRAINING_DATA *train_data, PARAMETERS *params, unsigned int *rstate,
		int *n_non_constant_features_checked, SPLIT_RECORD *best_split) {

	// Only non-constant features need to be checked. Here,
	// we do an early check based on the previous checks (we
	// keep track of constant features throughout the execution)
	if (!feature_is_constant(trecord->const_features, F, train_data->dXtrain)) {

		int feat_is_constant;
		FLOAT_TYPE threshold;

		// Combine data into a single array of structs (faster memory access afterwards)
		int n_XF_Y_W = trecord->end - trecord->start;
		PATTERN_LABEL_WEIGHT *XF_Y_W = (PATTERN_LABEL_WEIGHT*) malloc(
				(trecord->end - trecord->start) * sizeof(PATTERN_LABEL_WEIGHT));
		gather_feature_values(XF_Y_W, F, trecord->start, trecord->end,
				train_data, params);

		// Compute the "optimal" threshold for the current feature
		threshold = compute_threshold(F, XF_Y_W, n_XF_Y_W, &feat_is_constant,
				params, train_data, rstate, best_split);

		if (feat_is_constant) {

			// If feature is constant, keep track of it, but do nothing else
			set_feature_constant(trecord->const_features, F,
					train_data->dXtrain);

		} else {

			// Otherwise, we generate a new split record and compute
			// the impurity, improvement, ...
			*n_non_constant_features_checked += 1;

			SPLIT_RECORD *current_split = init_split_record();
			criterion_improvement_via_threshold(threshold, XF_Y_W,
					train_data, trecord, params, current_split);

			// Only update the best split in case the improvement
			// of the current one is better!
			if (current_split->improvement > best_split->improvement) {
				current_split->threshold = threshold;
				current_split->feature = F;
				copy_split_record(current_split, best_split);
			}

			free_split_record(current_split);

		}

		free(XF_Y_W);

	}

}

/* --------------------------------------------------------------------------------
 * Partitions the bootstrap indices according to a threshold value.
 * --------------------------------------------------------------------------------
 */
void inline cpu_partition_bindinces(int *bindinces, int *bindinces_weights,
		FLOAT_TYPE *Ytrain_mapped, int start, int end,
		FLOAT_TYPE threshold, int F, PARAMETERS *params,
		TRAINING_DATA *train_data) {

	int p = start;
	int partition_end = end;

	while (p < partition_end) {

		int idx = bindinces[p];

		FLOAT_TYPE xf;
		if (params->patterns_transposed == TRANSPOSED) {
			FLOAT_TYPE *XF = train_data->Xtrain + F * train_data->nXtrain;
			xf = XF[idx];
		} else {
			xf = train_data->Xtrain[idx * train_data->dXtrain + F];
		}

		if (xf <= threshold) {
			p++;
		} else {
			partition_end--;
			swap_ints(bindinces, partition_end, p);
			swap_ints(bindinces_weights, partition_end, p);
			swap_floats(Ytrain_mapped, partition_end, p);
		}

	}

}

/* --------------------------------------------------------------------------------
 * Computes a random threshold value
 * --------------------------------------------------------------------------------
 */
FLOAT_TYPE compute_threshold(int F, PATTERN_LABEL_WEIGHT *XF_Y_W, int n_XF_Y_W,
		int *feature_is_constant, PARAMETERS *params, TRAINING_DATA *train_data,
		unsigned int *random_state, SPLIT_RECORD *best_split) {

	unsigned int i;
	FLOAT_TYPE threshold;

	FLOAT_TYPE minf = MAX_FLOAT_TYPE;
	FLOAT_TYPE maxf = MIN_FLOAT_TYPE;

	// find minimum and maximum
	for (i = 0; i < n_XF_Y_W; i++) {
		minf = min(minf, XF_Y_W[i].pattern);
		maxf = max(maxf, XF_Y_W[i].pattern);
	}
	*feature_is_constant = (maxf <= minf + FEATURE_THRESHOLD);

	if (*feature_is_constant) {

		return -1;

	} else {

		if (params->tree_type == TREE_TYPE_STANDARD) {

			threshold = compute_optimal_threshold(XF_Y_W, n_XF_Y_W, params, train_data, best_split);

		} else if (params->tree_type == TREE_TYPE_RANDOMIZED) {

			threshold = rand_uniform(minf, maxf, random_state);

		} else {
			printf("Error: Unknown tree type: %i\n", params->tree_type);
			exit(EXIT_FAILURE);
		}

		return threshold;
	}

}

/* --------------------------------------------------------------------------------
 * Initializes bootstrap indices
 * --------------------------------------------------------------------------------
 */
void cpu_init_bindices(TRAINING_DATA *train_data, int n,
		unsigned int *random_state, int *bootstrap_indices,
		int *bootstrap_indices_weights, int n_bootstrap_indices,
		PARAMETERS *params) {

	int i;

	// allocate memory
	BINDICES* bindices = (BINDICES*) malloc(sizeof(BINDICES));

	if (n_bootstrap_indices >= 0) {

		bindices->indices = bootstrap_indices;
		bindices->indices_wmappings = bootstrap_indices_weights;
		bindices->n_indices = n_bootstrap_indices;

	} else {

		bindices->indices = (int*) malloc(n * sizeof(int));
		bindices->indices_wmappings = (int*) malloc(n * sizeof(int));

		// generate bootstrap indices
		if (params->bootstrap == USE_BOOTSTRAP_INDICES) {

			cpu_generate_bootstrap_indices(bindices->indices,
					bindices->indices_wmappings, &bindices->n_indices, n,
					random_state);

		} else {

			for (i = 0; i < n; i++) {
				bindices->indices[i] = i;
				bindices->indices_wmappings[i] = 1;
			}
			bindices->n_indices = n;
		}
	}

	train_data->bindices = bindices;

}

/* --------------------------------------------------------------------------------
 * Generates bootstrap indices with replacement.
 * --------------------------------------------------------------------------------
 */
void cpu_generate_bootstrap_indices(int *bootstrap_indices,
		int *bootstrap_indices_weights, int *n_bootstrap_indices, int n,
		unsigned int *random_state) {

	int i;

	// sample bootstrap indices (with replacement)
	int *all_indices = (int*) malloc(n * sizeof(int));
	for (i = 0; i < n; i++) {
		all_indices[i] = 0;
	}
	for (i = 0; i < n; i++) {
		all_indices[rand_int(n, random_state)] += 1;
	}

	int counter = 0;
	for (i = 0; i < n; i++) {
		if (all_indices[i] > 0) {

			// store weights (condensed storing)
			bootstrap_indices_weights[counter] = all_indices[i];

			// bootstrap indices (condensed storing)
			bootstrap_indices[counter] = i;
			counter += 1;

		}
	}

	*n_bootstrap_indices = counter;
	free(all_indices);

}

/* --------------------------------------------------------------------------------
 * Frees space for bootstrap indices
 * --------------------------------------------------------------------------------
 */
void cpu_free_bindices(TRAINING_DATA *train_data, int use_bindices) {

	if (!use_bindices) {
		free(train_data->bindices->indices);
		free(train_data->bindices->indices_wmappings);
	}
	free(train_data->bindices);

}

/* --------------------------------------------------------------------------------
 * Transpose training patterns
 * --------------------------------------------------------------------------------
 */
void cpu_transpose_training_patterns(FLOAT_TYPE* X, int nX, int dX,
		FLOAT_TYPE* X_transposed) {

	int i, j;

	for (j = 0; j < dX; j++) {
		for (i = 0; i < nX; i++) {
			X_transposed[j * nX + i] = X[i * dX + j];
		}
	}

}
