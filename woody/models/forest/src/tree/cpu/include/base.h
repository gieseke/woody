/*
 * cpu.h
 */
#ifndef ENSEMBLE_HUGE_FOREST_CPU
#define ENSEMBLE_HUGE_FOREST_CPU

#include <assert.h>
#include <omp.h>

#include "criteria.h"
#include "standard.h"

#include "../../include/global.h"
#include "../../include/tree.h"
#include "../../include/util.h"

#include "../../../include/util.h"
#include "../../../include/pqueue.h"

/* --------------------------------------------------------------------------------
 * Initializes components on host system
 * --------------------------------------------------------------------------------
 */
void cpu_init(PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Initializes components after training phase
 * --------------------------------------------------------------------------------
 */
void cpu_init_after_fitting(PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Frees resources
 * --------------------------------------------------------------------------------
 */
void cpu_free_resources(PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Computes the predictions using the CPU (for test patterns)
 * --------------------------------------------------------------------------------
 */
void cpu_predict(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
		FLOAT_TYPE *predictions,
		int *indices, int dindices,
		PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Queries the forest.
 * --------------------------------------------------------------------------------
 */
void cpu_query_forest(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
		FLOAT_TYPE *predictions, int *indices, int dindices, PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Queries the forest (all raw predictions)
 * --------------------------------------------------------------------------------
 */
void cpu_query_forest_all_preds(FLOAT_TYPE *Xtest, int nXtest, int dXtest,
FLOAT_TYPE *preds, int n_preds, int d_preds, int *indices,
int dindices, PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Queries a single tree
 * --------------------------------------------------------------------------------
 */
void cpu_query_tree(TREE tree, FLOAT_TYPE *Xtest, int nXtest, int dXtest,
FLOAT_TYPE *predictions, int *indices, int dindices, int prediction_type);

/* --------------------------------------------------------------------------------
 * Initializes the training data
 * --------------------------------------------------------------------------------
 */
TRAINING_DATA *cpu_init_training_data(FLOAT_TYPE *Xtrain, int nXtrain,
		int dXtrain, FLOAT_TYPE *Ytrain, PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Frees space allocated for training data
 * --------------------------------------------------------------------------------
 */
void cpu_free_training_data(TRAINING_DATA *train_data, PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Computes, for each traversal record, the corresponding split record
 * that contains the splitting feature and threshold
 * --------------------------------------------------------------------------------
 */
void cpu_compute_splits(TRAVERSAL_RECORD **trecords, int n_trecords,
		TRAINING_DATA *train_data, PARAMETERS *params,
		unsigned int *rstate);

/* --------------------------------------------------------------------------------
 * Checks a single feature
 * --------------------------------------------------------------------------------
 */
void cpu_check_single_feature(TRAVERSAL_RECORD *trecord, int F,
		TRAINING_DATA *train_data, PARAMETERS *params, unsigned int *rstate, int *n_non_constant_features_checked,
		SPLIT_RECORD *best_split);

/* --------------------------------------------------------------------------------
 * Partitions the bootstrap indices according to a threshold value.
 * --------------------------------------------------------------------------------
 */
void inline cpu_partition_bindinces(int *bindinces, int *bindinces_weights, FLOAT_TYPE *Ytrain_mapped, int start, int end,
		FLOAT_TYPE threshold, int F, PARAMETERS *params, TRAINING_DATA *train_data);

/* --------------------------------------------------------------------------------
 * Computes a random threshold value
 * --------------------------------------------------------------------------------
 */
FLOAT_TYPE compute_threshold(int F, PATTERN_LABEL_WEIGHT *XF_Y_W, int n_XF_Y_W, int *feature_is_constant, PARAMETERS *params, TRAINING_DATA *train_data, unsigned int *random_state, SPLIT_RECORD *best_split);


/* --------------------------------------------------------------------------------
 * Initializes bootstrap indices
 * --------------------------------------------------------------------------------
 */
void cpu_init_bindices(TRAINING_DATA *train_data, int n, unsigned int *random_state,
		int *bootstrap_indices, int *bootstrap_indices_weights, int n_bootstrap_indices,
		PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Generates bootstrap indices with replacement.
 * --------------------------------------------------------------------------------
 */
void cpu_generate_bootstrap_indices(int *bootstrap_indices,
		int *bootstrap_indices_weights, int *n_bootstrap_indices, int n,
		unsigned int *random_state);

/* --------------------------------------------------------------------------------
 * Frees space for bootstrap indices
 * --------------------------------------------------------------------------------
 */
void cpu_free_bindices(TRAINING_DATA *train_data, int use_bindices);

/* --------------------------------------------------------------------------------
 * Transpose training patterns
 * --------------------------------------------------------------------------------
 */
void cpu_transpose_training_patterns(FLOAT_TYPE* X, int nX, int dX,
		FLOAT_TYPE* X_transposed);

void gather_feature_values(PATTERN_LABEL_WEIGHT *XF_Y_W, int F, int start, int end, TRAINING_DATA *train_data, PARAMETERS *params);

#endif
