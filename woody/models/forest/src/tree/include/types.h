/*
 * types.h
 *
 *  Created on: 15.02.2016
 *      Author: fgieseke
 */

#ifndef ENSEMBLE_INCLUDE_TYPES_H_
#define ENSEMBLE_INCLUDE_TYPES_H_

#include "../../include/float.h"
#include "../../include/timing.h"

typedef struct parameters {

	int seed;
	int n_estimators;
	int min_samples_split;
	int max_features;
	int bootstrap;
	int max_depth;
	int min_samples_leaf;
	int num_threads;
	int verbosity_level;
	int tree_traversal_mode;
	int leaf_stopping_mode;
	int criterion;
	int learning_type;
	int tree_type;
	int prediction_type;
	int patterns_transposed;
	double lam_crit;
	int n_subset_check;

	// training
	FLOAT_TYPE *Xtrain;
	int nXtrain;
	int dXtrain;
	FLOAT_TYPE max_ytrain_value;

	TIMER timers[10];

} PARAMETERS;

typedef struct bootstrap_indices {
	int n_indices;
	int *indices;
	int *indices_wmappings;
} BINDICES;

typedef struct training_data {

	FLOAT_TYPE *Xtrain;
	FLOAT_TYPE *Ytrain;
	FLOAT_TYPE *Ytrain_mapped;
	int nXtrain;
	int dXtrain;
	BINDICES *bindices;
	int n_classes;
	FLOAT_TYPE *classes;

} TRAINING_DATA;

typedef struct split_record {

	unsigned int feature;
	int pos;
	FLOAT_TYPE threshold;
	FLOAT_TYPE improvement;
	FLOAT_TYPE impurity;
	FLOAT_TYPE impurity_left;
	FLOAT_TYPE impurity_right;
	FLOAT_TYPE prob_left;
	FLOAT_TYPE prob_right;
	int leaf_detected;

} SPLIT_RECORD;

typedef struct traversal_record {

	int start;
	int end;
	int depth;
	int parent_id;
	int is_left_child;
	int is_leaf;

	int n_constant_features;
	int *const_features;

	SPLIT_RECORD *split_record;

} TRAVERSAL_RECORD;

typedef struct tree_node {

	unsigned int left_id;
	unsigned int right_id;
	unsigned int feature;
	FLOAT_TYPE thres_or_leaf;
	unsigned int leaf_criterion;

} TREE_NODE;

typedef struct tree {

	TREE_NODE *root;
	int n_allocated;
	int node_counter;

} TREE;

typedef struct forest {

	TREE *trees;
	int n_trees;

} FOREST;

typedef struct pattern_label_weight PATTERN_LABEL_WEIGHT;

struct pattern_label_weight {
	FLOAT_TYPE pattern;
	FLOAT_TYPE label;
	int weight;
};

typedef struct criterion_record CRITERION_RECORD;

struct criterion_record {

	int current_pos;
	FLOAT_TYPE impurity;
	FLOAT_TYPE impurity_left;
	FLOAT_TYPE impurity_right;
	FLOAT_TYPE improvement;

	int weight_left;
	int weight_right;

	// needed for regression (MSE)
	FLOAT_TYPE sum_left;
	FLOAT_TYPE sum_right;
	FLOAT_TYPE sq_sum_left;
	FLOAT_TYPE sq_sum_right;

	// needed for classification (GINI and ENTROPY)
	int *class_counts_left;
	int *class_counts_right;

};

#endif /* ENSEMBLE_INCLUDE_TYPES_H_ */
