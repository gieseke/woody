/*
 * global.h
 */

#ifndef FOREST_STANDARD_GLOBAL_INCLUDE_H_
#define FOREST_STANDARD_GLOBAL_INCLUDE_H_

#include "../../include/timing.h"
#include "../../include/float.h"

#include "types.h"

#define NOT_TRANSPOSED 0
#define TRANSPOSED 1

// learning types
#define LEARNING_PROBLEM_TYPE_REGRESSION 		0
#define LEARNING_PROBLEM_TYPE_CLASSIFICATION 	1

// prediction types (e.g., for returning only the leaf ids)
#define PREDICTION_TYPE_NORMAL 0
#define PREDICTION_TYPE_LEAVES_IDS 1

// tree types
#define TREE_TYPE_STANDARD 				0
#define TREE_TYPE_RANDOMIZED 			1

#define USE_BOOTSTRAP_INDICES 	1

// tree traversal modes
#define TREE_TRAVERSAL_MODE_DFS 		0
#define TREE_TRAVERSAL_MODE_NODE_SIZE 	1
#define TREE_TRAVERSAL_MODE_PROB 	2

// criteria
#define CRITERION_MSE 		0
#define CRITERION_GINI 		1
#define CRITERION_ENTROPY 	2
#define CRITERION_EVEN_SPLIT_MSE 3
#define CRITERION_EVEN_SPLIT_GINI 4
#define CRITERION_EVEN_SPLIT_ENTROPY 5

// leaf stopping criteria
#define LEAF_CRIT_NO_LEAF 0
#define LEAF_CRIT_DETECTED 1
#define LEAF_CRIT_MAX_DEPTH 2
#define LEAF_CRIT_MIN_SAMPLES_SPLIT 3
#define LEAF_CRIT_MIN_SAMPLES_LEAF 4
#define LEAF_CRIT_MIN_IMPURITY 5
#define LEAF_CRIT_POS_END 6

// leaf stopping modes
#define LEAF_MODE_STOP_ALL 0
#define LEAF_MODE_STOP_IGNORE_IMPURITY 1

#define MIN_IMPURITY_SPLIT 	1e-10
#define FEATURE_THRESHOLD 	1e-10

#define NO_LEAF 0
#define LEAF 	1

#define TREE_ROOT_PARENT_ID -1
#define TREE_ROOT_ID 0
#define TREE_CHILD_ID_NOT_SET 0

#define NO_LEFT_CHILD 	0
#define LEFT_CHILD 		1

// float as default
#ifndef USE_DOUBLE
#define USE_DOUBLE 0
#endif

#if USE_DOUBLE > 0
#define FLOAT_TYPE double
#else
#define FLOAT_TYPE float
#endif

#define FREE_RESOURCES cpu_free_resources
#define COMPUTE_SPLITS cpu_compute_splits
#define INIT_BINDICES cpu_init_bindices
#define FREE_BINDICES cpu_free_bindices
#define INIT_TRAINING_DATA cpu_init_training_data
#define FREE_TRAINING_DATA cpu_free_training_data
#define INIT cpu_init
#define INIT_AFTER_FITTING cpu_init_after_fitting
#define PREDICT cpu_predict

#define PRINT(params) if ((params->verbosity_level) > 0) printf

#endif /* FOREST_STANDARD_GLOBAL_INCLUDE_H_ */
