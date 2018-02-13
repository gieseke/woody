/*
 * util.h
 */
#ifndef FOREST_STANDARD_INCLUDE_UTIL_H_
#define FOREST_STANDARD_INCLUDE_UTIL_H_

#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "global.h"

#include "../../include/util.h"

// for random number generation
#define RAND_R_MAX 0x7FFFFFFF

// for constant feature checking
#define CONST_FEATURE_BIT_LENGTH (8*sizeof(int))

/* --------------------------------------------------------------------------------
 * Sets default parameters
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Checks parameters for correctness.
 * --------------------------------------------------------------------------------
 */
void check_parameters(PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Maps the original classes labels to the new classes given in the array classes
 * --------------------------------------------------------------------------------
 */
void map_class_labels(FLOAT_TYPE *Ymapped, FLOAT_TYPE *Y, int nY,
		FLOAT_TYPE *classes, int nclasses);

/* --------------------------------------------------------------------------------
 * Maps a single label to a new class label via in the array classes
 * --------------------------------------------------------------------------------
 */
inline FLOAT_TYPE map_single_class_label(FLOAT_TYPE label,
		FLOAT_TYPE *classes, int nclasses);

/* --------------------------------------------------------------------------------
 * Retrieves classes that are given in a consecutive order
 * --------------------------------------------------------------------------------
 */
int retrieve_classes(FLOAT_TYPE *Ytrain, int nYtrain, FLOAT_TYPE **classes);

/* --------------------------------------------------------------------------------
 * Custom random number generator (similar to the scikit-learn implementation)
 * (see  http://www.jstatsoft.org/v08/i14/paper)
 * --------------------------------------------------------------------------------
 */
inline unsigned int custom_rand_r(unsigned int *random_state);

/* --------------------------------------------------------------------------------
 * Custom random integer generator (similar to the scikit-learn implementation)
 * (see  http://www.jstatsoft.org/v08/i14/paper)
 * --------------------------------------------------------------------------------
 */
inline int rand_int(int end, unsigned int* random_state);

/* --------------------------------------------------------------------------------
 * Custom random uniform number generator (similar to the scikit-learn implementation)
 * (see  http://www.jstatsoft.org/v08/i14/paper)
 * --------------------------------------------------------------------------------
 */
inline FLOAT_TYPE rand_uniform(FLOAT_TYPE low, FLOAT_TYPE high,
		unsigned int *random_state);

/* --------------------------------------------------------------------------------
 * Returns the number of integers needed for constant feature bit operations
 * --------------------------------------------------------------------------------
 */
inline int get_const_features_integers_bound(int dim);

/* --------------------------------------------------------------------------------
 * Allocates initial memory for constant features
 * --------------------------------------------------------------------------------
 */
int *allocate_initial_mem_const_features(int dim);

/* --------------------------------------------------------------------------------
 * Checks if feature F is constant
 * --------------------------------------------------------------------------------
 */
inline int feature_is_constant(int *const_features, int F, int dim);

/* --------------------------------------------------------------------------------
 * Sets feature vector according to feature F
 * --------------------------------------------------------------------------------
 */
inline void set_feature_constant(int *const_features, int F, int dim);

/* --------------------------------------------------------------------------------
 * Initializes array for keeping track of constant features
 * --------------------------------------------------------------------------------
 */
void init_const_features_array(int *const_features, int d);

/* --------------------------------------------------------------------------------
 * Updates the constant features array
 * --------------------------------------------------------------------------------
 */
void update_const_features_array(int *target_feature_array,
		int *source_feature_array, int d);

/* --------------------------------------------------------------------------------
 * Helper function to swap to integers
 * --------------------------------------------------------------------------------
 */
inline void swap_ints(int *arr, int i, int j);

/* --------------------------------------------------------------------------------
 * Helper function to swap to floats
 * --------------------------------------------------------------------------------
 */
inline void swap_floats(FLOAT_TYPE *arr, int i, int j);

/* --------------------------------------------------------------------------------
 * generates random subset of size m (without replacement)
 * --------------------------------------------------------------------------------
 */
void generate_random_feature_subset(int *m_random, int dim, unsigned int *random_state);

/* --------------------------------------------------------------------------------
 * Returns the dimension of a feature array
 * --------------------------------------------------------------------------------
 */
inline int get_feature_dimension(int *rfeatures, int i);

/* --------------------------------------------------------------------------------
 * Finds the index of the maximum in classes
 * --------------------------------------------------------------------------------
 */
int find_max_class(int *classes, int K);

/* --------------------------------------------------------------------------------
 * Initializes a traversal record
 * --------------------------------------------------------------------------------
 */
TRAVERSAL_RECORD *init_traversal_record(int start, int end, int dim, int depth,
		int parent_id, int is_left_child, int is_leaf);

/* --------------------------------------------------------------------------------
 * Frees memory allocated for a traversal record
 * --------------------------------------------------------------------------------
 */
void free_traversal_record(TRAVERSAL_RECORD *trecord);

/* --------------------------------------------------------------------------------
 * Copies a split record
 * --------------------------------------------------------------------------------
 */
void copy_split_record(SPLIT_RECORD *src, SPLIT_RECORD *dest);

/* --------------------------------------------------------------------------------
 * Initializes a split record
 * --------------------------------------------------------------------------------
 */
SPLIT_RECORD *init_split_record(void);

/* --------------------------------------------------------------------------------
 * Frees memory for split record
 * --------------------------------------------------------------------------------
 */
void free_split_record(SPLIT_RECORD *split);

/* --------------------------------------------------------------------------------
 * Copies a training data record
 * --------------------------------------------------------------------------------
 */
void flat_copy_train_data(TRAINING_DATA *src, TRAINING_DATA *dest);

/* --------------------------------------------------------------------------------
 * Computes the mapped labels for the training data
 * --------------------------------------------------------------------------------
 */
void compute_mapped_labels(PARAMETERS *params, TRAINING_DATA *train_data);


#endif /* FOREST_STANDARD_INCLUDE_UTIL_H_ */
