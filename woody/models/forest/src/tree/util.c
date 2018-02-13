/*
 * util.c
 */

#include "include/util.h"

/* --------------------------------------------------------------------------------
 * Sets default parameters
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(PARAMETERS *params) {

	params->num_threads = 1;
	params->verbosity_level = 0;
	params->n_estimators = 10;
	params->min_samples_split = 2;
	params->max_features = -1;
	params->bootstrap = 1;
	params->max_depth = 10000;
	params->min_samples_leaf = 1;
	params->tree_traversal_mode = TREE_TRAVERSAL_MODE_DFS;
	params->seed = 0;
	params->criterion = CRITERION_GINI;
	params->learning_type = LEARNING_PROBLEM_TYPE_CLASSIFICATION;
	params->leaf_stopping_mode = LEAF_MODE_STOP_ALL;
	params->tree_type = TREE_TYPE_RANDOMIZED;
	params->prediction_type = PREDICTION_TYPE_NORMAL;
	params->n_subset_check = -1;
	params->patterns_transposed = TRANSPOSED;

}

/* --------------------------------------------------------------------------------
 * Checks parameters for correctness
 * --------------------------------------------------------------------------------
 */
void check_parameters(PARAMETERS *params) {

}

/* --------------------------------------------------------------------------------
 * Maps the original classes labels to the new classes given in the array classes
 * --------------------------------------------------------------------------------
 */
void map_class_labels(FLOAT_TYPE *Ymapped, FLOAT_TYPE *Y, int nY,
		FLOAT_TYPE *classes, int nclasses) {

	int i;
	for (i = 0; i < nY; i++) {
		Ymapped[i] = map_single_class_label(Y[i], classes, nclasses);
	}

}

/* --------------------------------------------------------------------------------
 * Maps a single label to a new class label via in the array classes
 * --------------------------------------------------------------------------------
 */
inline FLOAT_TYPE map_single_class_label(FLOAT_TYPE label,
		FLOAT_TYPE *classes, int nclasses) {

	int k;

	for (k = 0; k < nclasses; k++) {
		FLOAT_TYPE diff = label - classes[k];
		if (diff * diff < 0.1) {
			return (FLOAT_TYPE) k;
		}
	}

	// this point should not be reached ...
	printf("Label could not be mapped! Exiting ...\n");
	exit(0);

}


/* --------------------------------------------------------------------------------
 * Retrieves classes that are given in a consecutive order
 * --------------------------------------------------------------------------------
 */
int retrieve_classes(FLOAT_TYPE *Ytrain, int nYtrain, FLOAT_TYPE **classes) {

	int i, j;

	int num_classes_found = 0;

	// allocate some space for the classes to be found
	FLOAT_TYPE *classes_tmp = (FLOAT_TYPE*) malloc(
			(num_classes_found + 1) * sizeof(FLOAT_TYPE));

	for (i = 0; i < nYtrain; i++) {

		// the original label
		FLOAT_TYPE label = Ytrain[i];

		// check for classification label
		FLOAT_TYPE diff_integer = label - (float) ((int) label);
		if (diff_integer * diff_integer > 0.0001) {
			printf("Classification task requires integers as class labels! Exiting ...\n");
			exit(0);
		}

		// check if label has already been found
		int found_new_class = 1;
		for (j = 0; j < num_classes_found; j++) {
			FLOAT_TYPE d = (label - classes_tmp[j]);
			if (d * d <= 0.0001) {
				found_new_class = 0;
				break;
			}
		}

		// if a new label is given
		if (found_new_class) {
			classes_tmp[num_classes_found] = (int) label;
			num_classes_found += 1;
			classes_tmp = (FLOAT_TYPE*) realloc(classes_tmp, (num_classes_found + 1) * sizeof(FLOAT_TYPE));
		}

	}

	// set pointer to array of classes
	*classes = classes_tmp;

	return num_classes_found;

}

/* --------------------------------------------------------------------------------
 * Custom random number generator (similar to the scikit-learn implementation)
 * (see  http://www.jstatsoft.org/v08/i14/paper)
 * --------------------------------------------------------------------------------
 */
inline unsigned int custom_rand_r(unsigned int *random_state) {

	*random_state ^= (unsigned int) (*random_state << 13);
	*random_state ^= (unsigned int) (*random_state >> 17);
	*random_state ^= (unsigned int) (*random_state << 5);

	return (unsigned int) ((*random_state) % ((unsigned int) RAND_R_MAX + 1));

}

/* --------------------------------------------------------------------------------
 * Custom random integer generator (similar to the scikit-learn implementation)
 * (see  http://www.jstatsoft.org/v08/i14/paper)
 * --------------------------------------------------------------------------------
 */
inline int rand_int(int end, unsigned int* random_state) {

	return (int) custom_rand_r(random_state) % end;

}

/* --------------------------------------------------------------------------------
 * Custom random uniform number generator (similar to the scikit-learn implementation)
 * (see  http://www.jstatsoft.org/v08/i14/paper)
 * --------------------------------------------------------------------------------
 */
inline FLOAT_TYPE rand_uniform(FLOAT_TYPE low, FLOAT_TYPE high,
		unsigned int *random_state) {

	// uniform random number (between 0.0 and 1.0)
	int rint = custom_rand_r(random_state);
	double uint = ((double) rint) / ((double) RAND_R_MAX);

	return low + (high - low) * ((FLOAT_TYPE) uint);

}

/* --------------------------------------------------------------------------------
 * Returns the number of integers needed for constant feature bit operations
 * --------------------------------------------------------------------------------
 */
inline int get_const_features_integers_bound(int dim){

	return (int)(dim / CONST_FEATURE_BIT_LENGTH) + 1;

}

/* --------------------------------------------------------------------------------
 * Allocates initial memory for constant features
 * --------------------------------------------------------------------------------
 */
int *allocate_initial_mem_const_features(int dim){

	return (int*) malloc(get_const_features_integers_bound(dim) * sizeof(int));

}

/* --------------------------------------------------------------------------------
 * Checks if feature F is constant
 * --------------------------------------------------------------------------------
 */
inline int feature_is_constant(int *const_features, int F, int dim) {

	// get pointer to appropriate feature bit
	int *feature_bit = const_features + F / CONST_FEATURE_BIT_LENGTH;

	// checks if corresponding bit is set to 1
	return ((*feature_bit) & (1 << (F % CONST_FEATURE_BIT_LENGTH)));

}

/* --------------------------------------------------------------------------------
 * Sets feature vector according to feature F
 * --------------------------------------------------------------------------------
 */
inline void set_feature_constant(int *const_features, int F, int dim) {

	// get pointer to appropriate feature bit
	int *feature_bit = const_features + F / CONST_FEATURE_BIT_LENGTH;

	// sets corresponding feature bit to 1
	*feature_bit |= (1 << (F % CONST_FEATURE_BIT_LENGTH));

}

/* --------------------------------------------------------------------------------
 * Initializes array for keeping track of constant features
 * --------------------------------------------------------------------------------
 */
void init_const_features_array(int *const_features, int d) {

	int i;
	int bound = get_const_features_integers_bound(d);

	for (i = 0; i < bound; i++) {
		const_features[i] = 0;
	}

}

/* --------------------------------------------------------------------------------
 * Updates the constant features array
 * --------------------------------------------------------------------------------
 */
void update_const_features_array(int *target_feature_array,
		int *source_feature_array, int d) {

	int i;
	int bound = get_const_features_integers_bound(d);

	for (i = 0; i < bound; i++) {
		target_feature_array[i] = source_feature_array[i];
	}

}


/* --------------------------------------------------------------------------------
 * Helper function to swap to integers
 * --------------------------------------------------------------------------------
 */
inline void swap_ints(int *arr, int i, int j) {

	int tmp;
	tmp = arr[i];
	arr[i] = arr[j];
	arr[j] = tmp;

}

/* --------------------------------------------------------------------------------
 * Helper function to swap to floats
 * --------------------------------------------------------------------------------
 */
inline void swap_floats(FLOAT_TYPE *arr, int i, int j) {

	FLOAT_TYPE tmp;
	tmp = arr[i];
	arr[i] = arr[j];
	arr[j] = tmp;

}

/* --------------------------------------------------------------------------------
 * generates random subset of size m (without replacement)
 * --------------------------------------------------------------------------------
 */
void generate_random_feature_subset(int *m_random, int dim, unsigned int *random_state) {

	int i;

	// allocate space for temporary indices
	int *all_feature_indices_tmp = (int*) malloc(dim * sizeof(int));

	// generate array of all indices
	for (i = 0; i < dim; i++) {
		all_feature_indices_tmp[i] = i;
	}

	// permute array
	for (i = 0; i < dim; i++) {
		int r_int = i + rand_int(dim - i, random_state);
		swap_ints(all_feature_indices_tmp, r_int, i);
	}

	// copy permuted indices to target array
	for (i = 0; i < dim; i++) {
		m_random[i] = all_feature_indices_tmp[i];
	}

	// free temporary memory
	free(all_feature_indices_tmp);

}

/* --------------------------------------------------------------------------------
 * Returns the dimension of a feature array
 * --------------------------------------------------------------------------------
 */
inline int get_feature_dimension(int *rfeatures, int i) {

	return rfeatures[i];

}

/* --------------------------------------------------------------------------------
 * Finds the index of the maximum in classes
 * --------------------------------------------------------------------------------
 */
int find_max_class(int *classes, int K) {

	int i;

	int max_class_value = -1;
	int max_class_index = -1;

	for (i = 0; i < K; i++) {

		if (classes[i] > max_class_value) {
			max_class_value = classes[i];
			max_class_index = i;
		}

	}

	return max_class_index;

}

/* --------------------------------------------------------------------------------
 * Initializes a traversal record
 * --------------------------------------------------------------------------------
 */
TRAVERSAL_RECORD *init_traversal_record(int start, int end, int dim, int depth,
		int parent_id, int is_left_child, int is_leaf) {

	TRAVERSAL_RECORD *trecord = (TRAVERSAL_RECORD*) malloc(sizeof(TRAVERSAL_RECORD));

	trecord->start = start;
	trecord->end = end;
	trecord->depth = depth;
	trecord->parent_id = parent_id;
	trecord->is_left_child = is_left_child;
	trecord->is_leaf = is_leaf;
	trecord->const_features = allocate_initial_mem_const_features(dim);
	trecord->split_record = (SPLIT_RECORD*) malloc(sizeof(SPLIT_RECORD));

	return trecord;

}

/* --------------------------------------------------------------------------------
 * Frees memory allocated for a traversal record
 * --------------------------------------------------------------------------------
 */
void free_traversal_record(TRAVERSAL_RECORD *trecord) {

	free(trecord->const_features);
	free(trecord->split_record);
	free(trecord);

}

/* --------------------------------------------------------------------------------
 * Copies a split record
 * --------------------------------------------------------------------------------
 */
void copy_split_record(SPLIT_RECORD *src, SPLIT_RECORD *dest) {

	dest->pos = src->pos;
	dest->feature = src->feature;
	dest->threshold = src->threshold;
	dest->improvement = src->improvement;
	dest->impurity = src->impurity;
	dest->impurity_left = src->impurity_left;
	dest->impurity_right = src->impurity_right;
	dest->leaf_detected = src->leaf_detected;
	dest->prob_left = src->prob_left;
	dest->prob_right = src->prob_right;

}

/* --------------------------------------------------------------------------------
 * Initializes a split record
 * --------------------------------------------------------------------------------
 */
SPLIT_RECORD *init_split_record() {

	SPLIT_RECORD *split = (SPLIT_RECORD*) malloc(sizeof(SPLIT_RECORD));

	split->impurity = MAX_FLOAT_TYPE;
	split->impurity_left = MAX_FLOAT_TYPE;
	split->impurity_right = MAX_FLOAT_TYPE;
	split->feature = 0;
	split->threshold = 0.0;
	split->improvement = MIN_FLOAT_TYPE;
	split->leaf_detected = 0;
	split->prob_left = 0.0;
	split->prob_right = 0.0;

	return split;

}

/* --------------------------------------------------------------------------------
 * Frees memory allocated for split record
 * --------------------------------------------------------------------------------
 */
void free_split_record(SPLIT_RECORD *split){

	free(split);

}

/* --------------------------------------------------------------------------------
 * Copies a training data record
 * --------------------------------------------------------------------------------
 */
void flat_copy_train_data(TRAINING_DATA *src, TRAINING_DATA *dest) {

	dest->Xtrain = src->Xtrain;
	dest->nXtrain = src->nXtrain;
	dest->dXtrain = src->dXtrain;
	dest->Ytrain_mapped = src->Ytrain_mapped;
	dest->bindices = src->bindices;

}

/* --------------------------------------------------------------------------------
 * Computes the mapped labels for the training data
 * --------------------------------------------------------------------------------
 */
void compute_mapped_labels(PARAMETERS *params, TRAINING_DATA *train_data) {

	int i, idx;

	FLOAT_TYPE *Ysub = (FLOAT_TYPE*) malloc(
			train_data->bindices->n_indices * sizeof(FLOAT_TYPE));

	for (i = 0; i < train_data->bindices->n_indices; i++) {

		idx = train_data->bindices->indices[i];
		Ysub[i] = train_data->Ytrain[idx];
		params->max_ytrain_value = max(params->max_ytrain_value, Ysub[i]);

	}

	// Generate mappings for classification tasks. NOTE: This mapping
	// depends on the particular tree considered, the indices might
	// be different here!
	if (params->learning_type == LEARNING_PROBLEM_TYPE_CLASSIFICATION) {

		train_data->n_classes = retrieve_classes(Ysub,
				train_data->bindices->n_indices, &(train_data->classes));
		map_class_labels(train_data->Ytrain_mapped, Ysub,
				train_data->bindices->n_indices, train_data->classes,
				train_data->n_classes);

	} else {

		memcpy(train_data->Ytrain_mapped, Ysub,
				train_data->bindices->n_indices * sizeof(FLOAT_TYPE));

	}

	free(Ysub);

}
