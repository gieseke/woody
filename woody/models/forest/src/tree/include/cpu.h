/*
 * cpu.h
 *
 *  Created on: 17.04.2015
 *      Author: fgieseke
 */

#ifndef FORESTS_STANDARD_INCLUDE_CPU_H_
#define FORESTS_STANDARD_INCLUDE_CPU_H_

#include <stdlib.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

#include "global.h"
#include "util.h"
#include "../cpu/include/base.h"




/* --------------------------------------------------------------------------------
 * Fits a model given the training data (and parameters)
 * --------------------------------------------------------------------------------
 */
void fit_forest(FLOAT_TYPE *Xtrain, int nXtrain, int dXtrain,
		FLOAT_TYPE *Ytrain, int *bootstrap_indices, int *bootstrap_indices_weights,
		int n_bootstrap_indices, int d_bootstrap_indices, int use_bindices, PARAMETERS *params, FOREST *forest);

/* --------------------------------------------------------------------------------
 * Builds a single tree.
 * --------------------------------------------------------------------------------
 */
void build_single_tree(TREE *tree, TRAINING_DATA *train_data,
		PARAMETERS *params, unsigned int *rstate);



/* --------------------------------------------------------------------------------
 * Process huge nodes
 * --------------------------------------------------------------------------------
 */
void process_all_nodes(TREE *tree, TRAINING_DATA *train_data, PQUEUE *huge_traversal_queue,
		PARAMETERS *params, unsigned int *rstate);


/* --------------------------------------------------------------------------------
 * Returns a chunk of traversal records.
 * --------------------------------------------------------------------------------
 */
TRAVERSAL_RECORD **get_chunk_trecords(PQUEUE *traversal_queue, int *n_trecords, int n_to_be_removed);

/* --------------------------------------------------------------------------------
 * Generate traversal records for the children of a given traversal record
 * --------------------------------------------------------------------------------
 */
void generate_traversal_records_children(TREE *tree, TRAINING_DATA *train_data, PQUEUE *huge_traversal_queue,
		TRAVERSAL_RECORD *trecord,
		PARAMETERS *params, int node_id);

void generate_next_leaf_node(int node_id, int start, int end, unsigned int leaf_criterion, int depth, int prio, int child_flag, TREE *tree,
		PQUEUE *huge_traversal_queue, TRAVERSAL_RECORD *trecord, TRAINING_DATA *train_data, PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Generates leaves and nodes
 * --------------------------------------------------------------------------------
 */
void generate_leaves_nodes(TREE *tree, TRAVERSAL_RECORD **trecords, int n_trecords,
		TRAINING_DATA *train_data, PARAMETERS *params, PQUEUE *huge_traversal_queue);

/* --------------------------------------------------------------------------------
 * Generates a single leaf node
 * --------------------------------------------------------------------------------
 */
void generate_leaf(TREE *tree, int start, int end, int parent_id, int is_left_child,
		unsigned int leaf_criterion, TRAINING_DATA *train_data, PARAMETERS *params);


#endif /* FORESTS_STANDARD_INCLUDE_CPU_H_ */
