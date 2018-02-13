/*
 * criteria.h
 *
 *  Created on: 08.01.2015
 *      Author: fgieseke
 */

#ifndef ENSEMBLE_HUGE_FOREST_INCLUDE_CRITERIA_H_
#define ENSEMBLE_HUGE_FOREST_INCLUDE_CRITERIA_H_

#include "../../include/global.h"
#include "../../include/util.h"

void criterion_improvement_via_threshold(FLOAT_TYPE threshold, PATTERN_LABEL_WEIGHT *XF_Y_W, TRAINING_DATA *train_data,
		TRAVERSAL_RECORD *trecord, PARAMETERS *params, SPLIT_RECORD *current_split);

/* --------------------------------------------------------------------------------
 * Computes the impurity for samples[start:end]
 * Similar to RegressionCriterion(Criterion) of sklearn
 * --------------------------------------------------------------------------------
 */
FLOAT_TYPE cpu_criterion_leaf(int start, int end,
		TRAINING_DATA *train_data, PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Initializes a splitting criterion (which can be updated).
 * --------------------------------------------------------------------------------
 */
void init_criterion_cpu(CRITERION_RECORD *crit_record, PATTERN_LABEL_WEIGHT *XF_Y_W,
		int n_XF_Y_W, PARAMETERS *params, TRAINING_DATA *train_data);

void free_criterion_cpu(CRITERION_RECORD *crit_record, PARAMETERS *params, TRAINING_DATA *train_data);

/* --------------------------------------------------------------------------------
 * Updates a criterion.
 * --------------------------------------------------------------------------------
 */
void inline update_criterion_cpu(CRITERION_RECORD *crit_record,
		PATTERN_LABEL_WEIGHT *XF_Y_W, int n_XF_Y_W, int new_pos, PARAMETERS *params, TRAINING_DATA *train_data);

#endif /* ENSEMBLE_HUGE_FOREST_INCLUDE_CRITERIA_H_ */
