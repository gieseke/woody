/*
 * criteria.c
 *
 *  Created on: 08.01.2015
 *      Author: fgieseke
 */

#include "include/criteria.h"

void criterion_improvement_via_threshold(FLOAT_TYPE threshold, PATTERN_LABEL_WEIGHT *XF_Y_W, TRAINING_DATA *train_data,
		TRAVERSAL_RECORD *trecord, PARAMETERS *params, SPLIT_RECORD *current_split){

	int start = trecord->start;
	int end = trecord->end;

	int splitting_pos = start;

	FLOAT_TYPE improvement = 0.0;
	FLOAT_TYPE impurity = 0.0;
	FLOAT_TYPE impurity_left = 0.0;
	FLOAT_TYPE impurity_right = 0.0;
	FLOAT_TYPE prob_left, prob_right;

	if (params->criterion == CRITERION_MSE || params->criterion == CRITERION_EVEN_SPLIT_MSE) {

		// set right sum/weights
		FLOAT_TYPE sum_left = 0.0;
		FLOAT_TYPE sum_right = 0.0;
		FLOAT_TYPE sq_sum_left = 0.0;
		FLOAT_TYPE sq_sum_right = 0.0;

		int i;

		// associated weights
		int sum_weights_left = 0;
		int sum_weights_right = 0;

		for (i = start; i < end; i++) {

			int weight = XF_Y_W[i-start].weight;
			FLOAT_TYPE val = XF_Y_W[i-start].label;

			if (XF_Y_W[i-start].pattern < threshold){
				sum_left += weight*val;
				sq_sum_left += weight*(val*val);
				sum_weights_left += weight;
				splitting_pos += 1;
			} else {
				sum_right += weight*val;
				sq_sum_right += weight*(val*val);
				sum_weights_right += weight;
			}

		}

		int sum_weights_all = sum_weights_left + sum_weights_right;
		FLOAT_TYPE fraction_left = sum_weights_left / (FLOAT_TYPE) sum_weights_all;
		FLOAT_TYPE fraction_right = sum_weights_right / (FLOAT_TYPE) sum_weights_all;

		// left impurity
		FLOAT_TYPE mean_left = sum_left / sum_weights_left;
		impurity_left = sq_sum_left / sum_weights_left - mean_left * mean_left;

		// right impurity
		FLOAT_TYPE mean_right = sum_right / sum_weights_right;
		impurity_right = sq_sum_right / sum_weights_right - mean_right * mean_right;

		// impurity
		FLOAT_TYPE mean = (sum_left + sum_right) / sum_weights_all;
		impurity = (sq_sum_left + sq_sum_right) / sum_weights_all - mean*mean;

		prob_left = ((FLOAT_TYPE)sum_weights_left) / (sum_weights_left + sum_weights_right);
		prob_right = ((FLOAT_TYPE)sum_weights_right) / (sum_weights_left + sum_weights_right);

		// improvement: we only adapt the improvement, because it is used for
		// selecting the best split. THe other quantities remain the same, such
		// as the impurity values ...
		if (params->criterion == CRITERION_EVEN_SPLIT_MSE){
			improvement = (1 - params->lam_crit) * (impurity - fraction_left * impurity_left - fraction_right * impurity_right);
			improvement = improvement - params->lam_crit*(fabs(fraction_left - fraction_right));
		} else {
			improvement = impurity - fraction_left * impurity_left - fraction_right * impurity_right;
		}

	} else if (params->criterion == CRITERION_GINI || params->criterion == CRITERION_EVEN_SPLIT_GINI) {

		int i;

		// initialize counting arrays
		int *class_counts_left = (int*) calloc(train_data->n_classes, sizeof(int));
		int *class_counts_right = (int*) calloc(train_data->n_classes, sizeof(int));

		// compute all class ratios
		int sum_weights_left = 0;
		int sum_weights_right = 0;
		for (i = start; i < end; i++) {
			int weight = XF_Y_W[i-start].weight;
			int label = XF_Y_W[i-start].label;

			if (XF_Y_W[i-start].pattern <= threshold){
				class_counts_left[label] += weight;
				sum_weights_left += weight;
				splitting_pos += 1;
			} else {
				class_counts_right[label] += weight;
				sum_weights_right += weight;
			}

		}

		// gini impurity
		impurity = 0.0;
		impurity_left = 0.0;
		impurity_right = 0.0;

		FLOAT_TYPE sum_weights = sum_weights_left + sum_weights_right;

		for (i = 0; i < train_data->n_classes; i++) {

			// left
			FLOAT_TYPE pmk_left = (1.0 / sum_weights_left) * (FLOAT_TYPE) class_counts_left[i];
			impurity_left += pmk_left * (1.0 - pmk_left);

			// right
			FLOAT_TYPE pmk_right = (1.0 / sum_weights_right) * (FLOAT_TYPE) class_counts_right[i];
			impurity_right += pmk_right * (1.0 - pmk_right);

			// both
			FLOAT_TYPE pmk = (1.0 / sum_weights) * (FLOAT_TYPE) (class_counts_left[i] + class_counts_right[i]);
			impurity += pmk * (1.0 - pmk);
		}

		free(class_counts_left);
		free(class_counts_right);

		int sum_weights_all = sum_weights_left + sum_weights_right;

		FLOAT_TYPE fraction_left = sum_weights_left / (FLOAT_TYPE) sum_weights_all;
		FLOAT_TYPE fraction_right = sum_weights_right / (FLOAT_TYPE) sum_weights_all;


		prob_left = ((FLOAT_TYPE)sum_weights_left) / (sum_weights_left + sum_weights_right);
		prob_right = ((FLOAT_TYPE)sum_weights_right) / (sum_weights_left + sum_weights_right);

		// improvement: we only adapt the improvement, because it is used for
		// selecting the best split. THe other quantities remain the same, such
		// as the impurity values ...
		if (params->criterion == CRITERION_EVEN_SPLIT_GINI){
			improvement = (1 - params->lam_crit) * (impurity - fraction_left * impurity_left - fraction_right * impurity_right);
			improvement = improvement - params->lam_crit*(fabs(fraction_left - fraction_right));
		} else {
			improvement = impurity - fraction_left * impurity_left - fraction_right * impurity_right;
		}

//	} else if (params->criterion == CRITERION_EVEN_SPLIT_MSE) {
//
//		int i;
//
//		// compute all class ratios
//		int sum_weights_left = 0;
//		int sum_weights_right = 0;
//
//		for (i = start; i < end; i++) {
//
//			int weight = XF_Y_W[i-start].weight;
//
//			if (XF_Y_W[i-start].pattern <= threshold){
//				sum_weights_left += weight;
//				splitting_pos += 1;
//			} else {
//				sum_weights_right += weight;
//			}
//
//		}
//
//		// "even distributed" impurity
//		// ignore impurity
//		impurity = 1.0;
//		impurity_left = 1.0;
//		impurity_right = 1.0;
//
//		int sum_weights_all = sum_weights_left + sum_weights_right;
//		FLOAT_TYPE fraction_left = sum_weights_left / (FLOAT_TYPE) sum_weights_all;
//		FLOAT_TYPE fraction_right = sum_weights_right / (FLOAT_TYPE) sum_weights_all;
//
//		// improvement
//		improvement = 1.0 / (fabs(fraction_left - fraction_right) + 0.00000001);
//
//		// probs
//		prob_left = ((FLOAT_TYPE)sum_weights_left) / (sum_weights_left + sum_weights_right);
//		prob_right = ((FLOAT_TYPE)sum_weights_right) / (sum_weights_left + sum_weights_right);
//
//	} else if (params->criterion == CRITERION_EVEN_SPLIT_GINI) {
//
//		// TODO

	} else {

		printf("Wrong criterion in cpu_init_criterion given: %i. Exiting ...\n", params->criterion);
		exit(0);

	}

	current_split->improvement = improvement;
	current_split->impurity = impurity;
	current_split->impurity_left = impurity_left;
	current_split->impurity_right = impurity_right;
	current_split->pos = splitting_pos;
	current_split->prob_left = prob_left;
	current_split->prob_right = prob_right;

}

/* --------------------------------------------------------------------------------
 * Computes the impurity for samples[start:end]
 * Similar to RegressionCriterion(Criterion) of sklearn
 * --------------------------------------------------------------------------------
 */
FLOAT_TYPE cpu_criterion_leaf(int start, int end,
		TRAINING_DATA *train_data, PARAMETERS *params) {

	int *samples_weights_mapping = train_data->bindices->indices_wmappings;
	if (params->criterion == CRITERION_MSE || params->criterion == CRITERION_EVEN_SPLIT_MSE) {

		FLOAT_TYPE sum_Y_total = 0.0;
		int weight_total = 0;

		int p;
		for (p = start; p < end; p++) {
			int weight = samples_weights_mapping[p];
			weight_total += weight;
			sum_Y_total += weight*train_data->Ytrain_mapped[p];
		}

		return sum_Y_total / weight_total;

	} else if (params->criterion == CRITERION_GINI || params->criterion == CRITERION_EVEN_SPLIT_GINI) {

		int *class_counts = (int*) calloc(train_data->n_classes, sizeof(int));
		int p;
		int weight_total = 0;

		for (p = start; p < end; p++) {
			int weight = samples_weights_mapping[p];
			int label = train_data->Ytrain_mapped[p];
			weight_total += weight;
			class_counts[label] += weight;
		}
		FLOAT_TYPE label = (FLOAT_TYPE) find_max_class(class_counts, train_data->n_classes);
		free(class_counts);

		return label;

	} else {

		printf("Wrong criterion in cpu_criterion_leaf given: %i. Exiting ...\n", params->criterion);
		exit(0);

	}
}

/* --------------------------------------------------------------------------------
 * Initializes a splitting criterion (which can be updated).
 * --------------------------------------------------------------------------------
 */
void init_criterion_cpu(CRITERION_RECORD *crit_record, PATTERN_LABEL_WEIGHT *XF_Y_W,
		int n_XF_Y_W, PARAMETERS *params, TRAINING_DATA *train_data) {

	crit_record->current_pos = 0;

	if (params->criterion == CRITERION_MSE || params->criterion == CRITERION_EVEN_SPLIT_MSE) {

		// set right sum/weights
		crit_record->sum_left = 0.0;
		crit_record->sum_right = 0.0;
		crit_record->sq_sum_left = 0.0;
		crit_record->sq_sum_right = 0.0;

		int i;

		// for bindex, we have an associated weight
		int sum_weights = 0;
		for (i = 0; i < n_XF_Y_W; i++) {
			int weight = XF_Y_W[i].weight;
			FLOAT_TYPE val = XF_Y_W[i].label;
			crit_record->sum_right += weight*val;
			crit_record->sq_sum_right += weight*(val * val);
			sum_weights += weight;

		}

		// impurity of node
		FLOAT_TYPE mean = crit_record->sum_right / sum_weights;
		crit_record->impurity = crit_record->sq_sum_right / sum_weights - mean * mean;
		crit_record->impurity_left = 0.0;
		crit_record->impurity_right = crit_record->impurity;
		crit_record->weight_left = 0;
		crit_record->weight_right = sum_weights;

		// improvement
		crit_record->improvement = MIN_FLOAT_TYPE;

	} else if (params->criterion == CRITERION_GINI || params->criterion == CRITERION_EVEN_SPLIT_GINI) {

		int i;

		// initialize records
		crit_record->class_counts_left = (int*) malloc(train_data->n_classes * sizeof(int));
		crit_record->class_counts_right = (int*) malloc(train_data->n_classes * sizeof(int));
		for (i = 0; i < train_data->n_classes; i++) {
			crit_record->class_counts_left[i] = 0;
			crit_record->class_counts_right[i] = 0;
		}

		// compute all class ratios (right side)
		int sum_weights = 0;
		for (i = 0; i < n_XF_Y_W; i++) {
			int label = XF_Y_W[i].label;
			int weight = XF_Y_W[i].weight;
			crit_record->class_counts_right[label] += weight;
			sum_weights += weight;
		}

		// gini impurity
		FLOAT_TYPE impurity = 0.0;
		for (i = 0; i < train_data->n_classes; i++) {
			FLOAT_TYPE pmk = (1.0 / sum_weights) * (FLOAT_TYPE) crit_record->class_counts_right[i];
			impurity += pmk * (1.0 - pmk);
		}

		// impurity of node
		crit_record->impurity = impurity;
		crit_record->impurity_left = 0.0;
		crit_record->impurity_right = crit_record->impurity;
		crit_record->weight_left = 0;
		crit_record->weight_right = sum_weights;

		// improvement
		crit_record->improvement = MIN_FLOAT_TYPE;

//	} else if (params->criterion == CRITERION_EVEN_SPLIT_MSE) {
//
//		int i;
//
//		crit_record->weight_left = 0;
//		crit_record->weight_right = 0;
//		for (i = 0; i < n_XF_Y_W; i++) {
//			crit_record->weight_right += XF_Y_W[i].weight;
//		}
//
//		// improvement
//		crit_record->improvement = MIN_FLOAT_TYPE;
//
//	} else if (params->criterion == CRITERION_EVEN_SPLIT_GINI) {
//
//		// TPDP

	} else {

		printf("Wrong criterion in init_criterion_cpu given: %i. Exiting ...\n", params->criterion);
		exit(0);

	}

}

/* --------------------------------------------------------------------------------
 * Frees memory taken by a criterion record.
 * --------------------------------------------------------------------------------
 */
void free_criterion_cpu(CRITERION_RECORD *crit_record, PARAMETERS *params, TRAINING_DATA *train_data) {

	if (params->criterion == CRITERION_MSE || params->criterion == CRITERION_EVEN_SPLIT_MSE) {

		// nothing to do

	} else if (params->criterion == CRITERION_GINI || params->criterion == CRITERION_EVEN_SPLIT_GINI) {

		free(crit_record->class_counts_left);
		free(crit_record->class_counts_right);

	} else {

		printf("Wrong criterion in free_criterion_cpu given: %i. Exiting ...\n", params->criterion);
		exit(0);

	}

	free(crit_record);

}

/* --------------------------------------------------------------------------------
 * Updates a criterion.
 * --------------------------------------------------------------------------------
 */
void inline update_criterion_cpu(CRITERION_RECORD *crit_record,
		PATTERN_LABEL_WEIGHT *XF_Y_W, int n_XF_Y_W, int new_pos,
		PARAMETERS *params, TRAINING_DATA *train_data) {

	if (params->criterion == CRITERION_MSE || params->criterion == CRITERION_EVEN_SPLIT_MSE) {

		int k;
		for (k = crit_record->current_pos; k < new_pos; k++) {
			int weight = XF_Y_W[k].weight;
			FLOAT_TYPE Yval = XF_Y_W[k].label;
			FLOAT_TYPE sq_Yval = Yval * Yval;
			crit_record->sum_left += weight*Yval;
			crit_record->sq_sum_left += weight*sq_Yval;
			crit_record->sum_right -= weight*Yval;
			crit_record->sq_sum_right -= weight*sq_Yval;
			crit_record->weight_left += weight;
			crit_record->weight_right -= weight;
		}
		crit_record->current_pos = new_pos;

		// left and right impurity
		crit_record->impurity_left = crit_record->sq_sum_left / crit_record->weight_left - (crit_record->sum_left / crit_record->weight_left)
				* (crit_record->sum_left / crit_record->weight_left);
		crit_record->impurity_right = crit_record->sq_sum_right / crit_record->weight_right - (crit_record->sum_right / crit_record->weight_right)
				* (crit_record->sum_right / crit_record->weight_right);

		int weight_all = crit_record->weight_left + crit_record->weight_right;
		FLOAT_TYPE fraction_left = crit_record->weight_left / (FLOAT_TYPE) weight_all;
		FLOAT_TYPE fraction_right = crit_record->weight_right / (FLOAT_TYPE) weight_all;

		// improvement
		if (params->criterion == CRITERION_EVEN_SPLIT_MSE){
			crit_record->improvement = (1 - params->lam_crit) * (crit_record->impurity - fraction_left * crit_record->impurity_left - fraction_right * crit_record->impurity_right);
			crit_record->improvement = crit_record->improvement - params->lam_crit*(fabs(fraction_left - fraction_right));
		} else {
			crit_record->improvement = crit_record->impurity - fraction_left * crit_record->impurity_left - fraction_right * crit_record->impurity_right;
		}

	} else if (params->criterion == CRITERION_GINI) {

		int i, k;
		for (k = crit_record->current_pos; k < new_pos; k++) {
			int weight = XF_Y_W[k].weight;
			int label = (int)XF_Y_W[k].label;
			crit_record->class_counts_left[label] += weight;
			crit_record->class_counts_right[label] -= weight;
			crit_record->weight_left += weight;
			crit_record->weight_right -= weight;
		}
		crit_record->current_pos = new_pos;

		crit_record->impurity_left = 0.0;
		crit_record->impurity_right = 0.0;

		for (i = 0; i < train_data->n_classes; i++) {

			// left impurity
			FLOAT_TYPE pmk_left = (1.0 / crit_record->weight_left) * (FLOAT_TYPE) crit_record->class_counts_left[i];
			crit_record->impurity_left += pmk_left * (1.0 - pmk_left);

			// right impurity
			FLOAT_TYPE pmk_right = (1.0 / crit_record->weight_right) * (FLOAT_TYPE) crit_record->class_counts_right[i];
			crit_record->impurity_right += pmk_right * (1.0 - pmk_right);

		}

		int weight_all = crit_record->weight_left + crit_record->weight_right;
		FLOAT_TYPE fraction_left = crit_record->weight_left / (FLOAT_TYPE) weight_all;
		FLOAT_TYPE fraction_right = crit_record->weight_right / (FLOAT_TYPE) weight_all;

		// improvement
		if (params->criterion == CRITERION_EVEN_SPLIT_GINI){
			crit_record->improvement = (1 - params->lam_crit) * (crit_record->impurity - fraction_left * crit_record->impurity_left - fraction_right * crit_record->impurity_right);
			crit_record->improvement = crit_record->improvement - params->lam_crit*(fabs(fraction_left - fraction_right));
		} else {
			crit_record->improvement = crit_record->impurity - fraction_left * crit_record->impurity_left - fraction_right * crit_record->impurity_right;
		}

//	} else if (params->criterion == CRITERION_EVEN_SPLIT_MSE ||
//			   params->criterion == CRITERION_EVEN_SPLIT_GINI) {
//
//		int k;
//
//		for (k = crit_record->current_pos; k < new_pos; k++) {
//			int weight = XF_Y_W[k].weight;
//			crit_record->weight_left += weight;
//			crit_record->weight_right -= weight;
//		}
//		crit_record->current_pos = new_pos;
//
//		// IGNORE: left and right impurity
//		crit_record->impurity_left = 1.0;
//		crit_record->impurity_right = 1.0;
//		crit_record->impurity = 1.0;
//
//		int weight_all = crit_record->weight_left + crit_record->weight_right;
//		FLOAT_TYPE fraction_left = crit_record->weight_left / (FLOAT_TYPE) weight_all;
//		FLOAT_TYPE fraction_right = crit_record->weight_right / (FLOAT_TYPE) weight_all;
//
//		// improvement
//		crit_record->improvement = 1.0 / (fabs(fraction_left - fraction_right) + 0.00000001);
//

	} else {

		printf("Wrong criterion in update_criterion_cpu given: %i. Exiting ...\n", params->criterion);
		exit(0);

	}

}
