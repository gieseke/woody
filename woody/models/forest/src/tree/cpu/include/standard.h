/*
 * standard.h
 *
 *  Created on: 23.01.2017
 *      Author: fgieseke
 */

#ifndef ENSEMBLE_CPU_INCLUDE_STANDARD_H_
#define ENSEMBLE_CPU_INCLUDE_STANDARD_H_

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "criteria.h"
#include "fastsort.h"

#include "../../include/global.h"
#include "../../include/util.h"


#define size_threshold 16

void intro_sort(PATTERN_LABEL_WEIGHT *a, int n);

FLOAT_TYPE compute_optimal_threshold(PATTERN_LABEL_WEIGHT *XF_Y_W, int n_XF_Y_W, PARAMETERS *params, TRAINING_DATA *train_data, SPLIT_RECORD *best_split);

#endif /* ENSEMBLE_CPU_INCLUDE_STANDARD_H_ */
