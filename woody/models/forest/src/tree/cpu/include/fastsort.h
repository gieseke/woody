/*
 * fastsort.h
 *
 *  Created on: 23.01.2017
 *      Author: fgieseke
 */

#ifndef ENSEMBLE_CPU_INCLUDE_FASTSORT_H_
#define ENSEMBLE_CPU_INCLUDE_FASTSORT_H_

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "criteria.h"

#include "../../include/global.h"
#include "../../include/util.h"


#define fast_size_threshold 64

void combined_sort(FLOAT_TYPE *XF, int *samples, int n);

#endif /* ENSEMBLE_CPU_INCLUDE_FASTSORT_H_ */
