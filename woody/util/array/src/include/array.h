/*
 * util.h
 */
#ifndef INCLUDE_ARRAY_H_
#define INCLUDE_ARRAY_H_

#include "global.h"
#include "util.h"

#include <stdlib.h>
#include <omp.h>

/* --------------------------------------------------------------------------------
 * Splits the array X according to the indices
 * --------------------------------------------------------------------------------
 */
void split_array(FLOAT_TYPE *X, int nX, int dX, FLOAT_TYPE *Xnew, int nXnew, int dXnew, int *indicator, int nindicator, int *chunks, int nchunks, int *cumsums_minus_counts, int ncumsums_minus_counts);

/* --------------------------------------------------------------------------------
 * Computes split offsets
 * --------------------------------------------------------------------------------
 */
void compute_split_offsets(int *offsets, int noffsets,
							int *indicator, int nindicator,
							int *chunks, int nchunks,
							int *cumsums_minus_counts, int ncumsums_minus_counts);

/* --------------------------------------------------------------------------------
 * Transposes an array
 * --------------------------------------------------------------------------------
 */
void transpose_array(FLOAT_TYPE* X, int nX, int dX, FLOAT_TYPE* XT, int nXT, int dXT);

#endif
