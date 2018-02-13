/*
 * util.h
 */
#ifndef COMMON_INCLUDE_UTIL_H_
#define COMMON_INCLUDE_UTIL_H_

#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>

#include "float.h"

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define ELEM_SWAP(a,b) { register FLOAT_TYPE t=(a);(a)=(b);(b)=t; }
#define median(a,n) kth_smallest(a,n,((n)/2))

/* --------------------------------------------------------------------------------
 * Transposes an array (float)
 * --------------------------------------------------------------------------------
 */
void transpose_array_float(float* array, int n, int d, float* array_transposed);

/* --------------------------------------------------------------------------------
 * Transposes an array (double)
 * --------------------------------------------------------------------------------
 */
void transpose_array_double(double* array, int n, int d,
		double* array_transposed);

int compare_floats(const void *p1, const void *p2);

int compare_ints(const void *p1, const void *p2);

FLOAT_TYPE kth_smallest(FLOAT_TYPE a[], int n, int k);
int kth_smallest_idx(FLOAT_TYPE a[], int n, int k);

#endif
