/*
 * fastsort.c
 *
 *  Created on: 23.01.2017
 *      Author: fgieseke
 */
#include "include/fastsort.h"


#define swap_fast(a1, a2, s1, s2) { \
register FLOAT_TYPE tmp = *(a1); \
*(a1) = *(a2); \
*(a2) = tmp; \
register int tmpint = *(s1); \
*(s1) = *(s2); \
*(s2) = tmpint; \
}

inline static int fast_partition(FLOAT_TYPE *a, int *samples, int lo, int hi, FLOAT_TYPE x);
inline static int fast_floor_lg(int a);
static FLOAT_TYPE fast_medianof3(FLOAT_TYPE *a, int lo, int mid, int hi);
static void fast_downheap(FLOAT_TYPE *a, int i, int n, int lo);
static void fast_heapsort(FLOAT_TYPE *a, int *samples, int lo, int hi);
static void fast_introsort_loop(FLOAT_TYPE *a, int *samples, int lo, int hi, int depth_limit);
static void fast_insertionsort(FLOAT_TYPE *a, int *samples, int lo, int hi);

void combined_sort(FLOAT_TYPE *XF, int *samples, int n) {

	fast_introsort_loop(XF, samples, 0, n, 2 * fast_floor_lg(n));
	fast_insertionsort(XF, samples, 0, n);
}

static void fast_introsort_loop(FLOAT_TYPE *a, int *samples, int lo, int hi, int depth_limit) {
	int p = -1;

	while (hi - lo > fast_size_threshold) {

		if (depth_limit == 0) {
			fast_heapsort(a, samples, lo, hi);
			return;
		}
		depth_limit--;

		p = fast_partition(a, samples, lo, hi, fast_medianof3(a, lo, lo + ((hi - lo) / 2) + 1, hi - 1));

		fast_introsort_loop(a, samples, p, hi, depth_limit);
		hi = p;
	}
}

inline static int fast_partition(FLOAT_TYPE *a, int *samples, int lo, int hi, FLOAT_TYPE x) {
	int i = lo, j = hi;
	while (1) {
		while (a[i] < x)
			i++;
		j--;
		while (x < a[j])
			j--;
		if (i >= j)
			return i;
		swap_fast(&a[i], &a[j], &samples[i], &samples[j]);
		i++;
	}
}

inline static FLOAT_TYPE fast_medianof3(FLOAT_TYPE *a, int lo, int mid, int hi) {

	if (a[mid] < a[lo]) {

		if (a[hi] < a[mid])
			return a[mid];
		else {
			if (a[hi] < a[lo])
				return a[hi];
			else
				return a[lo];
		}
	} else {
		if (a[hi] < a[mid]) {
			if (a[hi] < a[lo])
				return a[lo];
			else
				return a[hi];
		} else
			return a[mid];
	}
}

static void fast_heapsort(FLOAT_TYPE *a, int *samples, int lo, int hi) {
	int n = hi - lo;
	int i;
	for (i = n / 2; i >= 1; i--) {
		fast_downheap(a, i, n, lo);
	}
	for (i = n; i > 1; i--) {
		swap_fast(&a[lo], &a[lo + i - 1], &samples[lo], &samples[lo + i -1]);
		fast_downheap(a, 1, i - 1, lo);
	}
}

inline static void fast_downheap(FLOAT_TYPE *a, int i, int n, int lo) {
	FLOAT_TYPE d = a[lo + i - 1];
	int child;
	int n2 = n / 2;
	while (i <= n2) {
		child = 2 * i;
		if (child < n && a[lo + child - 1] < a[lo + child])
			child++;
		if (d >= a[lo + child - 1])
			break;
		a[lo + i - 1] = a[lo + child - 1];
		i = child;
	}
	a[lo + i - 1] = d;
}


static void fast_insertionsort(FLOAT_TYPE *a, int *samples, int lo, int hi) {
	int i, j;
	FLOAT_TYPE tfloat;
	int tint;
	for (i = lo; i < hi; i++) {
		j = i;
		tfloat = a[i];
		tint = samples[i];
		while (j != lo && tfloat < a[j - 1]) {
			a[j] = a[j - 1];
			samples[j] = samples[j - 1];
			j--;
		}
		a[j] = tfloat;
		samples[j] = tint;
	}
}

inline static int fast_floor_lg(int a) {
	return (int) floor(log(a) / log(2));
}
