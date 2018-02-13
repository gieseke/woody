/*
 * standard.c
 *
 *  Created on: 23.01.2017
 *      Author: fgieseke
 */
#include "include/standard.h"


FLOAT_TYPE compute_optimal_threshold(PATTERN_LABEL_WEIGHT *XF_Y_W, int n_XF_Y_W, PARAMETERS *params, TRAINING_DATA *train_data, SPLIT_RECORD *best_split){

	// Important: We are checking a non-constant feature here.

	// TODO: Scikit-Learn is still faster here since:
	// - it only sorts an array of features+samples (with only features being used for comparisons!)
	// - the subsequent updates only depend on the samples
	// -> about 2.5 faster!
	/*
	int i;

	FLOAT_TYPE *XF = (FLOAT_TYPE*) malloc(n_XF_Y_W * sizeof(FLOAT_TYPE));
	int *samples = (int*) malloc(n_XF_Y_W * sizeof(int));

	for(i=0; i<n_XF_Y_W; i++){
		XF[i] = XF_Y_W[i].pattern;
		samples[i] = i;
	}
	combined_sort(XF, samples, n_XF_Y_W);
*/

	// Generate local copy (since the patterns will be sorted here)
	PATTERN_LABEL_WEIGHT *XF_Y_W_copy = (PATTERN_LABEL_WEIGHT*) malloc(n_XF_Y_W * sizeof(PATTERN_LABEL_WEIGHT));
	memcpy(XF_Y_W_copy, XF_Y_W, n_XF_Y_W * sizeof(PATTERN_LABEL_WEIGHT));
	intro_sort(XF_Y_W_copy, n_XF_Y_W);

/*
	for(i=0; i<n_XF_Y_W; i++){
		XF_Y_W_copy[i].pattern = XF[i];
		XF_Y_W_copy[i].label = XF_Y_W[samples[i]].label;
		XF_Y_W_copy[i].weight = XF_Y_W[samples[i]].weight;
	}

	free(XF);
	free(samples);
	*/

	// init criterion record
	CRITERION_RECORD *criterion_record = (CRITERION_RECORD*) malloc(sizeof(CRITERION_RECORD));
	init_criterion_cpu(criterion_record, XF_Y_W_copy, n_XF_Y_W, params, train_data);
	FLOAT_TYPE best_improvement = MIN_FLOAT_TYPE;

	int start = 0;
	int end = n_XF_Y_W;

	FLOAT_TYPE threshold = 0.0;

	int p = start;
	while (p < end) {

		// Increase counter until feature difference is significant
		while ((p + 1 < end) && (XF_Y_W_copy[(p - start + 1)].pattern
						<= XF_Y_W_copy[(p - start)].pattern + FEATURE_THRESHOLD)) {
			p++;
		}

		// p==end possible; here, we already have p > start!
		p += 1;

		if (p < end) {

			// reject if min_samples_leaf is not guaranteed
			// NOTE: Can be done by starting p at start+min_samples_leaf to end-min_samples_leaf!
			if ((((p - start) < params->min_samples_leaf) || ((end - p) < params->min_samples_leaf))) {
				continue;
			}

			// update criterion w.r.t. new position p
			update_criterion_cpu(criterion_record, XF_Y_W_copy, n_XF_Y_W, p - start, params, train_data);

			// store results if improvement is better than before
			if (criterion_record->improvement > best_improvement) {
				// set threshold
				threshold = (XF_Y_W_copy[(p - 1 - start)].pattern + XF_Y_W_copy[(p - start)].pattern) / 2.0;
				if (threshold == XF_Y_W_copy[(p - start)].pattern) {
					threshold = XF_Y_W_copy[(p - 1 - start)].pattern;
				}
				best_improvement = criterion_record->improvement;
			}

		}

	}

	free_criterion_cpu(criterion_record, params, train_data);
	free(XF_Y_W_copy);

	return threshold;

}




#define swap(a, b) { \
register PATTERN_LABEL_WEIGHT tmp = *(a); \
*(a) = *(b); \
*(b) = tmp; \
}

inline static int floor_lg(int a);
static int partition(PATTERN_LABEL_WEIGHT *a, int lo, int hi, PATTERN_LABEL_WEIGHT x);
static PATTERN_LABEL_WEIGHT medianof3(PATTERN_LABEL_WEIGHT *a, int lo, int mid, int hi);
static void insertionsort(PATTERN_LABEL_WEIGHT *a, int lo, int hi);
static void downheap(PATTERN_LABEL_WEIGHT *a, int i, int n, int lo);
static void heapsort(PATTERN_LABEL_WEIGHT *a, int lo, int hi);
static void introsort_loop(PATTERN_LABEL_WEIGHT *a, int lo, int hi, int depth_limit);

void intro_sort(PATTERN_LABEL_WEIGHT *a, int n) {
	introsort_loop(a, 0, n, 2 * floor_lg(n));
	insertionsort(a, 0, n);
}

static void introsort_loop(PATTERN_LABEL_WEIGHT *a, int lo, int hi, int depth_limit) {
	int p;
	while (hi - lo > size_threshold) {
		if (depth_limit == 0) {
			heapsort(a, lo, hi);
			return;
		}
		depth_limit--;

		p = partition(a, lo, hi, medianof3(a, lo, lo + ((hi - lo) / 2) + 1, hi - 1));
		introsort_loop(a, p, hi, depth_limit);
		hi = p;
	}
}
inline static int partition(PATTERN_LABEL_WEIGHT *a, int lo, int hi, PATTERN_LABEL_WEIGHT x) {
	int i = lo, j = hi;
	while (1) {
		while (a[i].pattern < x.pattern)
			i++;
		j--;
		while (x.pattern < a[j].pattern)
			j--;
		if (i >= j)
			return i;
		swap(&a[i], &a[j]);
		i++;
	}
}

inline static PATTERN_LABEL_WEIGHT medianof3(PATTERN_LABEL_WEIGHT *a, int lo, int mid, int hi) {
	if (a[mid].pattern < a[lo].pattern) {
		if (a[hi].pattern < a[mid].pattern)
			return a[mid];
		else {
			if (a[hi].pattern < a[lo].pattern)
				return a[hi];
			else
				return a[lo];
		}
	} else {
		if (a[hi].pattern < a[mid].pattern) {
			if (a[hi].pattern < a[lo].pattern)
				return a[lo];
			else
				return a[hi];
		} else
			return a[mid];
	}
}

static void heapsort(PATTERN_LABEL_WEIGHT *a, int lo, int hi) {
	int n = hi - lo;
	int i;
	for (i = n / 2; i >= 1; i--) {
		downheap(a, i, n, lo);
	}
	for (i = n; i > 1; i--) {
		swap(&a[lo], &a[lo + i - 1]);
		downheap(a, 1, i - 1, lo);
	}
}

inline static void downheap(PATTERN_LABEL_WEIGHT *a, int i, int n, int lo) {
	PATTERN_LABEL_WEIGHT d = a[lo + i - 1];
	int child;
	int n2 = n / 2;
	while (i <= n2) {
		child = 2 * i;
		if (child < n && a[lo + child - 1].pattern < a[lo + child].pattern)
			child++;
		if (d.pattern >= a[lo + child - 1].pattern)
			break;
		a[lo + i - 1] = a[lo + child - 1];
		i = child;
	}
	a[lo + i - 1] = d;
}

static void insertionsort(PATTERN_LABEL_WEIGHT *a, int lo, int hi) {
	int i, j;
	PATTERN_LABEL_WEIGHT t;
	for (i = lo; i < hi; i++) {
		j = i;
		t = a[i];
		while (j != lo && t.pattern < a[j - 1].pattern) {
			a[j] = a[j - 1];
			j--;
		}
		a[j] = t;
	}
}

inline static int floor_lg(int a) {
	return (int) floor(log(a) / log(2));
}
