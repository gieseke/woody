#include "include/array.h"

/* --------------------------------------------------------------------------------
 * Splits the array X according to the indices
 * --------------------------------------------------------------------------------
 */
void split_array(FLOAT_TYPE *X, int nX, int dX,
		FLOAT_TYPE *Xnew, int nXnew, int dXnew,
		int *indicator, int nindicator,
		int *chunks, int nchunks,
		int *cumsums_minus_counts, int ncumsums_minus_counts){

    int i;

    int *offsets = (int*) malloc(nX * sizeof(int));

    compute_split_offsets(offsets, nX, indicator, nindicator, chunks, nchunks, cumsums_minus_counts, ncumsums_minus_counts);

    for(i=0; i<nX; i++){

    	if (offsets[i] < 0 || offsets[i] >= nX){
    		printf("Bad offset: %i [%i, %i]\n!", offsets[i], 0, nX);
    		exit(-1);
    	}
        copy_pattern(X + i * dX, Xnew + offsets[i] * dX, dX);
    }

    free(offsets);

}


void compute_split_offsets(int *offsets, int noffsets,
							int *indicator, int nindicator,
							int *chunks, int nchunks,
							int *cumsums_minus_counts, int ncumsums_minus_counts){

    int i;
    int *chunks_counters = (int*) calloc(nchunks, sizeof(int));

    for(i=0; i<noffsets; i++){

		int chunk = chunks[indicator[i]];
		//printf("chunk=%i\n", chunk);
		//offsets[i] = 0;
		//printf("cumsums[chunk-1]=%i\n", cumsums[chunk-1]);
		//printf("chunks_counters[chunk]=%i\n", chunks_counters[chunk]);

		// FIXME: Wrong access here? invalid chunk? chunk_counters wrong?
		// afterwards: offsets[i]==1892929992 (WRONG)
		// chunk -1 invalid? why -1? cumsums[chunk] should work
		// ++ wrong?
        offsets[i] = cumsums_minus_counts[chunk] + chunks_counters[chunk];
        chunks_counters[chunk]++;

    }

    free(chunks_counters);

}


/* --------------------------------------------------------------------------------
 * Transposes an array
 * --------------------------------------------------------------------------------
 */
void transpose_array(FLOAT_TYPE* X, int nX, int dX, FLOAT_TYPE* XT, int nXT, int dXT){

	int i, j;

	for (j = 0; j < dX; j++) {
		for (i = 0; i < nX; i++) {

			XT[j * nX + i] = X[i * dX + j];

		}
	}

}

