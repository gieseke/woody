#include "include/array.h"

/* --------------------------------------------------------------------------------
 * Copies a single pattern
 * --------------------------------------------------------------------------------
 */
inline void copy_pattern(FLOAT_TYPE *src, FLOAT_TYPE *dst, int dim){

	int j;

	// memcpy seems to be slower (function call)
    for (j=0; j<dim; j++){
        dst[j] = src[j];
    }

}




