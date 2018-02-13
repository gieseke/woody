%module wrapper_utils_cpu_float

%{
    #define SWIG_FILE_WITH_INIT
    #include "array.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE *X, int nX, int dX)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE *XT, int nXT, int dXT)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE *Xnew, int nXnew, int dXnew)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *y, int ny)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *ynew, int nynew)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *offsets, int noffsets)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *indicator, int nindicator)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *chunks, int nchunks)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *counts, int ncounts)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int *cumsums_minus_counts, int ncumsums_minus_counts)}

%include "array.h"      
