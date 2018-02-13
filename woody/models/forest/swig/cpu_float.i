%module wrapper_cpu_float

%{
    #define SWIG_FILE_WITH_INIT
    #include "base.h"
    #include "types.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* Xtrain, int nXtrain, int dXtrain)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *Ytrain, int nYtrain)}

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* Xtest, int nXtest, int dXtest)}
%apply (float* INPLACE_ARRAY1, int DIM1) {(FLOAT_TYPE *predictions, int npredictions)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* preds, int npreds, int dpreds)}

%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int *bootstrap_indices, int nbootstrap_indices, int dbootstrap_indices)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int *bootstrap_indices_weights, int nbootstrap_indices_weights, int dbootstrap_indices_weights)}

%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int *indices, int nindices, int dindices)}

%apply (int* INPLACE_ARRAY1, int DIM1) {(int *aforest, int naforest)}

%include "base.h"      
%include "types.h"  
