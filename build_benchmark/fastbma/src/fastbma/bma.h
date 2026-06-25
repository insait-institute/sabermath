#ifndef FASTBMA_BMA_H
#define FASTBMA_BMA_H

#include <stddef.h>
#include <stdlib.h>

double** c_bma_many_against_many(
    double** similarity_matrix, size_t size,
    size_t** lhs_idxs, size_t* lhs_lens, size_t lhs_count,
    size_t** rhs_idxs, size_t* rhs_lens, size_t rhs_count,
    int* invalid_idx_flag, size_t* invalid_idx_out
);

double* c_bma_one_against_many(
    double** similarity_matrix, size_t size,
    size_t* one_idxs, size_t one_len,
    size_t** many_idxs, size_t* many_lens, size_t many_count,
    int* invalid_idx_flag, size_t* invalid_idx_out
);

#endif // FASTBMA_BMA_H
