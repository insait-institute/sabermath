#include <float.h>
#include "bma.h"

double c_bma_one_way(
    double** similarity_matrix, size_t size,
    size_t* A, size_t A_len,
    size_t* B, size_t B_len,
    int* invalid_idx_flag, size_t* invalid_idx_out
) {
    if (invalid_idx_flag) {
        *invalid_idx_flag = 0;
    }

    if (A_len == 0 || B_len == 0) {
        return 0.0f;
    }

    double sum = 0.0f;

    for (size_t i = 0; i < A_len; ++i) {
        size_t a = A[i];
        if (a >= size && invalid_idx_flag) {
            *invalid_idx_flag = 1;
            if (invalid_idx_out) {
                *invalid_idx_out = a;
            }
            return 0.0f;
        }

        double max_sim = -DBL_MAX;

        for (size_t j = 0; j < B_len; ++j) {
            size_t b = B[j];
            if (b >= size && invalid_idx_flag) {
                *invalid_idx_flag = 1;
                if (invalid_idx_out) {
                    *invalid_idx_out = b;
                }
                return 0.0f;
            }

            double s = similarity_matrix[a][b];
            if (s > max_sim) {
                max_sim = s;
            }
        }

        sum += max_sim;
    }

    return sum / (double)(A_len + B_len);
}

double* c_bma_one_against_many(
    double** similarity_matrix, size_t size,
    size_t* one_idxs, size_t one_len,
    size_t** many_idxs, size_t* many_lens, size_t many_count,
    int* invalid_idx_flag, size_t* invalid_idx_out
) {
    if (invalid_idx_flag) {
        *invalid_idx_flag = 0;
    }

    double* output = (double*)malloc(many_count * sizeof(double));
    if (!output) {
        return NULL;
    }

    for (size_t k = 0; k < many_count; ++k) {
        size_t* rhs_idxs = many_idxs[k];
        size_t rhs_len = many_lens[k];

        double one_to_many_avg = c_bma_one_way(
            similarity_matrix, size,
            one_idxs, one_len,
            rhs_idxs, rhs_len,
            invalid_idx_flag, invalid_idx_out
        );

        if (invalid_idx_flag && *invalid_idx_flag) {
            free(output);
            return NULL;
        }

        double many_to_one_avg = c_bma_one_way(
            similarity_matrix, size,
            rhs_idxs, rhs_len,
            one_idxs, one_len,
            invalid_idx_flag, invalid_idx_out
        );

        if (invalid_idx_flag && *invalid_idx_flag) {
            free(output);
            return NULL;
        }

        output[k] = one_to_many_avg + many_to_one_avg;
    }

    return output;
}

double** c_bma_many_against_many(
    double** similarity_matrix, size_t size,
    size_t** lhs_idxs, size_t* lhs_lens, size_t lhs_count,
    size_t** rhs_idxs, size_t* rhs_lens, size_t rhs_count,
    int* invalid_idx_flag, size_t* invalid_idx_out
) {
    if (invalid_idx_flag) {
        *invalid_idx_flag = 0;
    }

    double** output = (double**)malloc(lhs_count * sizeof(double*));

    if (!output) {
        return NULL;
    }

    for (size_t i = 0; i < lhs_count; ++i) {
        double* row = c_bma_one_against_many(
            similarity_matrix, size,
            lhs_idxs[i], lhs_lens[i],
            rhs_idxs, rhs_lens, rhs_count,
            invalid_idx_flag, invalid_idx_out
        );

        if (!row) {
            for (size_t j = 0; j < i; ++j)
                free(output[j]);
            free(output);
            return NULL;
        }

        output[i] = row;
    }

    return output;
}
