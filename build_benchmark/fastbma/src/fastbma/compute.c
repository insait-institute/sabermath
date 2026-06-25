#include "bma.h"
#include "similarity_matrix.h"

int parse_list_of_lists(
    PyObject* obj,
    size_t*** values_out,
    size_t** lens_out,
    size_t* outer_len_out
) {
    *values_out = NULL;
    *lens_out = NULL;
    *outer_len_out = 0;

    if (!PyList_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of non-negative indices");
        return 0;
    }

    Py_ssize_t outer_len = PyList_Size(obj);

    if (outer_len < 0) {
        return 0;
    }

    size_t* lens = malloc(outer_len * sizeof(size_t));
    if (!lens) {
        PyErr_NoMemory();
        return 0;
    }

    size_t** values = malloc(outer_len * sizeof(size_t*));
    if (!values) {
        free(lens);
        PyErr_NoMemory();
        return 0;
    }

    for (Py_ssize_t i = 0; i < outer_len; ++i) {
        PyObject* row = PyList_GetItem(obj, i);
        if (!PyList_Check(row)) {
            free(lens);
            free(values);
            PyErr_SetString(PyExc_TypeError, "Inner elements must be lists of indices");
            return 0;
        }

        Py_ssize_t inner_len = PyList_Size(row);

        if (inner_len < 0) {
            free(lens);
            free(values);
            return 0;
        }

        lens[i] = (size_t)inner_len;
    }

    for (Py_ssize_t i = 0; i < outer_len; ++i) {
        size_t len = lens[i];

        values[i] = malloc(len * sizeof(size_t));
        if (!values[i]) {
            free(lens);
            for (Py_ssize_t k = 0; k < i; ++k) {
                free(values[k]);
            }
            free(values);
            PyErr_NoMemory();
            return 0;
        }

        PyObject* row = PyList_GetItem(obj, i);

        for (Py_ssize_t j = 0; j < (Py_ssize_t)len; ++j) {
            PyObject* item = PyList_GetItem(row, j);
            unsigned long long val = PyLong_AsUnsignedLongLong(item);
            if (PyErr_Occurred()) {
                for (Py_ssize_t k = 0; k <= i; ++k) {
                    free(values[k]);
                }
                free(lens);
                free(values);
                return 0;
            }

            if (val > (unsigned long long)SIZE_MAX) {
                for (Py_ssize_t k = 0; k <= i; ++k) {
                    free(values[k]);
                }
                free(lens);
                free(values);
                PyErr_SetString(PyExc_OverflowError, "Integer too large for size_t");
                return 0;
            }

            values[i][j] = (size_t)val;
        }
    }

    *values_out = values;
    *lens_out = lens;
    *outer_len_out = (size_t)outer_len;
    return 1;
}

PyObject* bma_compute(PyObject* self, PyObject* args) {
    PyObject* sim_matrix = NULL;
    PyObject* lhs_list = NULL;
    PyObject* rhs_list = NULL;

    if(!PyArg_ParseTuple(args, "OOO", &sim_matrix, &lhs_list, &rhs_list)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(sim_matrix, &SimilarityMatrixType)) {
        PyErr_SetString(PyExc_TypeError, "Expected a fastbma.SimilarityMatrix");
        return NULL;
    }

    SimilarityMatrixObject* S = (SimilarityMatrixObject*)sim_matrix;

    size_t** lhs_idxs = NULL;
    size_t* lhs_lens = NULL;
    size_t lhs_count;

    size_t** rhs_idxs = NULL;
    size_t* rhs_lens = NULL;
    size_t rhs_count;

    if (!parse_list_of_lists(lhs_list, &lhs_idxs, &lhs_lens, &lhs_count)) {
        return NULL;
    }

    if (!parse_list_of_lists(rhs_list, &rhs_idxs, &rhs_lens, &rhs_count)) {
        for (size_t i = 0; i < lhs_count; ++i) {
            free(lhs_idxs[i]);
        }
        free(lhs_idxs);
        free(lhs_lens);
        return NULL;
    }

    double** bma_scores = c_bma_many_against_many(
        S->data, S->cols,
        lhs_idxs, lhs_lens, lhs_count,
        rhs_idxs, rhs_lens, rhs_count,
        NULL, NULL
    );

    if (!bma_scores) {
        PyErr_SetString(PyExc_RuntimeError, "c_bma_many_against_many failed");
        for (size_t i = 0; i < lhs_count; ++i) free(lhs_idxs[i]);
        for (size_t i = 0; i < rhs_count; ++i) free(rhs_idxs[i]);
        free(lhs_idxs);
        free(rhs_idxs);
        free(lhs_lens);
        free(rhs_lens);
        return NULL;
    }

    PyObject* py_result = PyList_New((Py_ssize_t)lhs_count);
    if (!py_result) {
        for (size_t n = 0; n < lhs_count; ++n) free(bma_scores[n]);
        free(bma_scores);
        for (size_t n = 0; n < lhs_count; ++n) free(lhs_idxs[n]);
        for (size_t n = 0; n < rhs_count; ++n) free(rhs_idxs[n]);
        free(lhs_idxs); free(rhs_idxs);
        free(lhs_lens); free(rhs_lens);
        return NULL;
    }

    for (size_t i = 0; i < lhs_count; ++i) {
        PyObject* row = PyList_New((Py_ssize_t)rhs_count);
        if (!row) {
            Py_DECREF(py_result);
            for (size_t n = 0; n < lhs_count; ++n) free(bma_scores[n]);
            free(bma_scores);
            for (size_t n = 0; n < lhs_count; ++n) free(lhs_idxs[n]);
            for (size_t n = 0; n < rhs_count; ++n) free(rhs_idxs[n]);
            free(lhs_idxs); free(rhs_idxs);
            free(lhs_lens); free(rhs_lens);
            return NULL;
        }

        for (size_t j = 0; j < rhs_count; ++j) {
            PyObject* val = PyFloat_FromDouble(bma_scores[i][j]);
            if (!val) {
                Py_DECREF(row);
                Py_DECREF(py_result);
                for (size_t n = 0; n < lhs_count; ++n) free(bma_scores[n]);
                free(bma_scores);
                for (size_t n = 0; n < lhs_count; ++n) free(lhs_idxs[n]);
                for (size_t n = 0; n < rhs_count; ++n) free(rhs_idxs[n]);
                free(lhs_idxs); free(rhs_idxs);
                free(lhs_lens); free(rhs_lens);
                return NULL;
            }

            PyList_SET_ITEM(row, (Py_ssize_t)j, val);
        }

        PyList_SET_ITEM(py_result, (Py_ssize_t)i, row);
    }

    for (size_t i = 0; i < lhs_count; ++i) {
        free(bma_scores[i]);
    }
    free(bma_scores);

    for (size_t i = 0; i < lhs_count; ++i) free(lhs_idxs[i]);
    for (size_t i = 0; i < rhs_count; ++i) free(rhs_idxs[i]);
    free(lhs_idxs);
    free(rhs_idxs);
    free(lhs_lens);
    free(rhs_lens);

    return py_result;
}
