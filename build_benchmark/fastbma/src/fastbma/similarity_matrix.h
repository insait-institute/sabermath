#ifndef FASTBMA_SIMILARITY_MATRIX_H
#define FASTBMA_SIMILARITY_MATRIX_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    double** data;
    double* _buffer;
    Py_ssize_t rows;
    Py_ssize_t cols;
} SimilarityMatrixObject;

extern PyTypeObject SimilarityMatrixType;

#endif // FASTBMA_SIMILARITY_MATRIX_H
