#include "similarity_matrix.h"

static void SimilarityMatrix_dealloc(SimilarityMatrixObject* self) {
    if (self->data != NULL) {
        free(self->data);
        self->data = NULL;
    }

    if (self->_buffer != NULL) {
        free(self->_buffer);
        self->_buffer = NULL;
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int SimilarityMatrix_init(SimilarityMatrixObject* self, PyObject* args, PyObject* kwds) {
    PyObject* list_of_lists;

    self->data = NULL;
    self->_buffer = NULL;
    self->rows = 0;
    self->cols = 0;

    if (!PyArg_ParseTuple(args, "O", &list_of_lists))
        return -1;

    if (!PyList_Check(list_of_lists)) {
        PyErr_SetString(PyExc_TypeError, "Expected a matrix");
        return -1;
    }

    self->rows = PyList_Size(list_of_lists);
    if (self->rows == 0) {
        PyErr_SetString(PyExc_ValueError, "Similarity matrix cannot be empty");
        return -1;
    }

    PyObject *first_row = PyList_GetItem(list_of_lists, 0);
    if (!PyList_Check(first_row)) {
        PyErr_SetString(PyExc_ValueError, "Expected a matrix");
        return -1;
    }

    self->cols = PyList_Size(first_row);
    if (self->cols == 0) {
        PyErr_SetString(PyExc_ValueError, "Rows in similarity matrix cannot be empty");
        return -1;
    }

    if (self->rows != self->cols) {
        PyErr_SetString(PyExc_ValueError, "Similarity Matrix must be square");
        return -1;
    }

    self->_buffer = (double*)malloc(self->rows * self->cols * sizeof(double));
    if (!self->_buffer) {
        PyErr_NoMemory();
        return -1;
    }

    self->data = (double**)malloc(self->rows * sizeof(double*));
    if (!self->data) {
        free(self->_buffer);
        self->_buffer = NULL;
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t r = 0; r < self->rows; ++r) {
        self->data[r] = self->_buffer + r * self->cols;
    }

    for (Py_ssize_t r = 0; r < self->rows; ++r) {
        PyObject* row = PyList_GetItem(list_of_lists, r);
        if (!PyList_Check(row)) {
            PyErr_SetString(PyExc_TypeError, "Expected a matrix");
            if (self->data) {
                free(self->data);
                self->data = NULL;
            }
            if (self->_buffer) {
                free(self->_buffer);
                self->_buffer = NULL;
            }
            return -1;
        }
        if (PyList_Size(row) != self->cols) {
            PyErr_SetString(PyExc_ValueError, "All rows must have the same length");
            if (self->data) {
                free(self->data);
                self->data = NULL;
            }
            if (self->_buffer) {
                free(self->_buffer);
                self->_buffer = NULL;
            }
            return -1;
        }

        for (Py_ssize_t c = 0; c < self->cols; ++c) {
            PyObject* item = PyList_GetItem(row, c);
            double val = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                if (self->data) {
                free(self->data);
                self->data = NULL;
            }
            if (self->_buffer) {
                free(self->_buffer);
                self->_buffer = NULL;
            }
                return -1;
            }
            self->data[r][c] = val;
        }
    }

    return 0;
}

static PyObject* SimilarityMatrix_repr(SimilarityMatrixObject* self) {
    return PyUnicode_FromFormat("<fastbma.SimilarityMatrix %zd x %zd>", self->rows, self->cols);
}

static PyMemberDef SimilarityMatrix_members[] = {
    {"rows", T_PYSSIZET, offsetof(SimilarityMatrixObject, rows), READONLY, "number of rows"},
    {"cols", T_PYSSIZET, offsetof(SimilarityMatrixObject, rows), READONLY, "number of columns"},
    {NULL},
};

PyTypeObject SimilarityMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "fastbma.SimilarityMatrix",
    .tp_basicsize = sizeof(SimilarityMatrixObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Matrix of pairwise similarities",
    .tp_init = (initproc) SimilarityMatrix_init,
    .tp_dealloc = (destructor) SimilarityMatrix_dealloc,
    .tp_repr = (reprfunc) SimilarityMatrix_repr,
    .tp_members = SimilarityMatrix_members,
    .tp_new = PyType_GenericNew,
};
