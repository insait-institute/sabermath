#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "similarity_matrix.h"
#include "compute.h"

static PyMethodDef c_bma_methods[] = {
    {
        "bma_compute",
        (PyCFunction)bma_compute,
        METH_VARARGS,
        PyDoc_STR(
            "bma_compute(sim_matrix, lhs_list, rhs_list) -> list[list[float]]\n"
            "\n"
            "Compute BMA scores for many-to-many index sets using a SimilarityMatrix."
        )
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef c_bma_module = {
    PyModuleDef_HEAD_INIT,
    "_c_bma",
    "Best-Match Average implementations in C",
    -1,
    c_bma_methods,
};

PyMODINIT_FUNC PyInit__c_bma(void) {
    PyObject* m;

    if (PyType_Ready(&SimilarityMatrixType) < 0)
        return NULL;

    m = PyModule_Create(&c_bma_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&SimilarityMatrixType);
    if (PyModule_AddObject(
            m, "SimilarityMatrix", (PyObject *)&SimilarityMatrixType
    ) < 0) {
        Py_DECREF(&SimilarityMatrixType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
