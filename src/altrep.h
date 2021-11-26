#ifndef RETICULATE_ALTREP_H
#define RETIULATE_ALTREP_H

#include "libpython.h"

SEXP create_altrep_numpy_double_array(PyArrayObject*);

SEXP create_altrep_numpy_integer_array(PyArrayObject*);

SEXP create_altrep_numpy_logical_array(PyArrayObject*);

SEXP create_altrep_numpy_complex_array(PyArrayObject*);

#endif
