#ifndef RETICULATE_ALTREP_H
#define RETIULATE_ALTREP_H

#include "libpython.h"

SEXP create_altreal_from_numpy_array(PyObject*);

SEXP create_altinteger_from_numpy_array(PyObject*);

SEXP create_altlogical_from_numpy_array(PyObject*);

SEXP create_altcomplex_from_numpy_array(PyObject*);

#endif
