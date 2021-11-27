#include "libpython.h"
#include "reticulate_types.h"
#include "altrep.h"

#include <Rcpp.h>
#include "R_ext/Rdynload.h"
#include "R_ext/Altrep.h"

R_altrep_class_t altreal_from_numpy_array_t;
R_altrep_class_t altinteger_from_numpy_array_t;
R_altrep_class_t altlogical_from_numpy_array_t;
R_altrep_class_t altcomplex_from_numpy_array_t;

/********************************
 ****** Converter classes *******
 ********************************/

struct ConverterToReal {
private:
    template<typename From>
    static Rcpp::NumericVector copy_array_to_rvector(PyArrayObject* array) {
        int len = PyArray_SIZE(array);
        Rcpp::NumericVector y(len);

        From* pData = (From*)PyArray_DATA(array);
        for (int i = 0; i < len; ++i) {
            y[i] = pData[i];
        }

        return y;
    }

    template<typename From>
    static R_xlen_t copy_region_into_buffer(PyArrayObject* array, R_xlen_t start, R_xlen_t size, double* output) {
        From* ptr = static_cast<From*>(PyArray_DATA(array));
        R_xlen_t j = start, len = PyArray_SIZE(array);
        for (R_xlen_t i = 0; i < size && j < len; ++i, ++j) {
            output[i] = ptr[j];
        }
        return j - start;
    }

public:    
    static R_altrep_class_t* altrep_class() {
        return &altreal_from_numpy_array_t;
    }

    static SEXP materialize(PyArrayObject* array) {
        int type = PyArray_TYPE(array);
        switch (type) {
            case NPY_DOUBLE:
                return copy_array_to_rvector<npy_double>(array);
            case NPY_FLOAT:
                return copy_array_to_rvector<float>(array);
            case NPY_UINT:
                return copy_array_to_rvector<unsigned int>(array);
            case NPY_ULONG:
                return copy_array_to_rvector<unsigned long>(array);
            case NPY_ULONGLONG:
                return copy_array_to_rvector<unsigned long long>(array);
            case NPY_LONG:
                return copy_array_to_rvector<long>(array);
            case NPY_LONGLONG:
                return copy_array_to_rvector<long long>(array);
            case NPY_HALF: 
                // ????? We don't have a C standard type for this; we'll have
                // to cast the entire array on the Python side, in which case
                // it would be cheaper to not have an ALTREP at all. I will
                // omit this type in downstream methods under this assumption.
            default:
                // well, this better not happen. Not sure I'm allowed
                // to throw exceptions during altrep evaluation.
                return Rcpp::NumericVector(PyArray_SIZE(array));
        } 
    }

    typedef double value_type;

    static const char * type_as_string() {
        return "double";
    }

    static bool useable_pointer(PyArrayObject* array) {
        int type = PyArray_TYPE(array);
        return type == NPY_DOUBLE;
    }

    static double extract(PyArrayObject* array, R_xlen_t i) {
        int type = PyArray_TYPE(array);
        void* ptr = PyArray_DATA(array);
        switch (type) {
            case NPY_DOUBLE:
                return static_cast<npy_double*>(ptr)[i];
            case NPY_FLOAT:
                return static_cast<float*>(ptr)[i];
            case NPY_UINT:
                return static_cast<unsigned int*>(ptr)[i];
            case NPY_ULONG:
                return static_cast<unsigned long*>(ptr)[i];
            case NPY_ULONGLONG:
                return static_cast<unsigned long long*>(ptr)[i];
            case NPY_LONG:
                return static_cast<long*>(ptr)[i];
            case NPY_LONGLONG:
                return static_cast<long long*>(ptr)[i];
            default:
                return NA_REAL; // err. this better not happen!
        }
    }

    static R_xlen_t extract(PyArrayObject* array, R_xlen_t start, R_xlen_t size, double* output) {
        int type = PyArray_TYPE(array);
        switch (type) {
            case NPY_DOUBLE:
                return copy_region_into_buffer<npy_double>(array, start, size, output);
            case NPY_FLOAT:
                return copy_region_into_buffer<float>(array, start, size, output);
            case NPY_UINT:
                return copy_region_into_buffer<unsigned int>(array, start, size, output);
            case NPY_ULONG:
                return copy_region_into_buffer<unsigned long>(array, start, size, output);
            case NPY_ULONGLONG:
                return copy_region_into_buffer<unsigned long long>(array, start, size, output);
            case NPY_LONG:
                return copy_region_into_buffer<npy_long>(array, start, size, output);
            case NPY_LONGLONG:
                return copy_region_into_buffer<long long>(array, start, size, output);
            default:
                return 0; // err. this better not happen.
        }
    }
};

struct ConverterToInteger {
private:
    template<typename From>
    static Rcpp::IntegerVector copy_array_to_rvector(PyArrayObject* array) {
        int len = PyArray_SIZE(array);
        Rcpp::IntegerVector y(len);

        From* pData = (From*)PyArray_DATA(array);
        for (int i = 0; i < len; ++i) {
            y[i] = pData[i];
        }

        return y;
    }

    template<typename From>
    static R_xlen_t copy_region_into_buffer(PyArrayObject* array, R_xlen_t start, R_xlen_t size, int* output) {
        From* ptr = static_cast<From*>(PyArray_DATA(array));
        R_xlen_t j = start, len = PyArray_SIZE(array);
        for (R_xlen_t i = 0; i < size && j < len; ++i, ++j) {
            output[i] = ptr[j];
        }
        return j - start;
    }

public:    
    static R_altrep_class_t* altrep_class() {
        return &altinteger_from_numpy_array_t;
    }

    static SEXP materialize(PyArrayObject* array) {
        int type = PyArray_TYPE(array);
        switch (type) {
            case NPY_BYTE:
                return copy_array_to_rvector<char>(array);
            case NPY_UBYTE:
                return copy_array_to_rvector<unsigned char>(array);
            case NPY_SHORT:
                return copy_array_to_rvector<short>(array);
            case NPY_USHORT:
                return copy_array_to_rvector<unsigned short>(array);
            case NPY_INT:
                return copy_array_to_rvector<int>(array);
            default:
                // well, this better not happen. Not sure I'm allowed
                // to throw exceptions during altrep evaluation.
                return Rcpp::NumericVector(PyArray_SIZE(array));
        } 
    }

    typedef int value_type;

    static const char * type_as_string() {
        return "integer";
    }

    static bool useable_pointer(PyArrayObject* array) {
        int type = PyArray_TYPE(array);
        return type == NPY_INT;
    }

    static double extract(PyArrayObject* array, R_xlen_t i) {
        int type = PyArray_TYPE(array);
        void* ptr = PyArray_DATA(array);
        switch (type) {
            case NPY_BYTE:
                return static_cast<char*>(ptr)[i];
            case NPY_UBYTE:
                return static_cast<unsigned char*>(ptr)[i];
            case NPY_SHORT:
                return static_cast<short*>(ptr)[i];
            case NPY_USHORT:
                return static_cast<unsigned short*>(ptr)[i];
            case NPY_INT:
                return static_cast<int*>(ptr)[i];
            default:
                return NA_INTEGER; // err. this better not happen!
        }
    }

    static R_xlen_t extract(PyArrayObject* array, R_xlen_t start, R_xlen_t size, int* output) {
        int type = PyArray_TYPE(array);
        switch (type) {
            case NPY_BYTE:
                return copy_region_into_buffer<char>(array, start, size, output);
            case NPY_UBYTE:
                return copy_region_into_buffer<unsigned char>(array, start, size, output);
            case NPY_SHORT:
                return copy_region_into_buffer<short>(array, start, size, output);
            case NPY_USHORT:
                return copy_region_into_buffer<unsigned short>(array, start, size, output);
            case NPY_INT:
                return copy_region_into_buffer<int>(array, start, size, output);
            default:
                return 0; // err. this better not happen.
        }
    }
};

struct ConverterToLogical {
public:    
    static R_altrep_class_t* altrep_class() {
        return &altlogical_from_numpy_array_t;
    }

    static SEXP materialize(PyArrayObject* array) {
        int len = PyArray_SIZE(array);
        Rcpp::LogicalVector y(len);

        npy_bool* pData = (npy_bool*)PyArray_DATA(array);
        for (int i = 0; i < len; ++i) {
            y[i] = pData[i];
        }

        return y;
    }

    typedef int value_type;

    static const char * type_as_string() {
        return "logical";
    }

    static bool useable_pointer(PyArrayObject* array) {
        return false;
    }

    static double extract(PyArrayObject* array, R_xlen_t i) {
        return static_cast<npy_bool*>(PyArray_DATA(array))[i];
    }

    static R_xlen_t extract(PyArrayObject* array, R_xlen_t start, R_xlen_t size, int* output) {
        npy_bool* ptr = static_cast<npy_bool*>(PyArray_DATA(array));
        R_xlen_t j = start, len = PyArray_SIZE(array);
        for (R_xlen_t i = 0; i < size && j < len; ++i, ++j) {
            output[i] = ptr[j];
        }
        return j - start;
    }
};

typedef struct { float real, imag; } npy_complex64;

struct ConverterToComplex {
private:
    template<typename From>
    static Rcpp::ComplexVector copy_array_to_rvector(PyArrayObject* array) {
        int len = PyArray_SIZE(array);
        Rcpp::ComplexVector y(len);

        From* pData = (From*)PyArray_DATA(array);
        for (int i = 0; i < len; ++i) {
            From data = pData[i];
            Rcomplex cpx;
            cpx.r = data.real;
            cpx.i = data.imag;
            y[i] = cpx;
        }

        return y;
    }

    template<typename From>
    static Rcomplex fetch_single(PyArrayObject* array, R_xlen_t i) {
        From in = static_cast<From*>(PyArray_DATA(array))[i];
        Rcomplex cpx;
        cpx.r = in.real;
        cpx.i = in.imag;
        return cpx;
    }

    template<typename From>
    static R_xlen_t copy_region_into_buffer(PyArrayObject* array, R_xlen_t start, R_xlen_t size, Rcomplex* output) {
        From* ptr = static_cast<From*>(PyArray_DATA(array));
        R_xlen_t j = start, len = PyArray_SIZE(array);

        for (R_xlen_t i = 0; i < size && j < len; ++i, ++j) {
            From data = ptr[j];
            Rcomplex cpx;
            cpx.r = data.real;
            cpx.i = data.imag;
            output[i] = cpx;
        }

        return j - start;
    }

public:
    static R_altrep_class_t* altrep_class() {
        return &altcomplex_from_numpy_array_t;
    }

    static SEXP materialize(PyArrayObject* array) {
        int type = PyArray_TYPE(array);
        switch (type) {
            case NPY_CDOUBLE:
                return copy_array_to_rvector<npy_complex128>(array); 
            case NPY_CFLOAT:
                return copy_array_to_rvector<npy_complex64>(array); 
            default:
                // oops. Not sure what to do here.
                return Rcpp::ComplexVector(PyArray_SIZE(array));
        }
    }

    typedef Rcomplex value_type;

    static const char * type_as_string() {
        return "complex";
    }

    static bool useable_pointer(PyArrayObject* array) {
        return false;
    }

    static Rcomplex extract(PyArrayObject* array, R_xlen_t i) {
        int type = PyArray_TYPE(array);
        switch (type) {
            case NPY_CDOUBLE:
                return fetch_single<npy_complex128>(array, i);
            case NPY_CFLOAT:
                return fetch_single<npy_complex64>(array, i);
            default:
                // oops. Not sure what to do here.
                {
                    Rcomplex cpx;
                    cpx.i = NA_REAL;
                    cpx.r = NA_REAL;
                    return cpx;
                };
        }
    }

    static R_xlen_t extract(PyArrayObject* array, R_xlen_t start, R_xlen_t size, Rcomplex* output) {
        int type = PyArray_TYPE(array);
        switch (type) {
            case NPY_CDOUBLE:
                return copy_region_into_buffer<npy_complex128>(array, start, size, output); 
            case NPY_CFLOAT:
                return copy_region_into_buffer<npy_complex64>(array, start, size, output); 
            default:
                // oops. Not sure what to do here.
                return 0;
        }
    }
};

/********************************
 ******* Altrep handlers ********
 ********************************/

template<class Converter>
struct AltvecFromNumpyArray {
    // Need to tell XPtr to decref rather than attempt to free the Python-owned memory.
    static void PyObjectXPtr_finalizer(PyObject* x) {
        Py_DecRef(x);
        return;    
    } 

    typedef Rcpp::XPtr<PyObject, Rcpp::PreserveStorage, PyObjectXPtr_finalizer> PyObjectXPtr;

    static SEXP create(PyObject* x) {
        // protect from Python's GC until PyObjectXPtr's finalizer runs.
        Py_IncRef(x);
        return R_new_altrep(*Converter::altrep_class(), PyObjectXPtr(x, true), R_NilValue);
    }

    static PyArrayObject* as_array_obj(SEXP vec) {
        PyObjectXPtr data1(R_altrep_data1(vec));
        return reinterpret_cast<PyArrayObject*>(data1.get());                
    }

    static SEXP materialize(SEXP vec) {
        SEXP data2 = R_altrep_data2(vec);
        if (data2 == R_NilValue) {
            data2 = Converter::materialize(as_array_obj(vec));
            R_set_altrep_data2(vec, data2);

            // Technically, I guess we could decref it now and replace it with
            // a null pointer so that the finalizer is a no-op. This enables
            // the Python GC to collect memory ASAP to reduce overall memory
            // usage, but all methods MUST check for data2 non-NULL'ness before
            // attempting to access data1's array data.
        }
        return data2;
    }

    // ALTREP methods -------------------

    static R_xlen_t length(SEXP vec) {
        SEXP data2 = R_altrep_data2(vec);
        if (data2 == R_NilValue) {
            return PyArray_SIZE(as_array_obj(vec));
        } else {
            return LENGTH(data2);
        }
    }

    static Rboolean inspect(SEXP vec, int pre, int deep, int pvec, void (*inspect_subtree)(SEXP, int, int, int)){
        R_xlen_t len = length(vec);
        SEXP data2 = R_altrep_data2(vec);
        const char* mode = (data2 == R_NilValue ? "lazy" : "materialized");
        Rprintf("%s %s numpy vector (len=%d)\n", mode, Converter::type_as_string(), len);
        return TRUE;
    }

    // ALTVEC methods ------------------

    static const void* dataptr_or_null(SEXP vec){
        SEXP data2 = R_altrep_data2(vec);
        if (data2 == R_NilValue) {
            return nullptr;
        } else {
            return STDVEC_DATAPTR(data2);
        }
    }

    static void* dataptr(SEXP vec, Rboolean writeable){
        if (writeable) {
            return STDVEC_DATAPTR(materialize(vec));
        } 

        PyArrayObject* ptr = as_array_obj(vec);
        if (Converter::useable_pointer(ptr)) {
            return PyArray_DATA(ptr);
        } else {
            return STDVEC_DATAPTR(materialize(vec));
        }
    }

    // Initializer for the entire class.
  
    static void init() {
        R_altrep_class_t* class_t = Converter::altrep_class();
        R_set_altrep_Length_method(*class_t, length);
        R_set_altrep_Inspect_method(*class_t, inspect);
        R_set_altvec_Dataptr_method(*class_t, dataptr);
        R_set_altvec_Dataptr_or_null_method(*class_t, dataptr_or_null);
    } 
};

// ALTREAL/INTEGER/etc. methods -----------------

template<class Converter>
static typename Converter::value_type elt_from_numpy_array(SEXP vec, R_xlen_t i) {
    SEXP data2 = R_altrep_data2(vec);
    if (data2 == R_NilValue) {
        Rcpp::XPtr<PyArrayObject> data1(R_altrep_data1(vec));
        return Converter::extract(data1.get(), i);
    } else {
        return static_cast<const typename Converter::value_type*>(STDVEC_DATAPTR(data2))[i];
    }
}

template<class Converter>
R_xlen_t get_region_from_numpy_array(SEXP vec, R_xlen_t start, R_xlen_t size, typename Converter::value_type* out){
    SEXP data2=R_altrep_data2(vec);
    if (data2==R_NilValue) {
        PyArrayObject* data1 = AltvecFromNumpyArray<Converter>::as_array_obj(vec);
        return Converter::extract(data1, start, size, out);
    } else {
        const typename Converter::value_type* ptr = static_cast<const typename Converter::value_type*>(STDVEC_DATAPTR(data2));
        R_xlen_t j = start, len = AltvecFromNumpyArray<Converter>::length(vec);
        for (R_xlen_t i = 0; i < size && j < len; ++i, ++j) {
            out[i] = ptr[j];
        }
        return j - start;
    }
}

/********************************
 ******* Exported methods *******
 ********************************/

SEXP create_altreal_from_numpy_array(PyObject* ptr) {
    return AltvecFromNumpyArray<ConverterToReal>::create(ptr);
}

SEXP create_altinteger_from_numpy_array(PyObject* ptr) {
    return AltvecFromNumpyArray<ConverterToInteger>::create(ptr);
}

SEXP create_altlogical_from_numpy_array(PyObject* ptr) {
    return AltvecFromNumpyArray<ConverterToLogical>::create(ptr);
}

SEXP create_altcomplex_from_numpy_array(PyObject* ptr) {
    return AltvecFromNumpyArray<ConverterToComplex>::create(ptr);
}

// [[Rcpp::init]]
void init_altvec_from_numpy_array(DllInfo* dll) {
    altreal_from_numpy_array_t = R_make_altreal_class("altreal_from_numpy_array", "reticulate", dll); \
    AltvecFromNumpyArray<ConverterToReal>::init(); 
    R_set_altreal_Elt_method(altreal_from_numpy_array_t, elt_from_numpy_array<ConverterToReal>); 
    R_set_altreal_Get_region_method(altreal_from_numpy_array_t, get_region_from_numpy_array<ConverterToReal>); 

    altinteger_from_numpy_array_t = R_make_altinteger_class("altinteger_from_numpy_array", "reticulate", dll); \
    AltvecFromNumpyArray<ConverterToInteger>::init(); 
    R_set_altinteger_Elt_method(altinteger_from_numpy_array_t, elt_from_numpy_array<ConverterToInteger>); 
    R_set_altinteger_Get_region_method(altinteger_from_numpy_array_t, get_region_from_numpy_array<ConverterToInteger>); 

    altlogical_from_numpy_array_t = R_make_altlogical_class("altlogical_from_numpy_array", "reticulate", dll); \
    AltvecFromNumpyArray<ConverterToLogical>::init(); 
    R_set_altlogical_Elt_method(altlogical_from_numpy_array_t, elt_from_numpy_array<ConverterToLogical>); 
    R_set_altlogical_Get_region_method(altlogical_from_numpy_array_t, get_region_from_numpy_array<ConverterToLogical>); 

    altcomplex_from_numpy_array_t = R_make_altcomplex_class("altcomplex_from_numpy_array", "reticulate", dll); \
    AltvecFromNumpyArray<ConverterToComplex>::init(); 
    R_set_altcomplex_Elt_method(altcomplex_from_numpy_array_t, elt_from_numpy_array<ConverterToComplex>); 
    R_set_altcomplex_Get_region_method(altcomplex_from_numpy_array_t, get_region_from_numpy_array<ConverterToComplex>); 
}
