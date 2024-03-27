/*=============================================================================
 * X, [Obj, Dif] = pfdr_prox_tv_cpy(Y, edges, l22_metric, edge_weights, d1p,
 *      d1p_metric, rho, cond_min, dif_rcd, dif_tol, dif_it, it_max, verbose,
 *      real_t_double, compute_Obj, compute_Dif)
 * 
 * Raguet Hugo 2023
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "pfdr_prox_tv.hpp"

using namespace std;

/* vertex_t must be able to represent the number of vertices and of
 * (undirected) edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph, as well as the dimension D */
typedef uint32_t vertex_t;

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES NPY_REAL>
static PyObject* pfdr_prox_tv(PyArrayObject* py_Y, PyArrayObject* py_edges,
    PyArrayObject* py_l22_metric, PyArrayObject* py_edge_weights, 
    int d1p, PyArrayObject* py_d1p_metric, real_t rho, real_t cond_min,
    real_t dif_rcd, real_t dif_tol, int dif_it, int it_max, int verbose,
    int compute_Obj, int compute_Dif)
{
    /**  get inputs  **/

    /* observations */
    npy_intp * py_Y_dims = PyArray_DIMS(py_Y);
    size_t D = PyArray_NDIM(py_Y) > 1 ? py_Y_dims[0] : 1;
    vertex_t V = PyArray_NDIM(py_Y) > 1 ? py_Y_dims[1] : py_Y_dims[0];
    if (V == 1){ /* accept a column vector */ V = D; D = 1; }

    const real_t *Y = (real_t*) PyArray_DATA(py_Y);

    /* graph structure */
    vertex_t E = PyArray_SIZE(py_edges) / 2;
    const vertex_t *edges = (vertex_t*) PyArray_DATA(py_edges);

    /* penalizations */
    const real_t *l22_metric = (PyArray_SIZE(py_l22_metric) > 0) ?
        (real_t*) PyArray_DATA(py_l22_metric) : nullptr;

    typename Pfdr<real_t, vertex_t>::Condshape l22_metric_shape =
        (size_t) PyArray_SIZE(py_l22_metric) == D*V ?
            Pfdr<real_t, vertex_t>::MULTIDIM :
        (size_t) PyArray_SIZE(py_l22_metric) == V ?
            Pfdr<real_t, vertex_t>::MONODIM :
            Pfdr<real_t, vertex_t>::SCALAR;

    const real_t *edge_weights = (PyArray_SIZE(py_edge_weights) > 1) ?
        (real_t*) PyArray_DATA(py_edge_weights) : nullptr;
    real_t* ptr_edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = (PyArray_SIZE(py_edge_weights) == 1) ?
        ptr_edge_weights[0] : 1.0;

    const real_t* d1p_metric = (PyArray_SIZE(py_d1p_metric) > 0) ?
        (real_t*) PyArray_DATA(py_d1p_metric) : nullptr;

    /**  create output  **/
    /* NOTA: no check for successful allocations is performed */

    npy_intp size_py_X[] = {(npy_intp) D, V};
    PyArrayObject* py_X = (PyArrayObject*) PyArray_Zeros(2, size_py_X, 
        PyArray_DescrFromType(NPY_REAL), 1);
    real_t* X = (real_t*) PyArray_DATA(py_X);

    real_t* Obj = nullptr;
    if (compute_Obj){ Obj = (real_t*) malloc(sizeof(real_t)*(it_max + 1)); }

    real_t* Dif = nullptr;
    if (compute_Dif){ Dif = (real_t*) malloc(sizeof(real_t)*it_max); }

    /**  preconditioned forward-Douglas-Rachford  **/

    Pfdr_prox_tv<real_t, vertex_t> *pfdr =
        new Pfdr_prox_tv<real_t, vertex_t>
            (V, E, edges, Y, D, d1p == 1 ?
            Pfdr_d1<real_t, vertex_t>::D11 : Pfdr_d1<real_t, vertex_t>::D12,
            d1p_metric, l22_metric_shape, l22_metric);

    pfdr->set_edge_weights(edge_weights, homo_edge_weight);
    pfdr->set_conditioning_param(cond_min, dif_rcd);
    pfdr->set_relaxation(rho);
    pfdr->set_algo_param(dif_tol, dif_it, it_max, verbose);
    pfdr->set_monitoring_arrays(Obj, Dif);
    pfdr->set_iterate(X);
    pfdr->initialize_iterate();

    int it = pfdr->precond_proximal_splitting();

    pfdr->set_iterate(nullptr); // prevent X to be free()'d

    delete pfdr;

    /**  build output according to optional output requested  **/

    /* retrieve monitoring arrays */
    PyArrayObject* py_Obj = nullptr;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {it + 1};
        py_Obj = (PyArrayObject*) PyArray_Zeros(1, size_py_Obj,
            PyArray_DescrFromType(NPY_REAL), 1);
        real_t* Obj_ = (real_t*) PyArray_DATA(py_Obj);
        for (int i = 0; i < size_py_Obj[0]; i++){ Obj_[i] = Obj[i]; }
        free(Obj);
    }

    PyArrayObject* py_Dif = nullptr;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {it};
        py_Dif = (PyArrayObject*) PyArray_Zeros(1, size_py_Dif,
            PyArray_DescrFromType(NPY_REAL), 1);
        real_t* Dif_ = (real_t*) PyArray_DATA(py_Dif);
        for (int i = 0; i < size_py_Dif[0]; i++){ Dif_[i] = Dif[i]; }
        free(Dif);
    }

    if (compute_Obj && compute_Dif){
        return Py_BuildValue("OOO", py_X, py_Obj, py_Dif);
    }else if (compute_Obj){
        return Py_BuildValue("OO", py_X, py_Obj);
    }else if (compute_Dif){
        return Py_BuildValue("OO", py_X, py_Dif);
    }else{
        return Py_BuildValue("O", py_X);
    }
}

static PyObject* pfdr_prox_tv_cpy(PyObject* self, PyObject* args)
{ 
    PyArrayObject *py_Y, *py_edges, *py_l22_metric, *py_edge_weights,
        *py_d1p_metric;
    double rho, cond_min, dif_rcd, dif_tol;  
    int d1p, dif_it, it_max, verbose, real_t_double, compute_Obj, compute_Dif;

    if(!PyArg_ParseTuple(args, "OOOOiOddddiiiiii", &py_Y, &py_edges,
        &py_l22_metric, &py_edge_weights, &d1p, &py_d1p_metric,  
        &rho, &cond_min, &dif_rcd, &dif_tol, &dif_it, &it_max, &verbose,
        &real_t_double, &compute_Obj, &compute_Dif)){
        return NULL;
    }

    if (real_t_double){ /* real_t type is double */
        return pfdr_prox_tv<double, NPY_FLOAT64>(py_Y, py_edges, py_l22_metric,
            py_edge_weights, d1p, py_d1p_metric, rho, cond_min, dif_rcd,
            dif_tol, dif_it, it_max, verbose, compute_Obj, compute_Dif);
    }else{ /* real_t type is float */
        return pfdr_prox_tv<float, NPY_FLOAT32>(py_Y, py_edges, py_l22_metric,
            py_edge_weights, d1p, py_d1p_metric, rho, cond_min, dif_rcd,
            dif_tol, dif_it, it_max, verbose, compute_Obj, compute_Dif);
    }
}

static PyMethodDef pfdr_prox_tv_methods[] = {
    {"pfdr_prox_tv_cpy", pfdr_prox_tv_cpy, METH_VARARGS,
        "wrapper for PFDR prox TV"},
    {NULL, NULL, 0, NULL}
}; 

static struct PyModuleDef pfdr_prox_tv_module = {
    PyModuleDef_HEAD_INIT,
    "pfdr_prox_tv_cpy", /* name of module */
    NULL, /* module documentation, may be null */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    pfdr_prox_tv_methods, /* actual methods in the module */
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_pfdr_prox_tv_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&pfdr_prox_tv_module);
}
