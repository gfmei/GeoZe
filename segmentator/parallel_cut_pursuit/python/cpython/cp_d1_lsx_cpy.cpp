/*=============================================================================
 * Comp, rX, [List, Graph, Obj, Time, Dif] = cp_d1_lsx_cpy(loss, Y, first_edge,
 *  adj_vertices, edge_weights, loss_weights, d11_metric, cp_dif_tol,
 *  cp_it_max, K, split_iter_num, split_damp_ratio, split_values_init_num,
 *  split_values_iter_num, pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol,
 *  pfdr_it_max, verbose, max_num_threads, max_split_size,
 *  balance_parallel_split, real_is_double, compute_List, compute_Graph,
 *  compute_Obj, compute_Time, compute_Dif)
 * 
 *  Baudoin Camille 2019, Raguet Hugo 2021, 2022, 2023
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "cp_d1_lsx.hpp"

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned iterator in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    typedef int32_t index_t;
    #define NPY_IND NPY_INT32
    #ifndef COMP_T_ON_32_BITS
        typedef int16_t comp_t;
        #define NPY_COMP NPY_INT16
    #else /* more than 32767 components are expected */
        typedef int32_t comp_t;
        #define NPY_COMP NPY_INT32
    #endif
#else
    typedef uint32_t index_t;
    #define NPY_IND NPY_UINT32
    #ifndef COMP_T_ON_32_BITS
        typedef uint16_t comp_t;
        #define NPY_COMP NPY_UINT16
    #else /* more than 65535 components are expected */
        typedef uint32_t comp_t;
        #define NPY_COMP NPY_UINT32
    #endif
#endif

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES NPY_REAL>
static PyObject* cp_d1_lsx(real_t loss, PyArrayObject* py_Y,
    PyArrayObject* py_first_edge, PyArrayObject* py_adj_vertices,
    PyArrayObject* py_edge_weights, PyArrayObject* py_loss_weights,
    PyArrayObject* py_d11_metric, real_t cp_dif_tol, int cp_it_max, int K,
    int split_iter_num, real_t split_damp_ratio, int split_values_init_num,
    int split_values_iter_num, real_t pfdr_rho, real_t pfdr_cond_min,
    real_t pfdr_dif_rcd, real_t pfdr_dif_tol, int pfdr_it_max, int verbose,
    int max_num_threads, index_t max_split_size, int balance_parallel_split,
    int compute_List, int compute_Graph, int compute_Obj, int compute_Time,
    int compute_Dif)
{
    /**  get inputs  **/

    /* sizes and loss */
    npy_intp * py_Y_dims = PyArray_DIMS(py_Y);
    size_t D = py_Y_dims[0];
    index_t V = py_Y_dims[1]; 

    const real_t* Y = (real_t*) PyArray_DATA(py_Y);
    const real_t* loss_weights = (PyArray_SIZE(py_loss_weights) > 0) ?
        (real_t*) PyArray_DATA(py_loss_weights) : nullptr;

    /* graph structure */
    index_t E = PyArray_SIZE(py_adj_vertices);
    const index_t* first_edge = (index_t*) PyArray_DATA(py_first_edge);
    const index_t* adj_vertices = (index_t*) PyArray_DATA(py_adj_vertices);

    /* penalizations */
    const real_t* edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = PyArray_SIZE(py_edge_weights) == 1 ?
        edge_weights[0] : 1.0;
    if (PyArray_SIZE(py_edge_weights) <= 1){ edge_weights = nullptr; }

    const real_t* d11_metric = PyArray_SIZE(py_d11_metric) > 0 ?
        (real_t*) PyArray_DATA(py_d11_metric) : nullptr;

    /* number of threads */ 
    if (max_num_threads <= 0){ max_num_threads = omp_get_max_threads(); }

    /**  prepare output; rX is created later  **/
    /* NOTA: no check for successful allocations is performed */

    npy_intp size_py_Comp[] = {V};
    PyArrayObject* py_Comp = (PyArrayObject*) PyArray_Zeros(1,
        size_py_Comp, PyArray_DescrFromType(NPY_COMP), 1);
    comp_t* Comp = (comp_t*) PyArray_DATA(py_Comp); 

    real_t* Obj = nullptr;
    if (compute_Obj){ Obj = (real_t*) malloc(sizeof(real_t)*(cp_it_max + 1)); }

    double* Time = nullptr;
    if (compute_Time){
        Time = (double*) malloc(sizeof(double)*(cp_it_max + 1));
    }

    real_t* Dif = nullptr;
    if (compute_Dif){ Dif = (real_t*) malloc(sizeof(real_t)*cp_it_max); }

    /**  cut-pursuit with preconditioned forward-Douglas-Rachford  **/

    Cp_d1_lsx<real_t, index_t, comp_t>* cp =
        new Cp_d1_lsx<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices, D, Y);

    cp->set_loss(loss, Y, loss_weights);
    cp->set_d1_param(edge_weights, homo_edge_weight, d11_metric);
    cp->set_monitoring_arrays(Obj, Time, Dif);
    cp->set_components(0, Comp);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_split_param(max_split_size, K, split_iter_num, split_damp_ratio,
        split_values_init_num, split_values_iter_num);
    cp->set_pfdr_param(pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_it_max,
        pfdr_dif_tol);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);

    int cp_it = cp->cut_pursuit();

    /* get number of components and their lists of indices if necessary */
    const index_t* first_vertex;
    const index_t* comp_list;
    comp_t rV = cp->get_components(nullptr, &first_vertex, &comp_list);

    PyObject* py_List = nullptr;
    if (compute_List){
        py_List = PyList_New(rV); // list of arrays
        for (comp_t rv = 0; rv < rV; rv++){
            index_t comp_size = first_vertex[rv+1] - first_vertex[rv];
            npy_intp size_py_List_rv[] = {comp_size};
            PyArrayObject* py_List_rv = (PyArrayObject*) PyArray_Zeros(1,
                size_py_List_rv, PyArray_DescrFromType(NPY_IND), 1);
            index_t* List_rv = (index_t*) PyArray_DATA(py_List_rv);
            for (index_t i = 0; i < comp_size; i++){
                List_rv[i] = comp_list[first_vertex[rv] + i];
            }
            PyList_SetItem(py_List, rv, (PyObject*) py_List_rv);
        }
    }

    /* copy reduced values */
    const real_t* cp_rX = cp->get_reduced_values();
    npy_intp size_py_rX[] = {(npy_intp/* suppress warning*/) D, rV};
    PyArrayObject* py_rX = (PyArrayObject*) PyArray_Zeros(2, size_py_rX,
        PyArray_DescrFromType(NPY_REAL), 1);
    real_t* rX = (real_t*) PyArray_DATA(py_rX);
    for (size_t rvd = 0; rvd < rV*D; rvd++){ rX[rvd] = cp_rX[rvd]; }

    /* retrieve reduced graph structure */
    PyObject* py_Graph = nullptr;
    if (compute_Graph){
        const comp_t* reduced_edge_list;
        const real_t* reduced_edge_weights;
        size_t rE;
        /* get reduced edge list */
        rE = cp->get_reduced_graph(&reduced_edge_list, &reduced_edge_weights);

        /* numpy arrays for forward-star representation and weights */
        npy_intp size_py_red_first_edge[] = {rV + 1};
        PyArrayObject* py_red_first_edge = (PyArrayObject*)
            PyArray_Zeros(1, size_py_red_first_edge,
                PyArray_DescrFromType(NPY_IND), 1);
        index_t* red_first_edge = (index_t*) PyArray_DATA(py_red_first_edge);

        npy_intp size_py_red_adj_vertices[] = {(npy_intp/* supp. warning*/)rE};
        PyArrayObject* py_red_adj_vertices = (PyArrayObject*)
            PyArray_Zeros(1, size_py_red_adj_vertices,
                PyArray_DescrFromType(NPY_COMP), 1);
        comp_t* red_adj_vertices = (comp_t*) PyArray_DATA(py_red_adj_vertices);

        npy_intp size_py_red_edge_weights[] = {(npy_intp/* supp. warning*/)rE};
        PyArrayObject* py_red_edge_weights = (PyArrayObject*)
            PyArray_Zeros(1, size_py_red_edge_weights,
                PyArray_DescrFromType(NPY_REAL), 1);
        real_t* red_edge_weights = (real_t*) PyArray_DATA(py_red_edge_weights);

        /* reduced edge list is guaranteed to be in increasing order of
         * starting component; conversion to forward-star is straightforward */
        comp_t rv = 0;
        size_t re = 0;
        while (re < rE || rv < rV){
            red_first_edge[rv] = re;
            while (re < rE && reduced_edge_list[2*re] == rv){
                red_adj_vertices[re] = reduced_edge_list[2*re + 1];
                red_edge_weights[re] = reduced_edge_weights[re];
                re++;
            }
            rv++;
        }
        red_first_edge[rV] = rE;

        /* gather forward-star representation and weights in python tuple */
        py_Graph = PyTuple_New(3);
        PyTuple_SET_ITEM(py_Graph, 0, (PyObject*) py_red_first_edge);
        PyTuple_SET_ITEM(py_Graph, 1, (PyObject*) py_red_adj_vertices);
        PyTuple_SET_ITEM(py_Graph, 2, (PyObject*) py_red_edge_weights);
    }

    /* retrieve monitoring arrays */
    PyArrayObject* py_Obj = nullptr;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {cp_it + 1};
        py_Obj = (PyArrayObject*) PyArray_Zeros(1, size_py_Obj,
            PyArray_DescrFromType(NPY_REAL), 1);
        real_t* Obj_ = (real_t*) PyArray_DATA(py_Obj);
        for (int i = 0; i < size_py_Obj[0]; i++){ Obj_[i] = Obj[i]; }
        free(Obj);
    }

    PyArrayObject* py_Time = nullptr;
    if (compute_Time){
        npy_intp size_py_Time[] = {cp_it + 1};
        py_Time = (PyArrayObject*) PyArray_Zeros(1, size_py_Time,
            PyArray_DescrFromType(NPY_FLOAT64), 1);
        double* Time_ = (double*) PyArray_DATA(py_Time);
        for (int i = 0; i < size_py_Time[0]; i++){ Time_[i] = Time[i]; }
        free(Time);
    }

    PyArrayObject* py_Dif = nullptr;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {cp_it};
        py_Dif = (PyArrayObject*) PyArray_Zeros(1, size_py_Dif,
            PyArray_DescrFromType(NPY_REAL), 1);
        real_t* Dif_ = (real_t*) PyArray_DATA(py_Dif);
        for (int i = 0; i < size_py_Dif[0]; i++){ Dif_[i] = Dif[i]; }
        free(Dif);
    }
 
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /* build output according to optional output specified */
    Py_ssize_t nout = 2;
    if (compute_List){ nout++; }
    if (compute_Graph){ nout++; }
    if (compute_Obj){ nout++; }
    if (compute_Time){ nout++; }
    if (compute_Dif){ nout++; }
    PyObject* py_Out = PyTuple_New(nout);
    PyTuple_SET_ITEM(py_Out, 0, (PyObject*) py_Comp);
    PyTuple_SET_ITEM(py_Out, 1, (PyObject*) py_rX);
    nout = 2;
    if (compute_List){ PyTuple_SET_ITEM(py_Out, nout++, py_List); }
    if (compute_Graph){ PyTuple_SET_ITEM(py_Out, nout++, py_Graph); }
    if (compute_Obj){ PyTuple_SET_ITEM(py_Out, nout++, (PyObject*) py_Obj); }
    if (compute_Time){ PyTuple_SET_ITEM(py_Out, nout++, (PyObject*) py_Time); }
    if (compute_Dif){ PyTuple_SET_ITEM(py_Out, nout++, (PyObject*) py_Dif); }

    return py_Out;
}

/* actual python interface */
#if PY_VERSION_HEX >= 0x03040000 // Py_UNUSED suppress warning from 3.4
static PyObject* cp_d1_lsx_cpy(PyObject* Py_UNUSED(self), PyObject* args)
{ 
#else
static PyObject* cp_d1_lsx_cpy(PyObject* self, PyObject* args)
{   (void) self; // suppress unused parameter warning
#endif
    /* INPUT */
    PyArrayObject *py_Y, *py_first_edge, *py_adj_vertices, *py_edge_weights,
        *py_loss_weights, *py_d11_metric;
    double loss, cp_dif_tol, split_damp_ratio, pfdr_rho, pfdr_cond_min,
        pfdr_dif_rcd, pfdr_dif_tol;  
    int cp_it_max, K, split_iter_num, split_values_init_num,
        split_values_iter_num, pfdr_it_max, verbose, max_num_threads, 
        max_split_size, balance_parallel_split, real_is_double, compute_List,
        compute_Graph, compute_Obj, compute_Time, compute_Dif;

    /* parse the input, from Python Object to C PyArray, double, or int type */
    if(!PyArg_ParseTuple(args, "dOOOOOOdiiidiiddddiiiiiiiiiii", &loss, &py_Y,
        &py_first_edge, &py_adj_vertices, &py_edge_weights, &py_loss_weights,
        &py_d11_metric, &cp_dif_tol, &cp_it_max, &K, &split_iter_num,
        &split_damp_ratio, &split_values_init_num, &split_values_iter_num,
        &pfdr_rho, &pfdr_cond_min, &pfdr_dif_rcd, &pfdr_dif_tol, &pfdr_it_max,
        &verbose, &max_num_threads, &max_split_size, &balance_parallel_split,
        &real_is_double, &compute_List, &compute_Graph, &compute_Obj,
        &compute_Time, &compute_Dif)){
        return NULL;
    }

    if (real_is_double){
        return cp_d1_lsx<double, NPY_FLOAT64>(loss, py_Y, py_first_edge,
            py_adj_vertices, py_edge_weights, py_loss_weights,
            py_d11_metric, cp_dif_tol, cp_it_max, K, split_iter_num,
            split_damp_ratio, split_values_init_num, split_values_iter_num, 
            pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max,
            verbose, max_num_threads, max_split_size, balance_parallel_split,
            compute_List, compute_Graph, compute_Obj, compute_Time,
            compute_Dif);
    }else{ /* real_t type is float */
        return cp_d1_lsx<float, NPY_FLOAT32>(loss, py_Y, py_first_edge,
            py_adj_vertices, py_edge_weights, py_loss_weights,
            py_d11_metric, cp_dif_tol, cp_it_max, K, split_iter_num,
            split_damp_ratio, split_values_init_num, split_values_iter_num,
            pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_dif_tol, pfdr_it_max,
            verbose, max_num_threads, max_split_size, balance_parallel_split,
            compute_List, compute_Graph, compute_Obj, compute_Time,
            compute_Dif);
    }
}

static PyMethodDef cp_d1_lsx_methods[] = {
    {"cp_d1_lsx_cpy", cp_d1_lsx_cpy, METH_VARARGS,
        "wrapper for parallel cut-pursuit loss d1 simplex"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
static struct PyModuleDef cp_d1_lsx_module = {
    PyModuleDef_HEAD_INIT,
    "cp_d1_lsx_cpy", /* name of module */
    NULL, /* module documentation, may be null */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    cp_d1_lsx_methods, /* actual methods in the module */
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_cp_d1_lsx_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&cp_d1_lsx_module);
}
