/*=============================================================================
 * [Comp, rX, [List, Gtv, ...]] TODO: implement subgradient retrieval
 * [Comp, rX, [List, Graph, Obj, Time, Dif]] = cp_prox_tv(Y, first_edge,
 *      adj_vertices, [options])
 *
 * options is a struct with any of the following fields [with default values]:
 *
 *      l22_metric [none], edge_weights [1.0], d1p [2], d1p_metric [none],
 *      cp_dif_tol [1e-4], cp_it_max [10], K [2], split_iter_num [1],
 *      split_damp_ratio [1.0], split_values_init_num [2],
 *      split_values_iter_num [2], pfdr_rho [1.0], pfdr_cond_min [1e-2],
 *      pfdr_dif_rcd [0.0], pfdr_dif_tol [1e-2*cp_dif_tol], pfdr_it_max [1e4],
 *      verbose [1e3], max_num_threads [none], max_split_size [none],
 *      balance_parallel_split [true], compute_List [false],
 *      compute_Graph [false], compute_Obj [false], compute_Time [false],
 *      compute_Dif [false]
 * 
 *  Hugo Raguet 2023
 *===========================================================================*/
#include <cstdint>
#include <cstring>
#include "mex.h"
#include "cp_prox_tv.hpp"

using namespace std;

/* index_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph */
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned iterator in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
    typedef int32_t index_t;
    # define mxINDEX_CLASS mxINT32_CLASS
    # define INDEX_T_STRING "int32"
    #ifndef COMP_T_ON_32_BITS
        typedef int16_t comp_t;
        # define mxCOMP_CLASS mxINT16_CLASS
    #else /* more than 32767 components are expected */
        typedef int32_t comp_t;
        #define mxCOMP_CLASS mxINT32_CLASS
    #endif
#else
    typedef uint32_t index_t;
    # define mxINDEX_CLASS mxUINT32_CLASS
    # define INDEX_T_STRING "uint32"
    #ifndef COMP_T_ON_32_BITS
        typedef uint16_t comp_t;
        # define mxCOMP_CLASS mxUINT16_CLASS
    #else /* more than 65535 components are expected */
        typedef uint32_t comp_t;
        #define mxCOMP_CLASS mxUINT32_CLASS
    #endif
#endif

/* function for checking optional parameters */
static void check_opts(const mxArray* options)
{
    if (!options){ return; }

    if (!mxIsStruct(options)){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit prox TV: "
            "fourth parameter 'options' should be a structure, (%s given).",
            mxGetClassName(options));
    }

    const int num_allow_opts = 25;
    const char* opts_names[] = {"l22_metric", "edge_weights", "d1p", 
        "d1p_metric", "cp_dif_tol", "cp_it_max", "K", "split_iter_num",
        "split_damp_ratio", "split_values_init_num", "split_values_iter_num",
        "pfdr_rho", "pfdr_cond_min", "pfdr_dif_rcd", "pfdr_dif_tol",
        "pfdr_it_max", "verbose", "max_num_threads", "max_split_size",
        "balance_parallel_split", "compute_List", "compute_Graph",
        "compute_Obj", "compute_Time", "compute_Dif"};

    const int num_given_opts = mxGetNumberOfFields(options);

    for (int given_opt = 0; given_opt < num_given_opts; given_opt++){
        const char* opt_name = mxGetFieldNameByNumber(options, given_opt);
        int allow_opt;
        for (allow_opt = 0; allow_opt < num_allow_opts; allow_opt++){
            if (strcmp(opt_name, opts_names[allow_opt]) == 0){ break; }
        }
        if (allow_opt == num_allow_opts){
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit prox TV: "
                "option '%s' unknown.", opt_name);
        }
    }
}

/* function for checking parameter type */
static void check_arg_class(const mxArray* arg, const char* arg_name,
    mxClassID class_id, const char* class_name)
{
    if (mxGetNumberOfElements(arg) > 1 && mxGetClassID(arg) != class_id){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit prox TV: "
            "parameter '%s' should be of class %s (%s given).",
            arg_name, class_name, mxGetClassName(arg), class_name);
    }
}

/* resize memory buffer allocated by mxMalloc and create a row vector */
template <typename type_t>
static mxArray* resize_and_create_mxRow(type_t* buffer, size_t size,
    mxClassID class_id)
{
    mxArray* row = mxCreateNumericMatrix(0, 0, class_id, mxREAL);
    if (size){
        mxSetM(row, 1);
        mxSetN(row, size);
        buffer = (type_t*) mxRealloc((void*) buffer, sizeof(type_t)*size);
        mxSetData(row, (void*) buffer);
    }else{
        mxFree((void*) buffer);
    }
    return row;
}

/* template for handling both single and double precisions */
template <typename real_t, mxClassID mxREAL_CLASS>
static void cp_prox_tv_mex(int nlhs, mxArray *plhs[], int nrhs,
    const mxArray *prhs[])
{
    /***  get inputs  ***/

    const char* real_class_name = mxREAL_CLASS == mxDOUBLE_CLASS ?
        "double" : "single";

    /** quadratic functional **/

    size_t D = mxGetM(prhs[0]);
    index_t V = mxGetN(prhs[0]);
    if (V == 1){ /* accept a column vector */ V = D; D = 1; }
    const real_t* Y = (real_t*) mxGetData(prhs[0]);

    /**  graph structure  **/

    check_arg_class(prhs[1], "first_edge", mxINDEX_CLASS, INDEX_T_STRING);
    check_arg_class(prhs[2], "adj_vertices", mxINDEX_CLASS, INDEX_T_STRING);

    const index_t* first_edge = (index_t*) mxGetData(prhs[1]);
    const index_t* adj_vertices = (index_t*) mxGetData(prhs[2]);
    index_t E = mxGetNumberOfElements(prhs[2]);

    size_t first_edge_length = mxGetNumberOfElements(prhs[1]);
    if (first_edge_length != (size_t) V + 1){
        mexErrMsgIdAndTxt("MEX", "Cut-pursuit prox TV: "
            "third parameter 'first_edge' should contain |V| + 1 = %d "
            "elements (%d given).", (size_t) V + 1, first_edge_length);
    }

    /**  optional parameters  **/

    const mxArray* options = nrhs > 3 ? prhs[3] : nullptr;

    check_opts(options);
    const mxArray* opt;

    #define GET_REAL_OPT(NAME) \
        const real_t* NAME = nullptr; \
        if (opt = mxGetField(options, 0, #NAME)){ \
            check_arg_class(opt, #NAME, mxREAL_CLASS, real_class_name); \
            NAME = (real_t*) mxGetData(opt); \
        }

    GET_REAL_OPT(l22_metric)
    typename Cp_prox_tv<real_t, index_t, comp_t>::Metric_shape
        l22_metric_shape = Cp_prox_tv<real_t, index_t, comp_t>::IDENTITY;
    if (opt){
        l22_metric_shape = mxGetNumberOfElements(opt) == V ?
            Cp_prox_tv<real_t, index_t, comp_t>::MONODIM :
            Cp_prox_tv<real_t, index_t, comp_t>::MULTIDIM;
    }
    GET_REAL_OPT(d1p_metric);
    GET_REAL_OPT(edge_weights);
    real_t homo_edge_weight = 1.0;
    if (opt && mxGetNumberOfElements(opt) == 1){
        edge_weights = nullptr;
        homo_edge_weight = mxGetScalar(opt);
    }

    #define GET_SCAL_OPT(NAME, DFLT) \
        NAME = (opt = mxGetField(options, 0, #NAME)) ? mxGetScalar(opt) : DFLT;

    real_t GET_SCAL_OPT(d1p, (D == 1 ? 1 : 2));
    real_t GET_SCAL_OPT(cp_dif_tol, 1e-4);
    int GET_SCAL_OPT(cp_it_max, 10);
    comp_t GET_SCAL_OPT(K, 2);
    int GET_SCAL_OPT(split_iter_num, 1);
    real_t GET_SCAL_OPT(split_damp_ratio, 1.0);
    int GET_SCAL_OPT(split_values_init_num, 2);
    int GET_SCAL_OPT(split_values_iter_num, 2);
    real_t GET_SCAL_OPT(pfdr_rho, 1.0);
    real_t GET_SCAL_OPT(pfdr_cond_min, 1e-2);
    real_t GET_SCAL_OPT(pfdr_dif_rcd, 0.0);
    real_t GET_SCAL_OPT(pfdr_dif_tol, 1e-2*cp_dif_tol);
    int GET_SCAL_OPT(pfdr_it_max, 1e4);
    int GET_SCAL_OPT(verbose, 1e3);
    int GET_SCAL_OPT(max_num_threads, 0);
    int GET_SCAL_OPT(max_split_size, V);
    bool GET_SCAL_OPT(balance_parallel_split, true);
    bool GET_SCAL_OPT(compute_List, false);
    bool GET_SCAL_OPT(compute_Graph, false);
    bool GET_SCAL_OPT(compute_Obj, false);
    bool GET_SCAL_OPT(compute_Time, false);
    bool GET_SCAL_OPT(compute_Dif, false);

    /* check optional output requested */
    int nout = 2;
    if (compute_List){ nout++; }
    if (compute_Graph){ nout++; }
    if (compute_Obj){ nout++; }
    if (compute_Time){ nout++; }
    if (compute_Dif){ nout++; }
    if (nlhs != nout){
            mexErrMsgIdAndTxt("MEX", "Cut-pursuit prox TV: "
                "requested %i outputs, but %i captured", nout, nlhs);
    }

    /***  prepare output; rX (plhs[1]) is created later  ***/

    plhs[0] = mxCreateNumericMatrix(1, V, mxCOMP_CLASS, mxREAL);
    comp_t *Comp = (comp_t*) mxGetData(plhs[0]);

    /* TODO: implement subgradiant retrieval */
    // real_t* Gtv = nlhs > 3 ?
        // (real_t*) mxMalloc(sizeof(real_t)*E) : nullptr;

    real_t* Obj = nullptr;
    if (compute_Obj){
        Obj = (real_t*) mxMalloc(sizeof(real_t)*(cp_it_max + 1));
    }

    double* Time = nullptr;
    if (compute_Time){
        Time = (double*) mxMalloc(sizeof(double)*(cp_it_max + 1));
    }

    real_t* Dif = nullptr;
    if (compute_Dif){ Dif = (real_t*) mxMalloc(sizeof(real_t)*cp_it_max); }

    /***  cut-pursuit with preconditioned forward-Douglas-Rachford  ***/

    Cp_prox_tv<real_t, index_t, comp_t> *cp =
        new Cp_prox_tv<real_t, index_t, comp_t>
            (V, E, first_edge, adj_vertices, Y, D);

    cp->set_quadratic(l22_metric_shape, l22_metric);
    cp->set_d1_param(edge_weights, homo_edge_weight, d1p_metric,
        d1p == 1 ? Cp_prox_tv<real_t, index_t, comp_t>::D11
                 : Cp_prox_tv<real_t, index_t, comp_t>::D12);
    cp->set_edge_weights(edge_weights, homo_edge_weight);
    cp->set_cp_param(cp_dif_tol, cp_it_max, verbose);
    cp->set_split_param(max_split_size, K, split_iter_num, split_damp_ratio,
        split_values_init_num, split_values_iter_num);
    cp->set_pfdr_param(pfdr_rho, pfdr_cond_min, pfdr_dif_rcd, pfdr_it_max,
        pfdr_dif_tol);
    cp->set_parallel_param(max_num_threads, balance_parallel_split);
    cp->set_monitoring_arrays(Obj, Time, Dif);
    cp->set_components(0, Comp); // use the preallocated component array Comp
    /* TODO: implement subgradient retrieval */
    // cp->set_d1_subgradients(Gtv);

    int cp_it = cp->cut_pursuit();

    /* get number of components and list of indices */
    const index_t* first_vertex;
    const index_t* comp_list;
    comp_t rV = cp->get_components(nullptr, &first_vertex, &comp_list);

    /* copy reduced values */
    const real_t* cp_rX = cp->get_reduced_values();
    plhs[1] = mxCreateNumericMatrix(D, rV, mxREAL_CLASS, mxREAL);
    real_t* rX = (real_t*) mxGetData(plhs[1]);
    for (size_t rvd = 0; rvd < rV*D; rvd++){ rX[rvd] = cp_rX[rvd]; }

    /* get lists of indices if requested */
    mxArray* mx_List = nullptr;
    if (compute_List){
        mx_List = mxCreateCellMatrix(1, rV); // list of arrays
        for (comp_t rv = 0; rv < rV; rv++){
            index_t comp_size = first_vertex[rv+1] - first_vertex[rv];
            mxArray* mxList_rv = mxCreateNumericMatrix(1, comp_size,
                mxINDEX_CLASS, mxREAL);
            index_t* List_rv = (index_t*) mxGetData(mxList_rv);
            for (index_t i = 0; i < comp_size; i++){
                List_rv[i] = comp_list[first_vertex[rv] + i];
            }
            mxSetCell(mx_List, rv, mxList_rv);
        }
    }

    /* retrieve reduced graph structure */
    mxArray* mx_Graph = nullptr;
    if (compute_Graph){
        const comp_t* reduced_edge_list;
        const real_t* reduced_edge_weights;
        size_t rE;
        /* get reduced edge list */
        rE = cp->get_reduced_graph(&reduced_edge_list, &reduced_edge_weights);

        /* mex arrays for forward-star representation and weights */
        mxArray* mx_red_first_edge = mxCreateNumericMatrix(1, rV + 1,
            mxINDEX_CLASS, mxREAL);
        index_t* red_first_edge = (index_t*) mxGetData(mx_red_first_edge);

        mxArray* mx_red_adj_vertices = mxCreateNumericMatrix(1, rE,
            mxCOMP_CLASS, mxREAL);
        comp_t* red_adj_vertices = (comp_t*) mxGetData(mx_red_adj_vertices);

        mxArray* mx_red_edge_weights = mxCreateNumericMatrix(1, rE,
            mxREAL_CLASS, mxREAL);
        real_t* red_edge_weights = (real_t*) mxGetData(mx_red_edge_weights);

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

        /* gather forward-star representation and weights in mex cell array */
        mx_Graph = mxCreateCellMatrix(1, 3);
        mxSetCell(mx_Graph, 0, mx_red_first_edge);
        mxSetCell(mx_Graph, 1, mx_red_adj_vertices);
        mxSetCell(mx_Graph, 2, mx_red_edge_weights);
    }
    
    cp->set_components(0, nullptr); // prevent Comp to be free()'d
    delete cp;

    /**  assign optional outputs and resize monitoring arrays if necessary  **/
    nout = 2;
    if (compute_List){ plhs[nout++] = mx_List; }
    if (compute_Graph){ plhs[nout++] = mx_Graph; }
    if (compute_Obj){
        plhs[nout++] = resize_and_create_mxRow(Obj, cp_it + 1, mxREAL_CLASS);
    }
    if (compute_Time){
        plhs[nout++] = resize_and_create_mxRow(Time, cp_it+1, mxDOUBLE_CLASS);
    }
    if (compute_Dif){
        plhs[nout++] = resize_and_create_mxRow(Dif, cp_it, mxREAL_CLASS);
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by the first parameter Y */
    if (mxIsDouble(prhs[0])){
        cp_prox_tv_mex<double, mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        cp_prox_tv_mex<float, mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }
}
