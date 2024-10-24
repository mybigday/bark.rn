#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct bark_ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct bark_ggml_context * ctx = bark_ggml_init(params);
//
//       struct bark_ggml_tensor * x = bark_ggml_new_tensor_1d(ctx, BARK_GGML_TYPE_F32, 1);
//
//       bark_ggml_set_param(ctx, x); // x is an input variable
//
//       struct bark_ggml_tensor * a  = bark_ggml_new_tensor_1d(ctx, BARK_GGML_TYPE_F32, 1);
//       struct bark_ggml_tensor * b  = bark_ggml_new_tensor_1d(ctx, BARK_GGML_TYPE_F32, 1);
//       struct bark_ggml_tensor * x2 = bark_ggml_mul(ctx, x, x);
//       struct bark_ggml_tensor * f  = bark_ggml_add(ctx, bark_ggml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct bark_ggml_cgraph gf = bark_ggml_build_forward(f);
//
//       // set the input variable and parameter values
//       bark_ggml_set_f32(x, 2.0f);
//       bark_ggml_set_f32(a, 3.0f);
//       bark_ggml_set_f32(b, 4.0f);
//
//       bark_ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", bark_ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the bark_ggml_graph_compute() function.
//
// The bark_ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// bark_ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the bark_ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The bark_ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the bark_ggml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - bark_ggml_permute()
//   - bark_ggml_conv_1d_1s()
//   - bark_ggml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct bark_ggml_tensor)
//
// The tensors are stored in memory via the bark_ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct bark_ggml_tensor * c = bark_ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The bark_ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       const int nx = 2;
//       const int ny = 3;
//
//       struct bark_ggml_tensor * a = bark_ggml_new_tensor_2d(ctx, BARK_GGML_TYPE_F32, nx, ny);
//
//       for (int y = 0; y < ny; y++) {
//           for (int x = 0; x < nx; x++) {
//               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
//           }
//       }
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as bark_ggml_get_f32_1d() and bark_ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (bark_ggml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef BARK_GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef BARK_GGML_BUILD
#            define BARK_GGML_API __declspec(dllexport)
#        else
#            define BARK_GGML_API __declspec(dllimport)
#        endif
#    else
#        define BARK_GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define BARK_GGML_API
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define BARK_GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define BARK_GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define BARK_GGML_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define BARK_GGML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
#    define BARK_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define BARK_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define BARK_GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define BARK_GGML_FILE_VERSION 1

#define BARK_GGML_QNT_VERSION        2    // bump this on quantization format changes
#define BARK_GGML_QNT_VERSION_FACTOR 1000 // do not change this

#define BARK_GGML_MAX_DIMS          4
#define BARK_GGML_MAX_NODES         100000
#define BARK_GGML_MAX_PARAMS        1024
#define BARK_GGML_MAX_CONTEXTS      64
#define BARK_GGML_MAX_SRC           6
#define BARK_GGML_MAX_NAME          64
#define BARK_GGML_MAX_OP_PARAMS     32
#define BARK_GGML_DEFAULT_N_THREADS 4

#if UINTPTR_MAX == 0xFFFFFFFF
    #define BARK_GGML_MEM_ALIGN 4
#else
    #define BARK_GGML_MEM_ALIGN 16
#endif

#define BARK_GGML_EXIT_SUCCESS 0
#define BARK_GGML_EXIT_ABORTED 1

#define BARK_GGUF_MAGIC   0x46554747 // "GGUF"
#define BARK_GGUF_VERSION 2

#define BARK_GGUF_DEFAULT_ALIGNMENT 32

#define BARK_GGML_UNUSED(x) (void)(x)

#define BARK_GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define BARK_GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "BARK_GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifndef NDEBUG
#define BARK_GGML_UNREACHABLE() BARK_GGML_ASSERT(!"statement should not be reached")
#elif defined(__GNUC__)
#define BARK_GGML_UNREACHABLE() __builtin_unreachable()
#else
#define BARK_GGML_UNREACHABLE() ((void) 0)
#endif

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    BARK_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    BARK_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define BARK_GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    BARK_GGML_UNUSED(prefix##0);
#define BARK_GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    BARK_GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    BARK_GGML_UNUSED(prefix##1);
#define BARK_GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    BARK_GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    BARK_GGML_UNUSED(prefix##2);
#define BARK_GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    BARK_GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    BARK_GGML_UNUSED(prefix##3);

#ifdef  __cplusplus
extern "C" {
#endif

#if defined(__ARM_NEON) && defined(__CUDACC__)
    typedef half bark_ggml_fp16_t;
#elif defined(__ARM_NEON)
    typedef __fp16 bark_ggml_fp16_t;
#else
    typedef uint16_t bark_ggml_fp16_t;
#endif

    // convert FP16 <-> FP32
    BARK_GGML_API float       bark_ggml_fp16_to_fp32(bark_ggml_fp16_t x);
    BARK_GGML_API bark_ggml_fp16_t bark_ggml_fp32_to_fp16(float x);

    BARK_GGML_API void bark_ggml_fp16_to_fp32_row(const bark_ggml_fp16_t * x, float * y, int n);
    BARK_GGML_API void bark_ggml_fp32_to_fp16_row(const float * x, bark_ggml_fp16_t * y, int n);

    struct bark_ggml_object;
    struct bark_ggml_context;

    enum bark_ggml_type {
        BARK_GGML_TYPE_F32  = 0,
        BARK_GGML_TYPE_F16  = 1,
        BARK_GGML_TYPE_Q4_0 = 2,
        BARK_GGML_TYPE_Q4_1 = 3,
        // BARK_GGML_TYPE_Q4_2 = 4, support has been removed
        // BARK_GGML_TYPE_Q4_3 (5) support has been removed
        BARK_GGML_TYPE_Q5_0 = 6,
        BARK_GGML_TYPE_Q5_1 = 7,
        BARK_GGML_TYPE_Q8_0 = 8,
        BARK_GGML_TYPE_Q8_1 = 9,
        // k-quantizations
        BARK_GGML_TYPE_Q2_K = 10,
        BARK_GGML_TYPE_Q3_K = 11,
        BARK_GGML_TYPE_Q4_K = 12,
        BARK_GGML_TYPE_Q5_K = 13,
        BARK_GGML_TYPE_Q6_K = 14,
        BARK_GGML_TYPE_Q8_K = 15,
        BARK_GGML_TYPE_I8,
        BARK_GGML_TYPE_I16,
        BARK_GGML_TYPE_I32,
        BARK_GGML_TYPE_COUNT,
    };

    enum bark_ggml_backend_type {
        BARK_GGML_BACKEND_CPU = 0,
        BARK_GGML_BACKEND_GPU = 10,
        BARK_GGML_BACKEND_GPU_SPLIT = 20,
    };

    // model file types
    enum bark_ggml_ftype {
        BARK_GGML_FTYPE_UNKNOWN     = -1,
        BARK_GGML_FTYPE_ALL_F32     = 0,
        BARK_GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        BARK_GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
        BARK_GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
    };

    // available tensor operations:
    enum bark_ggml_op {
        BARK_GGML_OP_NONE = 0,

        BARK_GGML_OP_DUP,
        BARK_GGML_OP_ADD,
        BARK_GGML_OP_ADD1,
        BARK_GGML_OP_ACC,
        BARK_GGML_OP_SUB,
        BARK_GGML_OP_MUL,
        BARK_GGML_OP_DIV,
        BARK_GGML_OP_SQR,
        BARK_GGML_OP_SQRT,
        BARK_GGML_OP_LOG,
        BARK_GGML_OP_SUM,
        BARK_GGML_OP_SUM_ROWS,
        BARK_GGML_OP_MEAN,
        BARK_GGML_OP_ARGMAX,
        BARK_GGML_OP_REPEAT,
        BARK_GGML_OP_REPEAT_BACK,
        BARK_GGML_OP_CONCAT,
        BARK_GGML_OP_SILU_BACK,
        BARK_GGML_OP_NORM, // normalize
        BARK_GGML_OP_RMS_NORM,
        BARK_GGML_OP_RMS_NORM_BACK,
        BARK_GGML_OP_GROUP_NORM,

        BARK_GGML_OP_MUL_MAT,
        BARK_GGML_OP_OUT_PROD,

        BARK_GGML_OP_SCALE,
        BARK_GGML_OP_SET,
        BARK_GGML_OP_CPY,
        BARK_GGML_OP_CONT,
        BARK_GGML_OP_RESHAPE,
        BARK_GGML_OP_VIEW,
        BARK_GGML_OP_PERMUTE,
        BARK_GGML_OP_TRANSPOSE,
        BARK_GGML_OP_GET_ROWS,
        BARK_GGML_OP_GET_ROWS_BACK,
        BARK_GGML_OP_DIAG,
        BARK_GGML_OP_DIAG_MASK_INF,
        BARK_GGML_OP_DIAG_MASK_ZERO,
        BARK_GGML_OP_SOFT_MAX,
        BARK_GGML_OP_SOFT_MAX_BACK,
        BARK_GGML_OP_ROPE,
        BARK_GGML_OP_ROPE_BACK,
        BARK_GGML_OP_ALIBI,
        BARK_GGML_OP_CLAMP,
        BARK_GGML_OP_CONV_1D,
        BARK_GGML_OP_CONV_1D_STAGE_0,  // internal
        BARK_GGML_OP_CONV_1D_STAGE_1,  // internal
        BARK_GGML_OP_CONV_TRANSPOSE_1D,
        BARK_GGML_OP_CONV_2D,
        BARK_GGML_OP_CONV_2D_STAGE_0, // internal
        BARK_GGML_OP_CONV_2D_STAGE_1, // internal
        BARK_GGML_OP_CONV_TRANSPOSE_2D,
        BARK_GGML_OP_POOL_1D,
        BARK_GGML_OP_POOL_2D,
        BARK_GGML_OP_PAD_REFLEC_1D,

        BARK_GGML_OP_UPSCALE, // nearest interpolate

        BARK_GGML_OP_FLASH_ATTN,
        BARK_GGML_OP_FLASH_FF,
        BARK_GGML_OP_FLASH_ATTN_BACK,
        BARK_GGML_OP_WIN_PART,
        BARK_GGML_OP_WIN_UNPART,
        BARK_GGML_OP_GET_REL_POS,
        BARK_GGML_OP_ADD_REL_POS,

        BARK_GGML_OP_UNARY,

        BARK_GGML_OP_MAP_UNARY,
        BARK_GGML_OP_MAP_BINARY,

        BARK_GGML_OP_MAP_CUSTOM1_F32,
        BARK_GGML_OP_MAP_CUSTOM2_F32,
        BARK_GGML_OP_MAP_CUSTOM3_F32,

        BARK_GGML_OP_MAP_CUSTOM1,
        BARK_GGML_OP_MAP_CUSTOM2,
        BARK_GGML_OP_MAP_CUSTOM3,

        BARK_GGML_OP_CROSS_ENTROPY_LOSS,
        BARK_GGML_OP_CROSS_ENTROPY_LOSS_BACK,

        BARK_GGML_OP_COUNT,
    };

    enum bark_ggml_unary_op {
        BARK_GGML_UNARY_OP_ABS,
        BARK_GGML_UNARY_OP_SGN,
        BARK_GGML_UNARY_OP_NEG,
        BARK_GGML_UNARY_OP_STEP,
        BARK_GGML_UNARY_OP_TANH,
        BARK_GGML_UNARY_OP_ELU,
        BARK_GGML_UNARY_OP_RELU,
        BARK_GGML_UNARY_OP_GELU,
        BARK_GGML_UNARY_OP_GELU_QUICK,
        BARK_GGML_UNARY_OP_SILU,
    };

    enum bark_ggml_object_type {
        BARK_GGML_OBJECT_TENSOR,
        BARK_GGML_OBJECT_GRAPH,
        BARK_GGML_OBJECT_WORK_BUFFER
    };

    enum bark_ggml_log_level {
        BARK_GGML_LOG_LEVEL_ERROR = 2,
        BARK_GGML_LOG_LEVEL_WARN = 3,
        BARK_GGML_LOG_LEVEL_INFO = 4
    };

    // ggml object
    struct bark_ggml_object {
        size_t offs;
        size_t size;

        struct bark_ggml_object * next;

        enum bark_ggml_object_type type;

        char padding[4];
    };

    static const size_t BARK_GGML_OBJECT_SIZE = sizeof(struct bark_ggml_object);

    // n-dimensional tensor
    struct bark_ggml_tensor {
        enum bark_ggml_type         type;
        enum bark_ggml_backend_type backend;

        struct bark_ggml_backend_buffer * buffer;

        int     n_dims;
        int64_t ne[BARK_GGML_MAX_DIMS]; // number of elements
        size_t  nb[BARK_GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = bark_ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / bark_ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum bark_ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[BARK_GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        bool is_param;

        struct bark_ggml_tensor * grad;
        struct bark_ggml_tensor * src[BARK_GGML_MAX_SRC];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        struct bark_ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[BARK_GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[12];
    };

    static const size_t BARK_GGML_TENSOR_SIZE = sizeof(struct bark_ggml_tensor);

    // the compute plan that needs to be prepared for bark_ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct bark_ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `bark_ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `bark_ggml_graph_compute()`

        int n_threads;

        // the `n_tasks` of nodes, 1:1 mapping to cgraph nodes
        int n_tasks[BARK_GGML_MAX_NODES];

        // abort bark_ggml_graph_compute when true
        bool (*abort_callback)(void * data);
        void * abort_callback_data;
    };

    // next prime after BARK_GGML_MAX_NODES
    // #define BARK_GGML_GRAPH_HASHTABLE_SIZE 4099
    // next prime after BARK_GGML_MAX_NODES * 2 (nodes + leafs)
    // #define BARK_GGML_GRAPH_HASHTABLE_SIZE 8273
    // #define BARK_GGML_GRAPH_HASHTABLE_SIZE 16411
    #define BARK_GGML_GRAPH_HASHTABLE_SIZE 200003

    enum bark_ggml_cgraph_eval_order {
        BARK_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
        BARK_GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
        BARK_GGML_CGRAPH_EVAL_ORDER_COUNT
    };

    // computation graph
    struct bark_ggml_cgraph {
        int n_nodes;
        int n_leafs;

        struct bark_ggml_tensor * nodes[BARK_GGML_MAX_NODES];
        struct bark_ggml_tensor * grads[BARK_GGML_MAX_NODES];
        struct bark_ggml_tensor * leafs[BARK_GGML_MAX_NODES];

        void * visited_hash_table[BARK_GGML_GRAPH_HASHTABLE_SIZE];

        enum bark_ggml_cgraph_eval_order order;

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

    static const size_t BARK_GGML_GRAPH_SIZE = sizeof(struct bark_ggml_cgraph);

    // scratch buffer
    struct bark_ggml_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct bark_ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };


    // compute types

    // NOTE: the INIT or FINALIZE pass is not scheduled unless explicitly enabled.
    // This behavior was changed since https://github.com/ggerganov/llama.cpp/pull/1995.
    enum bark_ggml_task_type {
        BARK_GGML_TASK_INIT = 0,
        BARK_GGML_TASK_COMPUTE,
        BARK_GGML_TASK_FINALIZE,
    };

    struct bark_ggml_compute_params {
        enum bark_ggml_task_type type;

        // ith = thread index, nth = number of threads
        int ith, nth;

        // work buffer for all threads
        size_t wsize;
        void * wdata;
    };

    // misc

    BARK_GGML_API void    bark_ggml_time_init(void); // call this once at the beginning of the program
    BARK_GGML_API int64_t bark_ggml_time_ms(void);
    BARK_GGML_API int64_t bark_ggml_time_us(void);
    BARK_GGML_API int64_t bark_ggml_cycles(void);
    BARK_GGML_API int64_t bark_ggml_cycles_per_ms(void);

    BARK_GGML_API void    bark_ggml_numa_init(void); // call once for better performance on NUMA systems
    BARK_GGML_API bool    bark_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    BARK_GGML_API void    bark_ggml_print_object (const struct bark_ggml_object * obj);
    BARK_GGML_API void    bark_ggml_print_objects(const struct bark_ggml_context * ctx);

    BARK_GGML_API int64_t bark_ggml_nelements   (const struct bark_ggml_tensor * tensor);
    BARK_GGML_API int64_t bark_ggml_nrows       (const struct bark_ggml_tensor * tensor);
    BARK_GGML_API size_t  bark_ggml_nbytes      (const struct bark_ggml_tensor * tensor);
    BARK_GGML_API size_t  bark_ggml_nbytes_pad  (const struct bark_ggml_tensor * tensor); // same as bark_ggml_nbytes() but padded to BARK_GGML_MEM_ALIGN
    BARK_GGML_API size_t  bark_ggml_nbytes_split(const struct bark_ggml_tensor * tensor, int nrows_split);

    BARK_GGML_API int     bark_ggml_blck_size (enum bark_ggml_type type);
    BARK_GGML_API size_t  bark_ggml_type_size (enum bark_ggml_type type); // size in bytes for all elements in a block
    BARK_GGML_API float   bark_ggml_type_sizef(enum bark_ggml_type type); // bark_ggml_type_size()/bark_ggml_blck_size() as float

    BARK_GGML_API const char * bark_ggml_type_name(enum bark_ggml_type type);
    BARK_GGML_API const char * bark_ggml_op_name  (enum bark_ggml_op   op);
    BARK_GGML_API const char * bark_ggml_op_symbol(enum bark_ggml_op   op);

    BARK_GGML_API size_t  bark_ggml_element_size(const struct bark_ggml_tensor * tensor);

    BARK_GGML_API bool    bark_ggml_is_quantized(enum bark_ggml_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    BARK_GGML_API enum bark_ggml_type bark_ggml_ftype_to_bark_ggml_type(enum bark_ggml_ftype ftype);

    BARK_GGML_API bool bark_ggml_is_transposed(const struct bark_ggml_tensor * tensor);
    BARK_GGML_API bool bark_ggml_is_contiguous(const struct bark_ggml_tensor * tensor);
    BARK_GGML_API bool bark_ggml_is_permuted  (const struct bark_ggml_tensor * tensor);

    BARK_GGML_API bool bark_ggml_are_same_shape(const struct bark_ggml_tensor * t0, const struct bark_ggml_tensor * t1);

    // use this to compute the memory overhead of a tensor
    BARK_GGML_API size_t bark_ggml_tensor_overhead(void);

    // main

    BARK_GGML_API struct bark_ggml_context * bark_ggml_init(struct bark_ggml_init_params params);
    BARK_GGML_API void                  bark_ggml_free(struct bark_ggml_context * ctx);

    BARK_GGML_API size_t  bark_ggml_used_mem(const struct bark_ggml_context * ctx);

    BARK_GGML_API size_t  bark_ggml_set_scratch (struct bark_ggml_context * ctx, struct bark_ggml_scratch scratch);
    BARK_GGML_API bool    bark_ggml_get_no_alloc(struct bark_ggml_context * ctx);
    BARK_GGML_API void    bark_ggml_set_no_alloc(struct bark_ggml_context * ctx, bool no_alloc);

    BARK_GGML_API void *  bark_ggml_get_mem_buffer     (const struct bark_ggml_context * ctx);
    BARK_GGML_API size_t  bark_ggml_get_mem_size       (const struct bark_ggml_context * ctx);
    BARK_GGML_API size_t  bark_ggml_get_max_tensor_size(const struct bark_ggml_context * ctx);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_new_tensor(
            struct bark_ggml_context * ctx,
            enum   bark_ggml_type type,
            int    n_dims,
            const int64_t *ne);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_new_tensor_1d(
            struct bark_ggml_context * ctx,
            enum   bark_ggml_type type,
            int64_t ne0);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_new_tensor_2d(
            struct bark_ggml_context * ctx,
            enum   bark_ggml_type type,
            int64_t ne0,
            int64_t ne1);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_new_tensor_3d(
            struct bark_ggml_context * ctx,
            enum   bark_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_new_tensor_4d(
            struct bark_ggml_context * ctx,
            enum   bark_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_new_i32(struct bark_ggml_context * ctx, int32_t value);
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_new_f32(struct bark_ggml_context * ctx, float value);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_dup_tensor (struct bark_ggml_context * ctx, const struct bark_ggml_tensor * src);
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_view_tensor(struct bark_ggml_context * ctx, struct bark_ggml_tensor * src);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_get_tensor(struct bark_ggml_context * ctx, const char * name);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_zero(struct bark_ggml_tensor * tensor);
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_i32 (struct bark_ggml_tensor * tensor, int32_t value);
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_f32 (struct bark_ggml_tensor * tensor, float value);

    // Converts a flat index into coordinates
    BARK_GGML_API void    bark_ggml_unravel_index(const struct bark_ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);

    BARK_GGML_API int32_t bark_ggml_get_i32_1d(const struct bark_ggml_tensor * tensor, int i);
    BARK_GGML_API void    bark_ggml_set_i32_1d(const struct bark_ggml_tensor * tensor, int i, int32_t value);

    BARK_GGML_API int32_t bark_ggml_get_i32_nd(const struct bark_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    BARK_GGML_API void    bark_ggml_set_i32_nd(const struct bark_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    BARK_GGML_API float   bark_ggml_get_f32_1d(const struct bark_ggml_tensor * tensor, int i);
    BARK_GGML_API void    bark_ggml_set_f32_1d(const struct bark_ggml_tensor * tensor, int i, float value);

    BARK_GGML_API float   bark_ggml_get_f32_nd(const struct bark_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    BARK_GGML_API void    bark_ggml_set_f32_nd(const struct bark_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    BARK_GGML_API void *  bark_ggml_get_data    (const struct bark_ggml_tensor * tensor);
    BARK_GGML_API float * bark_ggml_get_data_f32(const struct bark_ggml_tensor * tensor);

    BARK_GGML_API enum bark_ggml_unary_op bark_ggml_get_unary_op(const struct bark_ggml_tensor * tensor);

    BARK_GGML_API const char *         bark_ggml_get_name   (const struct bark_ggml_tensor * tensor);
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_name   (      struct bark_ggml_tensor * tensor, const char * name);
    BARK_GGML_ATTRIBUTE_FORMAT(2, 3)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_format_name(      struct bark_ggml_tensor * tensor, const char * fmt, ...);

    //
    // operations on tensors with backpropagation
    //

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_dup(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_dup_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_add(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_add_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_add_cast(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            enum   bark_ggml_type      type);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_add1(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_add1_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_acc(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_acc_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sub(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sub_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_mul(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_mul_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_div(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_div_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sqr(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sqr_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sqrt(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sqrt_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_log(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_log_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // return scalar
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sum(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sum_rows(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // mean along rows
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_mean(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // argmax along rows
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_argmax(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_repeat(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // sums repetitions in a into shape of b
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_repeat_back(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // concat a and b on dim 2
    // used in stable-diffusion
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_concat(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_abs(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_abs_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sgn(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_sgn_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_neg(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_neg_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_step(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_step_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_tanh(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_tanh_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_elu(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_elu_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_relu(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_relu_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // TODO: double-check this computation is correct
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_gelu(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_gelu_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_gelu_quick(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_gelu_quick_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_silu(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_silu_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // a - x
    // b - dy
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_silu_back(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // normalize along rows
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_norm(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            float                 eps);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_norm_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            float                 eps);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rms_norm(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            float                 eps);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rms_norm_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            float                 eps);

    // group normalize along ne0*ne1*n_groups
    // used in stable-diffusion
    // TODO: eps is hardcoded to 1e-6 for now
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_group_norm(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   n_groups);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_group_norm_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   n_groups);

    // a - x
    // b - dy
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rms_norm_back(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            float                 eps);

    // A: k columns, n rows => [ne03, ne02, n, k]
    // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
    // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_mul_mat(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // A: m columns, n rows,
    // B: p columns, n rows,
    // result is m columns, p rows
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_out_prod(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_scale(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_scale_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_1d_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_2d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_set_2d_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // a -> b, return view(b)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cpy(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // a -> b, in-place, return view(b)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cpy_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // make contiguous
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cont(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // make contiguous, in-place
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cont_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // make contiguous, with new shape
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cont_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cont_2d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cont_3d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cont_4d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_reshape(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_reshape_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_reshape_2d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_reshape_3d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_reshape_4d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // offset in bytes
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_view_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_view_2d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_view_3d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_view_4d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_permute(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for bark_ggml_permute(ctx, a, 1, 0, 2, 3)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_transpose(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_get_rows(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_get_rows_back(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            struct bark_ggml_tensor  * c);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_diag(
        struct bark_ggml_context     * ctx,
        struct bark_ggml_tensor      * a);

    // set elements above the diagonal to -INF
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_diag_mask_inf(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_diag_mask_inf_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_diag_mask_zero(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_diag_mask_zero_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   n_past);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_soft_max(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_soft_max_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_soft_max_back(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_soft_max_back_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // rotary position embedding
    // if mode & 1 == 1, skip n_past elements (DEPRECATED)
    // if mode & 2 == 1, GPT-NeoX style
    // if mode & 4 == 1, ChatGLM style
    //
    // b is an int32 vector with size a->ne[2], it contains the positions
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rope(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rope_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // custom RoPE
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rope_custom(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rope_custom_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // xPos RoPE, in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rope_xpos_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   n_dims,
            float                 base,
            bool                  down);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_rope_back(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale,
            float                 xpos_base,
            bool                  xpos_down);

    // alibi position embedding
    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_alibi(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   n_past,
            int                   n_head,
            float                 bias_max);

    // clamp
    // in-place, returns view(a)
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_clamp(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            float                 min,
            float                 max);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_conv_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    // conv_1d with padding = half
    // alias for bark_ggml_conv_1d(a, b, s, a->ne[0]/2, d)
    BARK_GGML_API struct bark_ggml_tensor* bark_ggml_conv_1d_ph(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   s,
            int                   d);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_conv_transpose_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   s0,
            int                   p0,
            int                   d0);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_pad_reflec_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   p0,
            int                   p1);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_conv_2d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1,
            int                   d0,
            int                   d1);

    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_conv_2d_sk_p0(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    // kernel size is a->ne[0] x a->ne[1]
    // stride is 1
    // padding is half
    // example:
    // a:      3    3    256  256
    // b:     64   64    256    1
    // res:   64   64    256    1
    // used in sam
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_conv_2d_s1_ph(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_conv_transpose_2d_p0(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b,
            int                   stride);

    enum bark_ggml_op_pool {
        BARK_GGML_OP_POOL_MAX,
        BARK_GGML_OP_POOL_AVG,
        BARK_GGML_OP_POOL_COUNT,
    };

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_pool_1d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            enum bark_ggml_op_pool     op,
            int                   k0, // kernel size
            int                   s0, // stride
            int                   p0); // padding

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_pool_2d(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            enum bark_ggml_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1);

    // nearest interpolate
    // used in stable-diffusion
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_upscale(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   scale_factor);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_flash_attn(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * q,
            struct bark_ggml_tensor  * k,
            struct bark_ggml_tensor  * v,
            bool                  masked);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_flash_attn_back(
           struct bark_ggml_context * ctx,
           struct bark_ggml_tensor  * q,
           struct bark_ggml_tensor  * k,
           struct bark_ggml_tensor  * v,
           struct bark_ggml_tensor  * d,
           bool                  masked);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_flash_ff(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * b0,
            struct bark_ggml_tensor  * b1,
            struct bark_ggml_tensor  * c0,
            struct bark_ggml_tensor  * c1);

    // partition into non-overlapping windows with padding if needed
    // example:
    // a:   768   64   64    1
    // w:    14
    // res: 768   14   14    25
    // used in sam
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_win_part(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   w);

    // reverse of bark_ggml_win_part
    // used in sam
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_win_unpart(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   w0,
            int                   h0,
            int                   w);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_unary(
            struct bark_ggml_context * ctx,
             struct bark_ggml_tensor * a,
             enum bark_ggml_unary_op op);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_unary_inplace(
        struct bark_ggml_context * ctx,
        struct bark_ggml_tensor  * a,
        enum bark_ggml_unary_op op);

    // used in sam
    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_get_rel_pos(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            int                   qh,
            int                   kh);

    // used in sam

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_add_rel_pos(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * pw,
            struct bark_ggml_tensor  * ph);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_add_rel_pos_inplace(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * a,
            struct bark_ggml_tensor  * pw,
            struct bark_ggml_tensor  * ph);

    // custom operators

    typedef void (*bark_ggml_unary_op_f32_t) (const int, float *, const float *);
    typedef void (*bark_ggml_binary_op_f32_t)(const int, float *, const float *, const float *);

    typedef void (*bark_ggml_custom1_op_f32_t)(struct bark_ggml_tensor *, const struct bark_ggml_tensor *);
    typedef void (*bark_ggml_custom2_op_f32_t)(struct bark_ggml_tensor *, const struct bark_ggml_tensor *, const struct bark_ggml_tensor *);
    typedef void (*bark_ggml_custom3_op_f32_t)(struct bark_ggml_tensor *, const struct bark_ggml_tensor *, const struct bark_ggml_tensor *, const struct bark_ggml_tensor *);

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_unary_f32(
            struct bark_ggml_context        * ctx,
            struct bark_ggml_tensor         * a,
                   bark_ggml_unary_op_f32_t   fun),
        "use bark_ggml_map_custom1 instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_unary_inplace_f32(
            struct bark_ggml_context        * ctx,
            struct bark_ggml_tensor         * a,
                   bark_ggml_unary_op_f32_t   fun),
        "use bark_ggml_map_custom1_inplace instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_binary_f32(
            struct bark_ggml_context         * ctx,
            struct bark_ggml_tensor          * a,
            struct bark_ggml_tensor          * b,
                   bark_ggml_binary_op_f32_t   fun),
        "use bark_ggml_map_custom2 instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_binary_inplace_f32(
            struct bark_ggml_context         * ctx,
            struct bark_ggml_tensor          * a,
            struct bark_ggml_tensor          * b,
                   bark_ggml_binary_op_f32_t   fun),
        "use bark_ggml_map_custom2_inplace instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom1_f32(
            struct bark_ggml_context          * ctx,
            struct bark_ggml_tensor           * a,
                   bark_ggml_custom1_op_f32_t   fun),
        "use bark_ggml_map_custom1 instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom1_inplace_f32(
            struct bark_ggml_context          * ctx,
            struct bark_ggml_tensor           * a,
                   bark_ggml_custom1_op_f32_t   fun),
        "use bark_ggml_map_custom1_inplace instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom2_f32(
            struct bark_ggml_context          * ctx,
            struct bark_ggml_tensor           * a,
            struct bark_ggml_tensor           * b,
                   bark_ggml_custom2_op_f32_t   fun),
        "use bark_ggml_map_custom2 instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom2_inplace_f32(
            struct bark_ggml_context          * ctx,
            struct bark_ggml_tensor           * a,
            struct bark_ggml_tensor           * b,
                   bark_ggml_custom2_op_f32_t   fun),
        "use bark_ggml_map_custom2_inplace instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom3_f32(
            struct bark_ggml_context          * ctx,
            struct bark_ggml_tensor           * a,
            struct bark_ggml_tensor           * b,
            struct bark_ggml_tensor           * c,
                   bark_ggml_custom3_op_f32_t   fun),
        "use bark_ggml_map_custom3 instead");

    BARK_GGML_DEPRECATED(BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom3_inplace_f32(
            struct bark_ggml_context          * ctx,
            struct bark_ggml_tensor           * a,
            struct bark_ggml_tensor           * b,
            struct bark_ggml_tensor           * c,
                   bark_ggml_custom3_op_f32_t   fun),
        "use bark_ggml_map_custom3_inplace instead");

    // custom operators v2

    typedef void (*bark_ggml_custom1_op_t)(struct bark_ggml_tensor * dst , const struct bark_ggml_tensor * a, int ith, int nth, void * userdata);
    typedef void (*bark_ggml_custom2_op_t)(struct bark_ggml_tensor * dst , const struct bark_ggml_tensor * a, const struct bark_ggml_tensor * b, int ith, int nth, void * userdata);
    typedef void (*bark_ggml_custom3_op_t)(struct bark_ggml_tensor * dst , const struct bark_ggml_tensor * a, const struct bark_ggml_tensor * b, const struct bark_ggml_tensor * c, int ith, int nth, void * userdata);

    #define BARK_GGML_N_TASKS_MAX -1

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom1(
            struct bark_ggml_context   * ctx,
            struct bark_ggml_tensor    * a,
            bark_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom1_inplace(
            struct bark_ggml_context   * ctx,
            struct bark_ggml_tensor    * a,
            bark_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom2(
            struct bark_ggml_context   * ctx,
            struct bark_ggml_tensor    * a,
            struct bark_ggml_tensor    * b,
            bark_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom2_inplace(
            struct bark_ggml_context   * ctx,
            struct bark_ggml_tensor    * a,
            struct bark_ggml_tensor    * b,
            bark_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom3(
            struct bark_ggml_context   * ctx,
            struct bark_ggml_tensor    * a,
            struct bark_ggml_tensor    * b,
            struct bark_ggml_tensor    * c,
            bark_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_map_custom3_inplace(
            struct bark_ggml_context   * ctx,
            struct bark_ggml_tensor    * a,
            struct bark_ggml_tensor    * b,
            struct bark_ggml_tensor    * c,
            bark_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    // loss function

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cross_entropy_loss(
            struct bark_ggml_context         * ctx,
            struct bark_ggml_tensor          * a,
            struct bark_ggml_tensor          * b);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_cross_entropy_loss_back(
            struct bark_ggml_context         * ctx,
            struct bark_ggml_tensor          * a,
            struct bark_ggml_tensor          * b,
            struct bark_ggml_tensor          * c);

    //
    // automatic differentiation
    //

    BARK_GGML_API void bark_ggml_set_param(
            struct bark_ggml_context * ctx,
            struct bark_ggml_tensor  * tensor);


    BARK_GGML_API void bark_ggml_build_forward_expand (struct bark_ggml_cgraph * cgraph, struct bark_ggml_tensor * tensor);
    BARK_GGML_API void bark_ggml_build_backward_expand(struct bark_ggml_context * ctx, struct bark_ggml_cgraph * gf, struct bark_ggml_cgraph * gb, bool keep);

    BARK_GGML_API struct bark_ggml_cgraph bark_ggml_build_forward (struct bark_ggml_tensor * tensor);
    BARK_GGML_API struct bark_ggml_cgraph bark_ggml_build_backward(struct bark_ggml_context * ctx, struct bark_ggml_cgraph * gf, bool keep);

    // graph allocation in a context
    BARK_GGML_API struct bark_ggml_cgraph * bark_ggml_new_graph        (struct bark_ggml_context * ctx);
    BARK_GGML_API struct bark_ggml_cgraph * bark_ggml_build_forward_ctx(struct bark_ggml_context * ctx, struct bark_ggml_tensor * tensor);
    BARK_GGML_API size_t bark_ggml_graph_overhead(void);

    // bark_ggml_graph_plan() has to be called before bark_ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    BARK_GGML_API struct bark_ggml_cplan bark_ggml_graph_plan   (struct bark_ggml_cgraph * cgraph, int n_threads /*= BARK_GGML_DEFAULT_N_THREADS*/);
    BARK_GGML_API               int bark_ggml_graph_compute(struct bark_ggml_cgraph * cgraph, struct bark_ggml_cplan * cplan);
    BARK_GGML_API              void bark_ggml_graph_reset  (struct bark_ggml_cgraph * cgraph);

    // same as bark_ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    BARK_GGML_API void bark_ggml_graph_compute_with_ctx(struct bark_ggml_context * ctx, struct bark_ggml_cgraph * cgraph, int n_threads);

    BARK_GGML_API struct bark_ggml_tensor * bark_ggml_graph_get_tensor(struct bark_ggml_cgraph * cgraph, const char * name);

    BARK_GGML_API void               bark_ggml_graph_export(const struct bark_ggml_cgraph * cgraph, const char * fname);
    BARK_GGML_API struct bark_ggml_cgraph bark_ggml_graph_import(const char * fname, struct bark_ggml_context ** ctx_data, struct bark_ggml_context ** ctx_eval);

    // print info and performance information for the graph
    BARK_GGML_API void bark_ggml_graph_print(const struct bark_ggml_cgraph * cgraph);

    // dump the graph into a file using the dot format
    BARK_GGML_API void bark_ggml_graph_dump_dot(const struct bark_ggml_cgraph * gb, const struct bark_ggml_cgraph * gf, const char * filename);

    // build gradient checkpointing backward graph gb for gf using provided checkpoints
    // gb_tmp will contain original backward graph with rewritten backward process nodes,
    // but without the second forward pass nodes.
    BARK_GGML_API void bark_ggml_build_backward_gradient_checkpointing(
            struct bark_ggml_context   * ctx,
            struct bark_ggml_cgraph    * gf,
            struct bark_ggml_cgraph    * gb,
            struct bark_ggml_cgraph    * gb_tmp,
            struct bark_ggml_tensor  * * checkpoints,
            int                     n_checkpoints);
    //
    // optimization
    //

    // optimization methods
    enum bark_ggml_opt_type {
        BARK_GGML_OPT_ADAM,
        BARK_GGML_OPT_LBFGS,
    };

    // linesearch methods
    enum bark_ggml_linesearch {
        BARK_GGML_LINESEARCH_DEFAULT = 1,

        BARK_GGML_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
        BARK_GGML_LINESEARCH_BACKTRACKING_WOLFE        = 1,
        BARK_GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
    };

    // optimization return values
    enum bark_ggml_opt_result {
        BARK_GGML_OPT_OK = 0,
        BARK_GGML_OPT_DID_NOT_CONVERGE,
        BARK_GGML_OPT_NO_CONTEXT,
        BARK_GGML_OPT_INVALID_WOLFE,
        BARK_GGML_OPT_FAIL,
        BARK_GGML_OPT_CANCEL,

        BARK_GGML_LINESEARCH_FAIL = -128,
        BARK_GGML_LINESEARCH_MINIMUM_STEP,
        BARK_GGML_LINESEARCH_MAXIMUM_STEP,
        BARK_GGML_LINESEARCH_MAXIMUM_ITERATIONS,
        BARK_GGML_LINESEARCH_INVALID_PARAMETERS,
    };

    typedef void (*bark_ggml_opt_callback)(void * data, int accum_step, float * sched, bool * cancel);
    typedef void (*bark_ggml_log_callback)(enum bark_ggml_log_level level, const char * text, void * user_data);

    // optimization parameters
    //
    //   see ggml.c (bark_ggml_opt_default_params) for default values
    //
    struct bark_ggml_opt_params {
        enum bark_ggml_opt_type type;

        int n_threads;

        // delta-based convergence test
        //
        //   if past == 0 - disabled
        //   if past > 0:
        //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
        //
        int past;
        float delta;

        // maximum number of iterations without improvement
        //
        //   if 0 - disabled
        //   if > 0:
        //     assume convergence if no cost improvement in this number of iterations
        //
        int max_no_improvement;

        bool print_forward_graph;
        bool print_backward_graph;

        int n_gradient_accumulation;

        // ADAM parameters
        struct {
            int n_iter;

            float sched; // schedule multiplier (fixed, decay or warmup)
            float decay; // weight decay for AdamW, use 0.0f to disable
            int   decay_min_ndim; // minimum number of tensor dimension to apply weight decay
            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float eps_f; // epsilon for convergence test
            float eps_g; // epsilon for convergence test
            float gclip; // gradient clipping
        } adam;

        // LBFGS parameters
        struct {
            int m; // number of corrections to approximate the inv. Hessian
            int n_iter;
            int max_linesearch;

            float eps;      // convergence tolerance
            float ftol;     // line search tolerance
            float wolfe;
            float min_step;
            float max_step;

            enum bark_ggml_linesearch linesearch;
        } lbfgs;
    };

    struct bark_ggml_opt_context {
        struct bark_ggml_context * ctx;
        struct bark_ggml_opt_params params;

        int iter;
        int64_t nx; // number of parameter elements

        bool just_initialized;

        float loss_before;
        float loss_after;

        struct {
            struct bark_ggml_tensor * g;  // current gradient
            struct bark_ggml_tensor * m;  // first moment
            struct bark_ggml_tensor * v;  // second moment
            struct bark_ggml_tensor * pf; // past function values
            float fx_best;
            float fx_prev;
            int n_no_improvement;
        } adam;

        struct {
            struct bark_ggml_tensor * x;    // current parameters
            struct bark_ggml_tensor * xp;   // previous parameters
            struct bark_ggml_tensor * g;    // current gradient
            struct bark_ggml_tensor * gp;   // previous gradient
            struct bark_ggml_tensor * d;    // search direction
            struct bark_ggml_tensor * pf;   // past function values
            struct bark_ggml_tensor * lmal; // the L-BFGS memory alpha
            struct bark_ggml_tensor * lmys; // the L-BFGS memory ys
            struct bark_ggml_tensor * lms;  // the L-BFGS memory s
            struct bark_ggml_tensor * lmy;  // the L-BFGS memory y
            float fx_best;
            float step;
            int j;
            int k;
            int end;
            int n_no_improvement;
        } lbfgs;
    };

    BARK_GGML_API struct bark_ggml_opt_params bark_ggml_opt_default_params(enum bark_ggml_opt_type type);

    // optimize the function defined by the tensor f
    BARK_GGML_API enum bark_ggml_opt_result bark_ggml_opt(
            struct bark_ggml_context * ctx,
            struct bark_ggml_opt_params params,
            struct bark_ggml_tensor * f);

    // initialize optimizer context
    BARK_GGML_API void bark_ggml_opt_init(
            struct bark_ggml_context     * ctx,
            struct bark_ggml_opt_context * opt,
            struct bark_ggml_opt_params    params,
            int64_t                   nx);

    // continue optimizing the function defined by the tensor f
    BARK_GGML_API enum bark_ggml_opt_result bark_ggml_opt_resume(
            struct bark_ggml_context * ctx,
            struct bark_ggml_opt_context * opt,
            struct bark_ggml_tensor * f);

    // continue optimizing the function defined by the tensor f
    BARK_GGML_API enum bark_ggml_opt_result bark_ggml_opt_resume_g(
            struct bark_ggml_context * ctx,
            struct bark_ggml_opt_context * opt,
            struct bark_ggml_tensor * f,
            struct bark_ggml_cgraph * gf,
            struct bark_ggml_cgraph * gb,
            bark_ggml_opt_callback callback,
            void * callback_data);

    //
    // quantization
    //

    BARK_GGML_API size_t bark_ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
    BARK_GGML_API size_t bark_ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
    BARK_GGML_API size_t bark_ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
    BARK_GGML_API size_t bark_ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
    BARK_GGML_API size_t bark_ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);

    BARK_GGML_API size_t bark_ggml_quantize_chunk(enum bark_ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);

    //
    // gguf
    //

    enum bark_gguf_type {
        BARK_GGUF_TYPE_UINT8   = 0,
        BARK_GGUF_TYPE_INT8    = 1,
        BARK_GGUF_TYPE_UINT16  = 2,
        BARK_GGUF_TYPE_INT16   = 3,
        BARK_GGUF_TYPE_UINT32  = 4,
        BARK_GGUF_TYPE_INT32   = 5,
        BARK_GGUF_TYPE_FLOAT32 = 6,
        BARK_GGUF_TYPE_BOOL    = 7,
        BARK_GGUF_TYPE_STRING  = 8,
        BARK_GGUF_TYPE_ARRAY   = 9,
        BARK_GGUF_TYPE_UINT64  = 10,
        BARK_GGUF_TYPE_INT64   = 11,
        BARK_GGUF_TYPE_FLOAT64 = 12,
        BARK_GGUF_TYPE_COUNT,       // marks the end of the enum
    };

    struct bark_gguf_context;

    struct bark_gguf_init_params {
        bool no_alloc;

        // if not NULL, create a bark_ggml_context and allocate the tensor data in it
        struct bark_ggml_context ** ctx;
    };

    BARK_GGML_API struct bark_gguf_context * bark_gguf_init_empty(void);
    BARK_GGML_API struct bark_gguf_context * bark_gguf_init_from_file(const char * fname, struct bark_gguf_init_params params);
    //BARK_GGML_API struct bark_gguf_context * bark_gguf_init_from_buffer(..);

    BARK_GGML_API void bark_gguf_free(struct bark_gguf_context * ctx);

    BARK_GGML_API const char * bark_gguf_type_name(enum bark_gguf_type type);

    BARK_GGML_API int    bark_gguf_get_version    (const struct bark_gguf_context * ctx);
    BARK_GGML_API size_t bark_gguf_get_alignment  (const struct bark_gguf_context * ctx);
    BARK_GGML_API size_t bark_gguf_get_data_offset(const struct bark_gguf_context * ctx);
    BARK_GGML_API void * bark_gguf_get_data       (const struct bark_gguf_context * ctx);

    BARK_GGML_API int          bark_gguf_get_n_kv(const struct bark_gguf_context * ctx);
    BARK_GGML_API int          bark_gguf_find_key(const struct bark_gguf_context * ctx, const char * key);
    BARK_GGML_API const char * bark_gguf_get_key (const struct bark_gguf_context * ctx, int key_id);

    BARK_GGML_API enum bark_gguf_type bark_gguf_get_kv_type (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API enum bark_gguf_type bark_gguf_get_arr_type(const struct bark_gguf_context * ctx, int key_id);

    // will abort if the wrong type is used for the key
    BARK_GGML_API uint8_t      bark_gguf_get_val_u8  (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API int8_t       bark_gguf_get_val_i8  (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API uint16_t     bark_gguf_get_val_u16 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API int16_t      bark_gguf_get_val_i16 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API uint32_t     bark_gguf_get_val_u32 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API int32_t      bark_gguf_get_val_i32 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API float        bark_gguf_get_val_f32 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API uint64_t     bark_gguf_get_val_u64 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API int64_t      bark_gguf_get_val_i64 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API double       bark_gguf_get_val_f64 (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API bool         bark_gguf_get_val_bool(const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API const char * bark_gguf_get_val_str (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API int          bark_gguf_get_arr_n   (const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API const void * bark_gguf_get_arr_data(const struct bark_gguf_context * ctx, int key_id);
    BARK_GGML_API const char * bark_gguf_get_arr_str (const struct bark_gguf_context * ctx, int key_id, int i);

    BARK_GGML_API int    bark_gguf_get_n_tensors    (const struct bark_gguf_context * ctx);
    BARK_GGML_API int    bark_gguf_find_tensor      (const struct bark_gguf_context * ctx, const char * name);
    BARK_GGML_API size_t bark_gguf_get_tensor_offset(const struct bark_gguf_context * ctx, int i);
    BARK_GGML_API char * bark_gguf_get_tensor_name  (const struct bark_gguf_context * ctx, int i);

    // overrides existing values or adds a new one
    BARK_GGML_API void bark_gguf_set_val_u8  (struct bark_gguf_context * ctx, const char * key, uint8_t  val);
    BARK_GGML_API void bark_gguf_set_val_i8  (struct bark_gguf_context * ctx, const char * key, int8_t   val);
    BARK_GGML_API void bark_gguf_set_val_u16 (struct bark_gguf_context * ctx, const char * key, uint16_t val);
    BARK_GGML_API void bark_gguf_set_val_i16 (struct bark_gguf_context * ctx, const char * key, int16_t  val);
    BARK_GGML_API void bark_gguf_set_val_u32 (struct bark_gguf_context * ctx, const char * key, uint32_t val);
    BARK_GGML_API void bark_gguf_set_val_i32 (struct bark_gguf_context * ctx, const char * key, int32_t  val);
    BARK_GGML_API void bark_gguf_set_val_f32 (struct bark_gguf_context * ctx, const char * key, float    val);
    BARK_GGML_API void bark_gguf_set_val_u64 (struct bark_gguf_context * ctx, const char * key, uint64_t val);
    BARK_GGML_API void bark_gguf_set_val_i64 (struct bark_gguf_context * ctx, const char * key, int64_t  val);
    BARK_GGML_API void bark_gguf_set_val_f64 (struct bark_gguf_context * ctx, const char * key, double   val);
    BARK_GGML_API void bark_gguf_set_val_bool(struct bark_gguf_context * ctx, const char * key, bool     val);
    BARK_GGML_API void bark_gguf_set_val_str (struct bark_gguf_context * ctx, const char * key, const char * val);
    BARK_GGML_API void bark_gguf_set_arr_data(struct bark_gguf_context * ctx, const char * key, enum bark_gguf_type type, const void * data, int n);
    BARK_GGML_API void bark_gguf_set_arr_str (struct bark_gguf_context * ctx, const char * key, const char ** data, int n);

    // set or add KV pairs from another context
    BARK_GGML_API void bark_gguf_set_kv(struct bark_gguf_context * ctx, struct bark_gguf_context * src);

    // manage tensor info
    BARK_GGML_API void bark_gguf_add_tensor(struct bark_gguf_context * ctx, const struct bark_ggml_tensor * tensor);
    BARK_GGML_API void bark_gguf_set_tensor_type(struct bark_gguf_context * ctx, const char * name, enum bark_ggml_type type);
    BARK_GGML_API void bark_gguf_set_tensor_data(struct bark_gguf_context * ctx, const char * name, const void * data, size_t size);

    // writing gguf files can be done in 2 ways:
    //
    // - write the entire bark_gguf_context to a binary file in a single pass:
    //
    //   bark_gguf_write_to_file(ctx, fname);
    //
    // - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
    //
    //   FILE * f = fopen(fname, "wb");
    //   fseek(f, bark_gguf_get_meta_size(ctx), SEEK_SET);
    //   fwrite(f, ...);
    //   void * data = bark_gguf_meta_get_meta_data(ctx);
    //   fseek(f, 0, SEEK_SET);
    //   fwrite(f, data, bark_gguf_get_meta_size(ctx));
    //   free(data);
    //   fclose(f);
    //

    // write the entire context to a binary file
    BARK_GGML_API void bark_gguf_write_to_file(const struct bark_gguf_context * ctx, const char * fname, bool only_meta);

    // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
    BARK_GGML_API size_t bark_gguf_get_meta_size(const struct bark_gguf_context * ctx);
    BARK_GGML_API void   bark_gguf_get_meta_data(const struct bark_gguf_context * ctx, void * data);

    //
    // system info
    //

    BARK_GGML_API int bark_ggml_cpu_has_avx        (void);
    BARK_GGML_API int bark_ggml_cpu_has_avx2       (void);
    BARK_GGML_API int bark_ggml_cpu_has_avx512     (void);
    BARK_GGML_API int bark_ggml_cpu_has_avx512_vbmi(void);
    BARK_GGML_API int bark_ggml_cpu_has_avx512_vnni(void);
    BARK_GGML_API int bark_ggml_cpu_has_fma        (void);
    BARK_GGML_API int bark_ggml_cpu_has_neon       (void);
    BARK_GGML_API int bark_ggml_cpu_has_arm_fma    (void);
    BARK_GGML_API int bark_ggml_cpu_has_metal      (void);
    BARK_GGML_API int bark_ggml_cpu_has_f16c       (void);
    BARK_GGML_API int bark_ggml_cpu_has_fp16_va    (void);
    BARK_GGML_API int bark_ggml_cpu_has_wasm_simd  (void);
    BARK_GGML_API int bark_ggml_cpu_has_blas       (void);
    BARK_GGML_API int bark_ggml_cpu_has_cublas     (void);
    BARK_GGML_API int bark_ggml_cpu_has_clblast    (void);
    BARK_GGML_API int bark_ggml_cpu_has_gpublas    (void);
    BARK_GGML_API int bark_ggml_cpu_has_sse3       (void);
    BARK_GGML_API int bark_ggml_cpu_has_ssse3      (void);
    BARK_GGML_API int bark_ggml_cpu_has_vsx        (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
// restrict not standard in C++
#define BARK_GGML_RESTRICT
#else
#define BARK_GGML_RESTRICT restrict
#endif
    typedef void (*bark_ggml_to_float_t)  (const void  * BARK_GGML_RESTRICT x, float * BARK_GGML_RESTRICT y, int k);
    typedef void (*bark_ggml_from_float_t)(const float * BARK_GGML_RESTRICT x, void  * BARK_GGML_RESTRICT y, int k);
    typedef void (*bark_ggml_vec_dot_t)   (const int n, float * BARK_GGML_RESTRICT s, const void * BARK_GGML_RESTRICT x, const void * BARK_GGML_RESTRICT y);

    typedef struct {
        const char      * type_name;
        int               blck_size;
        size_t            type_size;
        bool              is_quantized;
        bark_ggml_to_float_t   to_float;
        bark_ggml_from_float_t from_float;
        bark_ggml_from_float_t from_float_reference;
        bark_ggml_vec_dot_t    vec_dot;
        enum bark_ggml_type    vec_dot_type;
    } bark_ggml_type_traits_t;

    BARK_GGML_API bark_ggml_type_traits_t bark_ggml_internal_get_type_traits(enum bark_ggml_type type);

#ifdef  __cplusplus
}
#endif
