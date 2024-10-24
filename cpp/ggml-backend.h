#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif
    struct bark_ggml_backend;
    struct bark_ggml_backend_buffer;

    // type-erased backend-specific types / wrappers
    typedef void * bark_ggml_backend_context_t;
    typedef void * bark_ggml_backend_graph_plan_t;
    typedef void * bark_ggml_backend_buffer_context_t;

    // avoid accessing internals of these types
    typedef struct bark_ggml_backend        * bark_ggml_backend_t;
    typedef struct bark_ggml_backend_buffer * bark_ggml_backend_buffer_t;

    //
    // backend buffer
    //

    struct bark_ggml_backend_buffer_i {
        void   (*free_buffer)   (bark_ggml_backend_buffer_t buffer);
        void * (*get_base)      (bark_ggml_backend_buffer_t buffer); // get base pointer
        size_t (*get_alloc_size)(bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor); // pre-allocation callback
        void   (*init_tensor)   (bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor); // post-allocation callback
        void   (*free_tensor)   (bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor); // pre-free callback
    };

    // TODO: hide behind API
    struct bark_ggml_backend_buffer {
        struct bark_ggml_backend_buffer_i iface;

        bark_ggml_backend_t                backend;
        bark_ggml_backend_buffer_context_t context;

        size_t size;
    };

    // backend buffer functions
    BARK_GGML_API bark_ggml_backend_buffer_t bark_ggml_backend_buffer_init(
            struct bark_ggml_backend                  * backend,
            struct bark_ggml_backend_buffer_i           iface,
                   bark_ggml_backend_buffer_context_t   context,
                   size_t                          size);

    BARK_GGML_API void   bark_ggml_backend_buffer_free          (bark_ggml_backend_buffer_t buffer);
    BARK_GGML_API size_t bark_ggml_backend_buffer_get_alignment (bark_ggml_backend_buffer_t buffer);
    BARK_GGML_API void * bark_ggml_backend_buffer_get_base      (bark_ggml_backend_buffer_t buffer);
    BARK_GGML_API size_t bark_ggml_backend_buffer_get_size      (bark_ggml_backend_buffer_t buffer);
    BARK_GGML_API size_t bark_ggml_backend_buffer_get_alloc_size(bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor);
    BARK_GGML_API void   bark_ggml_backend_buffer_init_tensor   (bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor);
    BARK_GGML_API void   bark_ggml_backend_buffer_free_tensor   (bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor);

    //
    // backend
    //

    struct bark_ggml_backend_i {
        const char * (*get_name)(bark_ggml_backend_t backend);

        void (*free)(bark_ggml_backend_t backend);

        // buffer allocation
        bark_ggml_backend_buffer_t (*alloc_buffer)(bark_ggml_backend_t backend, size_t size);

        // get buffer alignment
        size_t (*get_alignment)(bark_ggml_backend_t backend);

        // tensor data access
        // these functions can be asynchronous, helper functions are provided for synchronous access that automatically call synchronize
        void (*set_tensor_async)(bark_ggml_backend_t backend,       struct bark_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(bark_ggml_backend_t backend, const struct bark_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*synchronize)     (bark_ggml_backend_t backend);

        // (optional) copy tensor between different backends, allow for single-copy tranfers
        void (*cpy_tensor_from)(bark_ggml_backend_t backend, struct bark_ggml_tensor * src, struct bark_ggml_tensor * dst);
        void (*cpy_tensor_to)  (bark_ggml_backend_t backend, struct bark_ggml_tensor * src, struct bark_ggml_tensor * dst);

        // compute graph with a plan
        bark_ggml_backend_graph_plan_t (*graph_plan_create) (bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan);
        void                      (*graph_plan_compute)(bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan);

        // compute graph without a plan
        void (*graph_compute)(bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*supports_op)(bark_ggml_backend_t backend, const struct bark_ggml_tensor * op);
    };

    // TODO: hide behind API
    struct bark_ggml_backend {
        struct bark_ggml_backend_i iface;

        bark_ggml_backend_context_t context;
    };

    // backend helper functions
    BARK_GGML_API bark_ggml_backend_t bark_ggml_get_backend(const struct bark_ggml_tensor * tensor);

    BARK_GGML_API const char * bark_ggml_backend_name(bark_ggml_backend_t backend);
    BARK_GGML_API void         bark_ggml_backend_free(bark_ggml_backend_t backend);

    BARK_GGML_API bark_ggml_backend_buffer_t bark_ggml_backend_alloc_buffer(bark_ggml_backend_t backend, size_t size);

    BARK_GGML_API size_t bark_ggml_backend_get_alignment(bark_ggml_backend_t backend);

    BARK_GGML_API void bark_ggml_backend_tensor_set_async(      struct bark_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    BARK_GGML_API void bark_ggml_backend_tensor_get_async(const struct bark_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    BARK_GGML_API void bark_ggml_backend_tensor_set(      struct bark_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    BARK_GGML_API void bark_ggml_backend_tensor_get(const struct bark_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    BARK_GGML_API void bark_ggml_backend_synchronize(bark_ggml_backend_t backend);

    BARK_GGML_API bark_ggml_backend_graph_plan_t bark_ggml_backend_graph_plan_create (bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph);

    BARK_GGML_API void bark_ggml_backend_graph_plan_free   (bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan);
    BARK_GGML_API void bark_ggml_backend_graph_plan_compute(bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan);
    BARK_GGML_API void bark_ggml_backend_graph_compute     (bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph);
    BARK_GGML_API bool bark_ggml_backend_supports_op       (bark_ggml_backend_t backend, const struct bark_ggml_tensor * op);

    // tensor copy between different backends
    BARK_GGML_API void bark_ggml_backend_tensor_copy(struct bark_ggml_tensor * src, struct bark_ggml_tensor * dst);

    //
    // CPU backend
    //

    BARK_GGML_API bark_ggml_backend_t bark_ggml_backend_cpu_init(void);

    BARK_GGML_API bool bark_ggml_backend_is_cpu(bark_ggml_backend_t backend);

    BARK_GGML_API void bark_ggml_backend_cpu_set_n_threads(bark_ggml_backend_t backend_cpu, int n_threads);

    BARK_GGML_API bark_ggml_backend_buffer_t bark_ggml_backend_cpu_buffer_from_ptr(bark_ggml_backend_t backend_cpu, void * ptr, size_t size);

#ifdef  __cplusplus
}
#endif
