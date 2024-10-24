#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED BARK_GGML_UNUSED

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// backend buffer

bark_ggml_backend_buffer_t bark_ggml_backend_buffer_init(
        struct bark_ggml_backend                  * backend,
        struct bark_ggml_backend_buffer_i           iface,
               bark_ggml_backend_buffer_context_t   context,
               size_t                          size) {
    bark_ggml_backend_buffer_t buffer = malloc(sizeof(struct bark_ggml_backend_buffer));

    BARK_GGML_ASSERT(iface.get_base != NULL);

    (*buffer) = (struct bark_ggml_backend_buffer) {
        /* .interface = */ iface,
        /* .backend   = */ backend,
        /* .context   = */ context,
        /* .size      = */ size,
    };

    return buffer;
}

void bark_ggml_backend_buffer_free(bark_ggml_backend_buffer_t buffer) {
    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    free(buffer);
}

size_t bark_ggml_backend_buffer_get_alignment(bark_ggml_backend_buffer_t buffer) {
    return bark_ggml_backend_get_alignment(buffer->backend);
}

void * bark_ggml_backend_buffer_get_base(bark_ggml_backend_buffer_t buffer) {
    return buffer->iface.get_base(buffer);
}

size_t bark_ggml_backend_buffer_get_size(bark_ggml_backend_buffer_t buffer) {
    return buffer->size;
}

size_t bark_ggml_backend_buffer_get_alloc_size(bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor) {
    if (buffer->iface.get_alloc_size) {
        return buffer->iface.get_alloc_size(buffer, tensor);
    }
    return bark_ggml_nbytes(tensor);
}

void bark_ggml_backend_buffer_init_tensor(bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor) {
    if (buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}

void bark_ggml_backend_buffer_free_tensor(bark_ggml_backend_buffer_t buffer, struct bark_ggml_tensor * tensor) {
    if (buffer->iface.free_tensor) {
        buffer->iface.free_tensor(buffer, tensor);
    }
}

// backend

bark_ggml_backend_t bark_ggml_get_backend(const struct bark_ggml_tensor * tensor) {
    return tensor->buffer->backend;
}

const char * bark_ggml_backend_name(bark_ggml_backend_t backend) {
    return backend->iface.get_name(backend);
}

void bark_ggml_backend_free(bark_ggml_backend_t backend) {
    backend->iface.free(backend);
}

bark_ggml_backend_buffer_t bark_ggml_backend_alloc_buffer(bark_ggml_backend_t backend, size_t size) {
    return backend->iface.alloc_buffer(backend, size);
}

size_t bark_ggml_backend_get_alignment(bark_ggml_backend_t backend) {
    return backend->iface.get_alignment(backend);
}

void bark_ggml_backend_tensor_set_async(struct bark_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    bark_ggml_get_backend(tensor)->iface.set_tensor_async(bark_ggml_get_backend(tensor), tensor, data, offset, size);
}

void bark_ggml_backend_tensor_get_async(const struct bark_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    bark_ggml_get_backend(tensor)->iface.get_tensor_async(bark_ggml_get_backend(tensor), tensor, data, offset, size);
}

void bark_ggml_backend_tensor_set(struct bark_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    bark_ggml_get_backend(tensor)->iface.set_tensor_async(bark_ggml_get_backend(tensor), tensor, data, offset, size);
    bark_ggml_get_backend(tensor)->iface.synchronize(bark_ggml_get_backend(tensor));
}

void bark_ggml_backend_tensor_get(const struct bark_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    bark_ggml_get_backend(tensor)->iface.get_tensor_async(bark_ggml_get_backend(tensor), tensor, data, offset, size);
    bark_ggml_get_backend(tensor)->iface.synchronize(bark_ggml_get_backend(tensor));
}

void bark_ggml_backend_synchronize(bark_ggml_backend_t backend) {
    backend->iface.synchronize(backend);
}

bark_ggml_backend_graph_plan_t bark_ggml_backend_graph_plan_create(bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph) {
    return backend->iface.graph_plan_create(backend, cgraph);
}

void bark_ggml_backend_graph_plan_free(bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_free(backend, plan);
}

void bark_ggml_backend_graph_plan_compute(bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_compute(backend, plan);
}

void bark_ggml_backend_graph_compute(bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph) {
    backend->iface.graph_compute(backend, cgraph);
}

bool bark_ggml_backend_supports_op(bark_ggml_backend_t backend, const struct bark_ggml_tensor * op) {
    return backend->iface.supports_op(backend, op);
}

// backend copy

static bool bark_ggml_are_same_layout(const struct bark_ggml_tensor * a, const struct bark_ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < BARK_GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

void bark_ggml_backend_tensor_copy(struct bark_ggml_tensor * src, struct bark_ggml_tensor * dst) {
    //printf("src: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", src->name, (int)src->ne[0], (int)src->ne[1], (int)src->ne[2], (int)src->ne[3], (int)src->nb[0], (int)src->nb[1], (int)src->nb[2], (int)src->nb[3]);
    //printf("dst: %s ne: [%d %d %d %d] nb: [%d %d %d %d]\n", dst->name, (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3], (int)dst->nb[0], (int)dst->nb[1], (int)dst->nb[2], (int)dst->nb[3]);
    BARK_GGML_ASSERT(bark_ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    // printf("cpy tensor %s from %s to %s (%lu bytes)\n", src->name, bark_ggml_backend_name(src->backend), bark_ggml_backend_name(dst->backend), bark_ggml_nbytes(src));

    if (src == dst) {
        return;
    }

    // TODO: allow backends to support copy to/from same backend

    if (bark_ggml_get_backend(dst)->iface.cpy_tensor_from != NULL) {
        bark_ggml_get_backend(dst)->iface.cpy_tensor_from(bark_ggml_get_backend(dst)->context, src, dst);
    } else if (bark_ggml_get_backend(src)->iface.cpy_tensor_to != NULL) {
        bark_ggml_get_backend(src)->iface.cpy_tensor_to(bark_ggml_get_backend(src)->context, src, dst);
    } else {
        // shouldn't be hit when copying from/to CPU
        #ifndef NDEBUG
        fprintf(stderr, "bark_ggml_backend_tensor_copy: neither cpy_tensor_from nor cpy_tensor_to are implemented for backends %s and %s, falling back to get/set\n", bark_ggml_backend_name(src->buffer->backend), bark_ggml_backend_name(dst->buffer->backend));
        #endif
        size_t nbytes = bark_ggml_nbytes(src);
        void * data = malloc(nbytes);
        bark_ggml_backend_tensor_get(src, data, 0, nbytes);
        bark_ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

// backend CPU

struct bark_ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;
};

static const char * bark_ggml_backend_cpu_name(bark_ggml_backend_t backend) {
    return "CPU";

    UNUSED(backend);
}

static void bark_ggml_backend_cpu_free(bark_ggml_backend_t backend) {
    struct bark_ggml_backend_cpu_context * cpu_ctx = (struct bark_ggml_backend_cpu_context *)backend->context;
    free(cpu_ctx->work_data);
    free(cpu_ctx);
    free(backend);
}

static void * bark_ggml_backend_cpu_buffer_get_base(bark_ggml_backend_buffer_t buffer) {
    return (void *)buffer->context;
}

static void bark_ggml_backend_cpu_buffer_free_buffer(bark_ggml_backend_buffer_t buffer) {
    free(buffer->context);
    UNUSED(buffer);
}

static struct bark_ggml_backend_buffer_i cpu_backend_buffer_i = {
    /* .free_buffer    = */ bark_ggml_backend_cpu_buffer_free_buffer,
    /* .get_base       = */ bark_ggml_backend_cpu_buffer_get_base,
    /* .get_alloc_size = */ NULL, // defaults to bark_ggml_nbytes
    /* .init_tensor    = */ NULL, // no initialization required
    /* .free_tensor    = */ NULL, // no cleanup required
};

// for buffers from ptr, free is not called
static struct bark_ggml_backend_buffer_i cpu_backend_buffer_i_from_ptr = {
    /* .free_buffer    = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base       = */ bark_ggml_backend_cpu_buffer_get_base,
    /* .get_alloc_size = */ NULL, // defaults to bark_ggml_nbytes
    /* .init_tensor    = */ NULL,
    /* .free_tensor    = */ NULL,
};

static const size_t TENSOR_ALIGNMENT = 64; // should be enough for AVX 512

static bark_ggml_backend_buffer_t bark_ggml_backend_cpu_alloc_buffer(bark_ggml_backend_t backend, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: maybe use BARK_GGML_ALIGNED_MALLOC?

    return bark_ggml_backend_buffer_init(backend, cpu_backend_buffer_i, data, size);
}

static size_t bark_ggml_backend_cpu_get_alignment(bark_ggml_backend_t backend) {
    return TENSOR_ALIGNMENT;
    UNUSED(backend);
}

static void bark_ggml_backend_cpu_set_tensor_async(bark_ggml_backend_t backend, struct bark_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    BARK_GGML_ASSERT(offset + size <= bark_ggml_nbytes(tensor) && "tensor write out of bounds");
    BARK_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy((char *)tensor->data + offset, data, size);

    UNUSED(backend);
}

static void bark_ggml_backend_cpu_get_tensor_async(bark_ggml_backend_t backend, const struct bark_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    BARK_GGML_ASSERT(offset + size <= bark_ggml_nbytes(tensor) && "tensor read out of bounds");
    BARK_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    memcpy(data, (const char *)tensor->data + offset, size);

    UNUSED(backend);
}

static void bark_ggml_backend_cpu_synchronize(bark_ggml_backend_t backend) {
    UNUSED(backend);
}

static void bark_ggml_backend_cpu_cpy_tensor_from(bark_ggml_backend_t backend, struct bark_ggml_tensor * src, struct bark_ggml_tensor * dst) {
    bark_ggml_backend_tensor_get(src, dst->data, 0, bark_ggml_nbytes(src));

    UNUSED(backend);
}

static void bark_ggml_backend_cpu_cpy_tensor_to(bark_ggml_backend_t backend, struct bark_ggml_tensor * src, struct bark_ggml_tensor * dst) {
    // for a backend such as CUDA that can queue async calls, it is ok to do this asynchronously, but it may not be the case for other backends
    bark_ggml_backend_tensor_set_async(dst, src->data, 0, bark_ggml_nbytes(src));

    UNUSED(backend);
}

struct bark_ggml_backend_plan_cpu {
    struct bark_ggml_cplan cplan;
    struct bark_ggml_cgraph cgraph;
};

static bark_ggml_backend_graph_plan_t bark_ggml_backend_cpu_graph_plan_create(bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph) {
    struct bark_ggml_backend_cpu_context * cpu_ctx = (struct bark_ggml_backend_cpu_context *)backend->context;

    struct bark_ggml_backend_plan_cpu * cpu_plan = malloc(sizeof(struct bark_ggml_backend_plan_cpu));

    cpu_plan->cplan = bark_ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph;

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = malloc(cpu_plan->cplan.work_size);
    }

    return cpu_plan;
}

static void bark_ggml_backend_cpu_graph_plan_free(bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan) {
    struct bark_ggml_backend_plan_cpu * cpu_plan = (struct bark_ggml_backend_plan_cpu *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    UNUSED(backend);
}

static void bark_ggml_backend_cpu_graph_plan_compute(bark_ggml_backend_t backend, bark_ggml_backend_graph_plan_t plan) {
    struct bark_ggml_backend_plan_cpu * cpu_plan = (struct bark_ggml_backend_plan_cpu *)plan;

    bark_ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    UNUSED(backend);
}

static void bark_ggml_backend_cpu_graph_compute(bark_ggml_backend_t backend, struct bark_ggml_cgraph * cgraph) {
    struct bark_ggml_backend_cpu_context * cpu_ctx = (struct bark_ggml_backend_cpu_context *)backend->context;

    struct bark_ggml_cplan cplan = bark_ggml_graph_plan(cgraph, cpu_ctx->n_threads);

    if (cpu_ctx->work_size < cplan.work_size) {
        // TODO: may be faster to free and use malloc to avoid the copy
        cpu_ctx->work_data = realloc(cpu_ctx->work_data, cplan.work_size);
        cpu_ctx->work_size = cplan.work_size;
    }

    cplan.work_data = cpu_ctx->work_data;

    bark_ggml_graph_compute(cgraph, &cplan);
}

static bool bark_ggml_backend_cpu_supports_op(bark_ggml_backend_t backend, const struct bark_ggml_tensor * op) {
    return true;
    UNUSED(backend);
    UNUSED(op);
}

static struct bark_ggml_backend_i cpu_backend_i = {
    /* .get_name            = */ bark_ggml_backend_cpu_name,
    /* .free                = */ bark_ggml_backend_cpu_free,
    /* .alloc_buffer        = */ bark_ggml_backend_cpu_alloc_buffer,
    /* .get_alignment       = */ bark_ggml_backend_cpu_get_alignment,
    /* .set_tensor_async    = */ bark_ggml_backend_cpu_set_tensor_async,
    /* .get_tensor_async    = */ bark_ggml_backend_cpu_get_tensor_async,
    /* .synchronize         = */ bark_ggml_backend_cpu_synchronize,
    /* .cpy_tensor_from     = */ bark_ggml_backend_cpu_cpy_tensor_from,
    /* .cpy_tensor_to       = */ bark_ggml_backend_cpu_cpy_tensor_to,
    /* .graph_plan_create   = */ bark_ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free     = */ bark_ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_compute  = */ bark_ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute       = */ bark_ggml_backend_cpu_graph_compute,
    /* .supports_op         = */ bark_ggml_backend_cpu_supports_op,
};

bark_ggml_backend_t bark_ggml_backend_cpu_init(void) {
    struct bark_ggml_backend_cpu_context * ctx = malloc(sizeof(struct bark_ggml_backend_cpu_context));

    ctx->n_threads = BARK_GGML_DEFAULT_N_THREADS;
    ctx->work_data = NULL;
    ctx->work_size = 0;

    bark_ggml_backend_t cpu_backend = malloc(sizeof(struct bark_ggml_backend));

    *cpu_backend = (struct bark_ggml_backend) {
        /* .interface = */ cpu_backend_i,
        /* .context   = */ ctx
    };
    return cpu_backend;
}

bool bark_ggml_backend_is_cpu(bark_ggml_backend_t backend) {
    return backend->iface.get_name == bark_ggml_backend_cpu_name;
}

void bark_ggml_backend_cpu_set_n_threads(bark_ggml_backend_t backend_cpu, int n_threads) {
    BARK_GGML_ASSERT(bark_ggml_backend_is_cpu(backend_cpu));

    struct bark_ggml_backend_cpu_context * ctx = (struct bark_ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

bark_ggml_backend_buffer_t bark_ggml_backend_cpu_buffer_from_ptr(bark_ggml_backend_t backend_cpu, void * ptr, size_t size) {
    return bark_ggml_backend_buffer_init(backend_cpu, cpu_backend_buffer_i_from_ptr, ptr, size);
}
