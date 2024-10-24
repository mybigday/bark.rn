#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct bark_ggml_backend_buffer;

BARK_GGML_API struct bark_ggml_allocr * bark_ggml_allocr_new(void * data, size_t size, size_t alignment);
BARK_GGML_API struct bark_ggml_allocr * bark_ggml_allocr_new_measure(size_t alignment);
BARK_GGML_API struct bark_ggml_allocr * bark_ggml_allocr_new_from_buffer(struct bark_ggml_backend_buffer * buffer);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
BARK_GGML_API void   bark_ggml_allocr_set_parse_seq(struct bark_ggml_allocr * alloc, const int * list, int n);

BARK_GGML_API void   bark_ggml_allocr_free       (struct bark_ggml_allocr * alloc);
BARK_GGML_API bool   bark_ggml_allocr_is_measure (struct bark_ggml_allocr * alloc);
BARK_GGML_API void   bark_ggml_allocr_reset      (struct bark_ggml_allocr * alloc);
BARK_GGML_API void   bark_ggml_allocr_alloc      (struct bark_ggml_allocr * alloc, struct bark_ggml_tensor * tensor);
BARK_GGML_API size_t bark_ggml_allocr_alloc_graph(struct bark_ggml_allocr * alloc, struct bark_ggml_cgraph * graph);
BARK_GGML_API size_t bark_ggml_allocr_max_size   (struct bark_ggml_allocr * alloc);

BARK_GGML_API size_t bark_ggml_allocr_alloc_graph_n(
                    struct bark_ggml_allocr * alloc,
                    struct bark_ggml_cgraph ** graphs, int n_graphs,
                    struct bark_ggml_tensor *** inputs, struct bark_ggml_tensor *** outputs);

#ifdef  __cplusplus
}
#endif
