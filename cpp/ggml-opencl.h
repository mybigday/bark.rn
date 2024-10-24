#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void bark_ggml_cl_init(void);

void   bark_ggml_cl_mul(const struct bark_ggml_tensor * src0, const struct bark_ggml_tensor * src1, struct bark_ggml_tensor * dst);
bool   bark_ggml_cl_can_mul_mat(const struct bark_ggml_tensor * src0, const struct bark_ggml_tensor * src1, struct bark_ggml_tensor * dst);
size_t bark_ggml_cl_mul_mat_get_wsize(const struct bark_ggml_tensor * src0, const struct bark_ggml_tensor * src1, struct bark_ggml_tensor * dst);
void   bark_ggml_cl_mul_mat(const struct bark_ggml_tensor * src0, const struct bark_ggml_tensor * src1, struct bark_ggml_tensor * dst, void * wdata, size_t wsize);

void * bark_ggml_cl_host_malloc(size_t size);
void   bark_ggml_cl_host_free(void * ptr);

void bark_ggml_cl_free_data(const struct bark_ggml_tensor* tensor);

void bark_ggml_cl_transform_tensor(void * data, struct bark_ggml_tensor * tensor);

#ifdef  __cplusplus
}
#endif
