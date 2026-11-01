/*
 * iris_safetensors.h - Safetensors file format reader
 *
 * Safetensors format:
 *   - 8 bytes: uint64 little-endian header size
 *   - N bytes: JSON header with tensor metadata
 *   - Remaining: raw tensor data
 */

#ifndef IRIS_SAFETENSORS_H
#define IRIS_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

/* Maximum number of tensors per file */
#define SAFETENSORS_MAX_TENSORS 1024  /* Qwen3-VL in one file: 750 */

/* Tensor data types */
typedef enum {
    DTYPE_F32 = 0,
    DTYPE_F16 = 1,
    DTYPE_BF16 = 2,
    DTYPE_I32 = 3,
    DTYPE_I64 = 4,
    DTYPE_BOOL = 5,
    DTYPE_UNKNOWN = -1
} safetensor_dtype_t;

/* Tensor descriptor */
typedef struct {
    char name[256];
    safetensor_dtype_t dtype;
    int ndim;
    int64_t shape[8];
    size_t data_offset;
    size_t data_size;
} safetensor_t;

/* Safetensors file handle */
typedef struct {
    char *path;
    void *data;              /* mmap'd file data */
    size_t file_size;
    size_t header_size;
    char *header_json;
    int num_tensors;
    safetensor_t tensors[SAFETENSORS_MAX_TENSORS];
} safetensors_file_t;

/* Open a safetensors file (memory-mapped) */
safetensors_file_t *safetensors_open(const char *path);

/* Close and free resources */
void safetensors_close(safetensors_file_t *sf);

/* Find a tensor by name, returns NULL if not found */
const safetensor_t *safetensors_find(const safetensors_file_t *sf, const char *name);

/* Get raw pointer to tensor data (within mmap'd region) */
const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t);

/* Get tensor data as float32 array (allocates, caller must free)
 * Handles conversion from F16/BF16 */
float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t);

/* Get tensor data as raw bf16 array (allocates, caller must free)
 * Only works for BF16 tensors. Returns NULL for other dtypes. */
uint16_t *safetensors_get_bf16(const safetensors_file_t *sf, const safetensor_t *t);

/* Get direct pointer to bf16 data in mmap'd region (no copy, caller must NOT free)
 * Only works for BF16 tensors. Returns NULL for other dtypes. */
uint16_t *safetensors_get_bf16_direct(const safetensors_file_t *sf, const safetensor_t *t);

/* Check if tensor is stored in bf16 format */
int safetensor_is_bf16(const safetensor_t *t);

/* Get total number of elements in tensor */
int64_t safetensor_numel(const safetensor_t *t);

/* Print tensor info (for debugging) */
void safetensor_print(const safetensor_t *t);

/* Print all tensors in file */
void safetensors_print_all(const safetensors_file_t *sf);

#endif /* IRIS_SAFETENSORS_H */
