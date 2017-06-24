//
// Created by salmon on 17-6-24.
//
#include "simpla/utilities/cuda/cuda.h"

template <typename T>
__global__ void spCudaFill(T *dest, T const &src, size_t n) {
    size_t s = threadIdx.x * blockDim.x;
    if (s < n) { dest[s] = src; };
}
int spMemoryFill(float *dest, float v, size_t n) {
    SP_CALL_DEVICE_KERNEL(spCudaFill, (n + 256) / 256, 256, dest, v, n);
    return SP_SUCCESS;
}
int spMemoryFill(double *dest, double v, size_t n) {
    SP_CALL_DEVICE_KERNEL(spCudaFill, (n + 256) / 256, 256, dest, v, n);
    return SP_SUCCESS;
}
int spMemoryFill(int *dest, int v, size_t n) {
    SP_CALL_DEVICE_KERNEL(spCudaFill, (n + 256) / 256, 256, dest, v, n);
    return SP_SUCCESS;
}
