//
// Created by salmon on 17-6-24.
//
#include <cstring>
#include "../memory.h"
#include "cuda.h"
namespace simpla {

template <typename T>
__global__ void spCUDAFill(T *dest, T const &src, size_t n) {
    size_t s = threadIdx.x * blockDim.x;
    if (s < n) { dest[s] = src; };
}

int spMemoryAlloc(void **p, size_t s, int location) {
    //    *p = SingletonHolder<MemoryPool>::instance().pop(s, 0);

    CHECK(s);

    SP_DEVICE_CALL(cudaMalloc(p, s));

    //    switch (location) {
    //        case MANAGED_MEMORY:
    //            SP_DEVICE_CALL(cudaMallocManaged(p, s));
    //            break;
    //        case DEVICE_MEMORY:
    //            SP_DEVICE_CALL(cudaMalloc(p, s));
    //            break;
    //        case HOST_MEMORY:
    //        default:
    //            *p = malloc(s);
    //    }

    return SP_SUCCESS;
}
int spMemoryFree(void **p, size_t s, int location) {
    //    SingletonHolder<MemoryPool>::instance().push(*p, s, 0);
    SP_DEVICE_CALL(cudaFree(p));

    //    switch (location) {
    //        case MANAGED_MEMORY:
    //        case DEVICE_MEMORY:
    //            SP_DEVICE_CALL(cudaFree(p));
    //            break;
    //        case HOST_MEMORY:
    //        default:
    //            *p = malloc(s);
    //    }
    *p = nullptr;
    return SP_SUCCESS;
}

int spMemoryFill(void *dest, size_t n, void const *src, size_t else_size) {
    //    //    SP_DEVICE_CALL(cudaMemcpy(dest, src, ));
    //
    //    //    cudaDeviceSynchronize();
    //
    //    //#pragma omp parallel for
    //    //    for (size_t i = 0; i < ne; ++i) {
    //    //        reinterpret_cast<char *>(dest)[i] = reinterpret_cast<char const *>(src)[i % else_size];
    //    //    }
    return SP_SUCCESS;
}
int spMemoryCopy(void *dest, void const *src, size_t s) {
//    memcpy(dest, src, s);
    return SP_SUCCESS;
}
int spMemoryFill(float *dest, float v, size_t n) {
    SP_CALL_DEVICE_KERNEL(spCUDAFill, (n + 256) / 256, 256, dest, v, n)
    return SP_SUCCESS;
}

int spMemoryFill(double *dest, double v, size_t n) {
    SP_CALL_DEVICE_KERNEL(spCUDAFill, (n + 256) / 256, 256, dest, v, n)
    return SP_SUCCESS;
}

int spMemoryFill(int *dest, int v, size_t n) {
    SP_CALL_DEVICE_KERNEL(spCUDAFill, (n + 256) / 256, 256, dest, v, n)
    return SP_SUCCESS;
}
}  // namespace simpla