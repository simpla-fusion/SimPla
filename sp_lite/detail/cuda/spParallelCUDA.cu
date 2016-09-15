//
// Created by salmon on 16-7-25.
//

#include </usr/local/cuda/include/cuda_runtime_api.h>
#include "../../../../../../../usr/local/cuda/include/device_launch_parameters.h"

extern "C" {
#include "spParallelCUDA.h"
#include "../../spMPI.h"
}


int spParallelDeviceInitialize(int argc, char **argv)
{
    int error_code = SP_SUCCESS;
    int num_of_device = 0;
    SP_DEVICE_CALL(cudaGetDeviceCount(&num_of_device));
    SP_DEVICE_CALL(cudaSetDevice(spMPIRank() % num_of_device));
    SP_DEVICE_CALL(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
    SP_DEVICE_CALL(cudaGetLastError());
//    SP_DEVICE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    return error_code;
}

int spParallelDeviceFinalize()
{
    int error_code = SP_SUCCESS;

    SP_DEVICE_CALL(cudaDeviceReset());
    return error_code;

}

#define SP_DEFAULT_BLOCKS  128
#define SP_DEFAULT_THREADS 128

int spParallelGridDim()
{
    return SP_DEFAULT_THREADS;
};

int spParallelBlockDim()
{
    return SP_DEFAULT_BLOCKS;
};

int spParallelDeviceAlloc(void **p, size_type s)
{
    int error_code = SP_SUCCESS;

    SP_DEVICE_CALL(cudaMalloc(p, s));
    return error_code;
}

int spParallelDeviceFree(void **_P_)
{
    int error_code = SP_SUCCESS;
    if (*_P_ != NULL)
    {
        error_code = SP_DEVICE_CALL(cudaFree(*_P_));
        *_P_ = NULL;
    }
    return error_code;
};

int spParallelMemcpy(void *dest, void const *src, size_type s)
{
    int error_code = SP_SUCCESS;
    SP_DEVICE_CALL(cudaMemcpy(dest, src, s, cudaMemcpyDefault));
    return error_code;
}

int spParallelMemcpyToCache(const void *dest, void const *src, size_type s)
{

    int error_code = SP_SUCCESS;
    SP_DEVICE_CALL(cudaMemcpyToSymbol(dest, src, s));
    return error_code;


}

int spParallelMemset(void *dest, int v, size_type s)
{

    int error_code = SP_SUCCESS;
    SP_DEVICE_CALL (cudaMemset(dest, v, s));
    return error_code;

}

int spParallelDeviceSync()
{

    int error_code = SP_SUCCESS;
    SP_CALL(spParallelGlobalBarrier());
    SP_DEVICE_CALL (cudaDeviceSynchronize());
    return error_code;

}

int spParallelHostAlloc(void **p, size_type s)
{
    int error_code = SP_SUCCESS;
    SP_DEVICE_CALL (cudaHostAlloc(p, s, cudaHostAllocDefault));
    return error_code;

};

int spParallelHostFree(void **p)
{
    int error_code = SP_SUCCESS;
    if (*p != NULL)
    {
        error_code = SP_DEVICE_CALL(cudaFreeHost(*p));
        *p = NULL;
    }
    return error_code;
}


__global__
void spParallelDeviceFillIntKernel(int *d, int v, size_type max)
{
    for (size_t s = threadIdx.x + blockIdx.x * blockDim.x; s < max; s += gridDim.x * blockDim.x) { d[s] = v; }
};

int spParallelDeviceFillInt(int *d, int v, size_type s)
{
    int error_code = SP_SUCCESS;

    SP_DEVICE_CALL_KERNEL(spParallelDeviceFillIntKernel, 16, 256, d, v, s);

    return error_code;
};

__global__
void spParallelDeviceFillRealKernel(Real *d, Real v, size_type max)
{
    for (size_type s = threadIdx.x + blockIdx.x * blockDim.x; s < max; s += gridDim.x * blockDim.x) { d[s] = v; }
};

int spParallelDeviceFillReal(Real *d, Real v, size_type s)
{
    int error_code = SP_SUCCESS;
    SP_DEVICE_CALL_KERNEL(spParallelDeviceFillRealKernel, 16, 256, d, v, s);
    return error_code;
};


__global__
void spParallelAssignKernel(size_type max, size_type const *offset, Real *d, Real const *v)
{

    size_type num_of_thread = blockDim.x * gridDim.x * blockDim.x * gridDim.x * blockDim.x * gridDim.x;

    for (size_type s = (threadIdx.x + blockIdx.x * blockDim.x) +
                       (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x +
                       (threadIdx.x + blockIdx.x * blockDim.x) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
         s < max; s += num_of_thread) { d[offset[s]] = v[s]; }
};

int spParallelAssign(size_type num_of_point, size_type *offset, Real *d, Real const *v)
{
    int error_code = SP_SUCCESS;
    SP_DEVICE_CALL_KERNEL(spParallelAssignKernel, 16, 256, num_of_point, offset, d, v);
    return error_code;
};

int spMemoryDeviceToHost(void **p, void *src, size_type size_in_byte)
{
    int error_code = SP_SUCCESS;
    SP_CALL(spParallelHostAlloc(p, size_in_byte));
    SP_CALL(spParallelMemcpy(*p, src, size_in_byte));
    return error_code;

}

int spMemoryHostFree(void **p)
{
    int error_code = SP_SUCCESS;
    SP_CALL(spParallelHostFree(p));
    return error_code;

}