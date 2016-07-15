//
// Created by salmon on 16-7-6.
//
#include "sp_lite_def.h"
#include "spParallel.h"

#include <mpi.h>

// CUDA runtime
#include </usr/local/cuda/include/cuda_runtime.h>

void spParallelInitialize(int argc, char **argv)
{

    spMPIInitialize(argc, argv);

    int num_of_device = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&num_of_device));
    CUDA_CHECK_RETURN(cudaSetDevice(spMPIProcessNum() % num_of_device));
    CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN(cudaGetLastError());
}

void spParallelFinalize()
{
    CUDA_CHECK_RETURN(cudaDeviceReset());
    spMPIFinialize();

}

void spParallelDeviceSync()
{
    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
}

void spParallelHostMalloc(void **p, size_type s)
{
    CUDA_CHECK_RETURN(cudaHostAlloc(p, s, cudaHostAllocDefault););

}

void spParallelHostFree(void **p)
{
    if (*p != NULL)
    {
        cudaFreeHost(*p);
        *p = NULL;
    }
}

MC_HOST void spParallelDeviceMalloc(void **p, size_type s)
{
    CUDA_CHECK_RETURN(cudaMalloc(p, s));
}

MC_HOST void spParallelDeviceFree(void **p)
{
    if (*p != NULL)
    {
        CUDA_CHECK_RETURN(cudaFree(*p));
        *p = NULL;
    }
}

MC_HOST void spParallelMemcpy(void *dest, void const *src, size_type s)
{
    CUDA_CHECK_RETURN(cudaMemcpy(dest, src, s, cudaMemcpyDefault));
}

void spParallelMemcpyToSymbol(void *dest, void const *src, size_type s)
{
    CUDA_CHECK_RETURN(cudaMemcpyToSymbol(dest, src, s));

}

void spParallelMemset(void *dest, byte_type v, size_type s)
{
    CUDA_CHECK_RETURN(cudaMemset(dest, v, s));
}

MC_HOST_DEVICE inline int sp_is_device_ptr(void const *p)
{
    cudaPointerAttributes attribute;
    CUDA_CHECK(cudaPointerGetAttributes(&attribute, p));
    return (attribute.device == cudaMemoryTypeDevice);

}

MC_HOST_DEVICE inline int sp_pointer_type(void const *p)
{
    cudaPointerAttributes attribute;
    CUDA_CHECK(cudaPointerGetAttributes(&attribute, p));
    return (attribute.device);

}

//MC_DEVICE float SP_ATOMIC_ADD(float *v, float d) { return atomicAdd(v, d); }

