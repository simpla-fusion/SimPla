//
// Created by salmon on 16-7-6.
//
#include "sp_lite_def.h"
#include "spParallel.h"

void spParallelInitialize()
{
    CUDA_CHECK_RETURN(cudaThreadSynchronize()); // Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN(cudaGetLastError());
}

void spParallelFinalize()
{
    CUDA_CHECK_RETURN(cudaDeviceReset());

}

void spParallelGlobalSync()
{
    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
}


void spParallelDeviceMalloc(void **p, size_type s)
{
    CUDA_CHECK_RETURN(cudaMalloc(p, s));

}

void spParallelDeviceFree(void *p)
{
    CUDA_CHECK_RETURN(cudaFree(p));
}

void spParallelMemcpy(void *dest, void const *src, size_type s)
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