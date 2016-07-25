//
// Created by salmon on 16-7-25.
//

extern "C" {
#include "spParallel.h"
#include "spParallelCUDA.h"
}

__global__
void spParallelDeviceFillIntKernel(int *d, int v, size_type max)
{
    for (size_t s = threadIdx.x + blockIdx.x * blockDim.x; s < max; s += gridDim.x * blockDim.x) { d[s] = v; }
};
int spParallelDeviceFillInt(int *d, int v, size_type s)
{
    LOAD_KERNEL(spParallelDeviceFillIntKernel, 16, 256, d, v, s);

    return SP_SUCCESS;
};

__global__
void spParallelDeviceFillRealKernel(Real *d, Real v, size_type max)
{
    for (size_type s = threadIdx.x + blockIdx.x * blockDim.x; s < max; s += gridDim.x * blockDim.x) { d[s] = v; }
};
int spParallelDeviceFillReal(Real *d, Real v, size_type s)
{


    LOAD_KERNEL(spParallelDeviceFillRealKernel, 16, 256, d, v, s);

    return SP_SUCCESS;
};