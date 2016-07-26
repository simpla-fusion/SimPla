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


__global__
void spParallelAssignKernel(int num_of_sub, dim3 stride, size_type max, size_type *p[3], Real **d, Real const **v)
{

    size_type num_of_thread = blockDim.x * gridDim.x * blockDim.x * gridDim.x * blockDim.x * gridDim.x;

    for (size_type s = (threadIdx.x + blockIdx.x * blockDim.x) +
        (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x * gridDim.x +
        (threadIdx.x + blockIdx.x * blockDim.x) * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
         s < max; s += num_of_thread)
    {
        for (int i = 0; i < num_of_sub; ++i)
        {
            d[i][p[0][s] * stride.x + p[1][s] * stride.x + p[2][s] * stride.x] = v[i][s];
        }
    }
};

int spParallelAssign(int num_of_sub,
                     size_type num_of_point,
                     size_type *points[3],
                     size_type const *strides,
                     Real **d,
                     Real const**v)
{
    LOAD_KERNEL(spParallelAssignKernel, 16, 256, num_of_sub, sizeType2Dim3(strides), num_of_point, points, d, v);
    return SP_SUCCESS;
};