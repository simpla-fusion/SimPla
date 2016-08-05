//
// Created by salmon on 16-7-27.
//

extern "C" {
#include "../spMisc.h"

#include <math.h>

#include "spParallelCUDA.h"
}
__global__
void spFieldAssignValueSinKernel_g(Real *data, dim3 strides, Real3 k_dx, Real3 alpha0, Real amp)
{
    size_type x = threadIdx.x + blockIdx.x * blockDim.x;
    size_type y = threadIdx.y + blockIdx.y * blockDim.y;
    size_type z = threadIdx.z + blockIdx.z * blockDim.z;

    size_type s = x * strides.x + y * strides.y + z * strides.z;

    data[s] = amp *
        cos(k_dx.x * Real(x) + alpha0.x) *
        cos(k_dx.y * Real(y) + alpha0.y) *
        cos(k_dx.z * Real(z) + alpha0.z);

}
void spFieldAssignValueSinKernel(size_type const *block,
                                 size_type const *thread,
                                 Real *data,
                                 size_type const *strides,
                                 Real const *k_dx,
                                 Real const *alpha0,
                                 Real amp)
{
    SP_DEVICE_CALL_KERNEL(spFieldAssignValueSinKernel_g, sizeType2Dim3(block),
                sizeType2Dim3(thread),
                data,
                sizeType2Dim3(strides),
                real2Real3(k_dx),
                real2Real3(alpha0),
                amp
    );
};
