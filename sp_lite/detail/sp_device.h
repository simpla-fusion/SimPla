//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SP_DEVICE_H
#define SIMPLA_SP_DEVICE_H

#include "../spParallel.h"



#ifdef __CUDACC__
#   include "cuda/spParallelCUDA.h"

#else

#   include "openmp/spParallelOPENMP.h"

#endif


#ifdef REAL_IS_FLOAT

typedef float3 Real3;

#else
typedef double3 Real3;
#endif

INLINE __device__ __host__ dim3 sizeType2Dim3(size_type const *v)
{
    dim3 res;
    res.x = (int) v[0];
    res.y = (int) v[1];
    res.z = (int) v[2];
    return res;
}

INLINE __device__  __host__ Real3 real2Real3(Real const *v)
{
    Real3 res;
    res.x = (Real) v[0];
    res.y = (Real) v[1];
    res.z = (Real) v[2];
    return res;
}


#endif //SIMPLA_SP_DEVICE_H
