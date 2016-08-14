//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SPPARALLELDEVICE_H
#define SIMPLA_SPPARALLELDEVICE_H

#include "../spParallel.h"

#ifdef __CUDACC__
#   include "cuda/spParallelCUDA.h"
#else

#   include "cpu/spParallelCPU.h"

#endif


INLINE DEVICE HOST dim3 sizeType2Dim3(size_type const *v)
{
    dim3 res;
    res.x = (int) v[0];
    res.y = (int) v[1];
    res.z = (int) v[2];
    return res;
}

INLINE DEVICE  HOST Real3 real2Real3(Real const *v)
{
    Real3 res;
    res.x = (Real) v[0];
    res.y = (Real) v[1];
    res.z = (Real) v[2];
    return res;
}


#endif //SIMPLA_SPPARALLELDEVICE_H
