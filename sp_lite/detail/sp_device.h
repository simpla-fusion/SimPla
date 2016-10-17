//
// Created by salmon on 16-8-4.
//

#ifndef SIMPLA_SP_DEVICE_H
#define SIMPLA_SP_DEVICE_H

#include "../spParallel.h"


#ifdef __OMP__
#   include "openmp/spParallelOPENMP.h"
#else

#   include "cuda/spParallelCUDA.h"

#endif


#ifdef REAL_IS_FLOAT

typedef float3 Real3;

#else
typedef double3 Real3;
#endif

INLINE __device__ __host__ dim3 sizeType2Dim3(size_type const *v)
{
    dim3 res;
    res.x = (unsigned int) v[0];
    res.y = (unsigned int) v[1];
    res.z = (unsigned int) v[2];
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


INLINE int _show_dev_data_int(size_type const *d, size_type num)
{
    size_type *buffer;
    SP_CALL(spMemoryHostAlloc((void **) &buffer, num * sizeof(size_type)));
    SP_CALL(spMemoryCopy(buffer, d, num * sizeof(size_type)));


    for (int i = 0; i < (num); ++i)
    {
        if ((i) % 12 == 0)printf("\n %4d: ", i);

        printf("\t %ld", buffer[i]);
    }
    printf("\n");
    SP_CALL(spMemoryHostFree((void **) &buffer));
    return SP_SUCCESS;

}

INLINE int _show_dev_data_real(Real const *d, size_type num)
{
    Real *buffer;
    SP_CALL(spMemoryHostAlloc((void **) &buffer, num * sizeof(Real)));
    SP_CALL(spMemoryCopy(buffer, d, num * sizeof(Real)));


    for (int i = 0; i < (num); ++i)
    {
        if ((i) % 10 == 0)printf("\n %4d: ", i);

        printf("\t %f", buffer[i]);
    }
    printf("\n");
    SP_CALL(spMemoryHostFree((void **) &buffer));

    return SP_SUCCESS;
}

#endif //SIMPLA_SP_DEVICE_H
