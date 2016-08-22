//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLELOPENMP_H
#define SIMPLA_SPPARALLELOPENMP_H

#include <stdlib.h>
#include <string.h>
#include "../../sp_lite_def.h"

typedef struct { float x, y, z; } float3;

typedef struct { int x, y, z; } int3;

typedef struct { unsigned int x, y, z; } uint3;

typedef uint3 dim3;


#define SP_NUM_OF_THREADS_PER_BLOCK 1

#define INLINE  static inline
#define __device__
#define __host__
#define __constant__
#define __shared__

#define __register__

#define SP_DEVICE_DECLARE_KERNEL(_FUN_, ...)   void _FUN_ (dim3 gridDim,dim3 blockDim,dim3 blockIdx, dim3 threadIdx,__VA_ARGS__)

#define SP_DEVICE_CALL_KERNEL(_FUN_, gridDim, blockDim, ...)         \
{                                                                    \
  dim3   blockIdx;                                                   \
  _Pragma("omp parallel for")                                        \
  for (int x = 0; x < gridDim.x; ++x)                                \
  { blockIdx.x=x;                                                    \
    for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)       \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z)       \
    for (int j = 0; j < blockDim.y; ++j)                             \
    for (int k = 0; k < blockDim.z; ++k)                             \
    for (int i = 0; i < blockDim.x; ++i)                             \
    {  dim3 threadIdx = {i,j,k};                                     \
       _FUN_(gridDim, blockDim, blockIdx, threadIdx, __VA_ARGS__);   \
    }                                                                \
  }                                                                  \
}

#define SP_DEVICE_CALL(_CMD_)  SP_CALL(_CMD_)


INLINE unsigned int __umul24(unsigned int a, unsigned int b) { return a * b; }
INLINE int __mul24(int a, int b) { return a * b; }

INLINE int atomicAddInt(int *ptr, int val)
{
    int t;
#pragma omp atomic capture
    {
        t = *ptr;
        *ptr += val;
    }
    return t;
}
INLINE Real atomicAddReal(Real *ptr, float val)
{
    Real t;
#pragma omp atomic capture
    {
        t = *ptr;
        *ptr += val;
    }
    return t;
}

#endif //SIMPLA_SPPARALLELOPENMP_H
