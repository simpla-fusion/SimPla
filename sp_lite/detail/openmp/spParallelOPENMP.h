//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLELOPENMP_H
#define SIMPLA_SPPARALLELOPENMP_H

#include <stdlib.h>
#include <string.h>
#include "../../sp_lite_config.h"


typedef struct { float x, y, z; } float3;

typedef struct { double x, y, z; } double3;

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

#define SP_CHECK_DEVICE_ERROR

#define SP_DEVICE_DECLARE_KERNEL(_FUN_, ...)   void _FUN_ (dim3 gridDim,dim3 blockDim,dim3 blockIdx, dim3 threadIdx,__VA_ARGS__)

#define SP_CALL_DEVICE_KERNEL(_FUN_, gridDim, blockDim, ...)         \
{                                                                    \
   _Pragma("omp parallel for")                                       \
  for (int x = 0; x < gridDim.x; ++x)                                \
  { dim3 blockIdx,threadIdx;    blockIdx.x=x;                        \
    for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)       \
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z)       \
    for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)   \
    for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y)   \
    for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z)   \
    {                                                                \
       _FUN_(gridDim, blockDim, blockIdx, threadIdx, __VA_ARGS__);   \
    }                                                                \
  }                                                                  \
}

#define SP_DEVICE_CALL(_CMD_)  SP_CALL(_CMD_)

#define spParallelMemcpyToSymbol(_dest_, _src_, _s_)     spParallelMemcpyToCache(&_dest_, _src_, _s_);
#define spParallelSyncThreads()    _Pragma("omp barrier")


INLINE unsigned int __umul24(unsigned int a, unsigned int b) { return a * b; }

INLINE int __mul24(int a, int b) { return a * b; }
INLINE int atomicAdd(int *ptr, int val)
{
    return __sync_fetch_and_add(ptr,val);}
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

INLINE Real atomicAddReal(Real *ptr, Real val)
{
    Real t;
#pragma omp atomic capture
    {
        t = *ptr;
        *ptr += val;
    }
    return t;
}

INLINE int atomicCAS(int *address, int compare, int val)
{
    int t;
//#pragma omp atomic capture
//    {
//        if (*address == compare) { *address = val; }
//        t = *address;
//    }
    return t;
}

INLINE int atomicExch(int *address, int val)
{
    int t;
#pragma omp atomic capture
    {
        t = *address;
        *address = val;
    }
    return t;
}

#endif //SIMPLA_SPPARALLELOPENMP_H
