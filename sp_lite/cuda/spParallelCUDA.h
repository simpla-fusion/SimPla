//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLEL_CU_H
#define SIMPLA_SPPARALLEL_CU_H

#include "../sp_lite_def.h"
#include "../spParallel.h"
#include "../../../../../../usr/local/cuda/include/vector_types.h"
#include "../../../../../../usr/local/cuda/include/cuda_device_runtime_api.h"


#if !defined(__CUDA_ARCH__)
#define CUDA_CALL(_CMD_)                                            \
         printf(  "[line %d in file %s]\n %s = %d \n",                    \
                 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));
#else
#define CUDA_CALL(_CMD_) printf(  "[line %d in file %s : block=[%i,%i,%i] thread=[%i,%i,%i] ]\t %s = %d\n",					\
         __LINE__, __FILE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x , threadIdx.y, threadIdx.z, __STRING(_CMD_),(_CMD_));
#endif

#define SP_CUDA_CALL(_CMD_) {                                            \
    cudaError_t _m_cudaStat = _CMD_;                                        \
    if (_m_cudaStat != cudaSuccess) {                                        \
         printf("Error [code=0x%x] %s at line %d in file %s\n",                    \
                _m_cudaStat,cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);        \
        exit(1);                                                            \
    } }


#define SP_DEVICE_CALL_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...) _FUN_<<<_DIMS_,_N_THREADS_>>>(__VA_ARGS__)

//int SP_DEVICE_CALL_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...) _FUN_<<<_DIMS_,_N_THREADS_>>>(__VA_ARGS__)

#ifdef USE_FLOAT_REAL
typedef float3 Real3;
#else
typedef double3 Real3;
#endif
#define DEVICE_INLINE __inline__ __attribute__((always_inline)) __device__ __host__


DEVICE_INLINE dim3 sizeType2Dim3(size_type const *v)
{
    dim3 res;
    res.x = (int) v[0];
    res.y = (int) v[1];
    res.z = (int) v[2];
    return res;
}

DEVICE_INLINE Real3 real2Real3(Real const *v)
{
    Real3 res;
    res.x = (Real) v[0];
    res.y = (Real) v[1];
    res.z = (Real) v[2];
    return res;
}

#endif //SIMPLA_SPPARALLEL_CU_H
