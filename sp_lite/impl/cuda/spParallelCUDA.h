//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLEL_CU_H
#define SIMPLA_SPPARALLEL_CU_H

#include "../../sp_lite_def.h"
#include "../../spParallel.h"

#define SP_DEFAULT_BLOCKS  128
#define SP_DEFAULT_THREADS 128
#define  SP_DEVICE_GLOBAL __global__
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

#define SP_DEVICE_CALL_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...) _FUN_<<<(_DIMS_),(_N_THREADS_)>>>(__VA_ARGS__)

#define SP_DEVICE_DECLARE_KERNEL(_FUN_, ...) __global__ void _FUN_( __VA_ARGS__)

//int SP_DEVICE_CALL_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...) _FUN_<<<_DIMS_,_N_THREADS_>>>(__VA_ARGS__)

#ifdef USE_FLOAT_REAL
typedef float3 Real3;
#else
typedef double3 Real3;
#endif
#define INLINE __inline__ __attribute__((always_inline))
#define DEVICE __device__
#define HOST __host__


#endif //SIMPLA_SPPARALLEL_CU_H
