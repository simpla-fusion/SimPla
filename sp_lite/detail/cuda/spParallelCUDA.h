//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLEL_CU_H
#define SIMPLA_SPPARALLEL_CU_H

#include "../../sp_lite_def.h"
#include "../../spParallel.h"
#include </usr/local/cuda/include/driver_types.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>

#ifndef NUMBER_OF_THREADS_PER_BLOCK
#   define  NUMBER_OF_THREADS_PER_BLOCK 256
#endif

#define  SP_DEVICE_GLOBAL __global__
#if !defined(__CUDA_ARCH__)
#define CUDA_CALL(_CMD_)                                            \
         printf(  "[line %d in file %s]\n %s = %d \n",                    \
                 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));
#else
#define CUDA_CALL(_CMD_) printf(  "[line %d in file %s : block=[%i,%i,%i] thread=[%i,%i,%i] ]\t %s = %d\n",					\
         __LINE__, __FILE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x , threadIdx.y, threadIdx.z, __STRING(_CMD_),(_CMD_));
#endif


inline int
print_device_error(cudaError_t _m_cudaStat, char const *file, int line, char const *function, char const *cmd)
{
    if (_m_cudaStat != cudaSuccess)
    {
        printf("%s:%d:0:%s:  [code=0x%x:%s ] %s \n",
               file,
               line,
               function,
               _m_cudaStat,
               cudaGetErrorString(_m_cudaStat),
               cmd);
        return SP_FAILED;

    }
    return SP_SUCCESS;
}


#define SP_DEVICE_CALL(_CMD_) { if(SP_SUCCESS!=print_device_error((_CMD_),__FILE__, __LINE__,__PRETTY_FUNCTION__, __STRING(_CMD_))){return SP_FAILED;}}
//#define SP_DEVICE_CALL(_CMD_) {                                            \
//    cudaError_t _m_cudaStat = _CMD_;                                        \
//    if (_m_cudaStat != cudaSuccess) {                                        \
//        printf( "%s:%d:0:%s: %s [code=0x%x] %s \n", __FILE__, __LINE__,__PRETTY_FUNCTION__, __STRING(_CMD_),_m_cudaStat,cudaGetErrorString(_m_cudaStat));  \
//    } }


#define SP_CALL_DEVICE_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...) _FUN_<<<(_DIMS_),(_N_THREADS_)>>>(__VA_ARGS__);\
                    { if(SP_SUCCESS!=print_device_error(cudaPeekAtLastError(),__FILE__, __LINE__,__PRETTY_FUNCTION__, __STRING(_FUN_))){return SP_FAILED;}}\
                    { if(SP_SUCCESS!=print_device_error(cudaDeviceSynchronize(),__FILE__, __LINE__,__PRETTY_FUNCTION__, __STRING(_FUN_))){return SP_FAILED;}}
#define SP_DEVICE_CALL_KERNEL2(_FUN_, _DIMS_, _N_THREADS_, _SMEM_, ...) _FUN_<<<(_DIMS_),(_N_THREADS_),(_SMEM_)>>>(__VA_ARGS__);SP_DEVICE_CALL(cudaPeekAtLastError());SP_DEVICE_CALL(cudaDeviceSynchronize());

#define SP_DEVICE_DECLARE_KERNEL(_FUN_, ...) __global__ void _FUN_( __VA_ARGS__)


#define INLINE __inline__ __attribute__((always_inline))
#define __register__

#define spParallelMemcpyToSymbol(_dest_, _src_, _s_)      cudaMemcpyToSymbol(_dest_, _src_, _s_);
#define spParallelSyncThreads() __syncthreads()

INLINE __device__ int atomicAddInt(int *ptr, int val) { return atomicAdd(ptr, val); }

INLINE __device__ Real atomicAddReal(Real *ptr, float val) { return atomicAdd(ptr, val); }

#endif //SIMPLA_SPPARALLEL_CU_H
