//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLEL_CU_H
#define SIMPLA_SPPARALLEL_CU_H


#include "sp_lite_def.h"
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>


#ifdef USE_FLOAT_REAL
typedef float3 Real3;
#else

typedef double3 Real3;

#endif
//#define spParallelBlockNum()  ( blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x)
//
//#define spParallelNumOfBlocks() ( gridDim.x * gridDim.y * gridDim.z)
//
//#define spParallelBlockNumShift(shift)  ((blockIdx.x + shift.x + gridDim.x) % gridDim.x \
//                                       + ((blockIdx.y + shift.y + gridDim.y) % gridDim.y) * gridDim.x    \
//                                       + ((blockIdx.z + shift.z + gridDim.z) % gridDim.z) * gridDim.y * gridDim.x )
//
//#define spParallelThreadNum()  (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)
//
//#define spParallelNumOfThreads() (  blockDim.x * blockDim.y * blockDim.z)
//

#if !defined(__CUDA_ARCH__)
#define CUDA_CHECK(_CMD_)                                            \
         printf(  "[line %d in file %s]\n %s = %d \n",                    \
                 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));
#else
#	define CUDA_CHECK(_CMD_) printf(  "[line %d in file %s : block=[%i,%i,%i] thread=[%i,%i,%i] ]\t %s = %d\n",					\
         __LINE__, __FILE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x , threadIdx.y, threadIdx.z, __STRING(_CMD_),(_CMD_));
#endif

#define SP_PARALLEL_CHECK_RETURN(_CMD_) {                                            \
    cudaError_t _m_cudaStat = _CMD_;                                        \
    if (_m_cudaStat != cudaSuccess) {                                        \
         printf("Error [code=0x%x] %s at line %d in file %s\n",                    \
                _m_cudaStat,cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);        \
        exit(1);                                                            \
    } }
#define spParallelDeviceAlloc(_P_, _S_)      SP_PARALLEL_CHECK_RETURN(cudaMalloc(_P_, _S_));

#define spParallelDeviceFree(_P_)      {if (*_P_ != NULL) { SP_PARALLEL_CHECK_RETURN(cudaFree(*_P_)); *_P_ = NULL;   }};

#define spParallelMemcpy(_D_, _S_, _N_) SP_PARALLEL_CHECK_RETURN(cudaMemcpy(_D_, _S_,(_N_), cudaMemcpyDefault));

#define  spParallelMemcpyToSymbol(_D_, _S_, _N_)    SP_PARALLEL_CHECK_RETURN(cudaMemcpyToSymbol(_D_, _S_, _N_));

#define spParallelMemset(_D_, _V_, _N_)  SP_PARALLEL_CHECK_RETURN(cudaMemset(_D_, _V_, _N_));

#define spParallelDeviceSync()   {SP_CHECK_RETURN(spParallelGlobalBarrier()); SP_PARALLEL_CHECK_RETURN(cudaDeviceSynchronize())}

#define spParallelHostAlloc(_P_, _S_)    SP_PARALLEL_CHECK_RETURN(cudaHostAlloc(_P_, _S_, cudaHostAllocDefault))

#define spParallelHostFree(_P_)  if (*_P_ != NULL) { cudaFreeHost(*_P_);  *_P_ = NULL; }


#define LOAD_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...) _FUN_<<<_DIMS_,_N_THREADS_>>>(__VA_ARGS__)


dim3 sizeType2Dim3(size_type const *v);
Real3 real2Real3(Real const *v);

int spParallelAssign(size_type num_of_point, size_type *offset, Real *d, Real const *v);


#endif //SIMPLA_SPPARALLEL_CU_H
