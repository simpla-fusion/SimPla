//
// Created by salmon on 16-7-20.
//

#ifndef SIMPLA_SPPARALLEL_CU_H
#define SIMPLA_SPPARALLEL_CU_H


#include "sp_lite_def.h"
#include </usr/local/cuda/include/cuda_runtime.h>

#define Real3 float3

#ifndef NUMBER_OF_THREADS_PER_BLOCK
#	define NUMBER_OF_THREADS_PER_BLOCK 128
#endif //NUMBER_OF_THREADS_PER_BLOCK


//#define spAtomicAdd(_ADDR_, _V_) atomicAdd(_ADDR_,_V_)
//
//#define spAtomicSub(_ADDR_, _V_) atomicSub(_ADDR_,_V_)
//
//#define spParallelSyncThreads __syncthreads
//
//#define spParallelThreadIdx()  threadIdx
//
//#define spParallelBlockDim()  blockDim
//
//#define spParallelBlockIdx()  blockIdx
//
//#define spParallelGridDims()  gridDim

#define spParallelBlockNum()  ( blockIdx.x + (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x)

#define spParallelNumOfBlocks() ( gridDim.x * gridDim.y * gridDim.z)

#define spParallelBlockNumShift(shift)  ((blockIdx.x + shift.x + gridDim.x) % gridDim.x \
                                       + ((blockIdx.y + shift.y + gridDim.y) % gridDim.y) * gridDim.x    \
                                       + ((blockIdx.z + shift.z + gridDim.z) % gridDim.z) * gridDim.y * gridDim.x )

#define spParallelThreadNum()  (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)

#define spParallelNumOfThreads() (  blockDim.x * blockDim.y * blockDim.z)


#if !defined(__CUDA_ARCH__)
#define CUDA_CHECK(_CMD_)                                            \
         printf(  "[line %d in file %s]\n %s = %d \n",                    \
                 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));
#else
#	define CUDA_CHECK(_CMD_) printf(  "[line %d in file %s : block=[%i,%i,%i] thread=[%i,%i,%i] ]\t %s = %d\n",					\
         __LINE__, __FILE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x , threadIdx.y, threadIdx.z, __STRING(_CMD_),(_CMD_));
#endif

#define LOAD_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...) _FUN_<<<_DIMS_,_N_THREADS_>>>(__VA_ARGS__)

extern inline dim3 sizeType2Dim3(size_type const *v)
{
    dim3 res;
    res.x = v[0];
    res.y = v[1];
    res.z = v[2];
    return res;
}
#endif //SIMPLA_SPPARALLEL_CU_H
