//
// Created by salmon on 16-7-6.
//

#ifndef SIMPLA_SPPARALLEL_H
#define SIMPLA_SPPARALLEL_H

#include "sp_lite_def.h"

#ifndef __CUDACC__
typedef struct
{	int x, y, z;}int3;
typedef struct
{	int x, y, z, w;}int4;
typedef struct
{	float x, y, z;}float3;
typedef struct
{	float x, y, z, w;}float4;
typedef struct
{	size_t x, y, z;}dim3;

typedef struct
{	Real x, y, z;}Real3;

#define MC_HOST_DEVICE
#define MC_HOST
#define MC_DEVICE
#define MC_SHARED
#define MC_CONSTANT static
#define MC_GLOBAL

#define NUMBER_OF_THREADS_PER_BLOCK 1

#define CUDA_CHECK_RETURN(_CMD_) _CMD_

#define CUDA_CHECK(_CMD_)                                            \
         printf(  "[line %d in file %s]\n %s = %d \n",                    \
                 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));

#define MC_FOREACH_BLOCK(THREAD_IDX, BLOCK_DIM, BLOCK_IDX, GRID_DIM)              \
        dim3 THREAD_IDX, BLOCK_DIM, BLOCK_IDX, GRID_DIM;                             \
        size_type BLOCK_ID=0;                                                         \
        for (BLOCK_IDX.x = 0; BLOCK_IDX.x < GRID_DIM.x; ++BLOCK_IDX.x)               \
        for (BLOCK_IDX.y = 0; BLOCK_IDX.y < GRID_DIM.x; ++BLOCK_IDX.y)               \
        for (BLOCK_IDX.z = 0; BLOCK_IDX.z < GRID_DIM.z; ++BLOCK_IDX.z,               \
           BLOCK_ID= (BLOCK_IDX.x + (BLOCK_IDX.y + BLOCK_IDX.z * GRID_DIM.y) * GRID_DIM.x))               \
        for (THREAD_IDX.x = 0; THREAD_IDX.x < BLOCK_DIM.x; ++THREAD_IDX.x)           \
        for (THREAD_IDX.y = 0; THREAD_IDX.y < BLOCK_DIM.x; ++THREAD_IDX.y)           \
        for (THREAD_IDX.z = 0; THREAD_IDX.z < BLOCK_DIM.z; ++THREAD_IDX.z)
#define MC_FOREACH_BLOCK_ID(__BLOCK_ID__) size_type __BLOCK_ID__=0;

//#define spAtomicAdd(_ADDR_, _VAL_)  (*_ADDR_*=_VAL_)

#else  //__CUDACC__

typedef float3 Real3;

#ifndef NUMBER_OF_THREADS_PER_BLOCK
#	define NUMBER_OF_THREADS_PER_BLOCK 128
#endif //NUMBER_OF_THREADS_PER_BLOCK

#define MC_HOST_DEVICE __host__ __device__
#define MC_HOST __host__
#define MC_DEVICE  __device__
#define MC_SHARED __shared__
#define MC_CONSTANT __constant__
#define MC_GLOBAL  __global__

#define CUDA_CHECK_RETURN(_CMD_) {											\
    cudaError_t _m_cudaStat = _CMD_;										\
    if (_m_cudaStat != cudaSuccess) {										\
    	 printf("Error [code=0x%x] %s at line %d in file %s\n",					\
                _m_cudaStat,cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
        exit(1);															\
    } }

#if !defined(__CUDA_ARCH__)
#define CUDA_CHECK(_CMD_)  											\
         printf(  "[line %d in file %s]\n %s = %d \n",					\
                 __LINE__, __FILE__,__STRING(_CMD_),(_CMD_));
#else
#	define CUDA_CHECK(_CMD_) printf(  "[line %d in file %s : block=[%i,%i,%i] thread=[%i,%i,%i] ]\t %s = %x\n",					\
         __LINE__, __FILE__,blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x , threadIdx.y, threadIdx.z, __STRING(_CMD_),(_CMD_));
#endif

#endif //__CUDACC__

MC_HOST void spParallelInitialize();

MC_HOST void spParallelFinalize();

MC_HOST void spParallelDeviceSync();

MC_HOST void spParallelDeviceMalloc(void **, size_type s);

MC_HOST void spParallelDeviceFree(void *);

MC_HOST void spParallelMemcpy(void *dest, void const *src, size_type s);

MC_HOST void spParallelMemcpyToSymbol(void *dest, void const *src, size_type s);

MC_HOST void spParallelMemset(void *dest, byte_type v, size_type s);

MC_HOST_DEVICE int sp_is_device_ptr(void const *p);

MC_HOST_DEVICE int sp_pointer_type(void const *p);

MC_DEVICE unsigned int spParallelThreadNum();

MC_DEVICE unsigned int spParallelNumOfThreads();

MC_DEVICE dim3 spParallelBlockIdx();

MC_DEVICE dim3 spParallelGridDims();

MC_DEVICE unsigned int spParallelNumOfBlocks();

MC_DEVICE unsigned int spParallelBlockNum();

MC_HOST void spParallelMemcpy(void *, void const *, size_type);

#define spAtomicAdd(_ADDR_,_V_) atomicAdd(_ADDR_,_V_)
#define spAtomicSub(_ADDR_,_V_) atomicSub(_ADDR_,_V_)

#define spParallelSyncThreads __syncthreads

#define spParallelThreadIdx()  threadIdx

#define spParallelBlockDim()  blockDim

#define spParallelBlockIdx()  blockIdx

#define spParallelGridDims()  gridDim

#define spParallelBlockNum()  ( blockIdx.x + (blockIdx.y * gridDim.z + blockIdx.z) * gridDim.z)

#define spParallelNumOfBlocks() ( gridDim.x * gridDim.y * gridDim.z)

#define spParallelBlockNumShift(shift)  ((blockIdx.x + shift.x + gridDim.x) % gridDim.x \
		                               + ((blockIdx.y + shift.y + gridDim.y) % gridDim.y) * gridDim.x	\
		                               + ((blockIdx.z + shift.z + gridDim.z) % gridDim.z) * gridDim.y * gridDim.x )

#define spParallelThreadNum()  (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)

#define spParallelNumOfThreads() (  blockDim.x * blockDim.y * blockDim.z)

#endif //SIMPLA_SPPARALLEL_H
