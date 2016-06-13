//
// Created by salmon on 16-6-13.
//

#ifndef SIMPLA_SP_CUDA_CONFIG_H_H
#define SIMPLA_SP_CUDA_CONFIG_H_H

#ifdef __CUDACC__
#   define CUDA_DEVICE __device__
#   define CUDA_GLOBAL __global__
#   define CUDA_HOST   __host__
#   define CUDA_SHARED  __shared__
#   define INLINE_PREFIX __device__ __forceinline__
#else
#   define CUDA_DEVICE
#   define CUDA_GLOBAL
#   define CUDA_HOST
#   define CUDA_SHARED
#endif
#endif
#endif //SIMPLA_SP_CUDA_CONFIG_H_H
