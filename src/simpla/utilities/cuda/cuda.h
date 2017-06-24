//
// Created by salmon on 17-6-24.
//

#ifndef SIMPLA_CUDA_H
#define SIMPLA_CUDA_H
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/driver_types.h>
#include <simpla/SIMPLA_config.h>
#include <stdio.h>
#define SP_DEVICE_GLOBAL __global__
#if !defined(__CUDA_ARCH__)
#define CUDA_CALL(_CMD_) printf("[line %d in file %s]\n %s = %d \n", __LINE__, __FILE__, __STRING(_CMD_), (_CMD_));
#else
#define CUDA_CALL(_CMD_)                                                                                             \
    printf("[line %d in file %s : block=[%i,%i,%i] thread=[%i,%i,%i] ]\t %s = %d\n", __LINE__, __FILE__, blockIdx.x, \
           blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, __STRING(_CMD_), (_CMD_));
#endif

inline int print_device_error(cudaError_t _m_cudaStat, char const *file, int line, char const *function,
                              char const *cmd) {
    if (_m_cudaStat != cudaSuccess) {
        printf("%s:%d:0:%s:  [code=0x%x:%s ] %s \n", file, line, function, _m_cudaStat, cudaGetErrorString(_m_cudaStat),
               cmd);
        return SP_FAILED;
    }
    return SP_SUCCESS;
}
#ifdef CUDA_FOUND
#define SP_DEVICE_CALL(_CMD_)                                                                                       \
    {                                                                                                               \
        int err_code = (_CMD_);                                                                                     \
        if (SP_SUCCESS != print_device_error(err_code, __FILE__, __LINE__, __PRETTY_FUNCTION__, __STRING(_CMD_))) { \
            return SP_FAILED;                                                                                       \
        }                                                                                                           \
    }

#define SP_CALL_DEVICE_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...)                                                       \
    _FUN_<<<(_DIMS_), (_N_THREADS_)>>>(__VA_ARGS__);                                                                 \
    {                                                                                                                \
        if (SP_SUCCESS !=                                                                                            \
            print_device_error(cudaPeekAtLastError(), __FILE__, __LINE__, __PRETTY_FUNCTION__, __STRING(_FUN_))) {   \
            return SP_FAILED;                                                                                        \
        }                                                                                                            \
    }                                                                                                                \
    {                                                                                                                \
        if (SP_SUCCESS !=                                                                                            \
            print_device_error(cudaDeviceSynchronize(), __FILE__, __LINE__, __PRETTY_FUNCTION__, __STRING(_FUN_))) { \
            return SP_FAILED;                                                                                        \
        }                                                                                                            \
    }
#define SP_DEVICE_CALL_KERNEL2(_FUN_, _DIMS_, _N_THREADS_, _SMEM_, ...) \
    _FUN_<<<(_DIMS_), (_N_THREADS_), (_SMEM_)>>>(__VA_ARGS__);          \
    SP_DEVICE_CALL(cudaPeekAtLastError());                              \
    SP_DEVICE_CALL(cudaDeviceSynchronize());

#define SP_DEVICE_DECLARE_KERNEL(_FUN_, ...) __global__ void _FUN_(__VA_ARGS__)
#else
#define SP_DEVICE_CALL(_CMD_)
#endif
#endif  // SIMPLA_CUDA_H
