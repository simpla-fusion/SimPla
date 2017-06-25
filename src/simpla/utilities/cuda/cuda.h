//
// Created by salmon on 17-6-24.
//

#ifndef SIMPLA_CUDA_H
#define SIMPLA_CUDA_H
#include </usr/local/cuda/include/cuda_runtime_api.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/driver_types.h>
#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Log.h>
#ifdef CUDA_FOUND
#define SP_DEVICE_CALL(_CMD_)                                                                                        \
    {                                                                                                                \
        auto err_code = (_CMD_);                                                                                     \
        if (err_code != cudaSuccess) {                                                                               \
            RUNTIME_ERROR << "[code=0x" << err_code << ":" << cudaGetErrorString(err_code) << "]" << __STRING(_CMD_) \
                          << std::endl;                                                                              \
        }                                                                                                            \
    }

#define SP_CALL_DEVICE_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...)                                                       \
    _FUN_<<<(_DIMS_), (_N_THREADS_)>>>(__VA_ARGS__);                                                                 \
    {                                                                                                                \
        auto err_code = (cudaPeekAtLastError());                                                                     \
        if (err_code != cudaSuccess) {                                                                               \
            RUNTIME_ERROR << "[code=0x" << err_code << ":" << cudaGetErrorString(err_code) << "] :" << __STRING(_FUN_) \
                          << std::endl;                                                                              \
        }                                                                                                            \
    }                                                                                                                \
    {                                                                                                                \
        auto err_code = (cudaDeviceSynchronize());                                                                   \
        if (err_code != cudaSuccess) {                                                                               \
            RUNTIME_ERROR << "[code=0x" << err_code << ":" << cudaGetErrorString(err_code) << "] :" << __STRING(_FUN_) \
                          << std::endl;                                                                              \
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
