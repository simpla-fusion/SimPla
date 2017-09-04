//
// Created by salmon on 17-6-27.
//

#ifndef SAMRAI_DEVICE_COMMON_H
#define SAMRAI_DEVICE_COMMON_H

#include <cstddef>
#include <iostream>
#include <memory>
#include "Log.h"
#ifdef __CUDA__

#include </usr/local/cuda/include/cuda_runtime_api.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include </usr/local/cuda/include/driver_types.h>
//#define FILE_LINE_STAMP                                                                                        \
//    "\n\e[0m \e[1;37m From [" << (__FILE__) << ":" << (__LINE__) << ":0: " << (__PRETTY_FUNCTION__) << "] \n " \
//                                                                                                       "\e[1;31m\t"
#define SP_DEVICE_CALL(_CMD_)                                                                                        \
    {                                                                                                                \
        auto err_code = (_CMD_);                                                                                     \
        if (err_code != cudaSuccess) {                                                                               \
            RUNTIME_ERROR << "[code=0x" << err_code << ":" << cudaGetErrorString(err_code) << "]" << __STRING(_CMD_) \
                          << std::endl;                                                                              \
        }                                                                                                            \
    }

#define SP_CALL_DEVICE_KERNEL(_FUN_, _DIMS_, _N_THREADS_, ...)                                              \
                                                                                                            \
    _FUN_<<<(_DIMS_), (_N_THREADS_)>>>(__VA_ARGS__);                                                        \
    {                                                                                                       \
        auto err_code = (cudaGetLastError());                                                               \
        if (err_code != cudaSuccess) {                                                                      \
            RUNTIME_ERROR << FILE_LINE_STAMP << "CUDA Error:[code=0x" << err_code                           \
                          << "] : " << cudaGetErrorString(err_code) << ". ( Fun:" << __STRING(_FUN_) << ")" \
                          << std::endl;                                                                     \
        }                                                                                                   \
    }                                                                                                       \
    {                                                                                                       \
        auto err_code = (cudaDeviceSynchronize());                                                          \
        if (err_code != cudaSuccess) {                                                                      \
            RUNTIME_ERROR << FILE_LINE_STAMP << "CUDA Error:[code=0x" << err_code                           \
                          << "] : " << cudaGetErrorString(err_code) << ". ( Fun:" << __STRING(_FUN_) << ")" \
                          << std::endl;                                                                     \
        }                                                                                                   \
    }

inline void *_malloc(size_t s) {
    void *addr = nullptr;
    SP_DEVICE_CALL(cudaMallocManaged(&addr, s));
    return addr;
}
inline void _free(void *addr) {
    if (addr != nullptr) { SP_DEVICE_CALL(cudaFree(addr)); }
}

namespace detail {
    template <typename T>
    struct deleter_device_ptr_s {
        inline void operator()(T *ptr) {
            cudaPointerAttributes attributes;

            auto errcode = (cudaPointerGetAttributes(&attributes, ptr));

            if (errcode == cudaErrorInvalidValue) {
                cudaGetErrorString(errcode);
                free(ptr);
            } else {
                SP_DEVICE_CALL(cudaFree(ptr));
            }
        }
    };
}

template <typename T, int N = 3>  //
class ManagedArray {
public:
    typedef T value_type;
    ManagedArray(T *d, int const *min, int const *max);
    ManagedArray(T *d, int3 min, int3 max);
    ManagedArray(T *d, int x_min, int x_max, int y_min, int y_max, int z_min = 0, int z_max = 1);

    ManagedArray(ManagedArray<T, N> &&);
    ManagedArray(ManagedArray<T, N> const &);
    ~ManagedArray();
    template <typename... Args>
    __host__ __device__ value_type &operator()(Args &&... args) {
        return m_data_[hash(std::forward<Args>(args)...)];
    }
    template <typename... Args>
    __host__ __device__ value_type const &operator()(Args &&... args) const {
        return m_data_[hash(std::forward<Args>(args)...)];
    }

    __host__ __device__ int hash(int x, int y = 0, int z = 0, int n = 0) const;
    __host__ __device__ bool in_box(int x, int y = 0, int z = 0) const;
    __host__ __device__ size_t size() const { return static_cast<size_t>(m_offset_); };

private:
    std::remove_cv_t<T> *m_data_;
    std::remove_cv_t<T> *m_host_data_ = nullptr;
    std::shared_ptr<std::remove_cv_t<T>> m_holder_ = nullptr;
    int3 m_min_, m_max_;
    int m_offset_;
};

template <typename T, int N>
ManagedArray<T, N>::ManagedArray(T *d, int x_min, int x_max, int y_min, int y_max, int z_min, int z_max)
        : m_data_(const_cast<std::remove_cv_t<T> *>(d)),
          m_min_{x_min, y_min, z_min},
          m_max_{x_max, y_max, z_max},
          m_offset_((m_max_.z - m_min_.z) * (m_max_.y - m_min_.y) * (m_max_.x - m_min_.x)) {
    struct cudaPointerAttributes attributes;

    auto errcode = (cudaPointerGetAttributes(&attributes, m_data_));

    if (errcode == cudaErrorInvalidValue) {
        cudaGetLastError();
        m_host_data_ = m_data_;
        m_holder_.reset(reinterpret_cast<std::remove_cv_t<T> *>(_malloc(size() * sizeof(T))),
                        detail::deleter_device_ptr_s<std::remove_cv_t<T>>());
        m_data_ = m_holder_.get();
        SP_DEVICE_CALL(cudaMemcpy(reinterpret_cast<void *>(m_holder_.get()), reinterpret_cast<void *>(m_host_data_),
                                  size() * sizeof(T), cudaMemcpyHostToDevice));
    };
}

template <typename T, int N>
ManagedArray<T, N>::~ManagedArray() {
    if (m_host_data_ != nullptr && m_holder_.use_count() <= 1) {
        SP_DEVICE_CALL(cudaMemcpy(reinterpret_cast<void *>(m_host_data_), reinterpret_cast<void *>(m_holder_.get()),
                                  size() * sizeof(T), cudaMemcpyDeviceToHost));

        m_holder_.reset();
        m_data_ = nullptr;
    }
}
template <typename T, int N>
ManagedArray<T, N>::ManagedArray(T *d, int const *min, int const *max)
        : ManagedArray(d, min[0], max[0], min[1], max[1], min[2], max[2]) {}
template <typename T, int N>
ManagedArray<T, N>::ManagedArray(T *d, int3 min, int3 max)
        : ManagedArray(d, min.x, max.x, min.y, max.y, min.z, max.z) {}

template <typename T, int N>
ManagedArray<T, N>::ManagedArray(ManagedArray<T, N> const &other)
        : m_data_(other.m_data_),
          m_min_(other.m_min_),
          m_max_(other.m_max_),
          m_offset_(m_offset_),
          m_host_data_(other.m_host_data_),
          m_holder_(other.m_holder_) {}

template <typename T, int N>
ManagedArray<T, N>::ManagedArray(ManagedArray<T, N> &&other)
        : m_data_(other.m_data_),
          m_min_(other.m_min_),
          m_max_(other.m_max_),
          m_offset_(other.m_offset_),
          m_host_data_(other.m_host_data_),
          m_holder_(other.m_holder_) {
    other.m_holder_.reset();
}

template <typename T, int N>
__host__ __device__ int ManagedArray<T, N>::hash(int x, int y, int z, int n) const {
    // slow_first

    //  return n * m_offset_ +
    //         ((x - m_min_.x) * (m_max_.y - m_min_.y) + (y - m_min_.y)) *
    //             (m_max_.z - m_min_.z) +
    //         z - m_min_.z;

    return ((z - m_min_.z) * (m_max_.y - m_min_.y) + (y - m_min_.y)) * (m_max_.x - m_min_.x) + x - m_min_.x +
           n * m_offset_;
}
template <typename T, int N>
__host__ __device__ bool ManagedArray<T, N>::in_box(int x, int y, int z) const {
    return x >= m_min_.x && x < m_max_.x &&  //
           y >= m_min_.y && y < m_max_.y &&  //
           z >= m_min_.z && z < m_max_.z;
};

#endif  // __CUDA__
#endif  // SAMRAI_DEVICE_COMMON_H
