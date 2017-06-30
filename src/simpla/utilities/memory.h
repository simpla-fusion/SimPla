/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * @file memory_pool.h
 *
 *  created on: 2011-3-2
 *      Author: salmon
 */
#ifndef CORE_UTILITIES_MEMORY_POOL_H_
#define CORE_UTILITIES_MEMORY_POOL_H_

#include <simpla/SIMPLA_config.h>

#include "device_common.h"

#include <boost/mpl/size_t.hpp>
#include <cstddef>
#include <memory>

namespace simpla {

/** @ingroup toolbox
 * @addtogroup memory_pool Memory Pool
 * @{
 * @brief    design to speed up  frequently and repeatedly
 * allocate operation of moderate size array or memory block.
 *
 */

enum { MANAGED_MEMORY, HOST_MEMORY, DEVICE_MEMORY };

template <typename T>
int spMemoryAlloc(T **addr, size_t n, int location = MANAGED_MEMORY) {
    if (addr == nullptr) { return SP_FAILED; };
#ifndef __CUDA__
    *addr = reinterpret_cast<T *>(malloc(n * sizeof(T)));
#else
    ASSERT(addr != nullptr);
    SP_DEVICE_CALL(cudaMallocManaged(addr, n * sizeof(T)));
    SP_DEVICE_CALL(cudaDeviceSynchronize());
#endif
    return SP_SUCCESS;

    //    switch (location) {
    //        case MANAGED_MEMORY:
    //            SP_DEVICE_CALL(cudaMallocManaged(p, s));
    //            break;
    //        case DEVICE_MEMORY:
    //            SP_DEVICE_CALL(cudaMalloc(p, s));
    //            break;
    //        case HOST_MEMORY:
    //        default:
    //            *p = malloc(s);
    //    }
};


inline int spMemoryFree(void **dest, size_t n) {
    if (dest == nullptr) { return SP_FAILED; }
#ifndef __CUDA__
    free(*dest);
    *dest = nullptr;
#else
    SP_DEVICE_CALL(cudaFree(*dest));
    SP_DEVICE_CALL(cudaDeviceSynchronize());
#endif
    *dest = nullptr;
    return SP_SUCCESS;
};

template <typename T>
int spMemoryFree(T **addr, size_t n) {
    spMemoryFree((void **)addr, n * sizeof(T));
    return SP_SUCCESS;
};

namespace detail {
struct deleter_device_ptr_s {
    void *addr_;
    size_t m_size_;
    int m_loc_;

    deleter_device_ptr_s(void *p, size_t s, int loc) : addr_(p), m_size_(s), m_loc_(loc) {}

    ~deleter_device_ptr_s() = default;

    deleter_device_ptr_s(const deleter_device_ptr_s &) = default;

    deleter_device_ptr_s(deleter_device_ptr_s &&) = default;

    deleter_device_ptr_s &operator=(const deleter_device_ptr_s &) = default;

    deleter_device_ptr_s &operator=(deleter_device_ptr_s &&) = default;

    inline void operator()(void *ptr) { spMemoryFree(&ptr, m_size_); }
};
}

template <typename T>
std::shared_ptr<T> spMakeShared(T *d, size_t n, int location = MANAGED_MEMORY) {
    T *addr = nullptr;
    spMemoryAlloc(&addr, n, location);
    return std::shared_ptr<T>(addr, simpla::detail::deleter_device_ptr_s(addr, n * sizeof(T), location));
}
#ifdef __CUDA__
namespace detail {
template <typename T>
__global__ void spCUDA_Assign(T *dest, T src, size_t n) {
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < n) { dest[s] = src * threadIdx.x; };
}
template <typename T, typename U>
__global__ void spCUDA_Copy(T *dest, U const *src, size_t n) {
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < n) { dest[s] = src[s]; };
}
}
#endif
#define NUM_OF_THREAD 32

template <typename T>
int spMemoryFill(T *dest, T const &src, size_t n) {
#ifndef __CUDA__
    char *p_dest = reinterpret_cast<char *>(dest);
    char const *p_src = reinterpret_cast<char const *>(&src);
    static constexpr int m = sizeof(T);
#pragma omp parallel for
    for (int i = 0; i < m * n; ++i) { p_dest[i] = p_src[i % m]; }
#else
    SP_CALL_DEVICE_KERNEL(detail::spCUDA_Assign, (n + NUM_OF_THREAD) / NUM_OF_THREAD, NUM_OF_THREAD, dest, src, n);
#endif
    return SP_SUCCESS;
}

template <typename U, typename V>
int spMemoryCopy(U *dest, V const *src, size_t n) {
#ifndef __CUDA__
#else
    SP_CALL_DEVICE_KERNEL(detail::spCUDA_Copy, (n + NUM_OF_THREAD) / NUM_OF_THREAD, NUM_OF_THREAD, dest, src, n);

#endif
    return SP_SUCCESS;
}

template <typename T>
int spMemoryCopy(T *dest, T const *src, size_t n) {
#ifndef __CUDA__
#else
    SP_DEVICE_CALL(cudaMemcpy((void *)dest, (void const *)src, n * sizeof(T), cudaMemcpyDefault));

#endif
    return SP_SUCCESS;
}

}  // namespace simpla

#endif  // CORE_UTILITIES_MEMORY_POOL_H_
