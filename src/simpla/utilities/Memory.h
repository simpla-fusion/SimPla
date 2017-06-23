//
// Created by salmon on 16-10-17.
//

#ifndef SIMPLA_MEMORY_H
#define SIMPLA_MEMORY_H

#include <simpla/SIMPLA_config.h>

#include <cstring>

#include "Log.h"
#include "MemoryPool.h"

namespace simpla {
enum { HOST_MEMORY, DEVICE_MEMORY };

template <typename T>
std::shared_ptr<T> MemoryAllocT(size_type s, int location = HOST_MEMORY) {
    std::shared_ptr<T> res = nullptr;
    switch (location) {
        case DEVICE_MEMORY:
            break;
        case HOST_MEMORY:
        default:
            res = sp_alloc_array<T>(s);
            break;
    }
    return res;
}

std::shared_ptr<void> MemoryHostAlloc(size_type s) { return sp_alloc_memory(s); }

inline void MemorySet(void *d, int v, size_type s) { memset(d, v, s); }

inline void MemoryCopy(void *dest, void *src, size_type s) { memcpy(dest, src, s); }

}  // namespace simpla { namespace utilities

#endif  // SIMPLA_MEMORY_H
