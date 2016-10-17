//
// Created by salmon on 16-10-17.
//

#ifndef SIMPLA_MEMORY_H
#define SIMPLA_MEMORY_H

#include <cstring>
#include "SIMPLA_config.h"

#include "MemoryPool.h"

namespace simpla { namespace toolbox
{


template<typename T> std::shared_ptr<T> MemoryHostAllocT(size_type num) { return sp_alloc_array<T>(num); }

inline std::shared_ptr<void> MemoryHostAlloc(size_type s) { return sp_alloc_memory(s); }

inline void MemorySet(std::shared_ptr<void> d, int v, size_type s) { memset(d.get(), v, s); }

inline void MemoryCopy(std::shared_ptr<void> dest, std::shared_ptr<void> src, size_type s)
{
    memcpy(dest.get(), src.get(), s);
}

}}//namespace simpla { namespace toolbox

#endif //SIMPLA_MEMORY_H
