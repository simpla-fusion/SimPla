//
// Created by salmon on 16-10-17.
//

#ifndef SIMPLA_MEMORY_H
#define SIMPLA_MEMORY_H

#include <simpla/SIMPLA_config.h>

#include <cstring>

#include "MemoryPool.h"

namespace simpla { namespace toolbox
{


template<typename T> std::shared_ptr<T> MemoryHostAllocT(size_type num) { return sp_alloc_array<T>(num); }

inline std::shared_ptr<void> MemoryHostAlloc(size_type s) { return sp_alloc_memory(s); }

inline void MemorySet(void *d, int v, size_type s) { memset(d, v, s); }

inline void MemoryCopy(void *dest, void *src, size_type s) { memcpy(dest, src, s); }

}}//namespace simpla { namespace toolbox

#endif //SIMPLA_MEMORY_H
