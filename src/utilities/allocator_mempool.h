/*
 * allocator_mempool.h
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#ifndef ALLOCATOR_MEMPOOL_H_
#define ALLOCATOR_MEMPOOL_H_

#include <memory>
#include <cstddef>
#include <new>
#include <vector>

#include "log.h"
#include "memory_pool.h"

namespace simpla
{

template<typename T>
class MemPoolAllocator
{
public:
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;

	MemPoolAllocator() noexcept
	{
	}

	template<typename U>
	MemPoolAllocator(MemPoolAllocator<U> const &) noexcept
	{
	}

	inline pointer allocate(std::size_t num)
	{
		return static_cast<T*>(MEMPOOL.allocate(num*sizeof(T)));
	}

	inline void deallocate(pointer p, size_t num = 0)
	{
		MEMPOOL.deallocate(static_cast<void*>(p),num*sizeof(T));
	}

};
template<typename T1, typename T2> inline
bool operator==(const MemPoolAllocator<T1> &, const MemPoolAllocator<T2> &)
{
	return true;
}
template<typename T1, typename T2> inline
bool operator!=(const MemPoolAllocator<T1> &, const MemPoolAllocator<T2> &)
{
	return false;
}

}  // namespace simpla

#endif /* ALLOCATOR_MEMPOOL_H_ */
