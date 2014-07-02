/*
 * allocator_fixed_small.h
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#ifndef ALLOCATOR_FIXED_SMALL_H_
#define ALLOCATOR_FIXED_SMALL_H_

#include <cstddef>
#include <list>

#include "allocator_mempool.h"
#include "memory_pool.h"

namespace simpla
{

template<typename T>
class FixedSmallAllocator
{

	std::list<T, MemPoolAllocator<T> > pool_;
public:
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;

	FixedSmallAllocator() noexcept
	{
	}

	template<typename U>
	FixedSmallAllocator(FixedSmallAllocator<U> const &) noexcept
	{
	}

	inline pointer allocate(std::size_t num)
	{
		return static_cast<T*>(MEMPOOL.allocate(num * sizeof(T)));
	}

	inline void deallocate(pointer p, size_t num = 0)
	{
		MEMPOOL.deallocate(static_cast<void*>(p), num * sizeof(T));
	}

	template <typename U>
	struct rebind
	{
		typedef FixedSmallAllocator<U> other;
	};

};

template<typename T>
class FixedSmallAllocator<std::ListNode<T> >
{

	std::list<T, MemPoolAllocator<T> > pool_;
public:
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;

	FixedSmallAllocator() noexcept
	{
	}

	template<typename U>
	FixedSmallAllocator(FixedSmallAllocator<U> const &) noexcept
	{
	}

	inline pointer allocate(std::size_t num)
	{
		return static_cast<T*>(MEMPOOL.allocate(num * sizeof(T)));
	}

	inline void deallocate(pointer p, size_t num = 0)
	{
		MEMPOOL.deallocate(static_cast<void*>(p), num * sizeof(T));
	}

	template <typename U>
	struct rebind
	{
		typedef FixedSmallAllocator<U> other;
	};

};



}
 // namespace simpla

#endif /* ALLOCATOR_FIXED_SMALL_H_ */
