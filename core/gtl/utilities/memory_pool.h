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

#include <stddef.h>
#include <memory>

#include "../design_pattern/singleton_holder.h"

namespace simpla
{

/** @ingroup utilities
 * @addtogroup memory_pool Memory Pool
 * @{
 * @brief    design to speed up  frequently and repeatedly
 * allocate operation of moderate size array or memory block.
 *
 */
class MemoryPool
{

public:
	typedef char byte_type;

	MemoryPool();

	~MemoryPool();

	//!  unused memory will be freed when total memory size >= pool size
	void max_size(size_t s);

	/**
	 *  return the total size of memory in pool
	 * @return
	 */
	double size() const;

	/**
	 *  push memory into pool
	 * @param d memory address
	 * @param s size of memory in byte
	 */
	void push(void * d, size_t s);

	/**
	 * allocate an array TV[s] from local pool or system memory
	 * if s < MIN_BLOCK_SIZE or s > MAX_BLOCK_SIZE or
	 *    s + pool_depth_> max_pool_depth_ then directly allocate
	 *    memory from system
	 *
	 * @param s size of memory in byte
	 * @return shared point of memory
	 */
	void * pop(size_t s);

	void clear();

	template<typename TV>
	std::shared_ptr<TV> alloc(size_t s);

	struct deleter_s
	{
		void * addr_;
		size_t s_;

		deleter_s(void * p, size_t s)
				: addr_(p), s_(s)
		{
		}
		~deleter_s()
		{
		}

		inline void operator ()(void * ptr)
		{
			SingletonHolder<MemoryPool>::instance().push(addr_, s_);
		}
	};
private:
	struct pimpl_s;

	std::unique_ptr<pimpl_s> pimpl_;
};

template<typename TV>
std::shared_ptr<TV> MemoryPool::alloc(size_t s)
{
	s *= sizeof(TV) / sizeof(byte_type);
	void * addr = pop(s);
	return std::shared_ptr<TV>(reinterpret_cast<TV*>(addr), deleter_s(addr, s));

}
template<typename TV>
std::shared_ptr<TV> sp_make_shared_array(size_t s)
{

	return SingletonHolder<MemoryPool>::instance().template alloc<TV>(s);
}
std::shared_ptr<void> sp_alloc_memory(size_t s);
/** @} */

}
// namespace simpla

#endif  // CORE_UTILITIES_MEMORY_POOL_H_
