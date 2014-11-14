/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * MemoryPool.h
 *
 *  created on: 2011-3-2
 *      Author: salmon
 */
#ifndef INCLUDE_MEMORY_POOL_H_
#define INCLUDE_MEMORY_POOL_H_
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include "singleton_holder.h"
#include "log.h"
namespace simpla
{

////! \ingroup Utilities

/***
 * @brief MemoryPool is design to speed up  frequently and repeatedly
 * allocate operation of moderate size array or memory block.
 *
 */
class MemoryPool
{
private:

	typedef char byte_type;

	std::mutex locker_;

	std::multimap<size_t, void*> pool_;

	static constexpr size_t ONE_GIGA = 1024l * 1024l * 1024l;
	static constexpr size_t MAX_BLOCK_SIZE = 4 * ONE_GIGA; //std::numeric_limits<size_t>::max();
	static constexpr size_t MIN_BLOCK_SIZE = 256;

	size_t max_pool_depth_ = 16 * ONE_GIGA;
	size_t pool_depth_ = 0;

public:

	MemoryPool() :
			max_pool_depth_(4 * ONE_GIGA), pool_depth_(0) //2G
	{
	}
	~MemoryPool()
	{
		clear();
	}

	//!  unused memory will be freed when total memory size >= pool size
	inline void max_size(size_t s)
	{
		max_pool_depth_ = s;
	}

	/**
	 *  return the total size of memory in pool
	 * @return
	 */
	inline double size() const
	{
		return static_cast<double>(pool_depth_);
	}

	/**
	 *  push memory into pool
	 * @param d memory address
	 * @param s size of memory in byte
	 */
	inline void push(void * d, size_t s);

	/**
	 * allocate an array TV[s] from local pool or system memory
	 * if s < MIN_BLOCK_SIZE or s > MAX_BLOCK_SIZE or
	 *    s + pool_depth_> max_pool_depth_ then directly allocate
	 *    memory from system
	 *
	 * @param s size of memory in byte
	 * @return shared point of memory
	 */
	template<typename TV>
	std::shared_ptr<TV> pop(size_t s);

	void clear();

private:

	struct deleter_s
	{
		void * addr_;
		size_t s_;

		deleter_s(void * p, size_t s) :
				addr_(p), s_(s)
		{
		}

		void operator ()(void * ptr)
		{
			SingletonHolder<MemoryPool>::instance().push(addr_, s_);
		}
	};
};

inline void MemoryPool::clear()
{
	locker_.lock();
	for (auto & item : pool_)
	{
		delete[] reinterpret_cast<byte_type*>(item.second);
	}
	locker_.unlock();
}
inline void MemoryPool::push(void * p, size_t s)
{
	if ((s > MIN_BLOCK_SIZE) && (s < MAX_BLOCK_SIZE))
	{
		locker_.lock();

		if ((pool_depth_ + s < max_pool_depth_))
		{
			pool_.emplace(s, p);
			pool_depth_ += s;
			p = nullptr;
		}

		locker_.unlock();

	}
	if (p != nullptr)
		delete[] reinterpret_cast<byte_type*>(p);
}

template<typename TV>
std::shared_ptr<TV> MemoryPool::pop(size_t s)
{
	void * addr = nullptr;
	s *= sizeof(TV) / sizeof(byte_type);

	if ((s > MIN_BLOCK_SIZE) && (s < MAX_BLOCK_SIZE))
	{
		locker_.lock();

		// find memory block which is not smaller than demand size
		auto pt = pool_.lower_bound(s);

		if (pt != pool_.end())
		{
			size_t ts = 0;

			std::tie(ts, addr) = *pt;

			if (ts < s * 2)
			{
				s = ts;

				pool_.erase(pt);

				pool_depth_ -= s;
			}
			else
			{
				addr = nullptr;
			}
		}

		locker_.unlock();

	}

	if (addr != nullptr)
	{

		try
		{
			addr = reinterpret_cast<void*>(new byte_type[s]);

		} catch (std::bad_alloc const &error)
		{
			ERROR_BAD_ALLOC_MEMORY(s, error);
		}

	}
	return std::shared_ptr<TV>(reinterpret_cast<TV*>(addr), deleter_s(addr, s));

}

template<typename TV>
std::shared_ptr<TV> sp_make_shared_array(size_t s = 1)
{
	return SingletonHolder<MemoryPool>::instance().template pop<TV>(s);
}

}
// namespace simpla

#endif  // INCLUDE_MEMORY_POOL_H_
