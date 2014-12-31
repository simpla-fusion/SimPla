/*
 * memory_pool.cpp
 *
 *  Created on: 2014年11月14日
 *      Author: salmon
 */

#include "memory_pool.h"
namespace simpla
{
MemoryPool::MemoryPool() :
		max_pool_depth_(4 * ONE_GIGA), pool_depth_(0) //2G
{
}
MemoryPool::~MemoryPool()
{
	clear();
}
void MemoryPool::clear()
{
	locker_.lock();
	for (auto & item : pool_)
	{
		delete[] reinterpret_cast<byte_type*>(item.second);
	}
	locker_.unlock();
}
void MemoryPool::push(void * p, size_t s)
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

std::tuple<void*, size_t> MemoryPool::pop_(size_t s)
{
	void * addr = nullptr;


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
}  // namespace simpla
