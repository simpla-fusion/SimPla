/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * MemoryPool.h
 *
 *  Created on: 2011-3-2
 *      Author: salmon
 */
#ifndef INCLUDE_MEMORY_POOL_H_
#define INCLUDE_MEMORY_POOL_H_
#include <map>
#include <memory>
#include "singleton_holder.h"
#include "log.h"
namespace simpla
{

void deallocate_m(void *p);
//! @ingroup Utilities
class MemoryPool
{
private:
	enum
	{
		MAX_POOL_DEPTH = 128
	};

	typedef unsigned char byte_type;

	std::map<size_t, std::shared_ptr<byte_type>> released_raw_ptr_;

	std::multimap<size_t, std::shared_ptr<byte_type> > pool_;

	size_t MAX_POOL_SIZE;

	const size_t ONE_GIGA = 1024l * 1024l * 1024l;

	size_t ratio_; /// allocatro release free memory block when "demanded size"< block size < ratio_ * "demanded size"

public:

	MemoryPool()
			: MAX_POOL_SIZE(4), ratio_(2) //2G
	{
	}
	~MemoryPool()
	{
	}

	// unused memory will be freed when total memory size >= pool size
	void SetPoolSizeInGB(size_t s)
	{
		MAX_POOL_SIZE = s * ONE_GIGA;
	}

	/**
	 *
	 * @param not_used
	 * @param in_used
	 * @return
	 */
	size_t GetMemorySize(size_t * p_unused = nullptr, size_t * p_used = nullptr) const
	{
		size_t unused_memory = 0;
		size_t used_memory = 0;
		for (auto const & p : pool_)
		{
			if (p.second.unique())
			{
				unused_memory += p.first;
			}
			else
			{
				used_memory += p.first;
			}
		}
		if (p_unused != nullptr)
		{
			*p_unused = used_memory;
		}
		if (p_used != nullptr)
		{
			*p_used = used_memory;
		}

		return unused_memory + used_memory;
	}

	double GetMemorySizeInGB(double * p_unused = nullptr, double * p_used = nullptr) const
	{
		size_t unused_memory = 0;
		size_t used_memory = 0;
		size_t total = GetMemorySize(&unused_memory, &used_memory);

		if (p_unused != nullptr)
		{
			*p_unused = static_cast<double>(unused_memory) / static_cast<double>(ONE_GIGA);
		}

		if (p_used != nullptr)
		{
			*p_used = static_cast<double>(used_memory) / static_cast<double>(ONE_GIGA);
		}

		return static_cast<double>(total) / static_cast<double>(ONE_GIGA);
	}

	inline byte_type * allocate(size_t size)
	{
		std::shared_ptr<byte_type> res = _allocate_shared_ptr(size);

		released_raw_ptr_[std::hash<std::shared_ptr<byte_type>>()(res)] = res;

		return res.get();

	}

	inline void deallocate(void * p, size_t size = 0)
	{
		auto it = released_raw_ptr_.find(std::hash<byte_type *>()(reinterpret_cast<byte_type*>(p)));

		if (it != released_raw_ptr_.end())
		{
			it->second.reset();
		}
		ReleaseMemory();
	}

	inline void deallocate(std::shared_ptr<byte_type>& p, size_t size = 0)
	{
		p.reset();
		ReleaseMemory();
	}

	template<typename TV>
	inline std::shared_ptr<TV> allocate_shared_ptr(size_t demand)
	{
		return std::shared_ptr<TV>(reinterpret_cast<TV*>(allocate(demand * sizeof(TV))), deallocate_m);
	}

private:

	inline std::shared_ptr<byte_type> _allocate_shared_ptr(size_t demand)
	{
		std::shared_ptr<byte_type> res(nullptr);

		// find memory block which is not smaller than demand size
		auto pt = pool_.lower_bound(demand);

		for (auto & p : pool_)
		{
			//release memory if block is free and size < ratio_ * demand
			if (p.second.unique() && p.first < ratio_ * demand)
			{
				res = p.second;
				break;
			}
		}

		// if there is no proper memory block available , allocate new memory block
		if (res == nullptr)
		{
			try
			{
				res = std::shared_ptr<byte_type>(new byte_type[demand]);

			} catch (std::bad_alloc const &error)
			{
				ERROR_BAD_ALLOC_MEMORY(demand, error);
			}

			// put new memory into pool
			pool_.insert(std::make_pair(demand, res));
		}
		return res;
	}
	inline void ReleaseMemory()
	{
		// the size of allocated memory
		size_t total_size = GetMemorySize();

		auto it = pool_.begin();

		// release free memory until total_size < MAX_POOL_SIZE or no free memory is avaible
		while (total_size > MAX_POOL_SIZE && it != pool_.end())
		{
			if (it->second.unique())
			{
				total_size -= it->first;
				it = pool_.erase(it);
			}
			else
			{
				++it;
			}

		}
	}
};

#define MEMPOOL  SingletonHolder<MemoryPool>::instance()

inline void deallocate_m(void *p)
{
	SingletonHolder<MemoryPool>::instance().deallocate(p);
}
}  // namespace simpla

#endif  // INCLUDE_MEMORY_POOL_H_
