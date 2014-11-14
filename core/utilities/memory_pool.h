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
#include "singleton_holder.h"
#include "log.h"
namespace simpla
{

////! \ingroup Utilities
class MemoryPool
{
private:
	enum
	{
		MAX_POOL_DEPTH = 128
	};

	typedef char byte_type;

	std::multimap<size_t, void*> pool_;

	size_t MAX_POOL_SIZE;

	static constexpr size_t ONE_GIGA = 1024l * 1024l * 1024l;

	size_t ratio_; /// allocator release free memory block when
	///  "demanded size"< block size < ratio_ * "demanded size"

	size_t unused_size_ = 0;

public:

	MemoryPool();
	~MemoryPool();

	//! unused memory will be freed when total memory size >= pool size
	void max_size(size_t s)
	{
		MAX_POOL_SIZE = s;
	}

	double size() const
	{
		return static_cast<double>(unused_size_);
	}

	void push(void *, size_t s);

	template<typename TV>
	std::shared_ptr<TV> pop(size_t s);

	void clear();

//	template<typename TV>
//	size_t size(std::shared_ptr<TV> const &p) const;
//
//	/**
//	 * @return p_unused,p_used
//	 */
//	std::tuple<size_t, size_t> memory_size() const
//	{
//		size_t unused_memory = 0;
//		size_t used_memory = 0;
//		for (auto const & p : pool_)
//		{
//			if (p.second.unique())
//			{
//				unused_memory += p.first;
//			}
//			else
//			{
//				used_memory += p.first;
//			}
//		}
//
//		return std::make_tuple(used_memory, unused_memory);
//	}
//
//	size_t total_size() const
//	{
//		size_t unused_memory = 0;
//		size_t used_memory = 0;
//
//		std::tie(unused_memory, used_memory) = memory_size();
//
//		return unused_memory + used_memory;
//
//	}
//
//	double get_memory_sizeInGB(double * p_unused = nullptr, double * p_used =
//			nullptr) const
//	{
//		size_t unused_memory = 0;
//		size_t used_memory = 0;
//
//		std::tie(unused_memory, used_memory) = memory_size();
//
//		size_t total = unused_memory + used_memory;
//
//		if (p_unused != nullptr)
//		{
//			*p_unused = static_cast<double>(unused_memory)
//					/ static_cast<double>(ONE_GIGA);
//		}
//
//		if (p_used != nullptr)
//		{
//			*p_used = static_cast<double>(used_memory)
//					/ static_cast<double>(ONE_GIGA);
//		}
//
//		return static_cast<double>(total) / static_cast<double>(ONE_GIGA);
//	}

private:
//
//	template<typename TV>
//	inline std::shared_ptr<TV> allocate(size_t size)
//	{
//		std::shared_ptr<void> res = _allocate_shared_ptr(size);
//
//		released_raw_ptr_[std::hash<std::shared_ptr<void>>()(res)] = res;
//
//		return res.get();
//
//	}
//
//	inline void deallocate(void * p, size_t size = 0)
//	{
//		auto it = released_raw_ptr_.find(std::hash<void *>()(p));
//
//		if (it != released_raw_ptr_.end())
//		{
//			it->second.reset();
//		}
//		ReleaseMemory();
//	}
//
//	inline void deallocate(std::shared_ptr<void>& p, size_t size = 0)
//	{
//		p.reset();
//		ReleaseMemory();
//	}
//
//	inline void ReleaseMemory()
//	{
//		// the size of allocated memory
//		size_t t_size = total_size();
//
//		auto it = pool_.begin();
//
//		// release free memory until total_size < MAX_POOL_SIZE or no free memory is avaible
//		while (t_size > MAX_POOL_SIZE && it != pool_.end())
//		{
//			if (it->second.unique())
//			{
//				t_size -= it->first;
//				it = pool_.erase(it);
//			}
//			else
//			{
//				++it;
//			}
//
//		}
//	}

	struct Deleter
	{
		void * addr_;
		size_t s_;

		Deleter(void * p, size_t s) :
				addr_(p), s_(s)
		{
		}

		void operator ()(void * ptr)
		{
			ASSERT(ptr == addr_);

			SingletonHolder<MemoryPool>::instance().push(addr_, s_);
		}
	};
};

MemoryPool::MemoryPool() :
		MAX_POOL_SIZE(4 * ONE_GIGA), ratio_(2), unused_size_(0) //2G
{
}
MemoryPool::~MemoryPool()
{
	clear();
}
void MemoryPool::clear()
{
	for (auto & item : pool_)
	{
		delete[] reinterpret_cast<byte_type*>(item.second);
	}
}
void MemoryPool::push(void * p, size_t s)
{
	if (unused_size_ + s > MAX_POOL_SIZE)
	{
		delete[] reinterpret_cast<byte_type*>(p);
	}
	else
	{
		pool_.emplace(s, p);
		unused_size_ += s;
	}
}

template<typename TV>
std::shared_ptr<TV> MemoryPool::pop(size_t s)
{
	void * addr;
	s *= sizeof(TV);

// find memory block which is not smaller than demand size
	auto pt = pool_.lower_bound(s);

	if (pt != pool_.end())
	{
		std::tie(s, addr) = *pt;

		pool_.erase(pt);

		unused_size_ -= s;
	}
	else
	{

		try
		{
			addr = reinterpret_cast<void*>(new byte_type[s]);

		} catch (std::bad_alloc const &error)
		{
			ERROR_BAD_ALLOC_MEMORY(s, error);
		}

	}
	return std::shared_ptr<TV>(reinterpret_cast<TV*>(addr), Deleter(addr, s));
}

template<typename TV>
std::shared_ptr<TV> sp_make_shared(size_t demand = 1)
{
	return SingletonHolder<MemoryPool>::instance().template pop<TV>(demand);
}

}
// namespace simpla

#endif  // INCLUDE_MEMORY_POOL_H_
