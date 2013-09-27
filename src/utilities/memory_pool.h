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
#include "utilities/log.h"

class MemoryPool: public SingletonHolder<MemoryPool>
{
	enum
	{
		MAX_POOL_DEPTH = 128
	};

	typedef std::multimap<size_t, std::shared_ptr<int8_t> > MemoryMap;

	MemoryMap pool_;

	size_t MAX_POOL_DEPTH_IN_GB;
	const size_t ONE_GIGA = 1024l * 1024l * 1024l;
public:
	MemoryPool() :
			MAX_POOL_DEPTH_IN_GB(2)  //2G
	{
	}
	~MemoryPool()
	{
	}
	void set_pool_depth_in_GB(size_t s)
	{
		MAX_POOL_DEPTH_IN_GB = s;
	}
	inline std::shared_ptr<int8_t> alloc(size_t size)
	{
		std::shared_ptr<int8_t> res;

		bool isFound = false;

		std::pair<MemoryMap::iterator, MemoryMap::iterator> pt =
				pool_.equal_range(size);
		if (pt.first != pool_.end())
		{
			for (MemoryMap::iterator it = pt.first; it != pt.second; ++it)
			{
				if (it->second.unique())
				{
					std::shared_ptr<int8_t>(it->second).swap(res);
					isFound = true;
					break;
				}
			}
		}
		if (!isFound)
		{
			try
			{
				std::shared_ptr<int8_t>(new int8_t[size]).swap(res);

			} catch (std::bad_alloc const &error)
			{
				Log(-2) << __FILE__ << "[" << __LINE__ << "]:"
						<< "Can not get enough memory! [ ~" << size / ONE_GIGA
						<< " GiB ]" << std::endl;
				throw(error);
			}

			pool_.insert(MemoryMap::value_type(size, res));
		}
		return res;
	}

	inline void release()
	{
		size_t pool_depth = 0;

		for (MemoryMap::iterator it = pool_.begin(); it != pool_.end(); ++it)
		{
			if (it->second.unique())
			{
				if (pool_depth > MAX_POOL_DEPTH_IN_GB * ONE_GIGA)
				{
					pool_.erase(it);
				}
				else
				{
					pool_depth += it->first;
				}
			}
		}

		for (MemoryMap::iterator it = pool_.begin(); it != pool_.end(); ++it)
		{
			if (it->second.unique())
			{
				if (pool_depth > MAX_POOL_DEPTH_IN_GB * ONE_GIGA)
				{
					pool_.erase(it);
					pool_depth -= it->first;
				}
			}
		}
	}
};

#define MEMPOOL MemoryPool::instance()
#endif  // INCLUDE_MEMORY_POOL_H_
