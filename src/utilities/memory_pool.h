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
#include "singleton_holder.h"
#include "log.h"
class MemoryPool: public SingletonHolder<MemoryPool>
{
	enum
	{
		MAX_POOL_DEPTH = 128
	};

	typedef std::multimap<size_t, TR1::shared_ptr<int8_t> > MemoryMap;

	MemoryMap pool_;

	static size_t MAX_POOL_DEPTH_IN_Gbytes;
public:
	MemoryPool()  //2G
	{
	}
	~MemoryPool()
	{
	}
	static void set_max_pool_depth_in_Gb(size_t s)
	{
		MAX_POOL_DEPTH_IN_Gbytes = s;
	}

	inline TR1::shared_ptr<int8_t> alloc(size_t size)
	{
		TR1::shared_ptr<int8_t> res;

		bool isFound = false;

		std::pair<MemoryMap::iterator, MemoryMap::iterator> pt =
				pool_.equal_range(size);
		if (pt.first != pool_.end())
		{
			for (MemoryMap::iterator it = pt.first; it != pt.second; ++it)
			{
				if (it->second.unique())
				{
					TR1::shared_ptr<int8_t>(it->second).swap(res);
					isFound = true;
					break;
				}
			}
		}
		if (!isFound)
		{
			try
			{
				TR1::shared_ptr<int8_t>(new int8_t[size]).swap(res);

			} catch (std::bad_alloc const &error)
			{
				ERROR_BAD_ALLOC_MEMORY(size, error);
			}

			pool_.insert(MemoryMap::value_type(size, res));
		}
		return res;
	}

	inline void release()
	{
		static size_t pool_depth = 0;

		for (MemoryMap::iterator it = pool_.begin(); it != pool_.end(); ++it)
		{
			if (it->second.unique())
			{
				if (pool_depth > MAX_POOL_DEPTH_IN_Gbytes)
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
				if (pool_depth > MAX_POOL_DEPTH_IN_Gbytes)
				{
					pool_.erase(it);
					pool_depth -= it->first;
				}
			}
		}
	}
};

#endif  // INCLUDE_MEMORY_POOL_H_
