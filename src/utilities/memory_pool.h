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
#include "defs.h"
#include "singleton_holder.h"

class MemoryPool: public SingletonHolder<MemoryPool>
{
	static const int MAX_POOL_DEPTH = 128;
	typedef std::multimap<SizeType, TR1::shared_ptr<int8_t> > MemoryMap;
	MemoryMap pool_;

public:
	MemoryPool()
	{
	}
	~MemoryPool()
	{
	}

	inline TR1::shared_ptr<int8_t> alloc(SizeType size)
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
			TR1::shared_ptr<int8_t>(new int8_t[size]).swap(res);
			pool_.insert(MemoryMap::value_type(size, res));
		}
		return res;
	}

	inline void release()
	{
		if (pool_.size() > MAX_POOL_DEPTH)
		{
			for (MemoryMap::iterator it = pool_.begin(); it != pool_.end();
					++it)
			{
				if (it->second.unique())
				{
					pool_.erase(it);
					break;
				}
			}
		}
	}
};

#endif  // INCLUDE_MEMORY_POOL_H_
