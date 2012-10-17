/* SimPla Copyright 2007-2012 YU Zhi All right reserved.
 *   ____  _           ____  _
 * / ___|(_)_ __ ___ |  _ \| | __ _
 * \___ \| | '_ ` _ \| |_) | |/ _` |
 *  ___) | | | | | | |  __/| | (_| |
 * |____/|_|_| |_| |_|_|   |_|\__,_|
 *
 * A Framework of plasma simulation
 *
 *
 *
 *
 * object.h
 *
 *  Created on: 2012-2-6
 *      Author: salmon
 */

#ifndef OBJECT_H_
#define OBJECT_H_
#include "include/simpla_defs.h"
#include <typeinfo>
#include <iostream>
#include "utilities/properties.h"
#include "utilities/memory_pool.h"

namespace simpla
{
class Object
{

public:
	enum
	{
		MAX_NUM_OF_DIMS = 10
	};

	typedef TR1::shared_ptr<Object> Holder;

	ptree properties;

	Object(size_t es, std::string const & desc, size_t s) :
			nd(0), ele_size_in_bytes(es), ele_type_desc(desc)
	{
		size_t ss = s;
		ReAlloc(&ss, 1);
	}

	Object(size_t es, std::string const & desc = "", int ndims = 0, size_t *d =
			NULL) :
			nd(0), ele_size_in_bytes(es), ele_type_desc(desc)
	{
		if (ndims > 0)
		{
			ReAlloc(d, ndims);
		}
	}
	inline virtual ~Object()
	{
		MemoryPool::instance().release();
	}

// Metadata ------------------------------------------------------------

	virtual bool CheckType(std::type_info const &) const=0;

	virtual bool CheckValueType(std::type_info const &) const=0;

	inline bool Empty() const
	{
		return (data == NULL);
	}

	inline size_t get_element_size_in_bytes() const
	{
		return ele_size_in_bytes;
	}

	inline std::string get_element_type_desc() const
	{
		return ele_type_desc;
	}
	inline int8_t const * get_data(size_t s = 0) const
	{
		return data.get() + s * ele_size_in_bytes;
	}

	inline int8_t * get_data(size_t s = 0)
	{
		return data.get() + s * ele_size_in_bytes;
	}
	template<typename T>
	T & value(size_t s)
	{
		return *reinterpret_cast<T*>(data.get() + s * ele_size_in_bytes);
	}
	template<typename T>
	T const & value(size_t s) const
	{
		return *reinterpret_cast<T const*>(data.get() + s * ele_size_in_bytes);
	}
	/**
	 * Purpose: get the dimensions of object
	 * @Input:
	 * @Return: if dims!=NULL return the dimensions
	 * @Return: the number of dimensions
	 *
	 * */
	inline int get_dimensions(size_t* d = NULL) const
	{
		if (d != NULL)
		{
			for (int i = 0; i < nd; ++i)
			{
				d[i] = dims[i];
			}
		}
		return nd;
	}

	inline size_t get_size_in_bytes() const
	{
		size_t res = 1;
		for (int i = 0; i < nd; ++i)
		{
			res *= dims[i];
		}
		return res;
	}

	void ReAlloc(size_t *d, int ndims = 1)
	{

		size_t o_size_in_bytes = get_size_in_bytes();

		size_t size_in_bytes = ele_size_in_bytes;

		for (int i = 0; i < ndims; ++i)
		{
			dims[i] = d[i];
			size_in_bytes *= d[i];
		}

		data = MemoryPool::instance().alloc(size_in_bytes);

		nd = ndims;

	}
private:
	TR1::shared_ptr<int8_t> data;
	int nd;
	size_t dims[MAX_NUM_OF_DIMS];
	const size_t ele_size_in_bytes;
	const std::string ele_type_desc;
	static std::multimap<size_t, char*> pool_;
}
;

} //namespace simpla

#endif /* OBJECT_H_ */
