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
#include "utilities/memory_pool.h"
#include "datatype.h"
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

	Object(BaseDataType const & pdtype, std::vector<size_t> const & d) :
			ele_size_in_bytes(pdtype.size_in_bytes()), ele_type_desc(
					pdtype.desc())
	{
		ReAlloc(&d[0], d.size());
	}

	Object(BaseDataType const & pdtype, size_t d) :
			ele_size_in_bytes(pdtype.size_in_bytes()), ele_type_desc(
					pdtype.desc())
	{
		ReAlloc(&d, 1);
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

	inline int get_num_of_dimesnsions() const
	{
		return nd;
	}
	/**
	 * Purpose: get the dimensions of object
	 * @Input:
	 * @Return: if dims!=NULL return the dimensions
	 * @Return: the number of dimensions
	 *
	 * */
	template<typename TS>
	inline int get_dimensions(TS* d) const
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
	inline size_t get_num_of_elements() const
	{
		size_t res = 1;
		for (int i = 0; i < nd; ++i)
		{
			res *= dims[i];
		}
		return res;
	}
	inline size_t get_size_in_bytes() const
	{
		return get_num_of_elements() * get_element_size_in_bytes();
	}

	void ReAlloc(size_t const *d, int ndims = 1)
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

	template<typename T>
	void FullFill(T const & v)
	{
		if (!CheckValueType(typeid(T)))
		{
			ERROR << "Value type do not match !";
		}

		size_t num_of_ele = get_num_of_elements();

#pragma omp parallel for
		for (size_t s = 0; s < num_of_ele; ++s)
		{
			this->value<T>(s) = v;
		}
	}
	void Clear()
	{
		size_t size_in_bytes = get_size_in_bytes();
#pragma omp parallel for
		for (size_t s = 0; s < size_in_bytes; ++s)
		{
			*(data.get() + s) = 0;
		}
	}
private:
	TR1::shared_ptr<int8_t> data;
	int nd;
	size_t dims[MAX_NUM_OF_DIMS];
	const size_t ele_size_in_bytes;
	const std::string ele_type_desc;
}
;

} //namespace simpla

#endif /* OBJECT_H_ */
