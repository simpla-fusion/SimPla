/*
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
#include "primitives/properties.h"

namespace simpla
{
class Object
{

public:
	static const int MAX_NUM_OF_DIMS = 10;

	typedef TR1::shared_ptr<Object> Holder;

	ptree properties;

	Object(size_t es, std::string const & desc) :
			data(NULL), nd(0), ele_size_in_bytes(es), ele_type_desc(desc)
	{
		dims[0] = 0;
	}

	inline virtual ~Object()
	{
	}

	// Metadata ------------------------------------------------------------

	inline bool Empty() const
	{
		return (data == NULL);
	}
	virtual inline bool CheckType(std::type_info const &) const
	{
		return false;
	}

	virtual inline bool CheckValueType(std::type_info const &) const
	{
		return false;
	}

	inline size_t get_element_size_in_bytes() const
	{
		return ele_size_in_bytes;
	}

	inline std::string get_element_type_desc() const
	{
		return ele_type_desc;
	}
	inline char const * get_data(size_t s = 0) const
	{
		return data + s * ele_size_in_bytes;
	}

	inline char * get_data(size_t s = 0)
	{
		return data + s * ele_size_in_bytes;
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
	void ReleaseMemory()
	{
		nd = 0;
#pragma omp critical(OBJECT_ALLOC)
		{
			if (data != NULL)
			{
				delete data;
				data = NULL;
			}
		}

	}
	void ReAlloc(size_t *d, int ndims = 1)
	{
		size_t o_size_in_bytes = get_size_in_bytes();

		size_t size_in_bytes = ele_size_in_bytes;

		nd = ndims;

		for (int i = 0; i < ndims; ++i)
		{
			dims[i] = d[i];
			size_in_bytes *= d[i];
		}
		if (size_in_bytes > 0
				&& (size_in_bytes < o_size_in_bytes / 2
						|| size_in_bytes > o_size_in_bytes))
		{
			ReleaseMemory();
#pragma omp critical(OBJECT_ALLOC)
			{

				try
				{
					data = reinterpret_cast<char*>(operator new(size_in_bytes));

				} catch (std::bad_alloc const &error)
				{
					ERROR_BAD_ALLOC_MEMORY(size_in_bytes, error);
				}
			}
		}
	}
private:
	char * data;
	int nd;
	size_t dims[MAX_NUM_OF_DIMS];
	const size_t ele_size_in_bytes;
	const std::string ele_type_desc;
};
inline Object::~Object()
{
}
} //namespace simpla

#endif /* OBJECT_H_ */
