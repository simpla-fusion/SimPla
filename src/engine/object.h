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

namespace simpla
{
class Object
{
public:

	typedef TR1::shared_ptr<Object> Holder;

	Object()
	{
	}

	inline virtual ~Object()=0;

	// Metadata ------------------------------------------------------------

	virtual inline bool CheckType(std::type_info const &) const
	{
		return false;
	}

	virtual inline bool CheckValueType(std::type_info const &) const
	{
		return false;
	}
	virtual inline size_t get_element_size_in_bytes() const
	{
		return 0;
	}

	virtual inline std::string get_element_type_desc() const
	{
		return "";
	}
	virtual inline void const * get_data() const
	{
		return NULL;
	}

	virtual inline void * get_data()
	{
		return NULL;
	}

	/**
	 * Purpose: get the dimensions of object
	 * @Input:
	 * @Return: if dims!=NULL return the dimensions
	 * @Return: the number of dimensions
	 *
	 * */
	virtual inline int get_dimensions(size_t* dims = NULL) const
	{
		return 0;
	}

	virtual inline size_t get_size_in_bytes() const
	{
		return 0;
	}
	virtual inline bool Empty() const
	{
		return (true);
	}

	template<typename T>
	static TR1::shared_ptr<T> alloc(size_t num, size_t ele_size_in_bytes =
			sizeof(T))
	{
		TR1::shared_ptr<T> res;
		size_t size_in_bytes = num * ele_size_in_bytes;

#pragma omp critical(OBJECT_ALLOC)
		{

			if (size_in_bytes > 0)
			{
				try
				{
					res = TR1::shared_ptr<T>(
							reinterpret_cast<T*>(operator new(size_in_bytes)));

				} catch (std::bad_alloc const &error)
				{
					ERROR_BAD_ALLOC_MEMORY(size_in_bytes, error);
				}
			}
		}
		return res;
	}

};
inline Object::~Object()
{
}
} //namespace simpla

#endif /* OBJECT_H_ */
