/*
 * array.h
 *
 *  Created on: 2011-12-25
 *      Author: salmon
 */

#ifndef ARRAY_H_
#define ARRAY_H_

/**
 *  class Array
 *   base class of Fields
 * */
#include <vector>
#include "include/simpla_defs.h"
#include "engine/object.h"

namespace simpla
{
template<typename T, typename TExpr = NullType> class Array;
template<typename T>
class Array<T, NullType> : public NdArray
{
public:
	typedef T ValueType;
	typedef Array<T, NullType> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	Array() :
			num_of_elements_(0)
	{

	}
	Array(ThisType const & rhs) :
			data_(rhs.data_), num_of_elements_(rhs.num_of_elements_)
	{
	}

	Array(size_t num) :
			data_(NdArray::alloc<ValueType>(num)), num_of_elements_(num)
	{
	}

	Array(ValueType* data, size_t num) :
			data_(data), num_of_elements_(num)
	{
	}

	virtual ~Array()
	{
	}

	// Metadata ------------------------------------------------------------

	bool operator==(ThisType const & rhs)
	{
		return (data_.get() == rhs.data_.get());
	}

	virtual inline bool CheckType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(ThisType));
	}
	virtual inline bool CheckValueType(std::type_info const &tinfo) const
	{
		return (tinfo == typeid(ValueType));
	}

	virtual inline size_t get_element_size_in_bytes() const
	{
		return sizeof(ValueType);
	}

	virtual inline std::string get_element_type_desc() const
	{
		return "H5T_NATIVE_DOUBLE";
	}

	virtual inline void const * get_data() const
	{
		return reinterpret_cast<const void*>(data_.get());
	}

	virtual inline void * get_data()
	{
		return reinterpret_cast<void*>(data_.get());
	}

	virtual inline int get_dimensions(size_t* dims) const
	{
		if (dims != NULL)
		{
			dims[0] = num_of_elements_;
		}
		return 1;
	}

	virtual inline size_t get_size_in_bytes() const
	{
		return (num_of_elements_ * sizeof(ValueType));
	}

	virtual inline bool Empty() const
	{
		return (data_ == NULL);
	}

	//----------------------------------------------------------------------

	inline void Add(size_t s, ValueType const & v)
	{
#pragma omp atomic
		data_.get()[s] += v;
	}

	inline ValueType & operator[](size_t s)
	{
		return (data_.get()[s]);
	}

	inline ValueType const &operator[](size_t s) const
	{
		return (data_.get()[s]);
	}

private:

	TR1::shared_ptr<ValueType> data_;
	size_t num_of_elements_;
}
;
} //namespace simpla
#endif /* ARRAY_H_ */
