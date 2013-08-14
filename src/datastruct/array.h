/*
 * array.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef ARRAY_H_
#define ARRAY_H_
#include <algorithm>
#include <typeinfo>
#include <cassert>
#include "fetl/primitives.h"
#include "engine/object.h"
namespace simpla
{

template<typename T>
class Array: public Object
{
public:
	typedef T ValueType;
	typedef Array<T> ThisType;

	Array() :
			num_of_elements_(0), ele_size_in_byte_(sizeof(ValueType))
	{
	}
	Array(size_t num, size_t ele_size = sizeof(ValueType)) :
			Object(num * ele_size, typeid(ThisType).name()), num_of_elements_(
					num), ele_size_in_byte_(ele_size)
	{
	}

	~Array()
	{

	}

	Array(ThisType const&) = default;

	Array(ThisType && rhs) :
			Object(rhs), num_of_elements_(rhs.num_of_elements_), ele_size_in_byte_(
					rhs.ele_size_in_byte_)
	{

	}

	void swap(ThisType & rhs)
	{
		Object::swap(rhs);
		std::swap(num_of_elements_, rhs.num_of_elements_);
		std::swap(ele_size_in_byte_, rhs.ele_size_in_byte_);
	}

	bool IsEmpty() const
	{
		return (num_of_elements_ == 0);
	}
	bool IsSame(ThisType const & rhs) const
	{
		return (true);
	}

	template<typename TR>
	inline ThisType & operator =(TR const &rhs)
	{
#pragma omp parallel for
		for (size_t s = 0; s < num_of_elements_; ++s)
		{
			operator[](s) = index(rhs, s);
		}

		return (*this);
	}

	inline ThisType & operator =(ValueType const &rhs)
	{
#pragma omp parallel for
		for (size_t s = 0; s < num_of_elements_; ++s)
		{
			operator[](s) = rhs;
		}

		return (*this);
	}

	inline ValueType & operator[](size_t s)
	{
//		ASSERT(s<num_of_elements_);
		return (*reinterpret_cast<T*>(&(*Object::data_)
				+ (s % num_of_elements_) * ele_size_in_byte_));
	}

	inline ValueType const & operator[](size_t s) const
	{
//		ASSERT(s<num_of_elements_);
		return (*reinterpret_cast<T const*>(&(*Object::data_)
				+ (s % num_of_elements_) * ele_size_in_byte_));
	}

	template<typename TR>
	inline ThisType & operator +=(TR const &rhs)
	{
		*this = *this + rhs;
		return (*this);
	}
	template<typename TR>
	inline ThisType & operator -=(TR const &rhs)
	{
		*this = *this - rhs;
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator *=(TR const &rhs)
	{
		*this = *this * rhs;
		return (*this);
	}
	template<typename TR>
	inline ThisType & operator /=(TR const &rhs)
	{
		*this = *this / rhs;
		return (*this);
	}
	void fill(ValueType const &v)
	{
#pragma omp parallel for
		for (size_t i = 0; i < num_of_elements_; ++i)
		{
			operator[](i) = v;
		}
	}

private:
	size_t num_of_elements_;
	size_t ele_size_in_byte_;

};
template<typename T>
struct is_storage_type<Array<T> >
{
	static const bool value = true;
};

}  // namespace simpla

#endif /* ARRAY_H_ */
