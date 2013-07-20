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
#include "primitives/primitives.h"
#include "engine/object.h"
namespace simpla
{

template<typename T>
class Array: public Object
{
public:
	typedef size_t Index;
	typedef T Value;
	typedef Array<T> ThisType;

	Array() :
			num_of_elements_(0), ele_size_in_byte_(sizeof(Value))
	{
	}
	Array(size_t num, size_t ele_size = sizeof(Value)) :
			Object(num * ele_size, typeid(ThisType).name()), num_of_elements_(
					num), ele_size_in_byte_(ele_size)
	{

	}

	~Array()
	{

	}

	Array(ThisType const &)=delete;

	void swap(ThisType & rhs)
	{
		Object::swap(rhs);
		std::swap(num_of_elements_,rhs.num_of_elements_);
		std::swap(ele_size_in_byte_,rhs.ele_size_in_byte_);
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
		for (Index s = 0; s < num_of_elements_; ++s)
		{
			operator[](s) = index(rhs, s);
		}

		return (*this);
	}

	inline ThisType & operator =(Value const &rhs)
	{
#pragma omp parallel for
		for (Index s = 0; s < num_of_elements_; ++s)
		{
			operator[](s) = rhs;
		}

		return (*this);
	}

	inline Value & operator[](Index s)
	{
		return (*reinterpret_cast<T*>(&(*Object::data_) + s * ele_size_in_byte_));
	}

	inline Value const & operator[](Index s) const
	{
		return (*reinterpret_cast<T const*>(&(*Object::data_) + s * ele_size_in_byte_));
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

private:
	size_t num_of_elements_;
	size_t ele_size_in_byte_;

};

}  // namespace simpla

#endif /* ARRAY_H_ */
