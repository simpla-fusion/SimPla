/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_
#include "datastruct/array.h"
namespace simpla
{

template<typename TV, typename TPolicy = NullType>
struct Field: public Array<TV>
{
public:

	typedef Array<TV> BaseType;
	typedef Field<TV, TPolicy> ThisType;

	typedef ThisType const &ConstReference;

public:

	Field(size_t size = 0, size_t value_size = sizeof(Value)) :
			BaseType(size, value_size)
	{
	}

	virtual ~Field()
	{
	}

	bool CheckType(std::type_info const &rhs) const
	{
		return (typeid(ThisType) == rhs || BaseType::CheckType(rhs));
	}

// Assignment --------

	template<typename TR>
	inline ThisType & operator =(TR const &rhs)
	{
		Assign(*this, rhs);
		return (*this);
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

};

}  // namespace simpla

#endif /* FIELD_H_ */
