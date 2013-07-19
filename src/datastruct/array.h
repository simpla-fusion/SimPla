/*
 * array.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef ARRAY_H_
#define ARRAY_H_

namespace simpla
{

template<typename T>
class Array
{
public:
	typedef size_t Index;
	typedef T Value;
	typedef Array<T> ThisType;

	bool IsEmpty() const
	{
		return (true);
	}
	bool IsSame(ThisType const & rhs) const
	{
		return (true);
	}
	inline ThisType & operator *=(Real rhs)
	{
		*this = *this * rhs;
		return (*this);
	}
	inline ThisType & operator /=(Real rhs)
	{
		*this = *this / rhs;
		return (*this);
	}
};

}  // namespace simpla

#endif /* ARRAY_H_ */
