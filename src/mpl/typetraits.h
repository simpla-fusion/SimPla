/*
 * typetraits.h
 *
 *  Created on: 2012-10-16
 *      Author: salmon
 */

#ifndef TYPETRAITS_H_
#define TYPETRAITS_H_

namespace simpla
{

class NullType;

class EmptyType
{
};

template<typename T> struct TypeTraits
{
	typedef T Reference;
	typedef const T ConstReference;
};

template<int v>
struct Int2Type
{
	enum
	{
		value = v
	};
};

template<typename T>
struct Type2Type
{
	typedef T OriginalType;
};

}  // namespace simpla

#endif /* TYPETRAITS_H_ */
