/*
 * typetraits.h
 *
 *  Created on: 2012-3-29
 *      Author: salmon
 */

#ifndef TYPETRAITS_H_
#define TYPETRAITS_H_
namespace simpla
{

template<typename T> struct TypeTraits
{
	typedef T Reference;
	typedef const T ConstReference;
};
} // namespace simpla

#endif /* TYPETRAITS_H_ */
