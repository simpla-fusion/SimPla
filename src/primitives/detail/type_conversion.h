/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: TypeTraits.h 989 2010-12-12 02:47:22Z salmon $
 * TypeTraits.h
 *
 *  Created on: 2009-4-3
 *      Author: salmon
 */

#ifndef TYPE_CONVERSION_H_
#define TYPE_CONVERSION_H_

#include <complex>

namespace simpla
{
namespace _detail
{
template<typename T1, typename T2> struct ImplicitTypeConvert
{
	typedef decltype(T1()*T2()) Type;
};
template<typename T, int I> struct ImplicitTypeConvert<T, Int2Type<I> >
{
	typedef T Type;
};
template<typename T, int I> struct ImplicitTypeConvert<Int2Type<I>, T>
{
	typedef T Type;
};
//template<typename T1, typename T2> struct ImplicitTypeConvert;
//
//template<typename T> struct ImplicitTypeConvert<T, T>
//{
//	typedef T Type;
//};
//template<typename T> struct ImplicitTypeConvert<T, NullType>
//{
//	typedef T Type;
//};
//template<typename T> struct ImplicitTypeConvert<NullType, T>
//{
//	typedef T Type;
//};

//template<typename T> struct ImplicitTypeConvert<std::complex<T>, T>
//{
//	typedef std::complex<T> Type;
//};
//template<typename T> struct ImplicitTypeConvert<T, std::complex<T> >
//{
//	typedef std::complex<T> Type;
//};
//
//template<> struct ImplicitTypeConvert<long double, double>
//{
//	typedef long double Type;
//};
//template<> struct ImplicitTypeConvert<double, long double>
//{
//	typedef long double Type;
//};
//
//template<> struct ImplicitTypeConvert<double, int>
//{
//	typedef double Type;
//};
//template<> struct ImplicitTypeConvert<int, double>
//{
//	typedef double Type;
//};
//template<> struct ImplicitTypeConvert<double, long int>
//{
//	typedef double Type;
//};
//template<> struct ImplicitTypeConvert<long int, double>
//{
//	typedef double Type;
//};

}//namespace _detail
} //namespace simpla
//#endif
#endif  // TYPE_CONVERSION_H_
