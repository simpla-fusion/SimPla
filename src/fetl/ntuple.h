/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: nTuple.h 990 2010-12-14 11:06:21Z salmon $
 * nTuple.h
 *
 *  Created on: Jan 27, 2010
 *      Author: yuzhi
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

#ifndef INCLUDE_NTUPLE_H_
#define INCLUDE_NTUPLE_H_
#include <iostream>
#include <complex>
#include <sstream>

#include <utility>
#include "expression.h"
namespace simpla
{

/**
 *  nTuple :n-tuple
 *  @Semantics:
 *    n-tuple is a sequence (or ordered list) of n elements, where n is a positive
 *    integer. There is also one 0-tuple, an empty sequence. An n-tuple is defined
 *    inductively using the construction of an ordered pair. Tuples are usually
 *    written by listing the elements within parenthesis '( )' and separated by
 *    commas; for example, (2, 7, 4, 1, 7) denotes a 5-tuple.
 *    @url: http://en.wikipedia.org/wiki/Tuple
 *  @Implement
 *   template<int n,typename T> struct nTuple;
 *   nTuple<5,double> t={1,2,3,4,5};
 *
 * */

template<int N, typename T> struct nTuple;

//--------------------------------------------------------------------------------------------
template<int N, typename T>
struct nTuple
{
	static const int NDIM = N;
	typedef nTuple<NDIM, T> ThisType;
	typedef const ThisType & ConstReference;
	typedef T ValueType;

	ValueType v_[N];

	inline ValueType &
	operator[](int i)
	{
		return (v_[i]);
	}

	inline ValueType const&
	operator[](int i) const
	{
		return (v_[i]);
	}

	template<typename TR>
	inline operator nTuple<N,TR>() const
	{
		nTuple<N, TR> res;
		for (int i = 0; i < N; ++i)
		{
			res[i] = v_[i];
		}
		return (res);
	}

	inline void swap(ThisType & rhs)
	{
		for (int i = 0; i < N; ++i)
		{
			std::swap(v_[i], rhs[i]);
		}
	}

	template<typename TExpr>
	inline bool operator ==(nTuple<NDIM, TExpr> const &rhs)
	{
		bool res = true;
		for (int i = 0; i < NDIM; ++i)
		{
			res &= (v_[i] == rhs[i]);
		}
		return (res);
	}

	template<typename TExpr>
	inline bool operator !=(nTuple<NDIM, TExpr> const &rhs)
	{
		return (!(*this == rhs));
	}

	template<typename TR>
	inline ThisType & operator =(TR const &rhs)
	{
		for (int i = 0; i < NDIM; ++i)
		{
			v_[i] = index(rhs, i);
		}
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator =(nTuple<NDIM, TR> const &rhs)
	{
		for (int i = 0; i < NDIM; ++i)
		{
			v_[i] = index(rhs, i);
		}
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

#define IMPLICIT_TYPE_CONVERT                                       \
operator nTuple<NDIM,Value >()                          \
{                                                                   \
	nTuple<NDIM, Value> res;                                    \
	for (int s = 0; s < NDIM; ++s)                                  \
	{                                                               \
		res[s] = operator[](s);                                     \
                                                                    \
	}                                                               \
	return res;                                                     \
}

template<typename TLExpr, typename TRExpr>
inline auto Cross(nTuple<3, TLExpr> const &lhs, nTuple<3, TRExpr> const & rhs)
->nTuple<3,decltype(lhs[0]*rhs[0])>

{
	nTuple<3, decltype(lhs[0]*rhs[0])> res =
	{

	lhs[1] * rhs[2] - lhs[2] * rhs[1],

	lhs[2] * rhs[0] - lhs[0] * rhs[2],

	lhs[0] * rhs[1] - lhs[1] * rhs[0]

	};
	return (res);
}

template<int N, typename TL, typename TR> inline auto //
Dot(nTuple<N, TL> const &lhs,
		nTuple<N, TR> const &rhs)->decltype(lhs[0] * rhs[0])
{
	decltype(lhs[0] * rhs[0]) res = 0.0;

	for (int i = 0; i < N; ++i)
	{
		res += lhs[i] * rhs[i];
	}
	return (res);
}

//#define BI_FUN_OP(_OP_NAME_,_FUN_)                                                       \
//                                                                                     \
//struct Op##_OP_NAME_                                                                 \
//{                                                                                    \
//	template<typename TL, typename TR> inline static auto eval(TL const & l,         \
//			TR const &r)->decltype(_FUN_(l, r))                                        \
//	{                                                                                \
//		return (_FUN_(l, r));                                               \
//	}                                                                                \
//};                                                                                   \
//template<typename TL, typename TR> inline typename std::enable_if<                   \
//		!(is_arithmetic_scalar<TL>::value && is_arithmetic_scalar<TR>::value),       \
//		BiOp<Op##_OP_NAME_, TL, TR> >::type                                        \
//_OP_NAME_(TL const &lhs, TR const & rhs)                                                   \
//{                                                                                    \
//	return (BiOp<Op##_OP_NAME_, TL, TR>(lhs, rhs));                                  \
//}                                                                                    \
//
//BI_FUN_OP(Dot, Dot)
//BI_FUN_OP(Cross, Cross)
//
//#undef BI_FUN_OP

//template<typename TL, typename TR>
//inline auto Dot(nTuple<3, TL> const &lhs, nTuple<3, TR> const & rhs)
//DECL_RET_TYPE( (lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]))

template<int N, typename TL, typename TR>
inline bool operator==(nTuple<N, TL> const &lhs, nTuple<N, TR> const & rhs)
{
	bool res = true;
	for (int i = 0; i < N; ++i)
	{
		res &= (lhs[i] == rhs[i]);
	}
	return (res);
}

template<typename TL, typename TR>
inline bool operator==(nTuple<3, TL> const &lhs, nTuple<3, TR> const & rhs)
{

	return ((lhs[0] == rhs[0]) && (lhs[1] == rhs[1]) && (lhs[2] == rhs[2]));
}

template<int N, typename T> std::ostream &
operator<<(std::ostream& os, const nTuple<N, T> & tv)
{
	os << tv[0];
	for (int i = 1; i < N; ++i)
	{
		os << " " << tv[i];
	}
	return (os);
}

template<int N, typename T> std::istream &
operator>>(std::istream& is, nTuple<N, T> & tv)
{
	for (int i = 0; i < N && is; ++i)
	{
		is >> tv[i];
	}

	return (is);
}

template<int N, typename T> nTuple<N, T> ToNTuple(std::string const & str)
{
	std::istringstream ss(str);
	nTuple<N, T> res;
	ss >> res;
	return (res);
}

template<typename T> inline typename nTuple<3, nTuple<3, T> >::Value //
Determinant(nTuple<3, nTuple<3, T> > const & m)
{
	return ( //
	m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
	* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1] //
			* m[0][2] - m[1][2] * m[2][1] * m[0][0] //
	);
}

template<typename T> inline typename nTuple<4, nTuple<4, T> >::Value //
Determinant(nTuple<4, nTuple<4, T> > const & m)
{
	return ( //
	m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] //
			* m[3][0] - m[0][3] * m[1][1] * m[2][2] * m[3][0]
			+ m[0][1] * m[1][3] //
					* m[2][2] * m[3][0] + m[0][2] * m[1][1] * m[2][3] * m[3][0]
			- m[0][1] //
			* m[1][2] * m[2][3] * m[3][0]
			- m[0][3] * m[1][2] * m[2][0] * m[3][1] //
	+ m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2] //
			* m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1]
			- m[0][2] * m[1][0] //
					* m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1]
			+ m[0][3] //
			* m[1][1] * m[2][0] * m[3][2]
			- m[0][1] * m[1][3] * m[2][0] * m[3][2] //
			- m[0][3] * m[1][0] * m[2][1] * m[3][2]
			+ m[0][0] * m[1][3] * m[2][1] //
					* m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2]
			- m[0][0] * m[1][1] //
					* m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3]
			+ m[0][1] //
			* m[1][2] * m[2][0] * m[3][3]
			+ m[0][2] * m[1][0] * m[2][1] * m[3][3] //
	- m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2] //
			* m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3] //
	);
}

template<int N, typename T> auto abs(
		nTuple<N, T> const & m)-> decltype(std::abs(std::sqrt(Dot(m, m))))
{
	return std::abs(std::sqrt(Dot(m, m)));
}

template<int N, typename T> auto abs(
		nTuple<N, nTuple<N, T> > const & m)
		->typename std::remove_const<typename std::remove_reference<decltype(m[0][0])>::type>::type
{
	typedef typename std::remove_const<
			typename std::remove_reference<decltype(m[0][0])>::type>::type ValueType;

	nTuple<N, ValueType> res;

	for (size_t i = 0; i < N; ++i)
	{
		res[i] = Dot(m[i], m[i]);
	}

	return (sqrt(Dot(res, res)));
}

} //namespace simpla

#endif  // INCLUDE_NTUPLE_H_
