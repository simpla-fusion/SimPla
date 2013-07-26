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
#include <type_traits>
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
	operator[](size_t i)
	{
		return (v_[i]);
	}

	inline ValueType const&
	operator[](size_t i) const
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
//	template<typename TR> inline typename std::enable_if<
//	!is_indexable<TR>::value, ThisType &>::type //
//	operator =(TR const &rhs)
//	{
//		for (int i = 0; i < NDIM; ++i)
//		{
//			v_[i] = rhs;
//		}
//		return (*this);
//	}

	template<typename TR> inline ThisType & //
	operator =(TR const &rhs)
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

template<int N, typename T>
struct is_storage_type<nTuple<N, T> >
{
	static const bool value = is_storage_type<T>::value;
};

template<int N>
struct is_storage_type<nTuple<N, double> >
{
	static const bool value = true;
};

template<int N>
struct is_storage_type<nTuple<N, std::complex<double> > >
{
	static const bool value = true;
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

template<typename T> inline auto //
Determinant(
		nTuple<3, nTuple<3, T> > const & m) //
				DECL_RET_TYPE(//
						(
								m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1]//
								* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
								* m[0][2] - m[1][2] * m[2][1] * m[0][0]//
						)
						//
				)

template<typename T> inline auto //
Determinant(nTuple<4, nTuple<4, T> > const & m) DECL_RET_TYPE(
		(//
		m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1]//
		* m[3][0] - m[0][3] * m[1][1] * m[2][2] * m[3][0]
		+ m[0][1] * m[1][3]//
		* m[2][2] * m[3][0] + m[0][2] * m[1][1] * m[2][3] * m[3][0]
		- m[0][1]//
		* m[1][2] * m[2][3] * m[3][0]
		- m[0][3] * m[1][2] * m[2][0] * m[3][1]//
		+ m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2]//
		* m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1]
		- m[0][2] * m[1][0]//
		* m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1]
		+ m[0][3]//
		* m[1][1] * m[2][0] * m[3][2]
		- m[0][1] * m[1][3] * m[2][0] * m[3][2]//
		- m[0][3] * m[1][0] * m[2][1] * m[3][2]
		+ m[0][0] * m[1][3] * m[2][1]//
		* m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2]
		- m[0][0] * m[1][1]//
		* m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3]
		+ m[0][1]//
		* m[1][2] * m[2][0] * m[3][3]
		+ m[0][2] * m[1][0] * m[2][1] * m[3][3]//
		- m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2]//
		* m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3]//
))

template<int N, typename T> auto abs(
		nTuple<N, T> const & m)-> decltype(std::abs(std::sqrt(Dot(m, m))))
{
	return std::abs(std::sqrt(Dot(m, m)));
}

template<int N, typename T> auto abs(nTuple<N, nTuple<N, T> > const & m)
DECL_RET_TYPE( (sqrt(Determinant(m))))

// Expression template of nTuple

#define DEF_BIOP_CLASS(_NAME_,_OP_)                                                  \
template<typename, typename > struct Op##_NAME_;      \
template<int N, typename TL, typename TR>                              \
struct nTuple<N, Op##_NAME_<TL, TR> >                    \
{                                                                                   \
	typename ConstReferenceTraits<TL>::type l_;                                     \
	typename ConstReferenceTraits<TR>::type r_;                                     \
                                                                                    \
    nTuple(TL const & l, TR const & r) :                                             \
			 l_(l), r_(r)                                 \
	{                                                                               \
	}                                                                               \
	inline auto operator[](size_t s) const                                          \
	DECL_RET_TYPE((index(l_,s) _OP_ index(r_,s)))                              \
                                                                                    \
};\

DEF_BIOP_CLASS(PlusNTuple, +)
DEF_BIOP_CLASS(MinusNTuple, -)
DEF_BIOP_CLASS(MultipliesNTuple, *)
DEF_BIOP_CLASS(DividesNTuple, /)
#undef DEF_BIOP_CLASS

template<int N, typename TL, typename TR> inline auto   //
Plus(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE(
		( nTuple<N ,
				OpPlusNTuple<nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Plus(nTuple<N, TL> const & lhs, TR const & rhs)
ENABLE_IF_DECL_RET_TYPE((is_arithmetic_scalar<TR>::value) ,
		(nTuple<N,OpPlusNTuple<nTuple<N, TL>,TR > > (lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Plus(TL const & lhs, nTuple<N, TR> const & rhs)
ENABLE_IF_DECL_RET_TYPE(( is_arithmetic_scalar<TL>::value) ,
		(nTuple<N,OpPlusNTuple<TL,nTuple<N, TR> > > (lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Minus(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE(
		( nTuple<N ,
				OpMinusNTuple<nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Minus(nTuple<N, TL> const & lhs, TR const & rhs)
ENABLE_IF_DECL_RET_TYPE((is_arithmetic_scalar<TR>::value) ,
		(nTuple<N,OpMinusNTuple<nTuple<N, TL>,TR > > (lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Minus(TL const & lhs, nTuple<N, TR> const & rhs)
ENABLE_IF_DECL_RET_TYPE(( is_arithmetic_scalar<TL>::value) ,
		(nTuple<N,OpMinusNTuple<TL,nTuple<N, TR> > > (lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Multiplies(nTuple<N, TL> const & lhs, TR const & rhs)
ENABLE_IF_DECL_RET_TYPE((is_arithmetic_scalar<TR>::value) ,
		(nTuple<N,OpMultipliesNTuple<nTuple<N, TL>,TR > > (lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Multiplies(TL const & lhs, nTuple<N, TR> const & rhs)
ENABLE_IF_DECL_RET_TYPE(( is_arithmetic_scalar<TL>::value) ,
		(nTuple<N,OpMultipliesNTuple<TL,nTuple<N, TR> > > (lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Multiplies(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE(
		(nTuple<N,OpMultipliesNTuple<
				nTuple<N, TL>,
				nTuple<N, TR> > > (lhs, rhs)))

template<int N, typename TL, typename TR> inline auto   //
Divides(nTuple<N, TL> const & lhs, TR const & rhs)
ENABLE_IF_DECL_RET_TYPE(( is_arithmetic_scalar<TR>::value) ,
		(nTuple<N,OpDividesNTuple<nTuple<N, TL>,TR > > (lhs, rhs)))

template<typename > struct OpNegateNTuple;
template<int N, typename TL>
struct nTuple<N, OpNegateNTuple<TL> >
{
	typename ConstReferenceTraits<TL>::type expr;
	nTuple(TL const & l) :
			expr(l)
	{
	}
	inline auto operator[](size_t s) const
	DECL_RET_TYPE((- index(expr,s)))
};
template<int N, typename TL> inline  //
auto Negate(nTuple<N, TL> const & f)
DECL_RET_TYPE(( nTuple<N, OpNegateNTuple<nTuple<N, TL> > > (f)))

template<typename TLExpr, typename TRExpr> struct OpCrossNTuple;
template<int N, typename TL, typename TR>
struct nTuple<N, OpCrossNTuple<TL, TR> >
{

	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;

	nTuple(TL const & l, TR const & r) :
			l_(l), r_(r)
	{
	}

	inline auto operator[](size_t s) const
	DECL_RET_TYPE((l_[(s+1)%3] * r_[(s+2)%3] - l_[(s+2)%3] * r_[(s+1)%3]))

}
;
template<int N, typename TL, typename TR> inline auto   //
Cross(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE((nTuple<N,
				OpCrossNTuple<nTuple<N, TL>,nTuple<N, TR> > > (lhs, rhs)))

namespace _impl
{
template<int M, typename TL, typename TR> struct _dot
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE((l[M - 1] * r[M - 1] + _dot<M - 1, TL, TR>::eval(l, r)))
};
template<typename TL, typename TR> struct _dot<1, TL, TR>
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE(l[0]*r[0])
};
}   //namespace _impl

template<int N, typename TL, typename TR> inline auto //
Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE( (_impl::_dot<N,nTuple<N, TL>,nTuple<N, TR> >::eval(l,r)))

}
//namespace simpla

#endif  // INCLUDE_NTUPLE_H_
