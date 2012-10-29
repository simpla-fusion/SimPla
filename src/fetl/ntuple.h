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
#include <vector>
#include <sstream>
#include "include/simpla_defs.h"
#include "typeconvert.h"

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

typedef nTuple<THREE, Real> Vec3;
typedef nTuple<THREE, nTuple<THREE, Real> > Tensor3;
typedef nTuple<FOUR, nTuple<FOUR, Real> > Tensor4;

typedef nTuple<THREE, Integral> IVec3;
typedef nTuple<THREE, Real> RVec3;
typedef nTuple<THREE, Complex> CVec3;

typedef nTuple<THREE, nTuple<THREE, Real> > RTensor3;
typedef nTuple<THREE, nTuple<THREE, Complex> > CTensor3;

typedef nTuple<FOUR, nTuple<FOUR, Real> > RTensor4;
typedef nTuple<FOUR, nTuple<FOUR, Complex> > CTensor4;

namespace _impl
{
template<typename > class TypeTraits;

#define DEFINE_TYPETRAITS(_TV_)                                           \
template<int N >                                                          \
struct TypeTraits< nTuple<N, _TV_> >                                 \
{                                                                         \
	typedef  nTuple<N, _TV_> & Reference;                            \
	typedef const  nTuple<N, _TV_>& ConstReference;                  \
};                                                                        \
                                                                          \

DEFINE_TYPETRAITS(int)
DEFINE_TYPETRAITS(Real)
DEFINE_TYPETRAITS(Complex)
#undef DEFINE_TYPETRAITS

template<typename, typename > struct TypeConvertTraits;

#define DEFINE_NTUPLE_OP(_TV_ )                                                                         \
template<int N, typename TE>                                                                            \
struct TypeConvertTraits<nTuple<N, TE>, _TV_>                                                        \
{                                                                                                       \
	typedef nTuple<N,                                                                                   \
			typename TypeConvertTraits<_TV_,                                                         \
					typename nTuple<N, TE>::ValueType>::ValueType> ValueType;                           \
};                                                                                                      \
                                                                                                        \
template<int N, typename TE>                                                                            \
struct TypeConvertTraits<_TV_, nTuple<N, TE> >                                                       \
{                                                                                                       \
	typedef nTuple<N,                                                                                   \
			typename TypeConvertTraits<typename nTuple<N, TE>::ValueType,                               \
			_TV_>::ValueType> ValueType;                                                     \
};                                                                                                      \

DEFINE_NTUPLE_OP(int)
DEFINE_NTUPLE_OP(Real)
DEFINE_NTUPLE_OP(Complex)
//DEFINE_NTUPLE_OP(RVec3)
//DEFINE_NTUPLE_OP(CVec3)
#undef DEFINE_NTUPLE_OP

template<int N, typename TLExpr, typename TRExpr>
struct TypeConvertTraits<nTuple<N, TLExpr>, nTuple<N, TRExpr> >
{
	typedef nTuple<N,
			typename TypeConvertTraits<typename nTuple<N, TLExpr>::ValueType,
					typename nTuple<N, TRExpr>::ValueType>::ValueType> ValueType;
};

template<typename T> inline typename T::ValueType eval(T const & nt, size_t s)
{
	return nt[s];
}

template<typename T> inline //
T mapto_(T const & expr, size_t s)
{
	return expr;
}
template<int N, typename T> inline //
typename nTuple<N, T>::ValueType //
mapto_(nTuple<N, T> const & expr, size_t s)
{
	return expr[s];
}
template<int N, typename TL> inline //
typename nTuple<N, OpNegative<TL> >::ValueType //
eval(nTuple<N, OpNegative<TL> > const & expr, size_t s)
{
	return -mapto_(expr.lhs_, s);
}
template<int N, typename TL, typename TR> inline //
typename nTuple<N, OpAddition<TL, TR> >::ValueType //
eval(nTuple<N, OpAddition<TL, TR> > const & expr, size_t s)
{
	return mapto_(expr.lhs_, s) + mapto_(expr.rhs_, s);
}
template<int N, typename TL, typename TR> inline //
typename nTuple<N, OpSubtraction<TL, TR> >::ValueType //
eval(nTuple<N, OpSubtraction<TL, TR> > const & expr, size_t s)
{
	return mapto_(expr.lhs_, s) - mapto_(expr.rhs_, s);
}
template<int N, typename TL, typename TR> inline //
typename nTuple<N, OpMultiplication<TL, TR> >::ValueType //
eval(nTuple<N, OpMultiplication<TL, TR> > const & expr, size_t s)
{
	return mapto_(expr.lhs_, s) * mapto_(expr.rhs_, s);
}
template<int N, typename TL, typename TR> inline //
typename nTuple<N, OpDivision<TL, TR> >::ValueType //
eval(nTuple<N, OpDivision<TL, TR> > const & expr, size_t s)
{
	return mapto_(expr.lhs_, s) / mapto_(expr.rhs_, s);
}

} //namespace  _impl

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
		return res;
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

	inline ThisType & operator =(ValueType rhs)
	{
		for (int i = 0; i < NDIM; ++i)
		{
			v_[i] = rhs;
		}
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator =(TR const *rhs)
	{
		for (int i = 0; i < NDIM; ++i)
		{
			v_[i] = rhs[i];
		}
		return (*this);
	}
	template<typename TExpr>
	inline ThisType & operator =(nTuple<NDIM, TExpr> const &rhs)
	{
		for (int i = 0; i < NDIM; ++i)
		{
			v_[i] = rhs[i];
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
operator nTuple<NDIM,ValueType >()                          \
{                                                                   \
	nTuple<NDIM, ValueType> res;                                    \
	for (int s = 0; s < NDIM; ++s)                                  \
	{                                                               \
		res[s] = operator[](s);                                     \
                                                                    \
	}                                                               \
	return res;                                                     \
}

template<int N, typename TL, template<typename > class TOP>
struct nTuple<N, TOP<nTuple<N, TL> > >
{
	typedef nTuple<N, TOP<nTuple<N, TL> > > ThisType;

	static const int NDIM = N;

	typedef typename TOP<nTuple<N, TL> >::ValueType::ValueType ValueType;

	typename _impl::TypeTraits<nTuple<N, TL> >::ConstReference lhs_;

	nTuple(nTuple<N, TL> const & lhs) :
			lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t const & s) const
	{
		return _impl::eval(*this, s);
	}

	IMPLICIT_TYPE_CONVERT
};

template<int N, typename TL, typename TR,
		template<typename, typename > class TOP>
struct nTuple<N, TOP<TL, TR> >
{
	typedef typename TOP<TL, TR>::ValueType::ValueType ValueType;

	static const int NDIM = N;

	typename _impl::TypeTraits<TL>::ConstReference lhs_;
	typename _impl::TypeTraits<TR>::ConstReference rhs_;

	nTuple(TL const & lhs, TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return _impl::eval(*this, s);
	}

	IMPLICIT_TYPE_CONVERT
};

template<int N, typename TL> //
inline nTuple<N, _impl::OpNegative<nTuple<N, TL> > >                  //
operator -(nTuple<N, TL> const & lhs)
{
	return (nTuple<N, _impl::OpNegative<nTuple<N, TL> > >(lhs));
}

#define DECLARE_NTUPLE_ARITHMETIC_TYPE(_OP_,_OPNAME_,_TYPE_)                                \
template<int N, typename TExpr>                                                             \
inline nTuple<N, _impl::_OPNAME_<_TYPE_, nTuple<N, TExpr> > >                          \
_OP_(_TYPE_  const & lhs, nTuple<N, TExpr> const &rhs)                                      \
{                                                            \
	return nTuple<N, _impl::_OPNAME_<_TYPE_, nTuple<N, TExpr> > >(lhs, rhs);           \
}                                                                                           \
                                                                                            \
template<int N, typename TExpr>                                                             \
inline nTuple<N, _impl::_OPNAME_<nTuple<N, TExpr>, _TYPE_> >                           \
_OP_(nTuple<N, TExpr> const & lhs, _TYPE_ const & rhs)                                      \
{                                                                                         \
	return nTuple<N, _impl::_OPNAME_<nTuple<N, TExpr>, _TYPE_> >(lhs, rhs);            \
}

#define DECLARE_NTUPLE_ARITHMETIC(_OP_,_OPNAME_)                                            \
                                                                                            \
template<int N, typename TLExpr, typename TRExpr>                                           \
inline nTuple<N, _impl::_OPNAME_<nTuple<N, TLExpr>, nTuple<N, TRExpr> > >              \
_OP_(nTuple<N, TLExpr> const &lhs, nTuple<N, TRExpr> const & rhs)                           \
{                                                                                           \
return (nTuple<N,_impl::_OPNAME_<nTuple<N, TLExpr>, nTuple<N, TRExpr> > >(lhs,	rhs));  \
}                                                                                           \
DECLARE_NTUPLE_ARITHMETIC_TYPE(_OP_, _OPNAME_, int)                                         \
DECLARE_NTUPLE_ARITHMETIC_TYPE(_OP_, _OPNAME_, Real)                                        \
DECLARE_NTUPLE_ARITHMETIC_TYPE(_OP_, _OPNAME_, Complex)                                     \

DECLARE_NTUPLE_ARITHMETIC(operator*, OpMultiplication);
DECLARE_NTUPLE_ARITHMETIC(operator/, OpDivision);
DECLARE_NTUPLE_ARITHMETIC(operator+, OpAddition);
DECLARE_NTUPLE_ARITHMETIC(operator-, OpSubtraction);

#undef DECLARE_NTUPLE_ARITHMETIC
#undef DECLARE_NTUPLE_ARITHMETIC_TYPE

template<typename TLExpr, typename TRExpr>
inline nTuple<THREE,
		typename _impl::TypeConvertTraits<
				typename nTuple<THREE, TLExpr>::ValueType,
				typename nTuple<THREE, TRExpr>::ValueType>::ValueType>        //
Cross(nTuple<THREE, TLExpr> const &lhs, nTuple<THREE, TRExpr> const & rhs)
{
	nTuple<THREE,
			typename _impl::TypeConvertTraits<
					typename nTuple<THREE, TLExpr>::ValueType,
					typename nTuple<THREE, TRExpr>::ValueType>::ValueType> res =
	{

	lhs[1] * rhs[2] - lhs[2] * rhs[1],

	lhs[2] * rhs[0] - lhs[0] * rhs[2],

	lhs[0] * rhs[1] - lhs[1] * rhs[0]

	};
	return res;
}

template<int N, typename TL, typename TR>
inline typename _impl::TypeConvertTraits<typename nTuple<N, TL>::ValueType,
		typename nTuple<N, TR>::ValueType>::ValueType Dot(
		nTuple<N, TL> const &lhs, nTuple<N, TR> const & rhs)
{
	typename _impl::TypeConvertTraits<typename nTuple<N, TL>::ValueType,
			typename nTuple<N, TR>::ValueType>::ValueType res = 0.0;

	for (int i = 0; i < N; ++i)
	{
		res += lhs[i] * rhs[i];
	}
	return (res);
}

template<typename TL, typename TR>
inline typename _impl::TypeConvertTraits<typename nTuple<THREE, TL>::ValueType,
		typename nTuple<THREE, TR>::ValueType>::ValueType //
Dot(nTuple<THREE, TL> const &lhs, nTuple<THREE, TR> const & rhs)
{
	return (lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]);
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
inline bool operator==(nTuple<THREE, TL> const &lhs,
		nTuple<THREE, TR> const & rhs)
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
	return res;
}

template<typename T> inline typename nTuple<3, nTuple<3, T> >::ValueType //
Determinant(nTuple<3, nTuple<3, T> > const & m)
{
	return ( //
	m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
	* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1] //
			* m[0][2] - m[1][2] * m[2][1] * m[0][0] //
	);
}

template<typename T> inline typename nTuple<4, nTuple<4, T> >::ValueType //
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

} //namespace simpla

#endif  // INCLUDE_NTUPLE_H_
