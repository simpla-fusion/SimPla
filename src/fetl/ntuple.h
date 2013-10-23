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

namespace _impl
{

template<int M, typename TL, typename TR> struct _swap
{
	static inline void eval(TL const & l, TR const &r)
	{
		std::swap(l[M - 1], r[M - 1]);
		_swap<M - 1, TL, TR>::eval(l, r);
	}
};
template<typename TL, typename TR> struct _swap<1, TL, TR>
{
	static inline void eval(TL const & l, TR const &r)
	{
		std::swap(l[0], r[0]);
	}
};

template<int M, typename TL, typename TR> struct _assign
{
	static inline void eval(TL & l, TR const &r)
	{
		l[M - 1] = r[M - 1];
		_assign<M - 1, TL, TR>::eval(l, r);
	}
};
template<typename TL, typename TR> struct _assign<1, TL, TR>
{
	static inline void eval(TL & l, TR const &r)
	{
		l[0] = r[0];
	}
};

template<int M, typename TL, typename TR> struct _equal
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE((l[M - 1] ==r[M - 1] && _equal<M - 1, TL, TR>::eval(l, r)))
};
template<typename TL, typename TR> struct _equal<1, TL, TR>
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE(l[0]==r[0])
};
}  // namespace _impl
//--------------------------------------------------------------------------------------------
template<int N, typename T>
struct nTuple
{
	static const int NDIM = N;
	typedef nTuple<NDIM, T> ThisType;
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
		_impl::_assign<N, nTuple<N, TR>, ThisType>::eval(res, v_);
		return (res);
	}

	inline void swap(ThisType & rhs)
	{
		_impl::_swap<N, ThisType, ThisType>::eval(*this, rhs);
	}

	template<typename TR>
	inline bool operator ==(TR const &rhs) const
	{
		return (_impl::_equal<N, ThisType, TR>::eval(*this, rhs));
	}

	template<typename TExpr>
	inline bool operator !=(nTuple<NDIM, TExpr> const &rhs) const
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
		for (int i = 0; i < N; ++i)
		{
			v_[i] = rhs;
		}

		return (*this);
	}

	template<typename TR> inline ThisType & //
	operator =(TR rhs[])
	{
		for (int i = 0; i < N; ++i)
		{
			v_[i] = rhs[i];
		}

		return (*this);
	}
	template<typename TR> inline ThisType & //
	operator =(nTuple<N, TR> const &rhs)
	{
		_impl::_assign<N, ThisType, nTuple<N, TR>>::eval(*this, rhs);

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

template<typename T> struct is_nTuple
{
	static const bool value = false;
};

template<int N, typename TE> struct is_nTuple<nTuple<N, TE> >
{
	static const bool value = true;
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
struct is_storage_type<nTuple<N, int> >
{
	static const bool value = true;
};
template<int N>
struct is_storage_type<nTuple<N, std::complex<double> > >
{
	static const bool value = true;
};

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

// overloading operators
template<int N, template<typename, typename > class TOP, typename TL,
		typename TR>
struct nTuple<N, BiOp<TOP, TL, TR> >
{
	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;

	nTuple(TL const & l, TR const & r) :
			l_(l), r_(r)
	{
	}
	inline auto operator[](size_t s) const
	DECL_RET_TYPE((TOP<TL,TR>::eval(l_,r_,s)))

};

// Expression template of nTuple
#define _DEFINE_BINARY_OPERATOR(_NAME_,_OP_)                                                \
template<typename TL,typename TR> class Op##_NAME_;                                \
template<int N, typename TL, typename TR>                                           \
struct Op##_NAME_<nTuple<N, TL>, nTuple<N, TR> >                                    \
{                                                                                   \
	static inline auto eval(nTuple<N, TL> const & l, nTuple<N, TR> const &r,        \
			size_t s) DECL_RET_TYPE ((l[s] _OP_ r[s]))                              \
};                                                                                  \
                                                                                    \
template<int N, typename TL, typename TR>                                           \
struct Op##_NAME_<nTuple<N, TL>, TR>                                                \
{                                                                                   \
	static inline auto eval(nTuple<N, TL> const & l, TR const &r, size_t s)         \
	DECL_RET_TYPE ((l[s] _OP_ r))                                                   \
                                                                                    \
};                                                                                  \
                                                                                    \
template<int N, typename TL, typename TR>                                           \
struct Op##_NAME_<TL, nTuple<N, TR> >                                               \
{                                                                                   \
	static inline auto eval(TL const & l, nTuple<N, TR> const &r, size_t s)         \
	DECL_RET_TYPE ((l _OP_ r[s]))                                                   \
};                                                                                  \
                                                                                    \
                                                                                   \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)                   \
DECL_RET_TYPE(                                                                     \
		(nTuple<N, BiOp<Op##_NAME_ ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)))             \
                                                                                   \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (nTuple<N, TL> const & lhs, TR const & rhs)                              \
DECL_RET_TYPE((nTuple<N,BiOp<Op##_NAME_ ,nTuple<N, TL>,TR > > (lhs,rhs)))                    \
                                                                                   \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (TL const & lhs, nTuple<N, TR> const & rhs)                              \
DECL_RET_TYPE((nTuple<N,BiOp<Op##_NAME_,TL,nTuple<N, TR> > > (lhs,rhs)))                    \
                                                                                   \

_DEFINE_BINARY_OPERATOR(Plus, +)
_DEFINE_BINARY_OPERATOR(Minus, -)
_DEFINE_BINARY_OPERATOR(Multiplies, *)
_DEFINE_BINARY_OPERATOR(Divides, /)
_DEFINE_BINARY_OPERATOR(Modulus, %)
_DEFINE_BINARY_OPERATOR(BitwiseXOR, ^)
_DEFINE_BINARY_OPERATOR(BitwiseAND, &)
_DEFINE_BINARY_OPERATOR(BitwiseOR, |)
#undef _DEFINE_BINARY_OPERATOR

template<int N, template<typename > class TOP, typename TL>
struct nTuple<N, UniOp<TOP, TL> >
{
	typename ConstReferenceTraits<TL>::type l_;

	nTuple(TL const & l) :
			l_(l)
	{
	}
	inline auto operator[](size_t s) const
	DECL_RET_TYPE((TOP<TL >::eval(l_ ,s)))

};
template<typename > struct OpNegate;
template<int N, typename TL>
struct OpNegate<nTuple<N, TL> >
{
	static inline auto eval(nTuple<N, TL> const & l, size_t s)
	DECL_RET_TYPE ((-l[s] ))
};
template<int N, typename TL> inline  //
auto operator-(nTuple<N, TL> const & f)
DECL_RET_TYPE(( nTuple<N, UniOp<OpNegate,nTuple<N, TL> > > (f)))

template<int N, typename TL> inline  //
auto operator+(nTuple<N, TL> const & f)
DECL_RET_TYPE(f)

template<typename, typename > struct OpCross;
template<int N, typename TL, typename TR>
struct OpCross<nTuple<N, TL>, nTuple<N, TR> >
{
	static inline auto eval(nTuple<N, TL> const & l, nTuple<N, TR> const &r,
			size_t s)
					DECL_RET_TYPE ((l[(s+1)%3] * r[(s+2)%3] - l[(s+2)%3] * r[(s+1)%3]))
};

template<int N, typename TL, typename TR> inline auto   //
Cross(nTuple<N, TL> const & lhs,
		nTuple<N, TR> const & rhs)
				DECL_RET_TYPE(
						(nTuple<N,BiOp<OpCross, nTuple<N, TL>,nTuple<N, TR> > > (lhs, rhs)))

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
}
;
}   //namespace _impl

template<int N, typename TL, typename TR> inline auto //
Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((_impl::_dot<N,nTuple<N, TL>,nTuple<N, TR> >::eval(l,r)))

template<int N, typename T> using Matrix = nTuple<N,nTuple<N,T> >;

}
//namespace simpla

#endif  // INCLUDE_NTUPLE_H_
