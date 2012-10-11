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
#include "operation.h"

namespace simpla
{
namespace fetl
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

template<int N, typename T, typename TExpr = NullType> struct nTuple;

typedef nTuple<THREE, Real, NullType> Vec3;
typedef nTuple<THREE, nTuple<THREE, Real, NullType>, NullType> Tensor3;
typedef nTuple<FOUR, nTuple<FOUR, Real, NullType>, NullType> Tensor4;

typedef nTuple<THREE, Integral, NullType> IVec3;
typedef nTuple<THREE, Real, NullType> RVec3;
typedef nTuple<THREE, Complex, NullType> CVec3;

typedef nTuple<THREE, nTuple<THREE, Real, NullType>, NullType> RTensor3;
typedef nTuple<THREE, nTuple<THREE, Complex, NullType>, NullType> CTensor3;

typedef nTuple<FOUR, nTuple<FOUR, Real, NullType>, NullType> RTensor4;
typedef nTuple<FOUR, nTuple<FOUR, Complex, NullType>, NullType> CTensor4;

#define DEFINE_NTUPLE_OP(_TV_ )                                                                  \
template<int N, typename TVL, typename TLExpr,                                                   \
		template<typename, typename > class TOP>                                                 \
struct TypeOpTraits<nTuple<N, TVL, TLExpr>, _TV_, TOP>                                           \
{                                                                                                \
	typedef nTuple<N, typename TOP<TVL, _TV_>::ValueType> ValueType;                             \
};                                                                                               \
template<int N, typename TRExpr, typename TVR,                                                   \
		template<typename, typename > class TOP>                                                 \
struct TypeOpTraits<_TV_, nTuple<N, TVR, TRExpr>, TOP>                                           \
{                                                                                                \
	typedef nTuple<N, typename TypeOpTraits<_TV_, TVR, TOP>::ValueType> ValueType;               \
};
DEFINE_NTUPLE_OP(int)
DEFINE_NTUPLE_OP(Real)
DEFINE_NTUPLE_OP(Complex)
//DEFINE_NTUPLE_OP(RVec3)
//DEFINE_NTUPLE_OP(CVec3)
#undef DEFINE_NTUPLE_OP
template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr,
		template<typename, typename > class TOP>
struct TypeOpTraits<nTuple<N, TVL, TLExpr>, nTuple<N, TVR, TRExpr>, TOP>
{
	typedef nTuple<N, typename TypeOpTraits<TVL, TVR, TOP>::ValueType> ValueType;
};

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct TypeOpTraits<nTuple<N, TVL, TLExpr>, nTuple<N, TVR, TRExpr>,
		vector_calculus::OpDot>
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
};

template<int N, typename T>
struct nTuple<N, T, NullType>
{
	static const int NDIM = N;
	typedef nTuple<NDIM, T, NullType> ThisType;
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

	inline void swap(ThisType & rhs)
	{
		for (int i = 0; i < N; ++i)
		{
			std::swap(v_[i], rhs[i]);
		}
	}

	template<typename TR, typename TExpr>
	inline bool operator ==(nTuple<NDIM, TR, TExpr> const &rhs)
	{
		bool res = true;
		for (int i = 0; i < NDIM; ++i)
		{
			res &= (v_[i] == rhs[i]);
		}
		return (res);
	}

	template<typename TR, typename TExpr>
	inline bool operator !=(nTuple<NDIM, TR, TExpr> const &rhs)
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
	template<typename TR, typename TExpr>
	inline ThisType & operator =(nTuple<NDIM, TR, TExpr> const &rhs)
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

template<int N, typename T>
struct TypeTraits<nTuple<N, T, NullType> >
{
	typedef nTuple<N, T, NullType> & Reference;
	typedef const nTuple<N, T, NullType>& ConstReference;
};

#define IMPLICIT_TYPE_CONVERT                                       \
operator nTuple<NDIM,ValueType,NullType>()                                    \
{                                                                   \
	nTuple<NDIM, ValueType, NullType> res;                                    \
	for (int s = 0; s < NDIM; ++s)                                     \
	{                                                               \
		res[s] = operator[](s);                                     \
                                                                    \
	}                                                               \
	return res;                                                     \
}                                                                   \

template<int N, typename TV, typename TL, template<typename > class TOP>
struct nTuple<N, TV, TOP<nTuple<N, TV, TL> > >
{
	typedef nTuple<N, TV, TOP<nTuple<N, TV, TL> > > ThisType;
	static const int NDIM = N;
	typedef TV ValueType;
	typedef const ThisType ConstReference;
	typename nTuple<N, TV, TL>::ConstReference lhs_;

	nTuple(nTuple<N, TV, TL> const & lhs) :
			lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t const & s) const
	{
		return TOP<TV>::eval(lhs_[s]);
	}

	IMPLICIT_TYPE_CONVERT
};

template<int N, typename TV, typename TL> //
inline nTuple<N, TV, arithmetic::OpNegative<nTuple<N, TV, TL> > >             //
operator -(nTuple<N, TV, TL> const & lhs)
{
	return (nTuple<N, TV, arithmetic::OpNegative<nTuple<N, TV, TL> > >(lhs));
}

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr,
		template<typename, typename > class TOP>
struct nTuple<N, typename TOP<TVL, TVR>::ValueType,
		TOP<nTuple<N, TVL, TLExpr>, nTuple<N, TVR, TRExpr> > >
{
	typedef typename TOP<TVL, TVR>::ValueType ValueType;

	static const int NDIM = N;

	typedef nTuple<N, TVL, TLExpr> TL;
	typedef nTuple<N, TVR, TRExpr> TR;
	typename TypeTraits<nTuple<N, TVL, TLExpr> >::ConstReference lhs_;
	typename TypeTraits<nTuple<N, TVR, TRExpr> >::ConstReference rhs_;

	nTuple(TL const & lhs, TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return TOP<TVL, TVR>::eval(lhs_[s], rhs_[s]);
	}

	IMPLICIT_TYPE_CONVERT
};

template<int N, typename TVL, typename TLExpr, typename TVR, template<typename,
		typename > class TOP>
struct nTuple<N, typename TOP<TVL, TVR>::ValueType,
		TOP<nTuple<N, TVL, TLExpr>, TVR> >
{

	static const int NDIM = N;

	typedef typename TOP<TVL, TVR>::ValueType ValueType;

	typedef nTuple<N, TVL, TLExpr> TL;
	typedef TVR TR;
	typename TypeTraits<nTuple<N, TVL, TLExpr> >::ConstReference lhs_;
	TVR rhs_;
	nTuple(TL const & lhs, TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}
	inline ValueType operator[](size_t s) const
	{
		return TOP<TVL, TVR>::eval(lhs_[s], rhs_);
	}

	IMPLICIT_TYPE_CONVERT
};

template<typename TVL, int N, typename TVR, typename TRExpr, template<typename,
		typename > class TOP>
struct nTuple<N, typename TOP<TVL, TVR>::ValueType,
		TOP<TVL, nTuple<N, TVR, TRExpr> > >
{
	static const int NDIM = N;

	typedef typename TOP<TVL, TVR>::ValueType ValueType;

	typedef TVL TL;
	typedef nTuple<N, TVR, TRExpr> TR;

	TVL lhs_;
	typename TypeTraits<nTuple<N, TVR, TRExpr> >::ConstReference rhs_;

	nTuple(TL const & lhs, TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}
	inline ValueType operator[](size_t s) const
	{
		return TOP<TVL, TVR>::eval(lhs_, rhs_[s]);
	}
	IMPLICIT_TYPE_CONVERT
};

#define DECLARE_NTUPLE_ARITHMETIC(_OP_,_OP_NAME_)                                           \
                                                                                                       \
                                                                                                       \
template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>                          \
inline nTuple<N, typename arithmetic::_OP_NAME_<TVL, TVR>::ValueType,                           \
arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, nTuple<N, TVR, TRExpr> > >                                                        \
_OP_(	nTuple<N, TVL, TLExpr> const &lhs, nTuple<N, TVR, TRExpr> const & rhs)                     \
{                                                                                                      \
                                                                                                       \
	return (nTuple<N,                                                                                  \
			typename arithmetic::_OP_NAME_<TVL, TVR>::ValueType,                                \
			arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, nTuple<N, TVR, TRExpr> > >(lhs, rhs));                                        \
}                                                                                                      \
template<int N, typename TVR, typename TRExpr>                                                         \
inline nTuple<N, typename arithmetic::_OP_NAME_<Real, TVR>::ValueType,                          \
 arithmetic::_OP_NAME_<Real, nTuple<N, TVR, TRExpr> > >                             \
_OP_(	Real const & lhs, nTuple<N, TVR, TRExpr> const &rhs)                                       \
{                                                                                                      \
	return (nTuple<N,                                                                                  \
			typename arithmetic::_OP_NAME_<Real, TVR>::ValueType,                               \
			 arithmetic::_OP_NAME_<Real, nTuple<N, TVR, TRExpr> > >(                        \
			lhs, rhs));                                                                                \
}                                                                                                      \
template<int N, typename TVR, typename TRExpr>                                                         \
inline nTuple<N, typename arithmetic::_OP_NAME_<Complex, TVR>::ValueType,                       \
arithmetic::_OP_NAME_<Complex, nTuple<N, TVR, TRExpr> > >                          \
_OP_(	Complex const & lhs, nTuple<N, TVR, TRExpr> const &rhs)                                    \
{                                                                                                      \
	return (nTuple<N,                                                                                  \
			typename arithmetic::_OP_NAME_<Complex, TVR>::ValueType,                            \
			  arithmetic::_OP_NAME_<Complex, nTuple<N, TVR, TRExpr> > >(                     \
			lhs, rhs));                                                                                \
}                                                                                                      \
template<int N, typename TVR, typename TRExpr>                                                         \
inline nTuple<N, typename arithmetic::_OP_NAME_<int, TVR>::ValueType,                           \
		  arithmetic::_OP_NAME_<int, nTuple<N, TVR, TRExpr> > >                              \
_OP_(	int const & lhs, nTuple<N, TVR, TRExpr> const &rhs)                                        \
{                                                                                                      \
	return (nTuple<N,                                                                                  \
			typename arithmetic::_OP_NAME_<int, TVR>::ValueType,                                \
			  arithmetic::_OP_NAME_<int, nTuple<N, TVR, TRExpr> > >(                         \
			lhs, rhs));                                                                                \
}                                                                                                      \
                                                                                                       \
template<int N, typename TVL, typename TLExpr>                                                         \
inline nTuple<N, typename arithmetic::_OP_NAME_<TVL, int>::ValueType,                           \
		  arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, int > >                              \
_OP_(	nTuple<N, TVL, TLExpr> const & lhs, int const &rhs)                                        \
{                                                                                                      \
	return (nTuple<N,                                                                                  \
			typename arithmetic::_OP_NAME_<TVL, int>::ValueType,                                \
			  arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, int > >(                         \
			lhs, rhs));                                                                                \
}                                                                                                      \
template<int N, typename TVL, typename TLExpr>                                                         \
inline nTuple<N, typename arithmetic::_OP_NAME_<TVL, Real>::ValueType,                          \
		  arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, Real > >                             \
_OP_(	nTuple<N, TVL, TLExpr> const & lhs, Real const &rhs)                                       \
{                                                                                                      \
	return (nTuple<N,                                                                                  \
			typename arithmetic::_OP_NAME_<TVL, Real>::ValueType,                               \
			  arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, Real > >(                        \
			lhs, rhs));                                                                                \
}                                                                                                      \
template<int N, typename TVL, typename TLExpr>                                                         \
inline nTuple<N, typename arithmetic::_OP_NAME_<TVL, Complex>::ValueType,                       \
		  arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, Complex > >                          \
_OP_(	nTuple<N, TVL, TLExpr> const & lhs, Complex const &rhs)                                    \
{                                                                                                      \
	return (nTuple<N,                                                                                  \
			typename arithmetic::_OP_NAME_<TVL, Complex>::ValueType,                            \
			  arithmetic::_OP_NAME_<nTuple<N, TVL, TLExpr>, Complex > >(                     \
			lhs, rhs));                                                                                \
}                                                                                                      \


DECLARE_NTUPLE_ARITHMETIC(operator*, OpMultiplication);
DECLARE_NTUPLE_ARITHMETIC(operator/, OpDivision);
DECLARE_NTUPLE_ARITHMETIC(operator+, OpAddition);
DECLARE_NTUPLE_ARITHMETIC(operator-, OpSubtraction);

#undef DECLARE_NTUPLE_ARITHMETIC

//template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
//struct nTuple<THREE, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
//		vector_calculus::OpCross<nTuple<THREE, TVL, TLExpr>,
//				nTuple<THREE, TVR, TRExpr> > >
//{
//	static const int NDIM = THREE;
//	typedef typename arithmetic::OpMultiplication<TVL, TVR>::ValueType ValueType;
//
//	typedef nTuple<NDIM, TVL, TLExpr> TL;
//	typedef nTuple<NDIM, TVR, TRExpr> TR;
//	typename nTuple<NDIM, TVL, TLExpr>::ConstReference lhs_;
//	typename nTuple<NDIM, TVR, TRExpr>::ConstReference rhs_;
//
//	nTuple(TL const & lhs, TR const & rhs) :
//			lhs_(lhs), rhs_(rhs)
//	{
//	}
//	inline ValueType operator[](size_t s) const
//
//	{
//		return (lhs_[(s + 1) % 3] * rhs_[(s + 2) % 3]
//				- lhs_[(s + 2) % 3] * rhs_[(s + 1) % 3]);
//	}
//
//	IMPLICIT_TYPE_CONVERT
//};
//template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
//inline nTuple<THREE, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
//		vector_calculus::OpCross<nTuple<THREE, TVL, TLExpr>,
//				nTuple<THREE, TVR, TRExpr> > >                                //
//Cross(nTuple<THREE, TVL, TLExpr> const &lhs,
//		nTuple<THREE, TVR, TRExpr> const & rhs)
//{
//	return (nTuple<THREE,
//			typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
//			vector_calculus::OpCross<nTuple<THREE, TVL, TLExpr>,
//					nTuple<THREE, TVR, TRExpr> > >(lhs, rhs));
//}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline nTuple<THREE, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType>  //
Cross(nTuple<THREE, TVL, TLExpr> const &lhs,
		nTuple<THREE, TVR, TRExpr> const & rhs)
{
	nTuple<THREE, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType> res =
	{

	lhs[1] * rhs[2] - lhs[2] * rhs[1],

	lhs[2] * rhs[0] - lhs[0] * rhs[2],

	lhs[0] * rhs[1] - lhs[1] * rhs[0]

	};
	return res;
}

template<int N, typename TVL, typename TL, typename TVR, typename TR>
inline typename arithmetic::OpMultiplication<TVL, TVR>::ValueType //
Dot(nTuple<N, TVL, TL> const &lhs, nTuple<N, TVR, TR> const & rhs)
{
	typedef typename arithmetic::OpMultiplication<TVL, TVR>::ValueType ValueType;
	ValueType res = 0.0;
	for (int i = 0; i < N; ++i)
	{
		res += lhs[i] * rhs[i];
	}
	return (res);
}

template<typename TVL, typename TL, typename TVR, typename TR>
inline typename arithmetic::OpMultiplication<TVL, TVR>::ValueType //
Dot(nTuple<THREE, TVL, TL> const &lhs, nTuple<THREE, TVR, TR> const & rhs)
{
	return (lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]);
}

namespace vector_calculus
{
template<int N, typename TL, typename TLExpr, typename TR, typename TRExpr>
struct OpDot<nTuple<N, TL, TLExpr>, nTuple<N, TR, TRExpr> >
{
	typedef typename arithmetic::OpMultiplication<TL, TR>::ValueType ValueType;
	static ValueType eval(nTuple<N, TL, TLExpr> const & lhs,
			nTuple<N, TR, TRExpr> const & rhs)
	{
		return Dot(lhs, rhs);
	}

};

template<typename TL, typename TLExpr, typename TR, typename TRExpr>
struct OpCross<nTuple<THREE, TL, TLExpr>, nTuple<THREE, TR, TRExpr> >
{
	typedef nTuple<THREE,
			typename arithmetic::OpMultiplication<TL, TR>::ValueType> ValueType;
	static ValueType eval(nTuple<THREE, TL, TLExpr> const & lhs,
			nTuple<THREE, TR, TRExpr> const & rhs)
	{
		return Cross(lhs, rhs);
	}

};

} //namespace vector_calculus

template<int N, typename TVL, typename TL, typename TVR, typename TR>
inline bool operator==(nTuple<N, TVL, TL> const &lhs,
		nTuple<N, TVR, TR> const & rhs)
{
	bool res = true;
	for (int i = 0; i < N; ++i)
	{
		res &= (lhs[i] == rhs[i]);
	}
	return (res);
}

template<typename TVL, typename TL, typename TVR, typename TR>
inline bool operator==(nTuple<THREE, TVL, TL> const &lhs,
		nTuple<THREE, TVR, TR> const & rhs)
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
} //namespace fetl
} //namespace simpla

#endif  // INCLUDE_NTUPLE_H_
