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
#include "primitives/operation.h"

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

template<int N, typename T, typename TExpr = NullType> struct nTuple;

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

template<int N, typename TVL, typename TLExpr, typename TVR, typename TOP>
struct TypeOpTraits<nTuple<N, TVL, TLExpr> , TVR, TOP>
{
	typedef typename BiOp<nTuple<N, TVL, TLExpr> ,TVR ,TOP>::ResultType ValueType;
};
template<int N, typename TVL, typename TRExpr, typename TVR, typename TOP>
struct TypeOpTraits<TVL, nTuple<N, TVR, TRExpr> , TOP>
{
	typedef typename BiOp<TVR, nTuple<N, TVR, TRExpr> ,TOP>::ResultType ValueType;
};

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr,
		typename TOP>
struct TypeOpTraits<nTuple<N, TVL, TLExpr> , nTuple<N, TVR, TRExpr> , TOP>
{
	typedef typename BiOp<nTuple<N, TVL, TLExpr> , nTuple<N, TVR, TRExpr> , TOP>::ResultType ValueType;
};

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct TypeOpTraits<nTuple<N, TVL, TLExpr> , nTuple<N, TVR, TRExpr>
		, vector_calculus::OpDot>
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
};

template<int N, typename TVL, typename TLExpr>
struct TypeOpTraits<nTuple<N, TVL, TLExpr> , NullType, arithmetic::OpNegative>
{
	typedef nTuple<N, TVL,
			UniOp<nTuple<N, TVL, TLExpr> , arithmetic::OpNegative> > ValueType;
};

template<int N, typename T, typename TL, typename TR, typename TOP>
struct nTuple<N, T, BiOp<TL, TR, TOP> >
{
	typedef BiOp<TL, TR, TOP> OpType;

	typedef nTuple<BiOp<TL, TR, TOP>::NDIM
			, typename BiOp<TL, TR, TOP>::ValueType,
			BiOp<TL, TR, TOP> > ConstReference;

	typename OpType::LReference lhs_;
	typename OpType::RReference rhs_;

	nTuple(typename OpType::TL const &lhs, typename OpType::TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}

	inline typename OpType::ValueType operator[](size_t s) const
	{
		return (OpType::op(lhs_, rhs_, s));
	}

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

template<int N, typename TV, typename TL>
struct nTuple<N, TV, UniOp<nTuple<N, TV, TL> , arithmetic::OpNegative> >
{
	typedef nTuple<N, TV, UniOp<nTuple<N, TV, TL> , arithmetic::OpNegative> > ThisType;
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
		return (-lhs_[s]);
	}

};

template<int N, typename TV, typename TL> //
inline nTuple<N, TV, UniOp<nTuple<N, TV, TL> , arithmetic::OpNegative> >  //
operator -(nTuple<N, TV, TL> const & lhs)
{
	return (nTuple<N, TV, UniOp<nTuple<N, TV, TL> , arithmetic::OpNegative> >(
			lhs));
}

#define DECLARE_NTUPLE_ARITHMETIC(_OP_,_OP_NAME_)                                                                                                \
                                                                                                                                                 \
template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>                                                                    \
struct BiOp<nTuple<N, TVL, TLExpr> ,nTuple<N, TVR, TRExpr>                                                                                       \
		,arithmetic::_OP_NAME_>                                                                                                           \
{                                                                                                                                                \
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::_OP_NAME_>::ValueType ValueType;                                                          \
                                                                                                                                                 \
	typedef BiOp<nTuple<N, TVL, TLExpr> ,nTuple<N, TVR, TRExpr>                                                                                  \
			,arithmetic::_OP_NAME_> ThisType;                                                                                             \
                                                                                                                                                 \
	typedef nTuple<N, ValueType, ThisType> ResultType;                                                                                           \
                                                                                                                                                 \
	static const int NDIM = N;                                                                                                                   \
                                                                                                                                                 \
	typedef nTuple<N, TVL, TLExpr> TL;                                                                                                           \
	typedef nTuple<N, TVR, TRExpr> TR;                                                                                                           \
	typedef typename nTuple<N, TVL, TLExpr>::ConstReference LReference;                                                                          \
	typedef typename nTuple<N, TVR, TRExpr>::ConstReference RReference;                                                                          \
                                                                                                                                                 \
	static inline ValueType op(TL const & lhs, TR const &rhs,size_t s)                                                                                    \
	{                                                                                                                                            \
		return (lhs[s] _OP_ rhs[s]);                                                                                                                      \
	}                                                                                                                                            \
                                                                                                                                                 \
};                                                                                                                                               \
                                                                                                                                                 \
template<int N, typename TVL, typename TL, typename TVR, typename TR>                                                                            \
inline typename BiOp<nTuple<N, TVL, TL> ,nTuple<N, TVR, TR>                                                                                      \
		,arithmetic::_OP_NAME_>::ResultType                                                                                             \
operator _OP_(nTuple<N, TVL, TL> const &lhs , nTuple<N, TVR, TR> const & rhs)                                                                       \
{                                                                                                                                                \
	typedef typename BiOp<nTuple<N, TVL, TL> ,nTuple<N, TVR, TR>                                                                                 \
			,arithmetic::_OP_NAME_>::ResultType ResultType;                                                                               \
	return (ResultType(lhs, rhs));                                                                                                               \
}                                                                                                                                                \
                                                                                                                                                 \
template<typename TVL, int N, typename TVR, typename TRExpr>                                                                                     \
struct BiOp<TVL, nTuple<N, TVR, TRExpr> ,arithmetic::_OP_NAME_>                                                                           \
{                                                                                                                                                \
	typedef BiOp<TVL, nTuple<N, TVR, TRExpr> ,arithmetic::_OP_NAME_> ThisType;                                                            \
                                                                                                                                                 \
	static const int NDIM = N;                                                                                                                   \
                                                                                                                                                 \
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::_OP_NAME_>::ValueType ValueType;                                                          \
                                                                                                                                                 \
	typedef nTuple<NDIM, ValueType, ThisType> ResultType;                                                                                        \
                                                                                                                                                 \
	typedef TVL TL;                                                                                                                              \
	typedef nTuple<N, TVR, TRExpr> TR;                                                                                                           \
                                                                                                                                                 \
	typedef TVL LReference;                                                                                                                      \
	typedef typename nTuple<N, TVR, TRExpr>::ConstReference RReference;                                                                          \
                                                                                                                                                 \
	static inline ValueType op(TL const & lhs, TR const &rhs,size_t s)                                                                                    \
	{                                                                                                                                            \
		return (lhs _OP_ rhs[s]);                                                                                                                      \
	}                                                                                                                                            \
};                                                                                                                                               \
                                                                                                                                                 \
template<typename TVL, int N, typename TVR, typename TR>                                                                                         \
inline typename BiOp<TVL, nTuple<N, TVR, TR> ,arithmetic::_OP_NAME_>::ResultType                                                        \
operator _OP_(TVL const & lhs, nTuple<N, TVR, TR> const &rhs)                                                                                       \
{                                                                                                                                                \
	typedef typename BiOp<TVL, nTuple<N, TVR, TR> ,arithmetic::_OP_NAME_>::ResultType ResultType;                                         \
	return (ResultType(lhs, rhs));                                                                                                               \
}                                                                                                                                                \
                                                                                                                                                 \
template<int N, typename TVL, typename TLExpr, typename TVR>                                                                                     \
struct BiOp<nTuple<N, TVL, TLExpr> ,TVR, arithmetic::_OP_NAME_>                                                                           \
{                                                                                                                                                \
	typedef BiOp<nTuple<N, TVL, TLExpr> , TVR ,arithmetic::_OP_NAME_> ThisType;                                                           \
                                                                                                                                                 \
	static const int NDIM = N;                                                                                                                   \
                                                                                                                                                 \
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::_OP_NAME_>::ValueType ValueType;                                                          \
                                                                                                                                                 \
	typedef nTuple<NDIM, ValueType, ThisType> ResultType;                                                                                        \
                                                                                                                                                 \
	typedef nTuple<N, TVL, TLExpr> TL;                                                                                                           \
	typedef TVR TR;                                                                                                                              \
	typedef typename nTuple<N, TVL, TLExpr>::ConstReference LReference;                                                                               \
	typedef TVR RReference;                                                                                                                      \
                                                                                                                                                 \
	static inline ValueType op(TL const & lhs, TR const &rhs,size_t s)                                                                                    \
	{                                                                                                                                            \
		return (lhs[s] _OP_ rhs);                                                                                                                      \
	}                                                                                                                                            \
                                                                                                                                                 \
};                                                                                                                                               \
template<typename TL, int N, typename TVL, typename TVR>                                                                                         \
inline typename BiOp<nTuple<N, TVL, TL> ,TVR, arithmetic::_OP_NAME_>::ResultType                                                        \
operator _OP_(nTuple<N, TVL, TL> const & lhs,TVR const &rhs)                                                                                        \
{                                                                                                                                                \
	typedef typename BiOp<nTuple<N, TVL, TL> ,TVR, arithmetic::_OP_NAME_>::ResultType ResultType;                                         \
	return (ResultType(lhs, rhs));                                                                                                               \
}                                                                                                                                                \

DECLARE_NTUPLE_ARITHMETIC(*, OpMultiplication);
DECLARE_NTUPLE_ARITHMETIC(/, OpDivision);
DECLARE_NTUPLE_ARITHMETIC(+, OpAddition);
DECLARE_NTUPLE_ARITHMETIC(-, OpSubtraction);

#undef DECLARE_NTUPLE_ARITHMETIC

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<nTuple<THREE, TVL, TLExpr> ,nTuple<THREE, TVR, TRExpr>
		,vector_calculus::OpCross>
{
	static const int NDIM = THREE;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;

	typedef BiOp<nTuple<NDIM, TVL, TLExpr> ,nTuple<NDIM, TVR, TRExpr>
			,vector_calculus::OpCross> ThisType;

	typedef nTuple<NDIM, ValueType, ThisType> ResultType;

	typedef nTuple<NDIM, TVL, TLExpr> TL;
	typedef nTuple<NDIM, TVR, TRExpr> TR;
	typedef typename nTuple<NDIM, TVL, TLExpr>::ConstReference LReference;
	typedef typename nTuple<NDIM, TVR, TRExpr>::ConstReference RReference;

	static inline ValueType op(TL const & lhs, TR const &rhs, size_t s)

	{
		return (lhs[(s + 1) % 3] * rhs[(s + 2) % 3]
				- lhs[(s + 2) % 3] * rhs[(s + 1) % 3]);
	}

};
template<int N, typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<nTuple<THREE, TVL, TL> ,nTuple<THREE, TVR, TR>
		,vector_calculus::OpCross>::ResultType //
Cross(nTuple<N, TVL, TL> const &lhs , nTuple<N, TVR, TR> const & rhs)
{
	typedef typename BiOp<nTuple<THREE, TVL, TL> ,nTuple<THREE, TVR, TR>
			,vector_calculus::OpCross>::ResultType ResultType;
	return (ResultType(lhs, rhs));
}

template<int N, typename TVL, typename TL, typename TVR, typename TR>
inline typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType //
Dot(nTuple<N, TVL, TL> const &lhs , nTuple<N, TVR, TR> const & rhs)
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	ValueType res = 0.0;
	for (int i = 0; i < N; ++i)
	{
		res += lhs[i] * rhs[i];
	}
	return (res);
}

template<typename TVL, typename TL, typename TVR, typename TR>
inline typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType //
Dot(nTuple<THREE, TVL, TL> const &lhs , nTuple<THREE, TVR, TR> const & rhs)
{
	return (lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]);
}

template<int N, typename TVL, typename TL, typename TVR, typename TR>
inline bool operator==(nTuple<N, TVL, TL> const &lhs
		, nTuple<N, TVR, TR> const & rhs)
{
	bool res = true;
	for (int i = 0; i < N; ++i)
	{
		res &= (lhs[i] == rhs[i]);
	}
	return (res);
}

template<typename TVL, typename TL, typename TVR, typename TR>
inline bool operator==(nTuple<THREE, TVL, TL> const &lhs
		, nTuple<THREE, TVR, TR> const & rhs)
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
	for(int i=0; i<N && is; ++i )
	{
		is >> tv[i];
	}

	return (is);
}

template<int N,typename T> nTuple<N,T>
ToNTuple(std::string const & str)
{
	std::istringstream ss(str);
	nTuple<N,T> res;
	ss>>res;
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
	* m[3][0] - m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3] //
	* m[2][2] * m[3][0] + m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1] //
	* m[1][2] * m[2][3] * m[3][0] - m[0][3] * m[1][2] * m[2][0] * m[3][1] //
	+ m[0][2] * m[1][3] * m[2][0] * m[3][1] + m[0][3] * m[1][0] * m[2][2] //
	* m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] - m[0][2] * m[1][0] //
	* m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1] + m[0][3] //
	* m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2] //
	- m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1] //
	* m[3][2] + m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1] //
	* m[2][3] * m[3][2] - m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1] //
	* m[1][2] * m[2][0] * m[3][3] + m[0][2] * m[1][0] * m[2][1] * m[3][3] //
	- m[0][0] * m[1][2] * m[2][1] * m[3][3] - m[0][1] * m[1][0] * m[2][2] //
	* m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3] //
	);
}

} //namespace simpla

#endif  // INCLUDE_NTUPLE_H_
