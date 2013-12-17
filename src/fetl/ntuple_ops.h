/*
 * ntuple_ops.
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#ifndef NTUPLE_OPS_H_
#define NTUPLE_OPS_H_

#include "primitives.h"
#include <cstddef>
#include <sstream>
#include <string>
//#include "constant_ops.h"
//#include <utility>

namespace simpla
{
// Expression template of nTuple
template<int, typename > struct nTuple;

template<typename T>
struct nTupleTraits
{
	static const int NUM_OF_DIMS = 1;
	typedef T value_type;
};
template<int N, typename T>
struct nTupleTraits<nTuple<N, T>>
{
	static const int NUM_OF_DIMS = N;
	typedef T value_type;
};
//***********************************************************************************

namespace ntuple_impl
{

template<int N, typename T, typename TI>
inline auto OpEval(Int2Type<NEGATE>, nTuple<N, T> const & l, TI const & s)
DECL_RET_TYPE ((-l[s] ))

}  // namespace ntuple_impl

template<int N, typename TL> inline
auto operator-(nTuple<N, TL> const & f)
DECL_RET_TYPE(( nTuple<N, UniOp<NEGATE,nTuple<N, TL> > > (f)))
//***********************************************************************************
template<int N, typename TL> inline
auto operator+(nTuple<N, TL> const & f)
DECL_RET_TYPE(f)
//***********************************************************************************
namespace ntuple_impl
{

template<int M, typename TL, typename TR> struct _inner_product_s;

template<typename TL, typename TR>
inline auto _inner_product(TL const & l, TR const &r)
DECL_RET_TYPE((l*r))

template<int N, typename TL, typename TR>
inline auto _inner_product(nTuple<N, TL> const & l, nTuple<N, TR> const &r)
DECL_RET_TYPE(( _inner_product_s<N, nTuple<N, TL>, nTuple<N, TR> >::eval(l,r)))

template<int M, typename TL, typename TR>
struct _inner_product_s
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE((_inner_product(l[M - 1] , r[M - 1]) + _inner_product_s<M - 1, TL, TR>::eval(l, r)))
};
template<typename TL, typename TR>
struct _inner_product_s<1, TL, TR>
{
	static inline auto eval(TL const & l, TR const &r)
	DECL_RET_TYPE(_inner_product(l[0],r[0]))
}
;

}
//namespace ntuple_impl
template<int N, typename TL, typename TR>
inline auto Dot(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
DECL_RET_TYPE((ntuple_impl::_inner_product(l,r)))
//***********************************************************************************
namespace ntuple_impl
{
template<int N, typename TL, typename TR>
inline auto OpEval(Int2Type<CROSS>, nTuple<N, TL> const & l, nTuple<N, TR> const &r, size_t s)
DECL_RET_TYPE ((l[(s+1)%3] * r[(s+2)%3] - l[(s+2)%3] * r[(s+1)%3]))
}  // namespace ntuple_impl

template<int N, typename TL, typename TR> inline auto Cross(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE( (nTuple<N,BiOp<CROSS, nTuple<N, TL>,nTuple<N, TR> > > (lhs, rhs)))

//***********************************************************************************
// overloading operators
namespace ntuple_impl
{
template<typename TL, typename TR>
inline auto OpEval(Int2Type<PLUS>, TL const & l, TR const &r, size_t s)
DECL_RET_TYPE(((l[s] + r[s])))

template<typename TL, typename TR>
inline auto OpEval(Int2Type<MINUS>, TL const & l, TR const &r, size_t s)
DECL_RET_TYPE(((l[s] - r[s])))

}  // namespace ntuple_impl

template<int N, typename TL, typename TR> inline auto //
operator +(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE(((nTuple<N, BiOp<PLUS ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))

template<int N, typename TL, typename TR> inline auto //
operator -(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE(((nTuple<N, BiOp<MINUS ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))

namespace ntuple_impl
{
template<int N, typename TL, typename TR> inline auto OpEval(Int2Type<MULTIPLIES>, nTuple<N, TL> const & l,
        TR const & r, size_t s)
        DECL_RET_TYPE((l[s] * r))

template<int N, typename TL, typename TR> inline auto OpEval(Int2Type<MULTIPLIES>, TL const & l,
        nTuple<N, TR> const & r, size_t s)
        DECL_RET_TYPE((l*r[s]))

template<int N, typename TL, typename TR> inline auto OpEval(Int2Type<MULTIPLIES>, nTuple<N, TL> const & l,
        nTuple<N, TR> const & r, size_t s)
        DECL_RET_TYPE((l[s]*r[s]))

template<int N, typename TL, typename TR> inline auto OpEval(Int2Type<DIVIDES>, nTuple<N, TL> const & l, TR const & r,
        size_t s)
        DECL_RET_TYPE((l[s] / r))

template<int N, typename TL, typename TR> inline auto OpEval(Int2Type<DIVIDES>, nTuple<N, TL> const & l,
        nTuple<N, TR> const & r, size_t s)
        DECL_RET_TYPE((l[s] / r[s]))

} // namespace ntuple_impl

template<int N, typename TL, typename TR> inline auto operator *(nTuple<N, TL> const & lhs, TR const & rhs)
DECL_RET_TYPE(((nTuple<N, BiOp<MULTIPLIES, nTuple<N, TL>, TR > >(lhs, rhs))))
template<int N, typename TL, typename TR> inline auto operator *(TL const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE(((nTuple<N, BiOp<MULTIPLIES, TL , nTuple<N, TR> > >(lhs, rhs))))
template<int N, typename TL, typename TR> inline auto operator *(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
DECL_RET_TYPE(((nTuple<N, BiOp<MULTIPLIES, nTuple<N,TL> , nTuple<N, TR> > >(lhs, rhs))))
template<int N, typename TL, typename TR> inline auto operator /(nTuple<N, TL> const & lhs, TR const &rhs)
DECL_RET_TYPE(((nTuple<N, BiOp<DIVIDES, nTuple<N, TL>, TR > >(lhs, rhs))))

template<int N, typename TL, typename TR> inline void //
operator *(nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)
{

}

//***********************************************************************************

template<int N, typename T>
struct ConstReferenceTraits<nTuple<N, T>>
{
	typedef nTuple<N, T> const &type;
};

template<int N, int TOP, typename TL, typename TR>
struct ConstReferenceTraits<nTuple<N, BiOp<TOP, TL, TR> > >
{
	typedef const nTuple<N, BiOp<TOP, TL, TR> > type;
};

template<int N, int TOP, typename TL>
struct ConstReferenceTraits<nTuple<N, UniOp<TOP, TL> > >
{
	typedef const nTuple<N, UniOp<TOP, TL> > type;
};

template<int N, int TOP, typename TL, typename TR>
struct nTuple<N, BiOp<TOP, TL, TR> >
{
	typename ConstReferenceTraits<TL>::type l_;
	typename ConstReferenceTraits<TR>::type r_;

	nTuple(TL const & l, TR const & r)
			: l_(l), r_(r)
	{
	}

	typedef decltype(ntuple_impl::OpEval(Int2Type<TOP>(),std::declval<TL>() ,std::declval<TR>(),0)) value_type;

	inline operator nTuple<N,value_type>() const
	{
		nTuple<N, value_type> res;
		for (int i = 0; i < N; ++i)
		{
			res[i] = this->operator[](i);
		}
		return res;
	}

	inline auto operator[](size_t s) const DECL_RET_TYPE((ntuple_impl::OpEval(Int2Type<TOP>(),l_,r_,s)))
};

template<int N, int TOP, typename TL>
struct nTuple<N, UniOp<TOP, TL> >
{
	typename ConstReferenceTraits<TL>::type l_;

	typedef decltype(ntuple_impl::OpEval(Int2Type<TOP>(),std::declval<TL>() ,size_t ())) value_type;

	nTuple(TL const & l)
			: l_(l)
	{
	}

	inline operator nTuple<N,value_type>() const
	{
		nTuple<N, value_type> res;
		for (int i = 0; i < N; ++i)
		{
			res[i] = this->operator[](i);
		}
		return res;

	}

	inline value_type operator[](size_t s) const
	{
		return ntuple_impl::OpEval(Int2Type<TOP>(), l_, s);
	}

};

//***********************************************************************************

//***********************************************************************************

//***********************************************************************************
//
//template<typename TL, typename TR>
//inline auto TensorProduct(TL const & l, TR const &r)
//DECL_RET_TYPE((l*r))
//
//template<int N, typename TL, typename TR>
//inline auto TensorProduct(TL const & l, nTuple<N, TR> const &r)
//DECL_RET_TYPE((nTuple<nTupleTraits<TL>::NUM_OF_DIMS*N, BiOp<TENSOR_PRODUCT, TL, nTuple<N, TR> > >(l,r) ))
//
//template<int N, typename TL, typename TR>
//inline auto TensorProduct(nTuple<N, TL> const & l, TR const &r)
//DECL_RET_TYPE((nTuple<N*nTupleTraits<TR>::NUM_OF_DIMS, BiOp<TENSOR_PRODUCT, nTuple<N, TL>,TR > >(l,r) ))
//
//template<int N, typename TL, int M, typename TR>
//inline auto TensorProduct(nTuple<N, TL> const & l, nTuple<M, TR> const &r)
//DECL_RET_TYPE((nTuple<N*M, BiOp<TENSOR_PRODUCT, nTuple<N, TL>,nTuple<M,TR> > >(l,r) ))
//
//template<int N, typename TL, typename TR>
//struct nTuple<N, BiOp<TENSOR_PRODUCT, TL, TR> >
//{
//	typedef TL left_type;
//	typedef TR right_type;
//	typename ConstReferenceTraits<left_type>::type l_;
//	typename ConstReferenceTraits<right_type>::type r_;
//	static const int M = nTupleTraits<right_type>::NUM_OF_DIMS;
//	nTuple(left_type const & l, right_type const & r)
//			: l_(l), r_(r)
//	{
//	}
//
//	inline auto operator[](size_t s) const
//	DECL_RET_TYPE ((TensorProduct(index(l_, s/M), index(r_, s%M))))
//	;
//
//}
//;
//template<typename TL, typename TR>
//inline auto TensorContraction(TL const &l, TR const &r)
//DECL_RET_TYPE((l*r))
//
//template<int N, typename TL, typename TR>
//inline auto TensorContraction(nTuple<N, TL> const &l, nTuple<N, TR> const &r)
//ENABLE_IF_DECL_RET_TYPE((nTupleTraits<TL>::NUM_OF_DIMS==1),
//		(ntuple_impl::_inner_product(l,r)))
//
//template<int N, int M, int P, typename TL, typename TR>
//inline auto TensorContraction(nTuple<N, nTuple<M, TL>> const & l, nTuple<P, TR> const &r)
//DECL_RET_TYPE(
//		(nTuple<N, BiOp<TENSOR_CONTRACTION,
//				nTuple<N, nTuple<M, TL>>, nTuple<P, TR> > >(l,r)))
//
//template<int N, int M, int P, typename TL, typename TR>
//struct nTuple<N, BiOp<TENSOR_CONTRACTION, nTuple<N, nTuple<M, TL>>, nTuple<P, TR> > >
//{
//	typedef nTuple<N, nTuple<M, TL>> left_type;
//	typedef nTuple<P, TR> right_type;
//	typename ConstReferenceTraits<left_type>::type l_;
//	typename ConstReferenceTraits<right_type>::type r_;
//	nTuple(left_type const & l, right_type const & r)
//			: l_(l), r_(r)
//	{
//	}
//
//	inline auto operator[](size_t s) const
//	DECL_RET_TYPE((TensorContraction(index(l_, s), r_)))
//	;
//
//}
//;
template<typename T> inline auto Determinant(nTuple<3, nTuple<3, T> > const & m)
DECL_RET_TYPE(
		(
				m[0][0] * m[1][1] * m[2][2] - m[0][2] * m[1][1] * m[2][0] + m[0][1] //
		* m[1][2] * m[2][0] - m[0][1] * m[1][0] * m[2][2] + m[1][0] * m[2][1]//
		* m[0][2] - m[1][2] * m[2][1] * m[0][0]//
)

)

template<typename T> inline auto Determinant(nTuple<4, nTuple<4, T> > const & m) DECL_RET_TYPE(
		(//
		m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1]
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

template<int N, typename T> auto abs(simpla::nTuple<N, T> const & m)
DECL_RET_TYPE((std::sqrt(std::abs(Dot(m, m)))))

}
// namespace simpla

#endif /* NTUPLE_OPS_H_ */
