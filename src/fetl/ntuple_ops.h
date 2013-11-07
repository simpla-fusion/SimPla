/*
 * ntuple_ops.h
 *
 *  Created on: 2013年11月7日
 *      Author: salmon
 */

#ifndef NTUPLE_OPS_H_
#define NTUPLE_OPS_H_

#include <fetl/ntuple.h>
#include <fetl/primitives.h>

namespace simpla
{
// Expression template of nTuple
#define _DEFINE_BINARY_OPERATOR(_NAME_,_OP_)                                                \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (nTuple<N, TL> const & lhs, nTuple<N, TR> const & rhs)                   \
DECL_RET_TYPE(                                                                     \
		(nTuple<N, BiOp<_NAME_ ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)))             \
                                                                                   \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (nTuple<N, TL> const & lhs, TL const & rhs)                              \
DECL_RET_TYPE((nTuple<N,BiOp<_NAME_ ,nTuple<N, TL>,TR > > (lhs,rhs)))                    \
                                                                                   \
template<int N, typename TL, typename TR> inline auto                              \
operator  _OP_ (TL const & lhs, nTuple<N, TR> const & rhs)                              \
DECL_RET_TYPE((nTuple<N,BiOp<_NAME_,TL,nTuple<N, TR> > > (lhs,rhs)))                    \


_DEFINE_BINARY_OPERATOR(PLUS, +)
_DEFINE_BINARY_OPERATOR(MINUS, -)
_DEFINE_BINARY_OPERATOR(MULTIPLIES, *)
_DEFINE_BINARY_OPERATOR(DIVIDES, /)
//_DEFINE_BINARY_OPERATOR(BITWISEXOR, ^)
//_DEFINE_BINARY_OPERATOR(BITWISEAND, &)
//_DEFINE_BINARY_OPERATOR(BITWISEOR, |)
//_DEFINE_BINARY_OPERATOR(MODULUS, %)

#undef _DEFINE_BINARY_OPERATOR

template<int N, typename TL, typename TR> inline auto operator +(
		nTuple<N, TL> const & lhs,
		nTuple<N, TR> const & rhs)
		->decltype(((nTuple<N, BiOp<PLUS ,nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs))))
{
	return ((nTuple<N, BiOp<PLUS, nTuple<N, TL>, nTuple<N, TR> > >(lhs, rhs)));
}

template<int N, typename TL, typename TR> inline auto operator +(
		nTuple<N, TL> const & lhs,
		TL const & rhs)
		->decltype(((nTuple<N,BiOp<PLUS ,nTuple<N, TL>,TR > > (lhs,rhs))))
{
	return ((nTuple<N, BiOp<PLUS, nTuple<N, TL>, TR> >(lhs, rhs)));
}

template<int N, typename TL, typename TR> inline auto operator +(TL const & lhs,
		nTuple<N, TR> const & rhs)
		->decltype(((nTuple<N,BiOp<PLUS,TL,nTuple<N, TR> > > (lhs,rhs))))
{
	return ((nTuple<N, BiOp<PLUS, TL, nTuple<N, TR> > >(lhs, rhs)));
}
}
// namespace simpla

#endif /* NTUPLE_OPS_H_ */
