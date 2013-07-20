/*
 *  _fetl_impl::vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_
//#include "fetl_defs.h"
#include "primitives/expression.h"
namespace simpla
{

template<typename, typename > class Field;

struct OpGrad;
template<typename TG, typename TE> inline auto //
Grad(Field<ZeroForm<TG>, TE> const & lhs)
DECL_RET_TYPE( (Field<OneForm<TG>,
				UniOp<OpGrad, Field<ZeroForm<TG>, TE> > > (lhs)))
struct OpDiverge;
template<typename TG, typename TE> inline auto //
Diverge(Field<OneForm<TG>, TE> const & lhs)
DECL_RET_TYPE( (Field<ZeroForm<TG>,
				UniOp<OpDiverge, Field<OneForm<TG>, TE> > > (lhs)))

struct OpCurl;
template<typename TG, typename TE> inline auto //
Curl(Field<TwoForm<TG>, TE> const & lhs)
DECL_RET_TYPE( (Field<OneForm<TG>,
				UniOp<OpCurl, Field<TwoForm<TG>, TE> > > (lhs)))

template<typename TG, typename TE> inline auto //
Curl(Field<OneForm<TG>, TE> const & lhs)
DECL_RET_TYPE( (Field<TwoForm<TG>,
				UniOp<OpCurl, Field<OneForm<TG>, TE> > > (lhs)))

struct OpCurlPD;

template<int IPD, typename TG, typename TE> inline auto //
CurlPD(Int2Type<IPD>,
		Field<OneForm<TG>, TE> const & rhs)
				DECL_RET_TYPE( (Field<TwoForm<TG>,
								BiOp<OpCurlPD,Int2Type<IPD>, Field<OneForm<TG>, TE> > > (Int2Type<IPD>(),rhs)))

template<int IPD, typename TG, typename TE> inline auto //
CurlPD(Int2Type<IPD>,
		Field<TwoForm<TG>, TE> const & rhs)
				DECL_RET_TYPE( (Field<OneForm<TG>,
								BiOp<OpCurlPD,Int2Type<IPD>, Field<TwoForm<TG>, TE> > > (Int2Type<IPD>(),rhs)))


//
//template<typename TG, typename TLExpr, typename TRExpr>
//inline auto  //
//Dot(Field<TG, TL> const & lhs, Field<ZeroForm<TG>, TL> const & rhs)
//{
//	return (Field<TG, IZeroForm,
//			_impl::OpDot<Field<TG, IZeroForm, TLExpr>,
//					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
//}
//
//template<typename TG, int N, typename TLExpr, typename TRExpr>
//inline Field<TG, IZeroForm,
//		_impl::OpDot<nTuple<N, TLExpr>, Field<TG, IZeroForm, TRExpr> > >      //
//Dot(nTuple<N, TLExpr> const & lhs, Field<TG, IZeroForm, TRExpr> const &rhs)
//{
//
//	return (Field<TG, IZeroForm,
//			_impl::OpDot<nTuple<N, TLExpr>, Field<TG, IZeroForm, TRExpr> > >(
//			lhs, rhs));
//}
//
//template<typename TG, int N, typename TLExpr, typename TRExpr>
//inline Field<TG, IZeroForm,
//		_impl::OpDot<Field<TG, IZeroForm, TLExpr>, nTuple<N, TRExpr> > >      //
//Dot(Field<TG, IZeroForm, TLExpr> const & lhs, nTuple<N, TRExpr> const & rhs)
//{
//
//	return (Field<TG, IZeroForm,
//			_impl::OpDot<Field<TG, IZeroForm, TLExpr>, nTuple<N, TRExpr> > >(
//			lhs, rhs));
//}
//
//template<typename TG, typename TLExpr, typename TRExpr>
//inline Field<TG, IZeroForm,
//		_impl::OpCross<Field<TG, IZeroForm, TLExpr>,
//				Field<TG, IZeroForm, TRExpr> > >                              //
//Cross(Field<TG, IZeroForm, TLExpr> const & lhs,
//		Field<TG, IZeroForm, TRExpr> const & rhs)
//{
//	return (Field<TG, IZeroForm,
//			_impl::OpCross<Field<TG, IZeroForm, TLExpr>,
//					Field<TG, IZeroForm, TRExpr> > >(lhs, rhs));
//}
//
//template<typename TG, typename TLExpr, typename TRExpr>
//inline Field<TG, IZeroForm,
//		_impl::OpCross<nTuple<THREE, TLExpr>, Field<TG, IZeroForm, TRExpr> > >  //
//Cross(nTuple<THREE, TLExpr> const & lhs,
//		Field<TG, IZeroForm, TRExpr> const &rhs)
//{
//	return (Field<TG, IZeroForm,
//			_impl::OpCross<nTuple<THREE, TLExpr>, Field<TG, IZeroForm, TRExpr> > >(
//			lhs, rhs));
//}
//
//template<typename TG, typename TLExpr, typename TRExpr>
//inline Field<TG, IZeroForm,
//		_impl::OpCross<Field<TG, IZeroForm, TLExpr>, nTuple<THREE, TRExpr> > >  //
//Cross(Field<TG, IZeroForm, TLExpr> const & lhs,
//		nTuple<THREE, TRExpr> const & rhs)
//{
//	return (Field<TG, IZeroForm,
//			_impl::OpCross<Field<TG, IZeroForm, TLExpr>, nTuple<THREE, TRExpr> > >(
//			lhs, rhs));
//}

}        // namespace simpla
#endif /* VECTOR_CALCULUS_H_ */
