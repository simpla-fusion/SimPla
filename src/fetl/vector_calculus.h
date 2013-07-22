/*
 *  _fetl_impl::vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_

#include "expression.h"

namespace simpla
{
template<typename, typename > class Field;
template<typename, int> class Geometry;
namespace space3
{ //3-dimensional space

struct OpGrad;
template<typename TG, typename TE> inline auto //
Grad(Field<Geometry<TG, 0>, TE> const & f)
->Field<Geometry<TG, 1>, BiOp<OpGrad,Field<Geometry<TG, 0>, TE> > >
{
	return (Field<Geometry<TG, 1>, BiOp<OpGrad, Field<Geometry<TG, 0>, TE> > >(
			f));
}
//DECL_RET_TYPE( (Field<Geometry<TG, 1>,
//				BiOp<OpGrad, Field<Geometry<TG, 0>, TE> ,NullType > > (f)))

struct OpDiverge;
template<typename TG, typename TE> inline auto //
Diverge(Field<Geometry<TG, 1>, TE> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 0>,
				BiOp<OpDiverge, Field<Geometry<TG, 1>,TE >,NullType > > (f)))

struct OpCurl;
template<typename TG, typename TE> inline auto //
Curl(Field<Geometry<TG, 2>, TE> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 1>,
				BiOp<OpCurl, Field<Geometry<TG, 2>, TE> ,NullType> > (f)))

template<typename TG, typename TE> inline auto //
Curl(Field<Geometry<TG, 1>, TE> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 2>,
				BiOp<OpCurl, Field<Geometry<TG, 1>, TE> ,NullType> > (f)))

template<int IPD> struct OpCurlPD;
template<int IPD, typename TG, typename TE> inline auto //
CurlPD(Int2Type<IPD>, Field<Geometry<TG, 1>, TE> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 2>,
				BiOp< OpCurlPD<IPD>, Field<Geometry<TG, 1>, TE> ,NullType>
				> (f)))

template<int IPD, typename TG, typename TE> inline auto //
CurlPD(Int2Type<IPD>, Field<Geometry<TG, 2>, TE> const & f)
DECL_RET_TYPE( (Field<Geometry<TG, 1>,
				BiOp<OpCurlPD<IPD>, Field<Geometry<TG, 2>, TE>,NullType> > (f)))

struct OpHodgeStar;
template<typename TG, int IFORM, typename TL> inline auto  //
operator*(
		Field<Geometry<TG, IFORM>, TL> const & lexpr)
				DECL_RET_TYPE(
						( Field<Geometry<TG, 3-IFORM>,
								BiOp<OpHodgeStar, Field<Geometry<TG,IFORM>, TL>,NullType > > (lexpr )))

struct OpWedge;
template<typename TG, int IL, int IR, typename TL, typename TR> inline auto //
operator^(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, IR>, TR> const & rhs)
		DECL_RET_TYPE(
				(Field<Geometry<TG,IL+IR> ,
						BiOp< OpWedge,
						Field<Geometry<TG,IL> , TL>,
						Field<Geometry<TG,IR> , TR> > >(lhs, rhs) ) )

template<typename TG, int IL, typename TL, typename TR> inline auto //
operator^(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, 4 - IL>, TR> const & rhs)
		DECL_RET_TYPE( (Field<Geometry<TG,0> ,Zero>() ) )

template<typename TG, int IL, typename TL, typename TR> inline auto //
operator^(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, 5 - IL>, TR> const & rhs)
		DECL_RET_TYPE( (Field<Geometry<TG,0> ,Zero>() ) )

template<typename TG, int IL, typename TL, typename TR> inline auto //
operator^(Field<Geometry<TG, IL>, TL> const & lhs,
		Field<Geometry<TG, 6 - IL>, TR> const & rhs)
		DECL_RET_TYPE( (Field<Geometry<TG,0> ,Zero>() ) )

struct OpExtriorDerivative;

template<typename TG, int IFORM, typename TL> inline auto  //
d(Field<Geometry<TG, IFORM>, TL> const & lexpr)
DECL_RET_TYPE( ( Field<Geometry<TG, IFORM+1>,
				BiOp<OpExtriorDerivative,
				Field<Geometry<TG,IFORM>, TL> > > (lexpr )))

template<typename TG, typename TL> inline auto  //
d(Field<Geometry<TG, 3>, TL> const &)
DECL_RET_TYPE( ( Field<Geometry<TG, 0>,Zero > ()))
}
// namespace space3

}
// namespace simpla
#endif /* VECTOR_CALCULUS_H_ */
