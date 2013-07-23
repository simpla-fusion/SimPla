/*
 *  _fetl_impl::vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_

#include "expression.h"
#include "field.h"
#include "geometry.h"

namespace simpla
{

#define DEF_UNI_FIELDOP(_NAME_)                                                  \
struct Op##_NAME_; \
template<typename TR> inline auto                               \
_NAME_(TR const & f) DECL_RET_TYPE((UniOp<Op##_NAME_, TR>(f)))

DEF_UNI_FIELDOP(Grad)
DEF_UNI_FIELDOP(Curl)
DEF_UNI_FIELDOP(Diverge)
#undef DEF_UNI_FIELDOP

template<int> struct OpCurlPD;
template<int IPD, typename TR> inline auto                //
CurlPD(Int2Type<IPD>, TR const & f)
DECL_RET_TYPE((UniOp<OpCurlPD<IPD>, TR>(f)))

struct OpHodgeStar;
template<typename TL> inline typename std::enable_if<is_Field<TL>::value,
		UniOp<OpHodgeStar, TL> >::type                //
operator*(TL const & f)
{
	return (UniOp<OpHodgeStar, TL>(f));
}

struct OpExtriorDerivative;
template<typename TL> inline auto d(TL const &f)
->typename std::enable_if<is_Field<TL>::value,
typename std::conditional< order_of_form<TL>::value >=0 &&
(order_of_form<TL>::value < TL::Geometry::NUM_OF_DIMS ),
UniOp<OpExtriorDerivative, TL>, Zero>::type
>::type
{
	return (typename std::conditional<
			order_of_form<TL>::value >= 0
					&& (order_of_form<TL>::value < TL::Geometry::NUM_OF_DIMS),
			UniOp<OpExtriorDerivative, TL>, Zero>::type(f));
}

struct OpWedge;
template<typename TL, typename TR> inline auto             //
operator^(TL const & lhs,
		TR const & rhs)
		->typename std::enable_if<
		is_Field<TL>::value && is_Field<TR>::value,
		typename std::conditional<
		(order_of_form<TL>::value + order_of_form<TR>::value )>=0
		&& ((order_of_form<TL>::value + order_of_form<TR>::value ) < TL::Geometry::NUM_OF_DIMS ),
		UniOp<OpExtriorDerivative, TL>,
		Zero>::type>::type
{
	return ((typename std::conditional<
			(order_of_form<TL>::value + order_of_form<TR>::value) >= 0
					&& ((order_of_form<TL>::value + order_of_form<TR>::value)
							< TL::Geometry::NUM_OF_DIMS),
			UniOp<OpExtriorDerivative, TL>, Zero>::type(lhs, rhs)));
}

template<typename TF>
struct order_of_form<UniOp<OpDiverge, TF> >
{
	static const int value = (order_of_form<TF>::value == 1 ? 0 : 0);
};
template<typename TF>
struct order_of_form<UniOp<OpGrad, TF> >
{
	static const int value = (order_of_form<TF>::value == 0 ? 1 : 0);
};
template<typename TF>
struct order_of_form<UniOp<OpCurl, TF> >
{
	static const int value = (order_of_form<TF>::value == 1 ? 2 : 1);
};

} // namespace simpla
#endif /* VECTOR_CALCULUS_H_ */
