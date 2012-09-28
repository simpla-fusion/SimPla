/*
 * differential_geometry.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef DIFFERENTIAL_GEOMETRY_H_
#define DIFFERENTIAL_GEOMETRY_H_
#include "fetl/detail/field_expr_common.h"
namespace simpla
{
namespace differential_geometry
{
template<typename TL, typename TR> class OpWedge;
template<typename TL, typename TR> class OpLieDerivative;
template<typename TL, typename TR> class OpInteriorProduct;
template<typename TL> struct OpExteriorDerivative;
template<typename TL> struct OpHodgeStar;

template<typename TL, typename TR>
inline Field<OpWedge<Field<TL>, Field<TR> > > Wedge(Field<TL> const &lhs,
		Field<TL> const & rhs)
{
	return (Field<OpWedge<Field<TL>, Field<TR> > >(lhs, rhs));
}

template<typename TL, typename TR>
inline Field<OpLieDerivative<TL, Field<TR> > > LieDerivative(TL const &lhs,
		Field<TL> const & rhs)
{
	return (Field<OpLieDerivative<TL, Field<TR> > >(lhs, rhs));
}

template<typename TL, typename TR>
inline Field<OpInteriorProduct<TL, Field<TR> > > InteriorProduct(TL const &lhs,
		Field<TL> const & rhs)
{
	return (Field<OpInteriorProduct<TL, Field<TR> > >(lhs, rhs));
}

template<typename TL>
inline Field<OpExteriorDerivative<Field<TL> > > ExteriorDerivative(
		Field<TL> const& r)
{
	return (Field<OpExteriorDerivative<Field<TL> > >(r));
}

template<typename TL>
inline Field<OpHodgeStar<Field<TL> > > HodgeStar(Field<TL> const& r)
{
	return (Field<OpHodgeStar<Field<TL> > >(r));
}

template<typename TL, typename TR>
inline Field<OpInteriorProduct<TL, Field<TR> > > i(TL const &lhs,
		Field<TL> const & rhs)
{
	return (Field<OpInteriorProduct<TL, Field<TR> > >(lhs, rhs));
}

template<typename TL>
inline Field<OpExteriorDerivative<Field<TL> > > d(Field<TL> const& r)
{
	return (Field<OpExteriorDerivative<Field<TL> > >(r));
}

template<typename TL, typename TR>
inline Field<OpWedge<Field<TL>, Field<TR> > > operator^(Field<TL> const &lhs,
		Field<TL> const & rhs)
{
	return (Field<OpWedge<Field<TL>, Field<TR> > >(lhs, rhs));
}
template<typename TL>
inline Field<OpHodgeStar<Field<TL> > > operator*(Field<TL> const& r)
{
	return (Field<OpHodgeStar<Field<TL> > >(r));
}

} // namespace differential_geometry
} // namespace simpla

#endif /* DIFFERENTIAL_GEOMETRY_H_ */
