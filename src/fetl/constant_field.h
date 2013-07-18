/*
 * constant_field.h
 *
 *  Created on: 2012-3-15
 *      Author: salmon
 */

#ifndef CONSTANT_FIELD_H_
#define CONSTANT_FIELD_H_
#include "include/simpla_defs.h"
#include "fetl/fetl_defs.h"
namespace simpla
{

template<int IFORM, typename TV>
struct Field<IFORM, TV, Int2Type<0> >
{
};
template<typename TV>
struct Field<IZeroForm, TV, Int2Type<1> >
{
};

template<int IFORM, typename TVL, typename TLExpr, typename TVR>
Field<IFORM, TVL, TLExpr> const & //
operator +(Field<IFORM, TVL, TLExpr> const & lhs
		,Field<IFORM, TVR, Int2Type<0> > const &)
{
	return (lhs);
}

template<int IFORM, typename TVL, typename TVR, typename TRExpr>
Field<IFORM, TVR, TRExpr> const & //
operator +(Field<IFORM, TVL, Int2Type<0> > const &
		,Field<IFORM, TVR, TRExpr> const & rhs)
{
	return (rhs);
}

template<int IFORM, typename TVL, typename TVR, typename TRExpr> auto //
operator -(Field<IFORM, TVL, Int2Type<0> > const &
		,Field<IFORM, TVR, TRExpr> const & rhs)->decltype(-rhs)
{
	return (-rhs);
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR>
Field<IFORM, TVL, Int2Type<0> >  //
operator *(Field<IFORM, TVL, TLExpr> const & lhs
		,Field<IFORM, TVR, Int2Type<0> > const &)
{
	return (Field<IFORM, TVL, Int2Type<0> >());
}

template<int IFORM, typename TVL, typename TVR, typename TRExpr>
Field<IFORM, TVR, Int2Type<0> >  //
operator *(Field<IFORM, TVL, Int2Type<0>> const &
		,Field<IFORM, TVR, TRExpr> const &)
{
	return (Field<IFORM, TVR, Int2Type<0> >());
}

template<int IFORM, typename TVL, typename TVR, typename TRExpr>
Field<IFORM, TVR, Int2Type<0> >  //
operator /(Field<IFORM, TVL, Int2Type<0>> const &
		,Field<IFORM, TVR, TRExpr> const &)
{
	return (Field<IFORM, TVR, Int2Type<0> >());
}

} // namespace simpla

#endif /* CONSTANT_FIELD_H_ */
